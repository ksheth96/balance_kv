import math

import torch
import transformers
from flash_attn import flash_attn_func
from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers.models.llama.modeling_llama import (apply_rotary_pos_emb,
                                                      repeat_kv)

from .balanced_walk import uniform_sampling, balanced_walk,balanced_walk_needle_detection
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple, Union, Any, Dict



def merge_cache(cache_a, cache_b, dim=2):
    assert len(cache_a) == len(cache_b)
    cache_ab = ()
    for a, b in zip(cache_a, cache_b):
        ab = torch.cat((a, b), dim)
        cache_ab += (ab,)
    return cache_ab


def random_compress(key_states, value_states, capacity_option='default', window_size=32, rng=None):

    seq_len = key_states.shape[2]
    if capacity_option == 'default':
        max_capacity_prompt = max(int(seq_len * 3.875 / 64), window_size + 4)

    key_buf = key_states[:, :, -window_size:, :]
    val_buf = value_states[:, :, -window_size:, :]

    b, h, _, dim = key_states.shape
    assert seq_len > window_size
    front_size = seq_len - window_size

    indices = torch.stack([torch.randperm(front_size, device=key_states.device)[
                          :max_capacity_prompt] for _ in range(b * h)])
    indices = indices.reshape(b, h, max_capacity_prompt, 1)
    # indices = torch.randint(front_size, (b, h, max_capacity_prompt, 1), device=key_states.device)
    indices = indices.expand(-1, -1, -1, dim)

    k_compress = key_states[:, :, :-window_size,
                            :].gather(dim=2, index=indices)
    v_compress = value_states[:, :, :-window_size,
                              :].gather(dim=2, index=indices)
    k_cur = key_states[:, :, -window_size:, :]
    v_cur = value_states[:, :, -window_size:, :]

    key_states = torch.cat([k_compress, k_cur], dim=2)
    value_states = torch.cat([v_compress, v_cur], dim=2)
    return key_states, value_states


def manual_forward_llama(
    model,
    input_ids,
    kv_type,
    attention_mask=None,
    kv_cache=None, position_ids=None, cache_position=None, num_logits_to_keep=0,
    return_hidden_states=False,
):
    hh = model.model.embed_tokens(input_ids)
    if position_ids is None:
        position_ids = torch.arange(
            len(input_ids[0]), device=input_ids.device).unsqueeze(0)

    if int(transformers.__version__.split(".")[1]) >= 48:
        position_embeddings = model.model.rotary_emb(hh, position_ids)

    num_kv_heads = model.config.num_key_value_heads
    head_dim = getattr(model.config, "head_dim", None) or (
        model.config.hidden_size // model.config.num_attention_heads)

    past_kv_cache = []
    for i, decoder_layer in enumerate(model.model.layers):
        # hh = decoder_layer(hh, position_ids=position_ids)[0]
        res = hh
        hh = decoder_layer.input_layernorm(hh)

        # h1, _, kv = decoder_layer.self_attn(hh, position_ids=position_ids, use_cache=False)
        # <===
        q_len = hh.shape[1]
        kv_len = q_len
        qq = decoder_layer.self_attn.q_proj(hh).reshape(
            1, q_len, -1, head_dim)
        kk = decoder_layer.self_attn.k_proj(hh).reshape(
            1, kv_len, num_kv_heads, head_dim)
        vv = decoder_layer.self_attn.v_proj(hh).reshape(
            1, kv_len, num_kv_heads, head_dim).transpose(1, 2)

        # Qwen3-style QK-norm applies RMSNorm on the head_dim axis before RoPE.
        if hasattr(decoder_layer.self_attn, 'q_norm'):
            qq = decoder_layer.self_attn.q_norm(qq)
            kk = decoder_layer.self_attn.k_norm(kk)
        qq = qq.transpose(1, 2)
        kk = kk.transpose(1, 2)

        if int(transformers.__version__.split(".")[1]) >= 48:
            cos, sin = position_embeddings
        else:
            cos, sin = decoder_layer.self_attn.rotary_emb(vv, position_ids)
        qq, kk = apply_rotary_pos_emb(qq, kk, cos, sin)
        d = qq.shape[-1]

        if q_len > 1:
            attn_output = flash_attn_func(qq.transpose(
                1, 2), kk.transpose(1, 2), vv.transpose(1, 2), causal=True)

        if kv_cache is None:
            if kv_type in ['exact', 'dense']:  # or i > layer_cutoff:
                key_quant = kk
                val_quant = vv

            elif kv_type in [ 'uniform', 'weightedbw']:
                rng = model.config.rng
                gamma = model.config.gamma
                temp = model.config.temp
                beta = model.config.beta
                itrs = model.config.itrs
                block_size = model.config.block_size
                window_size = model.config.window_size
                sink_size = model.config.sink_size

                k_compressed = kk[:, :, sink_size:-window_size]
                v_compressed = vv[:, :, sink_size:-window_size]

                if kv_type == 'weightedbw':
                    
                    indices, weights = balanced_walk(
                        k_compressed, rng, gamma, temp, beta, itrs, block_size, value=v_compressed)
                    
                elif kv_type == 'uniform':
                    indices = uniform_sampling(
                        k_compressed, rng, itrs, block_size)

                if indices is not None:
                    k_bw = k_compressed.gather(
                        dim=2, index=indices.unsqueeze(-1).expand(-1, -1, -1, kk.shape[-1]))
                    v_bw = v_compressed.gather(
                        dim=2, index=indices.unsqueeze(-1).expand(-1, -1, -1, vv.shape[-1]))

                if kv_type == 'weightedbw':
                    weights = weights / 2**(itrs)
                    weights = weights.unsqueeze(-1)
                    v_bw = (v_bw * weights).to(vv.dtype)

                key_quant = torch.cat(
                    (kk[:, :, :sink_size], k_bw, kk[:, :, -window_size:]), dim=2)
                val_quant = torch.cat(
                    (vv[:, :, :sink_size], v_bw, vv[:, :, -window_size:]), dim=2)

                model.config.compress_size = k_bw.shape[2]

            else:
                import pdb
                pdb.set_trace()

            past_kv_cache += [(key_quant, val_quant)]
        else:
            if kv_type in ['exact', 'dense']:  
                assert len(kv_cache) == model.config.num_hidden_layers
                kk = torch.cat((kv_cache[i][0], kk), 2)
                vv = torch.cat((kv_cache[i][1], vv), 2)
                past_kv_cache += [(kk, vv)]

            elif kv_type in ['weightedbw', 'uniform']:
                itrs = model.config.itrs
                itrs = model.config.itrs
                compress_size = model.config.compress_size
                sink_size = model.config.sink_size

                key_old = kv_cache[i][0]
                val_old = kv_cache[i][1]
                kk = torch.cat((key_old, kk), dim=2)
                vv = torch.cat((val_old, vv), dim=2)

                qk = qq @ repeat_kv(kk, qq.shape[1] //
                                    kk.shape[1]).transpose(-1, -2) / d**0.5
                bias = torch.zeros_like(qk)
                bias[:, :, :, sink_size:sink_size +
                     compress_size] = math.log(2**itrs)
                attn_output = ((qk + bias).softmax(dim=-1) @
                               repeat_kv(vv, qq.shape[1]//vv.shape[1])).transpose(1, 2)
                past_kv_cache += [(kk, vv)]

            else:
                import pdb
                pdb.set_trace()

            if kv_type in ['exact', 'dense']:  
                attn_output = flash_attn_func(qq.transpose(
                    1, 2), kk.transpose(1, 2), vv.transpose(1, 2), causal=True)
            # attn_output = (pkvq_key_all[i].dequant_query(qq)/d**0.5).softmax(-1) @ repeat_heads(vv, qq.shape[1] // vv.shape[1])

        attn_output = attn_output.contiguous().view(
            qq.shape[0], qq.shape[2], -1)
        hh = decoder_layer.self_attn.o_proj(attn_output)
        # ===>
        hh = res + hh

        res = hh
        hh = decoder_layer.post_attention_layernorm(hh)
        hh = decoder_layer.mlp(hh)
        hh = res + hh

    hidden_states = model.model.norm(hh)  # (bsz, seq_len, dim)
    logits = model.lm_head(hidden_states[:, -num_logits_to_keep:, :])
    return logits, past_kv_cache, attention_mask
def manual_forward_llama_needle_detection(
    model,
    input_ids,
    kv_type,
    attention_mask=None,
    kv_cache=None, position_ids=None, cache_position=None, num_logits_to_keep=0,
):
    hh = model.model.embed_tokens(input_ids)
    if position_ids is None:
        position_ids = torch.arange(len(input_ids[0]), device=input_ids.device).unsqueeze(0)
    if int(transformers.__version__.split(".")[1]) >= 48:
        position_embeddings = model.model.rotary_emb(hh, position_ids)

    num_kv_heads = model.config.num_key_value_heads
    head_dim = getattr(model.config, "head_dim", None) or (
        model.config.hidden_size // model.config.num_attention_heads)

    needle_mask = None
    past_kv_cache = []
    num_layers = len(model.model.layers)
    for i, decoder_layer in enumerate(model.model.layers):
        res = hh
        hh = decoder_layer.input_layernorm(hh)

        q_len = hh.shape[1]
        kv_len = q_len
        qq = decoder_layer.self_attn.q_proj(hh).reshape(1, q_len, -1, head_dim)
        kk = decoder_layer.self_attn.k_proj(hh).reshape(1, kv_len, num_kv_heads, head_dim)
        vv = decoder_layer.self_attn.v_proj(hh).reshape(1, kv_len, num_kv_heads, head_dim).transpose(1, 2)

        # Qwen3-style QK-norm applies RMSNorm on the head_dim axis before RoPE.
        if hasattr(decoder_layer.self_attn, 'q_norm'):
            qq = decoder_layer.self_attn.q_norm(qq)
            kk = decoder_layer.self_attn.k_norm(kk)
        qq = qq.transpose(1, 2)
        kk = kk.transpose(1, 2)

        if int(transformers.__version__.split(".")[1]) >= 48:
            cos, sin = position_embeddings
        else:
            cos, sin = decoder_layer.self_attn.rotary_emb(vv, position_ids)
        qq, kk = apply_rotary_pos_emb(qq, kk, cos, sin)
        d = qq.shape[-1]

        if q_len > 1:
            attn_output = flash_attn_func(qq.transpose(1,2), kk.transpose(1,2), vv.transpose(1,2), causal=True)
        if kv_cache is None:
            if kv_type in ['exact']:
                key_quant = kk
                val_quant = vv
            elif kv_type in ['weightedbw', 'uniform', 'bw', 'balancedwalk', 'balancedwalk_rew']:
                rng = model.config.rng
                gamma = model.config.gamma
                temp = model.config.temp
                beta = model.config.beta
                itrs = model.config.itrs
                block_size = model.config.block_size
                window_size = model.config.window_size
                sink_size = model.config.sink_size

                k_compressed = kk[:, :, sink_size:-window_size]
                v_compressed = vv[:, :, sink_size:-window_size]

                if i==1: 
                    if kv_type in ['weightedbw', 'bw', 'balancedwalk']:
                        indices, weights, needle_mask = balanced_walk_needle_detection(k_compressed, rng, gamma, temp, beta, itrs, block_size, layer = i, value=v_compressed)
                    elif kv_type == 'uniform':
                        indices, weights, needle_mask = balanced_walk_needle_detection(k_compressed, rng, 0.0, temp, beta, itrs, block_size, layer = i, value=v_compressed)
                elif i > 1:
                    if kv_type in ['weightedbw', 'bw', 'balancedwalk']:
                        indices, weights, _ = balanced_walk_needle_detection(k_compressed, rng, gamma, temp, beta, itrs, block_size, layer = i, needle_mask = needle_mask, value=v_compressed)
                    elif kv_type == 'uniform':
                        indices, weights, _ = balanced_walk_needle_detection(k_compressed, rng, 0.0, temp, beta, itrs, block_size, layer = i, needle_mask = needle_mask, value=v_compressed)
                else:
                    if kv_type in ['weightedbw', 'bw', 'balancedwalk']:
                        indices, weights, _ = balanced_walk_needle_detection(k_compressed, rng, gamma, temp, beta, itrs, block_size, layer = i, value=v_compressed)
                    elif kv_type == 'uniform':
                        indices, weights, _ = balanced_walk_needle_detection(k_compressed, rng, 0.0, temp, beta, itrs, block_size, layer = i, value=v_compressed)

                k_bw = k_compressed.gather(dim=2, index=indices.unsqueeze(-1).expand(-1,-1,-1,kk.shape[-1]))
                v_bw = v_compressed.gather(dim=2, index=indices.unsqueeze(-1).expand(-1,-1,-1,vv.shape[-1]))

                if kv_type in ['weightedbw', 'bw', 'balancedwalk', 'uniform']:
                    if weights != None:
                      weights_zeros = weights > 0
                      weights_zeros = weights_zeros.unsqueeze(-1)
                      v_bw_num = v_bw*weights_zeros
                      v_bw_num = (v_bw_num).to(torch.bfloat16)
                    else:
                      v_bw_num = v_bw
                weights = weights.unsqueeze(-1)
                log_weights = torch.where(weights > 0, torch.log(weights), torch.full_like(weights, -1e9))
                    
                key_quant = torch.cat((kk[:,:,:sink_size], k_bw, kk[:, :, -window_size:]), dim=2)
                val_quant = torch.cat((vv[:,:,:sink_size], v_bw_num, vv[:, :, -window_size:]), dim=2)

                model.config.compress_size = k_bw.shape[2]
            else:
                import pdb; pdb.set_trace();
            past_kv_cache += [(key_quant, val_quant, log_weights)]
        else: 
            if kv_type in ['exact']:
                kk = torch.cat((kv_cache[i][0], kk), 2)
                vv = torch.cat((kv_cache[i][1], vv), 2)
                past_kv_cache += [(kk, vv)]

            elif kv_type in ['weightedbw', 'bw', 'balancedwalk', 'uniform']:
                itrs = model.config.itrs
                compress_size = model.config.compress_size
                sink_size = model.config.sink_size

                key_old = kv_cache[i][0]
                val_old = kv_cache[i][1]
                kk = torch.cat((key_old, kk), dim=2)
                vv = torch.cat((val_old, vv), dim=2)
                needle = kv_cache[i][2]
                needle_used = needle
                needle_used = needle_used.repeat_interleave(qq.shape[1]//kk.shape[1], dim=1).transpose(-1, -2)
                needle_used = needle_used.unsqueeze(0)

                qk = qq @ repeat_kv(kk, qq.shape[1]//kk.shape[1]).transpose(-1,-2) / d**0.5
                bias = torch.zeros_like(qk)
                bias[:, :, :, sink_size:sink_size+compress_size] = needle_used
                attn_output = ((qk + bias).softmax(dim=-1) @ repeat_kv(vv, qq.shape[1]//vv.shape[1])).transpose(1,2)
                past_kv_cache += [(kk, vv, needle)]


            if kv_type in ['exact']:
                attn_output = flash_attn_func(qq.transpose(1,2), kk.transpose(1,2), vv.transpose(1,2), causal=True)


        attn_output = attn_output.contiguous().view(qq.shape[0], qq.shape[2], -1)
        hh = decoder_layer.self_attn.o_proj(attn_output)

        hh = res + hh

        res = hh
        hh = decoder_layer.post_attention_layernorm(hh)
        hh = decoder_layer.mlp(hh)
        hh = res + hh

    hidden_states = model.model.norm(hh) 
    logits = model.lm_head(hidden_states[:, -num_logits_to_keep:, :])
    return logits, past_kv_cache, attention_mask


@torch.no_grad()
def greedy_generate(self, input_ids, max_new_tokens, kv_type, eos_token_id=[128009], return_dict_in_generate=False,needle_detection = False):
    position_ids = torch.arange(input_ids.shape[-1], device=input_ids.device).unsqueeze(0)
    if needle_detection:
        logits, cache, attention_mask = manual_forward_llama_needle_detection(self, input_ids, position_ids=position_ids, num_logits_to_keep=1, kv_type=kv_type)
    else:
        logits, cache, attention_mask = manual_forward_llama(self, input_ids, position_ids=position_ids, num_logits_to_keep=1, kv_type=kv_type)
    pred_token_idx = logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids = [pred_token_idx.item()]
    for i in range(max_new_tokens - 1):
        position_ids = torch.tensor(input_ids.shape[-1] + i, device=input_ids.device).reshape(1,1)
        if needle_detection: 
            logits, cache, _ = manual_forward_llama_needle_detection(self, pred_token_idx, position_ids=position_ids, attention_mask=attention_mask, kv_cache=cache, num_logits_to_keep=1, kv_type=kv_type)
        else:
            logits, cache, _ = manual_forward_llama(self, pred_token_idx, position_ids=position_ids,attention_mask=attention_mask, kv_cache=cache, num_logits_to_keep=1, kv_type=kv_type)
        pred_token_idx = logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids.append(pred_token_idx.item())
        if pred_token_idx in eos_token_id:
            break
    sequences = torch.tensor(input_ids[0].tolist() + generated_ids, device=input_ids.device).unsqueeze(0)
    if not return_dict_in_generate:
        return sequences
    return GenerateDecoderOnlyOutput(sequences=sequences, past_key_values=cache)
