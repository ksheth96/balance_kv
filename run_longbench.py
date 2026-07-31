import argparse
import datetime
import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from datasets import load_dataset
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer)

from src.llama_forward import greedy_generate
from src.metrics_longbench import (classification_score, code_sim_score,
                               count_score, qa_f1_score, qa_f1_zh_score,
                               retrieval_score, retrieval_zh_score,
                               rouge_score, rouge_zh_score)


dataset2prompt = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
    "passage_retrieval_zh": "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n答案是：",
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n"
}

dataset2maxlen = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "vcsum": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench-p": 64
}

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

BASELINE_SCORES = {
    'llama': {'longbench_v1_e': {'qasper': 0.42873338776034736, 'multifieldqa_en': 0.4854425606702108, 'hotpotqa': 0.5204788239428282, '2wikimqa': 0.38597740618514614, 'gov_report': 0.3130745811204385, 'multi_news': 0.22071233591968817, 'trec': 0.7166666666666667, 'triviaqa': 0.9184636591478696, 'samsum': 0.42360403153035575, 'passage_count': 0.20370370370370372, 'passage_retrieval_en': 0.9813164983164984, 'lcc': 0.49616666666666664, 'repobench-p': 0.4273}},
    'ministral': {'longbench_v1_e': {'qasper': 0.46835216476509606, 'multifieldqa_en': 0.560156533351397, 'hotpotqa': 0.6606288824121642, '2wikimqa': 0.4792426619132501, 'gov_report': 0.28389810170659513, 'multi_news': 0.2384837486956865, 'trec': 0.71, 'triviaqa': 0.9238518518518519, 'samsum': 0.4407199486739817, 'passage_count': 0.16, 'passage_retrieval_en': 1.0, 'lcc': 0.5410333333333334, 'repobench-p': 0.5494666666666667}}
}


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def truncate_input(input: list, max_length: int, manner="middle"):
    if max_length < 0:
        return input
    if len(input) <= max_length:
        return input
    if manner == "middle":
        split = max_length // 2
        return input[0:split] + input[-split:]
    else:
        return None


def truncate_by_tokens(input, tok, max_tokens, manner: str = "middle"):
    if type(input[0]) == int:  # .dtype in [torch.int32, torch.int64]:
        tokens = input
    else:
        tokens = tok.encode(input)
    len_before = len(tokens)
    # print(f"# tokens before: {len_before}")
    tokens = truncate_input(tokens, max_length=max_tokens, manner=manner)
    len_after = len(tokens)  # type: ignore
    # print(f"# tokens after: {len_after}")
    assert len_after <= len_before
    assert len_after <= max_tokens or max_tokens < 0
    return tokens


def compute_score(prediction, ground_truths, all_classes, dataset):
    score = 0.
    if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
        prediction = prediction.lstrip('\n').split('\n')[0]
    for ground_truth in ground_truths:
        score = max(score, dataset2metric[dataset](
            prediction, ground_truth, all_classes=all_classes))
    return score


def dump_jsonl(data, fname):
    with open(fname, "w", encoding="utf8") as fout:
        for line in data:
            fout.write(json.dumps(line, ensure_ascii=False) + "\n")


def reset_logging():
    loggers = [logging.getLogger(name)
               for name in logging.root.manager.loggerDict]
    loggers.append(logging.getLogger())
    for logger in loggers:
        handlers = logger.handlers[:]
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
        logger.setLevel(logging.NOTSET)
        logger.propagate = True


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        default="/home/ma-user/work/bucket-wulan-green/zhaoyusheng/checkpoints/Qwen/Qwen3-1.7B-Base/")
    # parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    # parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-14B-Instruct")
    # "THUDM/chatglm3-6b-32k")
    #
    # parser.add_argument("--output_dir", type=str, default="./result_longbench_v1", help="The directory of data.")
    parser.add_argument('--kv_type', type=str, default="exact")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prefix', type=str, default='new_')
    parser.add_argument('--datasets', type=str, default='triviaqa')
    parser.add_argument('--debug', action='store_true',
                        help="Evaluate on LongBench-E")
    parser.add_argument('--print_pred', action='store_true',
                        help="Evaluate on LongBench-E")
    parser.add_argument('--opt_template', action='store_true',
                        help="use optimized template")
    parser.add_argument('--e', action='store_true',
                        help="Evaluate on LongBench-E")

    parser.add_argument("--group_size", type=int, default=32, help="for KIVI")
    parser.add_argument("--bits", type=int, default=2,
                        choices=[1, 2, 3, 4], help="for KIVI")
    parser.add_argument("--residual_length", type=int,
                        default=32, help="for KIVI")

    parser.add_argument("--axis_key", type=int, default=0, help="for KIVI")
    parser.add_argument("--axis_value", type=int, default=0, help="for KIVI")

    parser.add_argument("--sink_size", type=int,
                        default=32, help="for StreamingLLM")
    parser.add_argument("--window_size", type=int,
                        default=32, help="for SnapKV,PyramidKV")
    parser.add_argument("--max_capacity_prompt", type=int,
                        default=4096, help="for SnapKV,PyramidKV")
    parser.add_argument("--kernel_size", type=int,
                        default=5, help="for SnapKV,PyramidKV")
    parser.add_argument("--pooling", type=str,
                        default="avgpool", help="for SnapKV,PyramidKV")

    parser.add_argument("--headkv_method", type=str,
                        default='adaptive', help="for HeadKV")
    parser.add_argument("--base_capacity", type=int,
                        default=128, help="for HeadKV")
    parser.add_argument("--block_size", type=int,
                        default=128, help="for HeadKV")
    parser.add_argument("--itrs", type=int, default=2, help="for BalancedWalk")
    parser.add_argument("--beta", type=float, default=0.0,
                        help="for BalancedWalk")
    parser.add_argument("--temp", type=float, default=1.0,
                        help="for BalancedWalk")
    parser.add_argument("--gamma", type=float, default=4.0,
                        help="for BalancedWalk")
    parser.add_argument("--floor", type=float, default=0.2, help="for HeadKV")
    parser.add_argument("--head_choice", type=str,
                        default='random', help="for HeadKV")

    parser.add_argument("--g", type=int, default=0, help="for Kernel Halving")

    return parser.parse_args(args)


def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)


def main():
    args = parse_args()
    seed_everything(args.seed)

    # change simple model name to full model name
    if args.model_name == 'llama':
        args.model_name = "meta-llama/Llama-3.1-8B-Instruct"
    elif args.model_name == 'mistral':
        args.model_name = "mistralai/Ministral-8B-Instruct-2410"

    print(f"model_name: {args.model_name}")

    model_name = args.model_name
    real_model_name = model_name.rstrip("/").split("/")[-1]

    exp_name = f"{real_model_name}_{args.kv_type}"
    if args.kv_type == 'kivi':
        exp_name += f"_g{args.group_size}_b{args.bits}_r{args.residual_length}"
    elif args.kv_type in ['snapkv', 'pyramidkv']:
        exp_name += f"_w{args.window_size}_nlinear_k{args.kernel_size}_p{args.pooling}"
    elif args.kv_type in ['weightedbw']:
        exp_name += f"_itr{args.itrs}_g{args.gamma}_t{args.temp}_b{args.block_size}_s{args.sink_size}_r{args.window_size}"
    elif args.kv_type in ['uniform']:
        exp_name += f"_itr{args.itrs}_b{args.block_size}_s{args.sink_size}_r{args.window_size}"
    elif args.kv_type in ['streamingllm', 'sink']:
        exp_name += f"_w{args.window_size}"
    print(f"exp_name: {exp_name}")

    datasets = args.datasets
    if datasets == 'all':
        if args.e:
            datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news",
                        "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
        else:
            datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique",
                        "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht",
                        "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
    elif type(datasets) == str:
        datasets = [datasets]
    else:
        import pdb
        pdb.set_trace()

    for i, dataset in enumerate(datasets):
        print(f"{i} of {len(datasets)}, dataset: {dataset}")

        # 0. set up logging
        datestr = datetime.datetime.now().strftime('%y%m%d%H%M')
        if not args.debug:
            result_dir = Path(os.path.join(
                f"./{args.prefix}results/longbench_v1" if not args.e else f"./{args.prefix}results/longbench_v1_e",
                exp_name))
            result_dir.mkdir(exist_ok=True, parents=True)
            output_path = result_dir / f"prediction_{dataset}_{datestr}.jsonl"
            log_path = os.path.join(
                f"{args.prefix}logs/longbench_v1" if not args.e else f"{args.prefix}logs/longbench_v1_e",
                exp_name)
            if not os.path.exists(log_path):
                os.makedirs(log_path)
            log_path = os.path.join(log_path, f"{dataset}_{datestr}.txt")
            # Configure the logger
            logging.basicConfig(
                level=logging.INFO,
                format=f'[%(asctime)s]{dataset}|{args.kv_type}| %(message)s',
                datefmt='%y%m%d %H:%M:%S',
                handlers=[logging.FileHandler(
                    log_path), logging.StreamHandler()],
            )
        else:
            logging.basicConfig(
                level=logging.INFO,
                format=f'[%(asctime)s]{dataset}|{args.kv_type}| %(message)s',
                datefmt='%y%m%d %H:%M:%S',
                handlers=[logging.StreamHandler()],
            )
            output_path = ""

        logger = logging.getLogger(__name__)
        logger.info(f"output_path: {output_path}")
        logger.info(f"exp_name: {exp_name}")

        for n_, v_ in args.__dict__.items():
            logger.info(f"{n_:<20} : {v_}")

        # 1. load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map='auto', trust_remote_code=True, torch_dtype=torch.bfloat16, _attn_implementation='flash_attention_2')
        model = model.eval().requires_grad_(False)
        # if 'gemm' in model_name.lower():
        #     # disable sliding_window
        #     for i in range(len(model.model.layers)):
        #         model.model.layers[i].is_sliding = False
        #         model.model.layers[i].self_attn.is_sliding = False
        if 'qwen' in model_name.lower():
            model._supports_num_logits_to_keep = model._supports_logits_to_keep
            if getattr(model.config, "head_dim", None) is None:
                model.config.head_dim = model.config.hidden_size // model.config.num_attention_heads

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True)
        if 'llama' in model_name or 'mistral' in model_name.lower():
            tokenizer.pad_token_id = tokenizer.eos_token_id
        # 2. patch model
        if args.kv_type == 'exact':
            pass

        elif args.kv_type in ['weightedbw',  'uniform']:
            logger.info("patching balanced walk KV cache ...")
            # from llama.llama_balanced_walk import greedy_generate
            # LlamaForCausalLM.generate = greedy_generate
            model.model.config.rng = torch.Generator(
                'cuda').manual_seed(args.seed)
            model.model.config.gamma = args.gamma
            model.model.config.beta = 0.
            model.model.config.temp = args.temp
            model.model.config.block_size = args.block_size
            # 2 if args.kv_type in ['bw', 'weightedbw'] else 1
            model.model.config.itrs = args.itrs
            if args.kv_type in ['weightedbw',  'uniform']:
                model.model.config.sink_size = args.sink_size
                model.model.config.window_size = args.window_size

        # 3. load dataset
        if args.e:
            examples = load_dataset(
                'THUDM/LongBench', f"{dataset}_e", split='test', download_mode="force_redownload").sort("_id")
        else:
            examples = load_dataset(
                'THUDM/LongBench', f"{dataset}", split='test')

        # for i in range(1, 4):

        # model.model.config.layer_cutoff = 8*i
        # logger.info(f"starting layer cutoff :{8*i}")
        prompt_format = dataset2prompt[dataset]
        maxlen = dataset2maxlen[dataset]
        max_input_length = 100_000

        # 4. inference
        tic0 = time.time()
        scores = []
        preds = []
        cnt = 0
        input_lens = []
        for eg in examples:
            tic = time.time()

            # 4.1 prepare input
            input_text = prompt_format.format(**eg)
            if 'llama-3' in model_name.lower() or 'llama-8b' in model_name.lower():
                # if dataset in ['hotpotqa']:
                #     msgs = [dict(role="system", content=input_text)]
                #     input_text = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
                # if dataset in ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa"]:
                # input_text = f"[INST] {input_text} [/INST]"
                pass

            elif 'mistral' in model_name.lower() or 'ministral' in model_name.lower():
                # msgs = [{"role": "user", "content": input_text},]
                # input_text = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
                pass
            elif 'glm' in model_name.lower():
                input_text = tokenizer.apply_chat_template([{"role": "user", "content": input_text}],
                                                           add_generation_prompt=True,
                                                           tokenize=False)
            elif 'gemma' in model_name.lower():
                messages = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are a helpful assistant."}]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": input_text}
                        ]
                    }
                ]
                inputs = tokenizer.apply_chat_template(
                    messages,  add_generation_prompt=True, tokenize=True, return_dict=True)
                input_text = inputs['input_ids']
            elif 'qwen' in model_name.lower():
                if 'base' in model_name.lower():
                    # base checkpoints are not instruction-tuned; feed the raw prompt
                    # text as a continuation, same as the other base LMs above.
                    pass
                elif dataset not in ["gov_report", "multi_news", "trec", "triviaqa", "samsum", 'lcc', 'repobench-p']:
                    messages = [
                        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                        {"role": "user", "content": input_text}
                    ]
                    inputs = tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=True
                    )
                    input_text = inputs['input_ids']

            # elif 'glm' in model_name.lower() or 'mistral' in model_name.lower():
            #     input_tokens = tokenizer(input_text, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
            else:
                import pdb
                pdb.set_trace()

            input_tokens = truncate_by_tokens(
                input_text, tokenizer, max_input_length)
            input_tensors = {"input_ids": torch.tensor(
                input_tokens).unsqueeze(0).to(model.device)}
            seq_len = len(input_tokens)

            # 4.2 set up KV cache type
            
            kv_cache = None
            if args.kv_type == 'sink':
                model.model.config.recent_size = seq_len//4 - args.window_size

            # 4.3 set up termination condition
            mem0 = torch.cuda.memory_reserved() / 1024 ** 3
            terminators = [tokenizer.eos_token_id]
            if 'llama-3' in model_name.lower():
                terminators.append(
                    tokenizer.convert_tokens_to_ids("<|eot_id|>"))
            elif 'glm' in model_name.lower():
                terminators.append(
                    tokenizer.convert_tokens_to_ids("<|endoftext|>"))

            # 4.4 generate
            if args.kv_type in ['weightedbw', 'uniform']:
                outputs = greedy_generate(model, input_tensors['input_ids'], max_new_tokens=maxlen,
                                          eos_token_id=terminators, kv_type=args.kv_type, return_dict_in_generate=True)
            else:
                outputs = None
                import pdb
                pdb.set_trace()

            if type(outputs) is not torch.Tensor:

                assert hasattr(outputs, "past_key_values")
                kv_cache_tensors = outputs.past_key_values
                outputs = outputs.sequences
            else:
                kv_cache_tensors = None

            kv_cache_size = 0
            kv_cache_shapes = []
            if kv_cache_tensors is not None:
                if type(kv_cache_tensors) == tuple:
                    for kv_per_layer in kv_cache_tensors:
                        for kv in kv_per_layer:
                            kv_cache_size += kv.numel() * kv.element_size()
                            kv_cache_shapes.append(kv.shape)
                
                elif type(kv_cache_tensors) == list:
                    for kv_per_layer in kv_cache_tensors:
                        for kv in kv_per_layer:
                            for kkvv in kv:
                                if kkvv is not None:
                                    kv_cache_size += kkvv.numel() * kkvv.element_size()
                                    kv_cache_shapes.append(kkvv.shape)
                                else:
                                    import pdb
                                    pdb.set_trace()
                else:
                    import pdb
                    pdb.set_trace()

                del kv_cache_tensors
            else:
                import pdb
                pdb.set_trace()

            kv_cache_size_ori = 2 * (outputs.shape[-1]-1) * model.config.num_hidden_layers * \
                model.config.head_dim * model.config.num_key_value_heads * 2

            if cnt == 0:
                print(
                    f"different kv_shape : {len(set(kv_cache_shapes))}, shape: {set(kv_cache_shapes)}")
                try:
                    baseline_score = BASELINE_SCORES[model_name.split(
                        "/")[-1].split("-")[0].lower()][f'longbench_v1_e' if args.e else 'longbench_v1'][dataset]
                    print(f"baseline_score: {baseline_score}")
                except:
                    pass
            output = outputs[0, seq_len:]
            output_token_len = len(output)
            output = tokenizer.decode(output, skip_special_tokens=True)
            pred = output.strip()
            if dataset not in ['lcc', 'repobench-p'] or 'llama' in model_name:
                pred = pred.lstrip('\n').split('\n')[0]
                pred = pred.split('  ')[0]

            ground_truths = eg['answers']

            score = compute_score(pred, ground_truths,
                                  eg['all_classes'], dataset)
            preds.append({"id": cnt, "prediction": pred,
                          "ground_truth": ground_truths, "score": score})
            scores.append(score)
            input_lens.append(seq_len)

            if args.print_pred:
                print("=" * 200)
                print(f"pred:\n{pred}")
                print(f"label:\n{eg['answers'][0]}")

            if not args.debug:
                dump_jsonl(preds, output_path)
            toc = time.time()
            mem_str = f"mem: ({torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB, {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB), " + \
                (f"kv_size: {kv_cache_size/1024**3:.2f} GB ({kv_cache_size_ori/1024**3:.2f} GB)" if kv_cache_size > 0 else "") + \
                f", {kv_cache_size*8/(model.config.head_dim*(outputs.shape[-1]-1)*2*model.config.num_key_value_heads*model.config.num_hidden_layers):.3f}-bits"
            logger.info(f"[{cnt}/{len(examples)}] score: {score:.3f} (avg: {torch.tensor(scores).mean():.3f}), in_len: {len(input_tokens)} (avg: {np.mean(input_lens):.1f}), out_len: {output_token_len}, time: ({toc-tic:.2f} sec, {toc-tic0:.2f} sec), {mem_str}")
            cnt += 1
            del outputs, kv_cache
            if args.kv_type == 'headkv':
                for layer in model.model.layers:
                    layer.self_attn.kv_seq_len = 0
                    del layer.self_attn.kv_cluster
            torch.cuda.empty_cache()

        logger.info("done.")
        reset_logging()

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
