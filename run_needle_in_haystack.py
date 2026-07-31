"""
This script is adapted from 
https://github.com/FranxYao/Long-Context-Data-Engineering
"""

import tiktoken
import os 
import pdb
import glob
import jieba

import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import argparse
from rouge_score import rouge_scorer

import sys
import os

from datetime import datetime, timezone
import time
import torch
import random


from src.llama_forward import greedy_generate

try:
    import sys; sys.path.append("/content/sample_data/HeadKV")
    
except:
    print("headkv not found")


scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

answer = "eat a sandwich and sit in Dolores Park on a sunny day."

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

class LLMNeedleHaystackTester:
    """
    This class is used to test the LLM Needle Haystack.
    """
    def __init__(self,
                 needle="\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n",#    \n在旧金山做的最棒的是事是吃一个三明治以及在晴天坐在多洛雷斯公园里\n
                 haystack_dir="/content/sample_data/HeadKV/data/PaulGrahamEssays", # PaulGrahamEssays  
                 retrieval_question="The best thing to do in San Francisco is: ", 
                 results_version = 1,
                 context_lengths_min = None,
                 context_lengths_max = None,
                 context_lengths_num_intervals = 40,
                 context_lengths = None,
                 document_depth_percent_min = 0,
                 document_depth_percent_max = 100,
                 document_depth_percent_intervals = 10,
                 document_depth_percents = None,
                 document_depth_percent_interval_type = "linear",
                 model_provider = "OpenAI",
                 model_name='',
                 model_name_suffix=None,
                 model_version=None, 
                 num_concurrent_requests = 1,
                 save_results = True,
                 save_contexts = True,
                 final_context_length_buffer = 200,
                 seconds_to_sleep_between_completions = None,
                 print_ongoing_status = True, 
                 step=100, 
                 method='pyramidkv', 
                 attn_implementation='flash_attention_2',
                 max_capacity_prompts=128, kv_type='exact'):
        """        
        :param needle: The needle to be found in the haystack. Default is None.
        :param haystack_dir: The directory of text files to use as background context (or a haystack) in which the needle is to be found. Default is Paul Graham Essays.
        :param retrieval_question: The question which with to prompt the model to do the retrieval.
        :param results_version: In case you would like to try the same combination of model, context length, and depth % multiple times, change the results version other than 1
        :param num_concurrent_requests: Due to volume, this object is set up to run concurrent requests, default = 1. Be careful of rate limits.
        :param save_results: Whether or not you would like to save your contexts to file. Warning: These will get long! Default = True
        :param save_contexts: Whether or not you would like to save your contexts to file. Warning: These will get long! Default is True.
        :param final_context_length_buffer: The amount of cushion you'd like to leave off the input context to allow for the output context. Default 200 tokens
        :param context_lengths_min: The minimum length of the context. Default is 0.
        :param context_lengths_max: The maximum length of the context. Default is 200000.
        :param context_lengths_num_intervals: The number of intervals for the context length. Default is 35.
        :param context_lengths: The lengths of the context. Default is None.
        :param document_depth_percent_min: The minimum depth percent of the document. Default is 0.
        :param document_depth_percent_max: The maximum depth percent of the document. Default is 100.
        :param document_depth_percent_intervals: The number of intervals for the document depth percent. Default is 35.
        :param document_depth_percents: The depth percentages of the document. Default is None.
        :param document_depth_percent_interval_type: The type of interval for the document depth percent. Must be either 'linear' or 'sigmoid'. Default is 'linear'.
        :param openai_api_key: The API key for OpenAI. Default is None.
        :param model_name: The name of the model. Default is 'gpt-4-1106-preview'.
        :param seconds_to_sleep_between_completions: The number of seconds to sleep between completions. Default is None.
        :param print_ongoing_status: Whether or not to print the ongoing status. Default is True.
        """
        if not needle or not haystack_dir or not retrieval_question:
            raise ValueError("Needle, haystack, and retrieval_question must be provided.")
        
        self.needle = needle
        self.haystack_dir = haystack_dir
        self.retrieval_question = retrieval_question
        self.results_version = results_version
        self.num_concurrent_requests = num_concurrent_requests
        self.save_results = save_results
        self.final_context_length_buffer = final_context_length_buffer
        self.save_contexts = save_contexts
        self.seconds_to_sleep_between_completions = seconds_to_sleep_between_completions
        self.print_ongoing_status = print_ongoing_status
        self.model_provider = model_provider
        self.testing_results = []
        self.step = step
        self.method = method
        self.max_capacity_prompts = max_capacity_prompts
        self.attn_implementation = attn_implementation
        self.kv_type = kv_type


        self.model_version = model_version
        if(model_name_suffix is not None): self.model_version += "_" + model_name_suffix

        if context_lengths is None:
            if context_lengths_min is None or context_lengths_max is None or context_lengths_num_intervals is None:
                raise ValueError("Either context_lengths_min, context_lengths_max, context_lengths_intervals need to be filled out OR the context_lengths_list needs to be supplied.")
            else:
                # self.context_lengths = np.round(np.linspace(context_lengths_min, context_lengths_max, num=context_lengths_num_intervals, endpoint=True)).astype(int)
                self.context_lengths = np.arange(context_lengths_min, context_lengths_max+1, step=self.step)
        else:
            self.context_lengths = context_lengths


        if document_depth_percents is None:
            if document_depth_percent_min is None or document_depth_percent_max is None or document_depth_percent_intervals is None:
                raise ValueError("Either document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals need to be filled out OR the document_depth_percents needs to be supplied.")
            else:
                if document_depth_percent_interval_type == 'linear':
                    self.document_depth_percents = np.round(np.linspace(document_depth_percent_min, document_depth_percent_max, num=document_depth_percent_intervals, endpoint=True)).astype(int)
                elif document_depth_percent_interval_type == 'sigmoid':
                    self.document_depth_percents = [self.logistic(x) for x in np.linspace(document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals)]
        else:
            self.document_depth_percents = document_depth_percents

        if document_depth_percent_interval_type not in [None, "linear", "sigmoid"]:
            raise ValueError("document_depth_percent_interval_type must be either None, 'linear' or 'sigmoid'. If you'd like your own distribution give a list of ints in via document_depth_percent_intervals")
        
        self.model_name = model_name

        # if(self.model_provider in ["LLaMA3", "Mistral"]):
        if args.model_name == 'mistralai/Mistral-7B-Instruct-v0.2':
            self.enc = AutoTokenizer.from_pretrained(
                args.model_name,
                use_fast=args.use_fast_tokenizer,
                padding_side="left",
                revision='dca6e4b60aca009ed25ffa70c9bb65e46960a573'
            )
        else:
            self.enc = AutoTokenizer.from_pretrained(
                args.model_name,
                use_fast=args.use_fast_tokenizer,
                padding_side="left"
            )
            if 'llama' in args.model_name or 'mistral' in args.model_name.lower():
                self.enc.pad_token_id = self.enc.eos_token_id
        # self.enc.add_special_tokens({'pad_token': '[PAD]'})
        print("loading from %s" % model_name)


        self.enc.padding_side = "left"
        if self.enc.pad_token is None:
            self.enc.pad_token = self.enc.eos_token
            self.enc.pad_token_id = self.enc.eos_token_id
    

        # if torch.cuda.device_count()>1:
        # if args.method.lower() != 'fullkv':
        #     from headkv.monkeypatch import replace_llama, replace_mistral 
        #     replace_llama(args.method)
        #     replace_mistral(args.method)

        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto",
            use_cache=args.use_cache,
            attn_implementation=args.attn_implementation
        ).eval()

        if args.kv_type in ['exact']:
            pass
        
        elif args.kv_type in ['sink']:
            self.model.model.config.sink_size = args.sink_size

        elif args.kv_type in ['balanced_walk', 'bw', 'twostagebw', 'weightedbw', 'weightedbwlayer', 'weightedbwlayer1', 'weightedbwlayer2', 'weightedbwlayer3']:
            self.model.model.config.rng = torch.Generator('cuda').manual_seed(123)
            self.model.model.config.gamma = args.gamma
            self.model.model.config.beta = 0.
            self.model.model.config.temp = args.temp
            self.model.model.config.block_size = args.block_size
            self.model.model.config.itrs = args.itrs #2 if args.kv_type in ['bw', 'weightedbw'] else 1
            if args.kv_type == 'twostagebw':
                self.model.model.config.pooling = args.pooling
                self.model.model.config.kernel_size = args.kernel_size
                self.model.model.config.window_size = args.window_size
            if args.kv_type in ['weightedbw', 'weightedbwlayer', 'weightedbwlayer1', 'weightedbwlayer2', 'weightedbwlayer3']:
                self.model.model.config.sink_size = args.sink_size
                self.model.model.config.window_size = args.window_size

        else:
            import pdb; pdb.set_trace();



    def logistic(self, x, L=100, x0=50, k=.1):
        if x == 0:
            return 0
        if x == 100:
            return 100
        return np.round(L / (1 + np.exp(-k * (x - x0))), 3)
    
    def bound_evaluate_and_log(self, *args):
        self.evaluate_and_log(*args)

    def run_test(self, args):

        # Run through each iteration of context_lengths and depths
        tasks = []
        for i, context_length in enumerate(self.context_lengths):
            if context_length < args.s_len or context_length > args.e_len: continue
            for j, depth_percent in enumerate(self.document_depth_percents):
                print(f"context_length: {context_length} ({i}/{len(self.context_lengths)}), depth_percent {depth_percent} ({j}/{len(self.document_depth_percents)})")
                task = self.bound_evaluate_and_log(context_length, depth_percent)
            print(f"Finished context_length: {context_length} ({i}/{len(self.context_lengths)}), avg_scores: {np.mean([t['recall_score'] for t in self.testing_results])}")


    def generate_prompt(self, context):
        # Generate the prompt for the Anthropic model
        # Replace the following line with the appropriate prompt structure
        if(self.model_provider not in ["OpenAI", "Anthropic"]):
            test_format=f"<|im_start|> This is a very long story book: <book> {context} </book>.\n Based on the content of the book, Question: {self.retrieval_question}\nAnswer:"
            return test_format
        else: 
            return [
                {
                    "role": "system",
                    "content": "You are a helpful AI bot that answers questions for a user. Keep your response short and direct"
                },
                {
                    "role": "user",
                    "content": context
                    },
                {
                    "role": "user",
                    "content": f"{self.retrieval_question} Don't give information outside the document or repeat your findings. The document definitely contains the answer, and I'm 100% sure. So try your best to find it."
                },
                {
                    "role": "assistant",
                    "content":"",
                },
                
            ]

    def evaluate_and_log(self, context_length, depth_percent):
        # Checks to see if you've already checked a length/percent/version.
        # This helps if the program stop running and you want to restart later
        if self.save_results:
            if self.result_exists(context_length, depth_percent):
                print("result exists, skipping")
                return
            else:
                print("result does not exist, testing")
        # Go generate the required length context and place your needle statement in
        context = self.generate_context(context_length, depth_percent)

        # Prepare your message to send to the model you're going to evaluate
        prompt = self.generate_prompt(context)
        test_start_time = time.time()
        
        # if(self.model_provider in ["LLaMA3", "Mistral"]):
        prompt = self.enc(prompt, return_tensors="pt")
        input_ids = prompt['input_ids'].to(self.model.device)
        seq_len = input_ids.shape[1]
        if self.kv_type in ['exact']:
            if self.kv_type == 'exact':
                kv_cache = None
            else:
                import pdb; pdb.set_trace();

            output_ids = self.model.generate(
                input_ids, 
                output_attentions=False,
                max_new_tokens=30,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                past_key_values=kv_cache, 
                eos_token_id=[self.enc.eos_token_id, self.enc.encode("\n", add_special_tokens=False)[-1]],
                return_dict_in_generate=True
            )
        
        elif self.kv_type in ['sink', 'weightedbw', 'adapkvq', 'adapkvqsimple', 'adapkvqrndrot', 'adapkvqrndrotsimple']:
            if self.kv_type == 'sink':
                self.model.model.config.recent_size = seq_len//4 - args.window_size
            terminators = [self.enc.eos_token_id, self.enc.encode("\n", add_special_tokens=False)[-1]]
            output_ids = greedy_generate(self.model, input_ids, max_new_tokens=30, eos_token_id=terminators, kv_type=args.kv_type, return_dict_in_generate=True,needle_detection=True)
        
        kv_cache_tensors = None
        if type(output_ids) is not torch.Tensor:
            assert hasattr(output_ids, "sequences")
            kv_cache_tensors = output_ids.past_key_values
            output_ids = output_ids.sequences
        
        try:
            response = self.enc.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        except:
            import pdb;pdb.set_trace();

        kv_cache_size = 0
        kv_cache_shapes = []
        if kv_cache_tensors is not None:
            if type(kv_cache_tensors) == tuple:
                for kv_per_layer in kv_cache_tensors:
                    for kv in kv_per_layer:
                        kv_cache_size += kv.numel() * kv.element_size()
                        kv_cache_shapes.append(kv.shape)
                # kv_cache_size == 2 (k,v) * 2 (bytes) * 8 (nheads) * 128 (headim) * seq_len * 32 (num_layers) / 1024**3 if kv_type == 'exact'
            elif type(kv_cache_tensors) == list:
                for kv_per_layer in kv_cache_tensors:
                    for kv in kv_per_layer:
                        for kkvv in kv:
                            if kkvv is not None:
                                kv_cache_size += kkvv.numel() * kkvv.element_size()
                                kv_cache_shapes.append(kkvv.shape)
                            else:
                                import pdb; pdb.set_trace();

        kv_cache_size_ori = 2 * (output_ids.shape[-1]-1) * self.model.config.num_hidden_layers * self.model.config.head_dim * self.model.config.num_key_value_heads * 2
        print(f"kv_cache_size: {kv_cache_size / 1024**3:.2f} GB ({kv_cache_size_ori / 1024**3:.2f} GB)")
        print(response)
        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time
        if len(response) != 0:
            score = scorer.score(self.needle, response)['rouge1'].fmeasure*10
            expected_answer = answer.lower().split()
            recall_score = len(set(response.lower().split()).intersection(set(expected_answer))) / len(set(expected_answer))
        else:
            score = 0.0
            recall_score = 0.0

        results = {
            'model' : self.model_name,
            'context_length' : int(context_length),
            'depth_percent' : float(depth_percent),
            'version' : self.results_version,
            'needle' : self.needle,
            'model_response' : response,
            'score' : score,
            'recall_score' : recall_score,
            'test_duration_seconds' : test_elapsed_time,
            'test_timestamp_utc' : datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z'), 
            'kv_cache_size_ori': kv_cache_size_ori,
            'kv_cache_size': kv_cache_size,
        }

        self.testing_results.append(results)

        del output_ids, kv_cache_tensors, kv_cache_size, kv_cache_shapes
        import gc; gc.collect()
        print(f"[before] mem: ({torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB, {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB), ")
        torch.cuda.empty_cache()
        print(f"[after ] mem: ({torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB, {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB), ")

        if self.print_ongoing_status:
            print (f"-- Test Summary -- ")
            print (f"Duration: {test_elapsed_time:.1f} seconds")
            print (f"Context: {context_length} tokens")
            print (f"Depth: {depth_percent}%")
            print (f"Score: {score}")
            print (f"Recall Score: {recall_score}")
            print (f"Response: {response}\n")

        context_file_location = f'{self.model_version.replace(".", "_")}_len_{context_length}_depth_{int(depth_percent*100)}'

        if self.save_contexts:
            results['file_name'] : context_file_location

            # Save the context to file for retesting
            if not os.path.exists(f'{args.save_dir}/contexts'):
                os.makedirs(f'{args.save_dir}/contexts')

            # if not os.path.exists(f'{args.save_dir}/contexts/{self.model_version}'):
            #     os.makedirs(f'{args.save_dir}/contexts/{self.model_version}')

            with open(f'{args.save_dir}/contexts/{context_file_location}_context.txt', 'w') as f:
                f.write(context)
            
        if self.save_results:
            # Save the context to file for retesting
            if not os.path.exists(f'{args.save_dir}/results'):
                os.makedirs(f'{args.save_dir}/results')
            
            # if not os.path.exists(f'{args.save_dir}/results/{self.model_version}'):
            #     os.makedirs(f'{args.save_dir}/results/{self.model_version}')

            # Save the result to file for retesting
            p = f'{args.save_dir}/results/{context_file_location}_results.json'
            print("Writing at %s" % p)
            with open(p, 'w') as f:
                json.dump(results, f, ensure_ascii=False)

    def result_exists(self, context_length, depth_percent):
        """
        Checks to see if a result has already been evaluated or not
        """

        results_dir = '{args.save_dir}/results/'
        print("Searching existing results at %s" % results_dir)
        if not os.path.exists(results_dir):
            return False
        for filename in os.listdir(results_dir):
            if filename.endswith('.json'):
                with open(os.path.join(results_dir, filename), 'r') as f:
                    result = json.load(f)
                    context_length_met = result['context_length'] == context_length
                    depth_percent_met = result['depth_percent'] == depth_percent
                    version_met = result.get('version', 1) == self.results_version
                    model_met = result['model'] == self.model_name
                    # import ipdb; ipdb.set_trace()
                    if context_length_met and depth_percent_met and version_met and model_met:
                        return True
        return False

    def generate_context(self, context_length, depth_percent):
        # Load up tiktoken so we navigate tokens more easily

        # Get your Paul Graham files loaded into a string
        
        context = self.read_context_files()
        

        # Truncate the Paul Graham essays to the context length you desire
        context = self.encode_and_trim(context, context_length)

        # Insert your random statement according to your depth percent
        context = self.insert_needle(context, depth_percent, context_length)

        return context
    
    def encode_text_to_tokens(self, text):
        if self.model_provider in ["Mistral", "LLaMA3"]:
            return self.enc.encode(text)
        elif self.model_provider == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            return self.enc.encode(text).ids
        else:
            return self.enc.encode(text)
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")
    
    def insert_needle(self, context, depth_percent, context_length):
        tokens_needle = self.encode_text_to_tokens(self.needle)
        tokens_context = self.encode_text_to_tokens(context)

        # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
        context_length -= self.final_context_length_buffer

        # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[:context_length - len(tokens_needle)]

        if depth_percent == 100:
            # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
            tokens_new_context = tokens_context + tokens_needle
        else:
            # Go get the position (in terms of tokens) to insert your needle
            insertion_point = int(len(tokens_context) * (depth_percent / 100))

            # tokens_new_context represents the tokens before the needle
            tokens_new_context = tokens_context[:insertion_point]

            # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
            if(self.model_provider in ["LLaMA", "LongLLaMA"]): period_tokens = [29889, 869]
            elif(self.model_provider == "LLaMA3"): period_tokens = [13]
            elif(self.model_provider == "Mistral"): period_tokens = [842, 28723]
            elif(self.model_provider == "GLM"): period_tokens = [918, 30930]
            else: period_tokens = self.encode_text_to_tokens('.')
            
            # Then we iteration backwards until we find the first period
            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                tokens_new_context = tokens_context[:insertion_point]

            print("insertion at %d" % insertion_point)
            # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
            # Now we have a needle in a haystack
            tokens_new_context += tokens_needle + tokens_context[insertion_point:]

        # Convert back to a string and return it
        new_context = self.decode_tokens(tokens_new_context)
        return new_context

    def get_context_length_in_tokens(self, context):
        if self.model_provider in ["Mistral", "LLaMA3"]:
            return len(self.enc.encode(context))
        else:
            return len(self.enc.encode(context))
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")

    def read_context_files(self):
        context = ""
        max_context_length = max(self.context_lengths)

        while self.get_context_length_in_tokens(context) < max_context_length:
            for file in glob.glob(f"{self.haystack_dir}/*.txt"):
                with open(file, 'r') as f:
                    context += f.read()
        return context

    def get_tokens_from_context(self, context):
        if self.model_provider in ["Mistral", "LLaMA3"]:
            return self.enc.encode(context)
        else:
            return self.enc.encode(context)
            # raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")
        
    def decode_tokens(self, tokens, context_length=None):
        if self.model_provider in ["Mistral", "LLaMA3"]:
            return self.enc.decode(tokens[:context_length])
        else:
            return self.enc.decode(tokens[:context_length])
            # raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")

    def encode_and_trim(self, context, context_length):
        tokens = self.get_tokens_from_context(context)
        if len(tokens) > context_length:
            context = self.decode_tokens(tokens, context_length)
        return context
    
    def get_results(self):
        return self.testing_results
    
    def print_start_test_summary(self):
        print ("\n")
        print ("Starting Needle In A Haystack Testing...")
        print (f"- Model: {self.model_name}")
        print (f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        print (f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%")
        print (f"- Needle: {self.needle.strip()}")
        print ("\n\n")

    def start_test(self, args):
        if self.print_ongoing_status:
            self.print_start_test_summary()
        #asyncio.run(self.run_test())
        self.run_test(args)


if __name__ == "__main__":
    # Tons of defaults set, check out the LLMNeedleHaystackTester's init for more info
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--s_len', metavar='N', default=1024, type=int, help='a number')
    parser.add_argument('-e', '--e_len', metavar='N', default=819200000, type=int, help='a number')
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.1-8B-Instruct", help='name of model')
    parser.add_argument('--kv_type', type=str, default="exact")
    parser.add_argument("--use_fast_tokenizer", type=bool, default=True, help="")
    parser.add_argument("--attn_implementation", type=str,  default="flash_attention_2", choices=["flash_attention_2", "sdpa", "None"])
    parser.add_argument('--model_version', type=str, default="llama-3.1-8b-instruct", help='provider of model')
    parser.add_argument('--model_name_suffix', type=str, default=None, help='name of model')
    parser.add_argument('--model_provider', type=str, default="LLaMA3", help='which model to use')
    parser.add_argument('--api_key', type=str, default="", help='OpenAI API Key')
    parser.add_argument('--step', type=int, default=1000)
    parser.add_argument('--max_capacity_prompts', type=int, default=128)
    parser.add_argument('--haystack_dir', type=str, default="/content/sample_data/HeadKV/data/PaulGrahamEssays")

    parser.add_argument('--method', type=str, default=None, help='method')
    parser.add_argument('--save_dir', type=str, default="./new_results/needle/", help='method')
    parser.add_argument('--use_cache', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument("--group_size", type=int, default=32, help="for KIVI")
    parser.add_argument("--bits", type=int, default=2, choices=[2,4], help="for KIVI")
    parser.add_argument("--residual_length", type=int, default=32, help="for KIVI")

    parser.add_argument("--sink_size", type=int, default=32, help="for StreamingLLM")
    parser.add_argument("--window_size", type=int, default=32, help="for SnapKV,PyramidKV")
    parser.add_argument("--max_capacity_prompt", type=int, default=4096, help="for SnapKV,PyramidKV")
    parser.add_argument("--kernel_size", type=int, default=5, help="for SnapKV,PyramidKV")
    parser.add_argument("--pooling", type=str, default="avgpool", help="for SnapKV,PyramidKV")

    parser.add_argument("--block_size", type=int, default=128, help="for HeadKV")
    parser.add_argument("--itrs", type=int, default=2, help="for BalancedWalk")
    parser.add_argument("--beta", type=float, default=0.0, help="for BalancedWalk")
    parser.add_argument("--temp", type=float, default=1.0, help="for BalancedWalk")
    parser.add_argument("--gamma", type=float, default=4.0, help="for BalancedWalk")

    # parser.add_argument("--aug_capacity", type=int, default=-1, help='-1 means use the full kv cache')
    # parser.add_argument("--head_choice", type=str, default='random', choices=['random', 'copy', 'musique', 'reason', 'mix', 'mix_top3', 'musique_top3', 'merge', 'final', 'final_reason_top3', 'final_copy'])
    # parser.add_argument("--top_num", type=int, default=10)
    # parser.add_argument('--beta', type=float, default=1.5)
    # parser.add_argument('--temp', type=float, default=1.0)
    # parser.add_argument('--alpha', type=float, default=0.56, help='should be [0, 1]. 1-- total copy.')
    parser.add_argument('--debug', action='store_true', help="debug mode")
    args = parser.parse_args()
    set_seed(args.seed)

    model_name = args.model_name
    real_model_name = model_name.rstrip("/").split("/")[-1]
    exp_name = f"{real_model_name}_{args.kv_type}"
    if args.kv_type == 'kivi':
        exp_name += f"_g{args.group_size}_b{args.bits}_r{args.residual_length}"
    elif args.kv_type in ['snapkv', 'pyramidkv']:
        exp_name += f"_w{args.window_size}_nlinear_k{args.kernel_size}_p{args.pooling}"
    elif args.kv_type in ['headkv']:
        args.kernel_size = 7
        args.pooling = 'maxpool'
        args.beta = 1.5
        args.temp = 1.0
        exp_name += f"_{args.headkv_method}_w{args.window_size}_nlinear_k{args.kernel_size}_p{args.pooling}_b{args.beta}_t{args.temp}"
    elif args.kv_type in ['bw', 'balacned_walk', 'twostagebw']:
        #  best_param: (0.1, 0.2, 128)
        exp_name += f"_g{args.gamma}_t{args.temp}_b{args.block_size}_r{args.window_size}"
    elif args.kv_type in ['weightedbw']:
        exp_name += f"_itr{args.itrs}_g{args.gamma}_t{args.temp}_b{args.block_size}_s{args.sink_size}_r{args.window_size}"
    elif args.kv_type in ['weightedbwlayer', 'weightedbwlayer1', 'weightedbwlayer2', 'weightedbwlayer3']:
        exp_name += f"_itr{args.itrs}_g{args.gamma}_t{args.temp}_b{args.block_size}_s{args.sink_size}_r{args.window_size}"
    elif args.kv_type in ['adapkvq', 'adapkvqsimple', 'adapkvqrndrot', 'adapkvqrndrotsimple']:
        exp_name += f"_s{args.seed}"
    elif args.kv_type == 'streamingllm':
        exp_name += f"_w{args.window_size}"
    elif args.kv_type == 'sink':
        exp_name += f"_s{args.sink_size}"
    print(f"exp_name: {exp_name}")

    # context_lengths = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    # context_lengths = [1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 64_000, 128_000]
    # if args.debug:
    # else:
    context_lengths = np.logspace(10, 17, num=8 + 7*2, base=2).astype(int).tolist()[6:-1] # 8 + 7 * 3
    #context_lengths = [4096]
    args.save_dir = os.path.join(args.save_dir, exp_name)
    if not args.debug:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
    ht = LLMNeedleHaystackTester(model_name=args.model_name, 
                                 model_name_suffix=args.model_name_suffix,
                                 model_provider=args.model_provider,
                                 haystack_dir=args.haystack_dir,
                                 model_version=args.model_version, 
                                 context_lengths=context_lengths,
                                 save_contexts=True if not args.debug else False,
                                 save_results=True if not args.debug else False,
                                #  context_lengths_max=args.e_len, 
                                 step=args.step, 
                                 method=args.method, 
                                 max_capacity_prompts=args.max_capacity_prompts,
                                 attn_implementation=args.attn_implementation, 
                                 kv_type=args.kv_type,
                                 )
    ht.start_test(args)
