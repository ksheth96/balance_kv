# Streaming Attention Approximation via Discrepancy Theory

This repository contains code for evaluating our proposed BalanceKV algorithm in our NeurIPS 2025 spotlight paper https://arxiv.org/abs/2502.07861 on the LongBench and NIAH benchmarks. 

### Files and Directories

- **src/balanced_walk.py**: Implements the balanced walk algorithm.
- **src/llama_forward.py**: Contains functions for inference using the Llama model and patching our custom KV cache.
- **src/metrics_longbench.py**: Defines various metrics used for LongBench evaluation.
- **src/single_layer_approx.ipynb**: Notebook containing ablation study experiments regarding single layer attention approximation.
- **run_longbench.py**: Main script for running the LongBench evaluation.
- **run_needle_in_haystack.py**: Main script for running the NIAH evaluation.

## Requirements

Install requirements via
```sh
pip install -r requirements.txt
```

## Usage

To run the LongBench evaluation, use the following command (default model is ``meta-llama/Llama-3.1-8B-Instruct'')

```sh
python run_longbench.py --kv_type weightedbw --datasets qasper --e
```

To run the NIAH evaluation, use the following command (default model is ``meta-llama/Llama-3.1-8B-Instruct'')
```sh
python run_needle_in_haystack.py --kv_type "weightedbw" --haystack_dir "<CurrentPath>/data/PaulGrahamEssays"
```

### Using other base models (e.g. Qwen)

`src/llama_forward.py` derives `head_dim`, `num_key_value_heads`, and (when present) Qwen3-style
QK-norm directly from `model.config`/`model.self_attn`, so any Llama-3-family or Qwen2/Qwen2.5/Qwen3
`AutoModelForCausalLM` checkpoint can be dropped in via `--model_name`. `--model_name` also accepts a
local path, so an already-downloaded checkpoint works without any Hub download, e.g.:

```sh
python run_longbench.py --model_name /path/to/local/Qwen2.5-7B-Instruct --kv_type weightedbw --datasets qasper --e
```

For `run_needle_in_haystack.py`, also pass `--model_provider` to a value other than `LLaMA`, `LLaMA3`,
`Mistral`, `LongLLaMA`, or `GLM` (e.g. `--model_provider Qwen`) so the needle-insertion sentence-boundary
detection tokenizes `.` with the model's own tokenizer instead of a Llama-specific token id.

### TODO
To add the multimodal vlm evaluations, will be done in an upcoming commit.