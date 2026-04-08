# LLamA2_demo

A lightweight, educational Llama2-style training demo with tokenizer training, pretraining, SFT, and inference examples.

## Contents

- `train_demo.ipynb`: Original end-to-end demo notebook.
- `train_demo_rewrite.ipynb`: Teaching-oriented rewrite with detailed comments.
- `train_tokenizer.py`: Standalone tokenizer training script.
- `data/`: Datasets (JSON/JSONL) used for tokenizer, pretraining, and SFT.
- `tokenizer/`: Trained tokenizer artifacts.
- `output/`: Pretraining checkpoints.
- `sft_output/`: SFT checkpoints.

|-- README.md
|-- data
|   |-- BelleGroup_sft.jsonl
|   |-- mobvoi_seq_monkey_general_open_corpus.jsonl
|   |-- mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2
|   `-- train_3.5M_CN.json
|-- sft_output
|   |-- sft_dim768_layers12_vocab_size6144.pth
|-- pretrain_output
|   |-- checkpoint-4364
|   |   |-- added_tokens.json
|   |   |-- chat_template.jinja
|   |   |-- config.json
|   |   |-- generation_config.json
|   |   |-- merges.txt
|   |   |-- model-00001-of-00002.safetensors
|   |   |-- model-00002-of-00002.safetensors
|   |   |-- model.safetensors.index.json
|   |   |-- optimizer.pt
|   |   |-- rng_state.pth
|   |   |-- scheduler.pt
|   |   |-- special_tokens_map.json
|   |   |-- tokenizer.json
|   |   |-- tokenizer_config.json
|   |   |-- trainer_state.json
|   |   |-- training_args.bin
|   |   `-- vocab.json
|   |-- pretrain
|   |   |-- added_tokens.json
|   |   |-- chat_template.jinja
|   |   |-- checkpoint-1456
|   |   |   |-- added_tokens.json
|   |   |   |-- chat_template.jinja
|   |   |   |-- config.json
|   |   |   |-- generation_config.json
|   |   |   |-- global_step1456
|   |   |   |   |-- bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt
|   |   |   |   |-- bf16_zero_pp_rank_1_mp_rank_00_optim_states.pt
|   |   |   |   |-- bf16_zero_pp_rank_2_mp_rank_00_optim_states.pt
|   |   |   |   `-- mp_rank_00_model_states.pt
|   |   |   |-- latest
|   |   |   |-- merges.txt
|   |   |   |-- model.safetensors
|   |   |   |-- rng_state_0.pth
|   |   |   |-- rng_state_1.pth
|   |   |   |-- rng_state_2.pth
|   |   |   |-- special_tokens_map.json
|   |   |   |-- tokenizer.json
|   |   |   |-- tokenizer_config.json
|   |   |   |-- trainer_state.json
|   |   |   |-- training_args.bin
|   |   |   |-- vocab.json
|   |   |   `-- zero_to_fp32.py
|   |   |-- config.json
|   |   |-- generation_config.json
|   |   |-- merges.txt
|   |   |-- model.safetensors
|   |   |-- special_tokens_map.json
|   |   |-- tokenizer.json
|   |   |-- tokenizer_config.json
|   |   |-- training_args.bin
|   |   `-- vocab.json
|   |-- pretrain_768_12_6144.pth
|-- tokenizer
|   |-- special_tokens_map.json
|   |-- tokenizer.json
|   `-- tokenizer_config.json
|-- train_demo.ipynb
|-- train_demo_rewrite.ipynb
`-- train_tokenizer.py

## Quick Start

1. Open and run the teaching notebook:
   - `train_demo_rewrite.ipynb`
2. Follow the sections in order. Heavy steps (tokenizer training and model training) are gated by `RUN_HEAVY` in the notebook.

## Environment

This demo depends on common Python ML packages.

Suggested (not exhaustive):
- Python 3.9+
- PyTorch
- transformers
- tokenizers
- numpy
- tqdm

If you use the training sections with logging, install your preferred logger (e.g., swanlab) as needed.

## Data Preparation

Place data files under `data/` (paths match the notebook defaults):

- Pretraining data: `data/mobvoi_seq_monkey_general_open_corpus.jsonl`
- SFT data: `data/BelleGroup_sft.jsonl`

Each line should be JSON. For pretraining, each line must contain a `text` field.

## Tokenizer Training

You can train a BPE tokenizer either in the notebook or with `train_tokenizer.py`.

Notebook path:
- Section: "Tokenizer 训练" in `train_demo_rewrite.ipynb`
- Set `RUN_HEAVY = True`

Output:
- `tokenizer/tokenizer.json`
- `tokenizer/tokenizer_config.json`
- `tokenizer/special_tokens_map.json`

## Pretraining

Notebook path:
- Section: "预训练循环" in `train_demo_rewrite.ipynb`
- Set `RUN_HEAVY = True`

Output:
- `output/pretrain_<dim>_<layers>_<vocab>.pth`
- Periodic checkpoints: `output/pretrain_<...>_stepXXXX.pth`

## SFT (Supervised Fine-Tuning)

Notebook path:
- Section: "SFT 训练循环" in `train_demo_rewrite.ipynb`
- Set `RUN_HEAVY = True`

Output:
- `sft_output/sft_dim<...>_layers<...>_vocab_size<...>.pth`

## Inference

Notebook path:
- Section: "推理与采样" in `train_demo_rewrite.ipynb`

By default, inference loads `output/pretrain_768_12_6144.pth` and the tokenizer from `tokenizer/`.

## Notes

- The notebooks are educational and trade speed for clarity.
- Some steps require large datasets and GPU resources.
- Update paths if your data/checkpoints live elsewhere.

## License

Specify your license here (e.g., MIT, Apache-2.0), or remove this section if not applicable.
