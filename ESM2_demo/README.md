# ESM2 Demo

This folder contains a single notebook that fine-tunes ESM2 for two tasks:

1) Sequence classification: cytosolic vs membrane proteins
2) Token classification: secondary structure labels (HELIX/STRAND/none)

## Files

- finetune.ipynb: end-to-end workflow (data download, preprocessing, tokenization, training)
- sft_output/: training outputs and checkpoints

## Requirements

- Python 3.10+
- transformers
- datasets
- torch
- accelerate
- pandas
- scikit-learn
- requests
- numpy

## How It Works

### Sequence Classification

- Downloads UniProt data with subcellular location annotations.
- Filters for cytosolic-only and membrane-only proteins.
- Builds binary labels (0 = cytosolic, 1 = membrane).
- Tokenizes sequences with ESM2 tokenizer.
- Trains ESM2 (sequence classification head) with Trainer.

### Token Classification (Secondary Structure)

- Downloads UniProt features (HELIX and STRAND regions).
- Parses HELIX/STRAND ranges with regex and builds per-token labels:
  - 0: none
  - 1: HELIX
  - 2: STRAND
- Tokenizes sequences and aligns labels.
- Trains ESM2 (token classification head) with Trainer and
  DataCollatorForTokenClassification.

## Run

Open the notebook and run cells in order:

1) Load model/tokenizer
2) Download and prepare data
3) Tokenize
4) Build datasets
5) Train

The notebook uses Hugging Face mirror:

- HF_ENDPOINT = https://hf-mirror.com

Change or remove it if you do not use the mirror.

## Outputs

Checkpoints and logs are written under:

- sft_output/esm2_t33_650M_UR50D-finetuned-subcellular-location
- sft_output/esm2_t33_650M_UR50D-finetuned-secondary-structure

## Notes

- Tokenization uses truncation with max_length=512.
- UniProt data is pulled live; network access is required.
- The dataset is filtered by sequence length 80..500 in the API query.
