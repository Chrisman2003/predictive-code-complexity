# Predictive Modelling of Code Complexity from Problem Descriptions
An expandable project skeleton for an interdisciplinary project (IDP) focused on transformer-based estimation of code complexity directly from natural language problem descriptions.

This project aims to automate the estimation of software task complexity (such as Agile Story Points) by applying Natural Language Processing (NLP) to Jira tickets and GitHub issues. Because human estimation is highly subjective and inconsistent, this project leverages pre-trained Transformer models to read task descriptions and predict the underlying effort required.

The repository acts as an end-to-end Machine Learning pipeline. It includes a custom Command Line Interface (CLI) capable of dynamically extracting, shuffling, and partitioning datasets directly from live open-source issue trackers (like the Apache Software Foundation). The extracted data is then fed into a continuous deep learning training loop utilizing ordinal loss functions and cosine learning rate scheduling to accurately map textual requirements to discrete complexity integers.


## Encoder-Only Transformer Backbone Table
| HuggingFace Model Name | Parameters | VRAM | Transformer Blocks |
|---|---|---|---|
| `microsoft/codebert-base`          | 125M | ~1.86 GiB   | 12 |
| `microsoft/graphcodebert-base`     | 125M | ~1.86 GiB   | 12 |
| `microsoft/unixcoder-base`         | 125M | ~1.86 GiB   | 12 |
| `FacebookAI/roberta-base`          | 125M | ~1.86 GiB   | 12 |
| `answerdotai/ModernBERT-base`      | 149M | ~2.22 GiB   | 22 |
| `microsoft/deberta-v3-base`        | 184M | ~2.74 GiB   | 12 |
| `FacebookAI/roberta-large`         | 355M | ~5.29 GiB   | 24 |
| `answerdotai/ModernBERT-large`     | 395M | ~5.89 GiB   | 28 |
| `microsoft/deberta-v3-large`       | 435M | ~6.48 GiB   | 24 |
| `microsoft/deberta-v2-xlarge`      | 900M | ~13.4 GiB   | 24 |
| `microsoft/deberta-v2-xxlarge`     | 1.5B | ~22.35 GiB  | 48 |

## Decoder-Only Transformer Backbone Table
| HuggingFace Model Name | Parameters | VRAM | Transformer Blocks |
|---|---|---|---|
| `sshleifer/tiny-gpt2`              | 3M   | ~0.045 GiB  |  2 |
| `gpt2`                             | 124M | ~1.85 GiB   | 12 |
| `gpt2-medium`                      | 345M | ~5.14 GiB   | 24 |
| `gpt2-large`                       | 774M | ~11.53 GiB  | 36 |
| `EleutherAI/gpt-neo-1.3B`          | 1.3B | ~19.37 GiB  | 24 |
| `gpt2-xl`                          | 1.5B | ~22.35 GiB  | 48 |
| `microsoft/Phi-3-mini-4k-instruct` | 3.8B | ~56.62 GiB  | 32 |
| `EleutherAI/gpt-j-6B`              | 6B   | ~89.41 GiB  | 28 |
| `Qwen/Qwen2-7B`                    | 7B   | ~104.31 GiB | 28 |
| `mistralai/Mistral-7B-v0.3`        | 7.2B | ~107.29 GiB | 32 |
| `meta-llama/Llama-3.1-8B`          | 8B   | ~119.21 GiB | 32 |
| `Qwen/Qwen3-8B`                    | 8B   | ~119.21 GiB | 36 |
| `EleutherAI/gpt-neox-20b`          | 20B  | ~298.02 GiB | 44 |


## Repository Structure
- `data/` – datasets and raw assets
  - `raw/` – unpartitioned data dumps
  - `processed/` – dynamically split datasets (e.g., `Mesos/mesos_dataset_train.json`)
- `src/` – package source code
  - `dataloader/` – dataset extraction (Jira/GitHub APIs) and preprocessing logic
  - `models/` – model definitions, loss functions, and prediction pipelines
  - `evaluation/` – validation metrics (MAE) and analysis scripts
  - `__main__.py` – the core CLI entry point (`predict-complexity`)
- `binaries/` – saved model checkpoints (e.g., `.pt` files)
- `tests/` – automated tests
- `notebooks/` – exploratory data analysis notebooks
- `pyproject.toml` – Python project configuration



# EXAMPLE PROMP [HOMEONE]:
CUDA_VISIBLE_DEVICES=2 \
  predict-complexity predict \
    --train \
    --train-data data/processed/MESOS/mesos_dataset_train.json \
    --val-data data/processed/MESOS/mesos_dataset_val.json \
    --model gpt2