# Predictive Modelling of Code Complexity from Problem Descriptions
An expandable project skeleton for an interdisciplinary project (IDP) focused on transformer-based estimation of code complexity directly from natural language problem descriptions.

This project aims to automate the estimation of software task complexity (such as Agile Story Points) by applying Natural Language Processing (NLP) to Jira tickets and GitHub issues. Because human estimation is highly subjective and inconsistent, this project leverages pre-trained Transformer models to read task descriptions and predict the underlying effort required.

The repository acts as an end-to-end Machine Learning pipeline. It includes a custom Command Line Interface (CLI) capable of dynamically extracting, shuffling, and partitioning datasets directly from live open-source issue trackers (like the Apache Software Foundation). The extracted data is then fed into a continuous deep learning training loop utilizing ordinal loss functions and cosine learning rate scheduling to accurately map textual requirements to discrete complexity integers.


## Encoder-Only Transformer Backbones
| HuggingFace Model Name | Parameters | VRAM | Transformer Blocks |
|---|---|---|---|
| `microsoft/codebert-base`          | 125M | ~1.86 GiB   | 12 |
| `microsoft/graphcodebert-base`     | 125M | ~1.86 GiB   | 12 |
| `microsoft/unixcoder-base`         | 125M | ~1.88 GiB   | 12 |
| `FacebookAI/roberta-base`          | 125M | ~1.86 GiB   | 12 |
| `answerdotai/ModernBERT-base`      | 149M | ~2.22 GiB   | 22 |
| `microsoft/deberta-v3-base`        | 184M | ~2.74 GiB   | 12 |
| `FacebookAI/roberta-large`         | 355M | ~5.30 GiB   | 24 |
| `answerdotai/ModernBERT-large`     | 395M | ~5.88 GiB   | 28 |
| `microsoft/deberta-v3-large`       | 435M | ~6.47 GiB   | 24 |
| `microsoft/deberta-v2-xlarge`      | 900M | ~13.18 GiB  | 24 |
| `microsoft/deberta-v2-xxlarge`     | 1.5B | ~23.31 GiB  | 48 |
| `facebook/xlm-roberta-xl`          | 3.5B | ~51.89 GiB  | 36 |
| `facebook/xlm-roberta-xxl`         |10.7B | ~159.63 GiB | 48 |

## Decoder-Only Transformer Backbones
| HuggingFace Model Name | Parameters | VRAM | Transformer Blocks |
|---|---|---|---|
| `sshleifer/tiny-gpt2`              | 3M   | ~1.57 MiB   |  2 |
| `gpt2`                             | 124M | ~1.85 GiB   | 12 |
| `gpt2-medium`                      | 345M | ~5.29 GiB   | 24 |
| `gpt2-large`                       | 774M | ~11.53 GiB  | 36 |
| `EleutherAI/gpt-neo-1.3B`          | 1.3B | ~19.65 GiB  | 24 |
| `gpt2-xl`                          | 1.5B | ~23.21 GiB  | 48 |
| `microsoft/Phi-3-mini-4k-instruct` | 3.8B | ~56.94 GiB  | 32 |
| `EleutherAI/gpt-j-6B`              | 6B   | ~87.14 GiB  | 28 |
| `Qwen/Qwen2-7B`                    | 7B   | ~105.36 GiB | 28 |
| `mistralai/Mistral-7B-v0.3`        | 7.2B | ~106.00 GiB | 32 |
| `meta-llama/Llama-3.1-8B`          | 8B   | ~119.21 GiB | 32 |
| `Qwen/Qwen3-8B`                    | 8B   | ~112.78 GiB | 36 |
| `EleutherAI/gpt-neox-20b`          | 20B  | ~301.67 GiB | 44 |

# VRAM Estimation:
pip install accelerate
accelerate estimate-memory "HuggingFace model path"

## Datasets: 
- Apache Repository:
  --> Aurora Project
  --> Mesos Project
  --> Usergrid Project

# TRAINING - EXAMPLE PROMPT [HOMEONE]:
CUDA_VISIBLE_DEVICES=2 \
  predict-complexity predict \
    --train \
    --train-data data/repositories/Apache/MESOS/mesos_dataset_train.json \
    --val-data data/repositories/Apache/MESOS/mesos_dataset_val.json \
    --model gpt2

# DATASET DOWNLOADING - EXAMPLE PROMPT:
predict-complexity extract \
  --source jira \ 
  --target MESOS \
  --domain issues.apache.org/jira \
  --out "Apache/NEWMESOS"  \
  --split 60 20 20