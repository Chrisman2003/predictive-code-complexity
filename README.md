# Predictive Code Complexity
An expandable project skeleton for an interdisciplinary project (IDP) focused on transformer-based estimation of code complexity directly from natural language problem descriptions.

## Description
This project aims to automate the estimation of software task complexity (such as Agile Story Points) by applying Natural Language Processing (NLP) to Jira tickets and GitHub issues. Because human estimation is highly subjective and inconsistent, this project leverages pre-trained Transformer models to read task descriptions and predict the underlying effort required.

The repository acts as an end-to-end Machine Learning pipeline. It includes a custom Command Line Interface (CLI) capable of dynamically extracting, shuffling, and partitioning datasets directly from live open-source issue trackers (like the Apache Software Foundation). The extracted data is then fed into a continuous deep learning training loop utilizing ordinal loss functions and cosine learning rate scheduling to accurately map textual requirements to discrete complexity integers.

## Current Status
*   **Ranking Pipeline:** The core architecture for the ranking pipeline is mostly functional, but its inference accuracy requires further refinement and improvement.
*   **Predictive Pipeline:** The predictive pipeline currently features a standard deep learning training architecture (including Early Stopping, Cosine Annealing with Warmup, and `CoralOrdinalLoss`). While the training loop is fully operational and has been validated on smaller test models (e.g., `tiny-gpt2`), comprehensive training is yet to be completed. The next major phase involves training on a significantly larger dataset evaluating a variety of pre-trained transformer backbones (such as CodeBERT, RoBERTa, or standard GPT-2) to prevent mode collapse and establish a robust language-predictive backbone. This training will be performed on a GPU-Cluster.

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

## Next steps
- Scale up the Predictive Pipeline training using larger, code-aware transformer backbones (e.g., CodeBERT) to overcome early mode collapse.
- Improve the Ranking Pipeline's inference accuracy.
- Expand extraction fallback logic to capture alternative complexity metrics (like "Lines Of Code") for non-Agile open-source projects.
- Conduct final evaluations across multiple diverse project datasets.