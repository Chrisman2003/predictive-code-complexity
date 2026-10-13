# TRANSFORMER PIPELINE
Estimate Agile **story points** from the text of an issue by fine-tuning a pretrained transformer.
- **Input:** issue text (title and description as one string).
- **Output:** story points on a Fibonacci-style scale, by default `1, 2, 3, 5, 8, 13, 20`.
- **Backbones:** any Hugging Face encoder or decoder. The defaults target `microsoft/codebert-base`; decoder-only models such as `gpt2` are supported.
- **Two output heads:** an ordinal classification head (CORAL, the default) or a scalar regression head.
- **Built for small datasets** (hundreds to a few thousand labelled issues): two-phase fine-tuning, layer-wise learning-rate decay, early stopping, and restoring the best epoch.
This package is the training and inference engine behind the `predict-complexity predict` command.


## Data format
Training, validation and test files are JSON (or JSONL / CSV) with one record per issue:
```json
[
  {"story": "Add retry logic to the scheduler when an agent disconnects", "points": 3.0},
  {"story": "Fix NPE in the log rotation path", "points": 1.0}
]
```
- Records with a missing story or a missing/NaN `points` value are skipped, and the count is printed.
- **Ordinal head:** each point value is snapped to the nearest scale value (ties go to the lower one, for example 4 → 3, 6 → 5, 40 → 21) to form the training label. Metrics are still computed against the original, unsnapped points.
- **Regression head:** the original point value is the target, so it is not limited to the scale.


## Training protocol
1. **Phase A, frozen backbone.** For the first `--freeze-epochs` epochs only the randomly initialised head trains. The backbone runs in eval mode, so dropout does not add noise to its features.
2. **Phase B, full fine-tuning.** The backbone is unfrozen. Learning rates use **layer-wise decay (LLRD)**: the head has its own rate, the top transformer block gets the backbone rate, and each lower block gets a smaller one, with the embeddings lowest. This protects the pretrained lower layers. Biases and normalisation weights get no weight decay.
3. **Schedule.** Each phase has its own linear warmup (a fraction of that phase's steps). Phase B then follows a cosine decay to zero over the remaining steps. The freeze state is derived from the epoch number, so a resumed run is always correct.
4. **Early stopping and model selection.** Training stops when validation loss has not improved for `--patience` epochs. The weights of the **best epoch** (lowest validation loss) are restored before saving, not the last epoch's.
5. **Diagnostics.** Each epoch prints train/validation loss, validation MAE in story points and in ranks, and the spread of the predictions. A warning appears if predictions collapse to a single value.


## Evaluation
After training, the CLI prints metrics on the validation set (and the test set with `--test-data`). All are in **raw story points**, so they are comparable with the effort-estimation literature:
- **MAE** and **MdAE** (mean and median absolute error)
- **macro-MAE**, the mean of the per-true-value MAEs, which a constant predictor scores badly on
- **SA**, standardized accuracy against random guessing
- **Median and mean baselines** (a predictor that always outputs the training median or mean) and the percentage improvement over the median baseline

Always read the model's MAE next to the baseline line. On small, noisy datasets, always predicting the median is a strong baseline that is easy to match without learning anything.


## Outputs
| Path | Content |
|---|---|
| `binaries/runs/<model>_<time>_s<seed>/` (or `--run-dir`) | one folder per run |
| `.../checkpoint_latest.pt` | model, optimizer, schedule position and best-epoch state, written after every epoch (for `--resume`) |
| `.../best_model.pt` | weights of the best epoch |
| `.../run_config.json` | hyperparameters, head settings and final metrics of the run |
| `binaries/story_point_model.pt` (`--weights`) | final model for inference. It stores the architecture settings, scale, tokenizer name and max length, so it loads without extra flags |
| `binaries/tuning/trial_*` | isolated checkpoints of each tuning trial |

Checkpoints written before this refactor use a different head and cannot be loaded. Retrain to produce a new one.


## Hyperparameter tuning
`--tune` runs a two-stage search before the final training:
1. **Random search** over head learning rate, backbone learning rate, weight decay, dropout, head width and batch size.
2. **Focused grid** around the best random configuration.

Each configuration trains for a shortened schedule and is scored by mean validation loss over several seeds. The winner is then trained with the normal epoch and patience settings. Tuning uses the same training code as the final run. Expect it to take many times longer than a single training run.



=================================================================================================================================
     TENSOR PROCESSING THROUGHOUT ARCHITECTURE (TRANSFORMER BACKBONE) + (OUTPUT HEAD) (Assuming Batch=32, Seq=512, Dim=768)
=================================================================================================================================
# General Transformer Computation Chain:  INPUT TENSOR -> Embedding -> Encoder/Decoder Blocks -> Pooling -> Head -> OUTPUT TENSOR
**Encoder-Only Transformer Block Outline** : NORMAL Self-Attention -> Residual -> LayerNorm -> Feedforward -> Residual -> LayerNorm
**Decoder-Only Transformer Block Outline** : MASKED Self-Attention -> Residual -> LayerNorm -> Feedforward -> Residual -> LayerNorm

# Tensor ProcessFlow
STAGE A: Tokenization (The Input)
- Format : 2D matrix of integers
- Shape  : (B, S) 
- Example: [[101, 4532, 883, 102, 0, 0, ...]]

STAGE B: The Embedding Layer (The Dictionary Lookup)
  - Action : Replaces integer IDs with 768-dim vectors + positional vectors.
  - Format : 3D floating-point tensor
  - Shape  : (B, S, D) -> (32, 512, 768)

STAGE C: The 12 Transformer Blocks (The Engine)
  - Action : Attention & feed-forward math. Dimensionality is preserved. Output of Layer N is the exact input shape of Layer N+1.
  - Format : 3D floating-point tensor (Hidden States)
  - Shape  : (B, S, D) -> (32, 512, 768) 

STAGE D: The Pooler (The Bottleneck)
  - Action : [CLS] extraction or a different pooling Version. Collapses the 512 words into a single 768-dimensional summary vector.
  - Format : 2D floating-point tensor
  - Shape  : (B, D) -> (32, 768)

STAGE E: The True CORAL Head (The Output)
  - Action : Pooled tensor multiplied by shared weight vector (compressing 
             768 to 1). K-1 biases are then broadcasted to create thresholds.
  - Format : 2D floating-point tensor (Logits)
  - Shape  : (B, K-1) -> (32, 6)
