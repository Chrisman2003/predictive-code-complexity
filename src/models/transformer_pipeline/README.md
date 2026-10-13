# General Transformer Computation Chain:  INPUT TENSOR -> Embedding -> Encoder/Decoder Blocks -> Pooling -> Head -> OUTPUT TENSOR
# **Encoder-Only Transformer Block Outline** : NORMAL Self-Attention -> Residual -> LayerNorm -> Feedforward -> Residual -> LayerNorm
# **Decoder-Only Transformer Block Outline** : MASKED Self-Attention -> Residual -> LayerNorm -> Feedforward -> Residual -> LayerNorm

=================================================================================================================================
     TENSOR PROCESSING THROUGHOUT ARCHITECTURE (TRANSFORMER BACKBONE) + (OUTPUT HEAD) (Assuming Batch=32, Seq=512, Dim=768)
=================================================================================================================================
STAGE A: Tokenization (The Input)
- Format : 2D matrix of integers
- Shape  : (B, S) 
- Example: [[101, 4532, 883, 102, 0, 0, ...]]

STAGE B: The Embedding Layer (The Dictionary Lookup)
  - Action : Replaces integer IDs with 768-dim vectors + positional vectors.
  - Format : 3D floating-point tensor
  - Shape  : (B, S, D) -> (32, 512, 768)

STAGE C: The 12 Transformer Blocks (The Engine)
  - Action : Attention & feed-forward math. Dimensionality is preserved.
             Output of Layer N is the exact input shape of Layer N+1.
  - Format : 3D floating-point tensor (Hidden States)
  - Shape  : (B, S, D) -> (32, 512, 768) 

STAGE D: The Pooler (The Bottleneck)
  - Action : [CLS] extraction or Masked Mean Pooling. Collapses the 512 
             words into a single 768-dimensional summary vector.
  - Format : 2D floating-point tensor
  - Shape  : (B, D) -> (32, 768)

STAGE E: The True CORAL Head (The Output)
  - Action : Pooled tensor multiplied by shared weight vector (compressing 
             768 to 1). K-1 biases are then broadcasted to create thresholds.
  - Format : 2D floating-point tensor (Logits)
  - Shape  : (B, K-1) -> (32, 6)
=========================================================================





===================================================================================
                  PREDICTION PIPELINE CALL-CHAIN ARCHITECTURE
===================================================================================

                    +-----------------------------------+
                    |    User / External Calling Code   |
                    +-----------------------------------+
                                │           │
               1. tune_and_fit()│           │ 6. predict()
                                ▼           ▼
            +───────────────────────────────────────────────────+
            |  pipeline.py (StoryPointPredictionPipeline API)   |
            +───────────────────────────────────────────────────+------------|
                 │              │              │              │              |
    2. Tokenize  │   3. Tune    │  4. Build    │  5. Train    │ 7. Batch     |
       & Vector  │   (Opt)      │     Model    │     Loop     │    Inference |
                 ▼              ▼              ▼              ▼              ▼
           +──────────+   +──────────+   +──────────+   +──────────+   +──────────+
           | dataset  |   |  tuner   |   |   arch   |   | trainer  |   |   arch   |
           +──────────+   +──────────+   +──────────+   +──────────+   +──────────+
           |StoryPoint|   |Hyperparam|   |Transformr|   |StoryPoint|   | Model    |
           | Dataset  |   |  Tuner   |   | StoryPt  |   | Trainer  |   | Forward  |
           +──────────+   +────┬─────+   +────┬─────+   +────┬─────+   +────┬─────+
                               │              │              │              │
                          Executes       Configures     Runs Mini-      Thresholds
                          Trials         Freezing &     Batches         Sigmoids  to
                               │          LLRD LR            │          Fibonacci
                               ▼              ▼              │              │
                          +─────────+    +─────────+         │              ▼
                          | trainer |    | freeze/ |         │         +──────────+
                          +─────────+    |  llrd   |         │         | Output   |
                                         +─────────+         │         | Points   |
                                                             │         +──────────+
 ┌───────────────────────────────────────────────────────────┘
 │
 ▼
===================================================================================
                        TRAINING EXECUTION ENGINE (trainer.py)
===================================================================================

  +─────────────────────────────────────────────────────────────────────────────+
  |  StoryPointTrainer.train()                                                  |
  |                                                                             |
  |  FOR epoch IN range(epochs):                                                |
  |  │                                                                          |
  |  ├──► FOR batch IN train_loader:                                            |
  |  │    │                                                                     |
  |  │    ├──► architecture.py ──► TransformerStoryPointModel.forward()         |
  |  │    │                                                                     |
  |  │    ├──► loss.py         ──► CoralOrdinalLoss.forward()                   |
  |  │    │                                                                     |
  |  │    ├──► PyTorch Engine  ──► loss.backward() & optimizer.step()           |
  |  │    │                                                                     |
  |  │    └──► trainer.py      ──► _adjust_lr()  [Cosine Warmup Scheduler]      |
  |  │                                                                          |
  |  ├──► trainer.py ────────────► _eval()       [Validation Loss & MAE]        |
  |  │                                                                          |
  |  └──► trainer.py ────────────► EarlyStopping.check()                        |
  |                                 ├──► Continue training                      |
  |                                 └──► Stop early (Patience met)              |
  +─────────────────────────────────────────────────────────────────────────────+