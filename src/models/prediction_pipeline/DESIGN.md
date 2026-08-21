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