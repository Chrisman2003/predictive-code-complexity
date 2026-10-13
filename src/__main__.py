"""
Global Command Line Interface for Predictive Code Complexity.
Routes commands to the ranking, prediction, and generative pipelines.
"""
import argparse
import json
import os
import sys
from typing import List, Tuple, Dict
from src.dataloader.build_dataset import run_mining_pipeline

# Import pipeline runners
from src.models.ranking_pipeline.pipeline import run_ranking_pipeline
from src.models.transformer_pipeline.pipeline import TransformerPredictionPipeline
from src.dataloader.extractor import run_extraction_pipeline


def print_metrics(title: str, m: Dict[str, float]) -> None:
    """Pretty-print the dict returned by StoryPointPredictionPipeline.evaluate()."""
    print(f"--- {title} (n={m['n']}) ---")
    print(f"MAE {m['mae']:.3f} | MdAE {m['mdae']:.3f} | macro-MAE {m['macro_mae']:.3f} | pred std {m['pred_std']:.3f}")
    if "baseline_median_mae" in m:
        print(f"Median baseline MAE {m['baseline_median_mae']:.3f} | Mean baseline MAE {m['baseline_mean_mae']:.3f} "
              f"| improvement vs median {m['improvement_vs_median_pct']:+.1f}% | SA {m['sa']:.1f}%")
 



def load_dataset_file(filepath: str) -> Tuple[List[str], List[float]]:
    """
    Loads user stories and target story points from a JSON or CSV file.
    Expected JSON format: [{"story": "...", "points": 3.0}, ...]
    Expected CSV columns: 'story', 'points'
    """
    if not os.path.exists(filepath):
        print(f"[!] Error: File not found at {filepath}")
        sys.exit(1)

    stories, points = [], []

    if filepath.endswith(".json") or filepath.endswith(".jsonl"):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                stories.append(item["story"])
                points.append(float(item["points"]))

    elif filepath.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(filepath)
        stories = df["story"].astype(str).tolist()
        points = df["points"].astype(float).tolist()

    else:
        print(f"[!] Error: Unsupported file format '{filepath}'. Use .json or .csv.")
        sys.exit(1)

    return stories, points


def setup_parser() -> argparse.ArgumentParser:
    """Sets up the CLI argument parser with sub-commands for each pipeline."""
    parser = argparse.ArgumentParser(
        prog="predict-complexity",
        description="CLI for mapping Problem Descriptions to Code Complexity metrics."
    )
    
    subparsers = parser.add_subparsers(
        title="Pipelines",
        dest="command",
        help="Choose which pipeline to run"
    )

    # ==========================================
    # 1. RANKING PIPELINE SUBCOMMAND
    # ==========================================
    rank_parser = subparsers.add_parser(
        "rank", 
        help="Predict relative complexity ranking of GPU paradigms"
    )
    rank_parser.add_argument(
        "--repo", 
        type=str, 
        required=False,
        help="URL or path to GitHub repository to analyze (Future Integration)"
    )
    rank_parser.add_argument(
        "--model", 
        type=str, 
        default="qwen2.5-coder",
        help="Local Ollama model to use for inference"
    )
    rank_parser.add_argument(
        "--use-few-shot", 
        action="store_true",
        help="Enable few-shot examples in the prompt"
    )
    
    # ==========================================
    # 2. PREDICTION PIPELINE SUBCOMMAND
    # ==========================================
    predict_parser = subparsers.add_parser(
        "predict", 
        help="Train or predict absolute story point complexity from User Stories"
    )
    predict_parser.add_argument( # Training System
        "--train",
        action="store_true",
        help="Trigger the training pipeline"
    )
    predict_parser.add_argument( # Inference System
        "--story",
        type=str,
        help="Single User Story text string to run inference on"
    )
    predict_parser.add_argument(
        "--train-data",
        type=str,
        help="Path to training dataset (.json, .jsonl or .csv)"
    )
    predict_parser.add_argument(
        "--val-data",
        type=str,
        help="Path to validation dataset (.json, .jsonl or .csv)"
    )
    predict_parser.add_argument(
        "--test-data",
        type=str,
        help="Optional held-out test set; evaluated ONCE after training with the best checkpoint"
    )
    predict_parser.add_argument(
        "--tune",
        action="store_true",
        help="Run hyperparameter tuning (Random Search -> Grid Search) before training"
    )
    predict_parser.add_argument("--tune-trials", type=int, default=6, help="Random-search trials when --tune is set")
    predict_parser.add_argument("--tune-seeds", type=int, default=2, help="Seeds averaged per tuning config")
    predict_parser.add_argument(
        "--model",
        type=str,
        default="microsoft/codebert-base",
        help="Pre-trained Transformer backbone model"
    )
    predict_parser.add_argument(
        "--weights",
        type=str,
        default="binaries/story_point_model.pt",
        help="Path to pre-trained model weights (.pt file)"
    )
    # Model / data options
    predict_parser.add_argument(
        "--pooling", type=str, choices=["cls", "mean", "last"], default=None,
        help="Pooling over backbone outputs (default: mean for encoders, last token for decoders)"
    )
    predict_parser.add_argument("--max-length", type=int, default=256, help="Max tokens per story")
    predict_parser.add_argument(
        "--head-type", type=str, choices=["ordinal", "regression"], default="ordinal",
        help="Output head: CORAL ordinal classification (default) or scalar regression"
    )
    predict_parser.add_argument(
        "--target-transform", type=str, choices=["none", "log1p"], default="none",
        help="Regression head only: predict raw points or log(1 + points)"
    )
    predict_parser.add_argument(
        "--regression-loss", type=str, choices=["l1", "huber", "mse"],
        help="Regression head only: loss function (default l1, which optimizes MAE directly)"
    )
    predict_parser.add_argument("--huber-delta", type=float, help="Regression head only: delta of the huber loss")
    # Training hyperparameters (unset = use training.TRAINING_DEFAULTS)
    predict_parser.add_argument("--lr", type=float, help="Backbone top-block learning rate")
    predict_parser.add_argument("--head-lr", type=float, help="Learning rate of the ordinal head")
    predict_parser.add_argument("--llrd-decay", type=float, help="Per-layer LR decay factor")
    predict_parser.add_argument("--weight-decay", type=float)
    predict_parser.add_argument("--dropout", type=float, help="Head dropout")
    predict_parser.add_argument("--hidden-dim", type=int, help="Head hidden size (0 = linear CORAL head)")
    predict_parser.add_argument("--batch-size", type=int)
    predict_parser.add_argument("--epochs", type=int)
    predict_parser.add_argument("--patience", type=int, help="Early-stopping patience (epochs)")
    predict_parser.add_argument("--freeze-epochs", type=int, help="Epochs with a frozen backbone")
    predict_parser.add_argument("--warmup-ratio", type=float, help="Warmup fraction of each phase's steps")
    predict_parser.add_argument("--seed", type=int)
    # Checkpointing
    predict_parser.add_argument(
        "--run-dir", type=str,
        help="Directory for checkpoints and run_config.json (default: binaries/runs/<model>_<timestamp>_s<seed>)"
    )
    predict_parser.add_argument(
        "--resume", type=str,
        help="Path to a checkpoint_latest.pt to resume training from (not combinable with --tune)"
    )
    


    """
    predict_parser = subparsers.add_parser(
        "predict", 
        help="Train or predict absolute story point complexity from User Stories"
    )
    predict_parser.add_argument( # Training System
        "--train",
        action="store_true",
        help="Trigger the training pipeline"
    )
    predict_parser.add_argument( # Inference System
        "--story",
        type=str,
        help="Single User Story text string to run inference on"
    )
    predict_parser.add_argument(
        "--train-data",
        type=str,
        help="Path to training dataset (.json or .csv)"
    )
    predict_parser.add_argument(
        "--val-data",
        type=str,
        help="Path to validation dataset (.json or .csv)"
    )
    predict_parser.add_argument(
        "--tune",
        action="store_true",
        help="Run hyperparameter tuning (Random Search -> Grid Search) before training"
    )
    predict_parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Pre-trained Transformer backbone model (e.g., gpt2, microsoft/deberta-v3-base)"
    )
    predict_parser.add_argument(
        "--weights",
        type=str,
        default="binaries/story_point_model.pt",
        help="Path to pre-trained model weights (.pt file)"
    ) 
    """

    # ==========================================
    # 3. GENERATIVE PIPELINE SUBCOMMAND (Placeholder)
    # ==========================================
    generate_parser = subparsers.add_parser(
        "generate", 
        help="Generate synthetic problem descriptions or implementation code"
    )
    generate_parser.add_argument("--repo", type=str, help="GitHub repository")

    # ==========================================
    # 4. DATA EXTRACTION PIPELINE SUBCOMMAND
    # ==========================================
    extract_parser = subparsers.add_parser(
        "extract", 
        help="Extract User Stories and Story Points from Jira or GitHub"
    )
    extract_parser.add_argument(
        "--source", 
        type=str, 
        choices=["github", "jira"], 
        required=True, 
        help="Source platform to extract data from"
    )
    extract_parser.add_argument(
        "--target", 
        type=str, 
        required=True, 
        help="GitHub repo (e.g., 'owner/repo') or Jira Project Key (e.g., 'PROJ')"
    )
    extract_parser.add_argument(
        "--out", 
        type=str, 
        default="data/raw_dataset.json", 
        help="Path to save the extracted JSON dataset"
    )
    extract_parser.add_argument(
        "--split", 
        type=int, 
        nargs=3, 
        metavar=("TRAIN", "VAL", "TEST"),
        help="Space-separated percentages for Train/Val/Test splits (e.g., --split 80 10 10). If provided, it overrides --out."
    )
    # Platform specific arguments
    extract_parser.add_argument("--domain", type=str, help="Jira domain (e.g., yourcompany.atlassian.net)")
    extract_parser.add_argument("--email", type=str, help="Jira account email (or set JIRA_EMAIL env var)")
    extract_parser.add_argument("--token", type=str, help="API token (or set GITHUB_TOKEN / JIRA_API_TOKEN env vars)")

    # ==========================================
    # 5. REPOSITORY MINING SUBCOMMAND
    # ==========================================
    mine_parser = subparsers.add_parser(
        "mine", 
        help="Mine a local git repository to append LOC to an existing dataset"
    )
    mine_parser.add_argument(
        "--data", 
        required=True, 
        help="Path to the extracted JSON dataset (e.g., data/processed/MESOS/mesos_dataset_train.json)"
    )
    mine_parser.add_argument(
        "--repo", 
        required=True, 
        help="Path to the local cloned git repository (e.g., ../mesos)"
    )
    mine_parser.add_argument(
        "--out", 
        help="Optional: Path to save the updated dataset (defaults to overwriting --data)"
    )
    return parser


def main():
    """Main entry point for the CLI routing."""
    parser = setup_parser()
    args = parser.parse_args()

    if args.command == "rank":
        print(f"[*] Initializing Ranking Pipeline with model: {args.model}")
        # Note: Eventually, you will pass args.repo to a dataloader here
        run_ranking_pipeline(
            model_name=args.model,
            use_few_shot=args.use_few_shot,
            repo_url=args.repo  # Pass the repo flag down to the pipeline
        )
    elif args.command == "predict":
        weights_path = args.weights
        
        if args.train:
            if not args.train_data or not args.val_data:
                print("[!] Error: --train-data and --val-data paths are required for training.")
                sys.exit(1)
            if args.resume and args.tune:
                print("[!] Error: --resume cannot be combined with --tune.")
                sys.exit(1)
 
            print(f"[*] Loading training data from: {args.train_data}")
            train_stories, train_points = load_dataset_file(args.train_data)
            
            print(f"[*] Loading validation data from: {args.val_data}")
            val_stories, val_points = load_dataset_file(args.val_data)
 
            if args.head_type == "ordinal" and (
                args.target_transform != "none" or args.regression_loss or args.huber_delta is not None
            ):
                print("[!] --target-transform / --regression-loss / --huber-delta only apply to "
                      "--head-type regression and are ignored.")
 
            print(f"[*] Initializing Prediction Pipeline Backbone: {args.model} | Head: {args.head_type}")
            pipeline = TransformerPredictionPipeline(
                model_name=args.model, pooling=args.pooling, max_length=args.max_length,
                #head_type=args.head_type, target_transform=args.target_transform,
            )
 
            print("================================================================================")
            print("STARTING STORY POINT PREDICTION PIPELINE TRAINING")
            print(f"Backbone: {args.model} | Hyperparameter Tuning: {args.tune}")
            print("================================================================================")
 
            # Only flags the user actually set; everything else falls back to TRAINING_DEFAULTS.
            fit_kwargs = {k: v for k, v in {
                "lr": args.lr, "head_lr": args.head_lr, "llrd_decay": args.llrd_decay,
                "weight_decay": args.weight_decay, "dropout": args.dropout, "hidden_dim": args.hidden_dim,
                "batch_size": args.batch_size, "epochs": args.epochs, "patience": args.patience,
                "freeze_epochs": args.freeze_epochs, "warmup_ratio": args.warmup_ratio,
                "seed": args.seed, "run_dir": args.run_dir,
                "regression_loss": args.regression_loss, "huber_delta": args.huber_delta,
            }.items() if v is not None}
 
            metrics = pipeline.tune_and_fit(
                train_stories=train_stories,
                train_points=train_points,
                val_stories=val_stories,
                val_points=val_points,
                run_tuning=args.tune,
                resume_path=args.resume,
                tune_trials=args.tune_trials,
                tune_seeds=args.tune_seeds,
                **fit_kwargs,
            )
            # AUTOMATIC WEIGHT SAVING ON TRAINING COMPLETION (the trainer has already restored the best epoch)
            pipeline.save_weights(weights_path)
 
            print("================================================================================")
            print(f"TRAINING COMPLETE | Best epoch: {metrics.get('best_epoch')} | "
                  f"Val Loss: {metrics.get('best_val_loss', 0.0):.4f} | "
                  f"Val MAE (points): {metrics.get('val_mae', 0.0):.4f} | "
                  f"Val MAE (rank): {metrics.get('val_mae_rank', 0.0):.4f}")
            print("================================================================================")
 
            print_metrics("VALIDATION", pipeline.evaluate(val_stories, val_points, train_points))
            if args.test_data:
                print(f"[*] Loading test data from: {args.test_data}")
                test_stories, test_points = load_dataset_file(args.test_data)
                print_metrics("TEST", pipeline.evaluate(test_stories, test_points, train_points))
    
    elif args.command == "generate":
        print("[!] Generative pipeline is not yet implemented.")
        sys.exit(1)
    
    elif args.command == "extract":
        print("================================================================================")
        print(f"STARTING DATA EXTRACTION PIPELINE | Source: {args.source.upper()}")
        print("================================================================================")
        run_extraction_pipeline(args)

    elif args.command == "mine":
        run_mining_pipeline(args)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()