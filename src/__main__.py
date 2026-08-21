"""
Global Command Line Interface for Predictive Code Complexity.
Routes commands to the ranking, prediction, and generative pipelines.
"""
import argparse
import json
import os
import sys
from typing import List, Tuple

# Import pipeline runners
from src.models.ranking_pipeline.pipeline import run_ranking_pipeline
from src.models.prediction_pipeline.pipeline import StoryPointPredictionPipeline
from src.dataloader.extractor import run_extraction_pipeline

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
        default="weights/story_point_model.pt",
        help="Path to pre-trained model weights (.pt file)"
    ) 
    
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
    
    # Platform specific arguments
    extract_parser.add_argument("--domain", type=str, help="Jira domain (e.g., yourcompany.atlassian.net)")
    extract_parser.add_argument("--email", type=str, help="Jira account email (or set JIRA_EMAIL env var)")
    extract_parser.add_argument("--token", type=str, help="API token (or set GITHUB_TOKEN / JIRA_API_TOKEN env vars)")

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
        # Set default weights path if not provided
        weights_path = getattr(args, "weights", "weights/story_point_model.pt")
        
        if args.train:
            if not args.train_data or not args.val_data:
                print("[!] Error: --train-data and --val-data paths are required for training.")
                sys.exit(1)

            print(f"[*] Loading training data from: {args.train_data}")
            train_stories, train_points = load_dataset_file(args.train_data)
            
            print(f"[*] Loading validation data from: {args.val_data}")
            val_stories, val_points = load_dataset_file(args.val_data)

            print(f"[*] Initializing Prediction Pipeline Backbone: {args.model}")
            pipeline = StoryPointPredictionPipeline(model_name=args.model)

            print("================================================================================")
            print("STARTING STORY POINT PREDICTION PIPELINE TRAINING")
            print(f"Backbone: {args.model} | Hyperparameter Tuning: {args.tune}")
            print("================================================================================")

            metrics = pipeline.tune_and_fit(
                train_stories=train_stories,
                train_points=train_points,
                val_stories=val_stories,
                val_points=val_points,
                run_tuning=args.tune
            )
            # AUTOMATIC WEIGHT SAVING ON TRAINING COMPLETION
            pipeline.save_weights(weights_path)

            print("================================================================================")
            print(f"TRAINING COMPLETE | Final Validation Loss: {metrics.get('best_val_loss', 0.0):.4f} | Val MAE: {metrics.get('val_mae', 0.0):.4f}")
            print("================================================================================")

        elif args.story:
            # AUTOMATIC WEIGHT LOADING FOR INFERENCE
            print(f"[*] Running inference with model: {args.model}")
            pipeline = StoryPointPredictionPipeline(model_name=args.model)
            if os.path.exists(weights_path):
                pipeline.load_weights(weights_path)
            else: 
                print(f"[!] Warning: No saved weights found at '{weights_path}'. Predicting with initialized base weights.")

            predicted_point = pipeline.predict([args.story])[0]
            print(f"\n[+] Input Story: {args.story}")
            print(f"[+] Predicted Story Points: {predicted_point}\n")

        else:
            print("[!] Specify either --train (with dataset paths) or --story for prediction.")
            parser.print_help()
            sys.exit(1)

    elif args.command == "generate":
        print("[!] Generative pipeline is not yet implemented.")
        sys.exit(1)
    
    elif args.command == "extract":
        print("================================================================================")
        print(f"STARTING DATA EXTRACTION PIPELINE | Source: {args.source.upper()}")
        print("================================================================================")
        run_extraction_pipeline(args)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()