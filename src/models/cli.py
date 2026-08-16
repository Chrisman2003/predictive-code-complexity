"""
Global Command Line Interface for Predictive Code Complexity.
Routes commands to the ranking, prediction, and generative pipelines.
"""
import argparse
import sys

# Import the ranking pipeline runner
from src.models.ranking_pipeline.pipeline import run_ranking_pipeline


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
    # 2. PREDICTION PIPELINE SUBCOMMAND (Placeholder)
    # ==========================================
    predict_parser = subparsers.add_parser(
        "predict", 
        help="Predict absolute code complexity metric from description"
    )
    predict_parser.add_argument("--repo", type=str, help="GitHub repository")
    predict_parser.add_argument("--problem", type=str, help="Problem description file")

    # ==========================================
    # 3. GENERATIVE PIPELINE SUBCOMMAND (Placeholder)
    # ==========================================
    generate_parser = subparsers.add_parser(
        "generate", 
        help="Generate synthetic problem descriptions or implementation code"
    )
    generate_parser.add_argument("--repo", type=str, help="GitHub repository")

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
        print("[!] Prediction pipeline is not yet implemented.")
        sys.exit(1)

    elif args.command == "generate":
        print("[!] Generative pipeline is not yet implemented.")
        sys.exit(1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()