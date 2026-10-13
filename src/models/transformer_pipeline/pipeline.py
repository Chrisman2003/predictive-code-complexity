import torch
import os
import random
import json
import numpy as np
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from torch.utils.data import DataLoader
from transformers import PretrainedConfig, AutoTokenizer
from src.models.transformer_pipeline.dataset import OrdinalDataCollator
from src.models.transformer_pipeline.architecture import TransformerArchitecture
from src.models.transformer_pipeline.trainer import TransformerTrainer
from src.models.transformer_pipeline.dataset import StoryPointDataset #, make_collate_fn
from src.models.transformer_pipeline.hyperparameter_tuner import HyperparameterTuner
from src.validation.metrics import compute_metrics

TRAINING_DEFAULTS: Dict[str, Any] = {
    "base_lr": 3e-5,            # backbone top-block LR
    "head_lr": 1e-3,       # randomly initialised head
    "llrd_decay": 0.9,
    "weight_decay": 0.01,
    "dropout": 0.2,
    "mlp_head_width": 256,     # 0 = linear CORAL head
    "batch_size": 16,
    "epochs": 20,
    "patience": 8,
    "freeze_epochs": 1,
    "warmup_ratio": 0.1,
    "seed": 42,
    # Further Parameters
}

CHECKPOINT_VERSION = 2  # 1 = old Linear(K-1) head; 2 = CORAL head + config-carrying checkpoint
DEFAULT_SCALE = [1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0]

class TransformerPredictionPipeline:
    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        scale: Optional[List[float]] = None,
        pooling: str = "cls",   # None = "last" for decoders, "mean" for encoders
        max_length: int = 256,
    ):
        self.model_name = model_name
        self.scale = sorted(scale if scale is not None else DEFAULT_SCALE)
        self.pooling = pooling
        self.max_length = max_length
        self.tokenizer = self._build_tokenizer(model_name)
        self.model = None

    # ------------------------------------------------------------ configuration

    @staticmethod
    def _build_tokenizer(model_name: str):
        """Initializes and configures a Hugging Face tokenizer for the model.

        Loads the pre-trained tokenizer associated with the specified model identifier. If the tokenizer lacks a dedicated padding token, it assigns the end-of-sequence
        (EOS) token as the padding token. It also strictly enforces right-side padding, which is a mathematical prerequisite for accurately extracting the final sequence 
        embedding when using last-token pooling.

        Args:
            model_name (str): The Hugging Face model identifier (e.g., 'gpt2' or 
                'microsoft/codebert-base').

        Returns:
            PreTrainedTokenizer: The instantiated and configured Hugging Face tokenizer.
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"  # required by last-token pooling
        return tokenizer

    def _default_run_dir(self, seed: int) -> str:
        """Generates a unique, filesystem-safe directory path for saving run artifacts.

        Sanitizes the Hugging Face model identifier by replacing special characters (like slashes) with hyphens. It then appends a highly readable timestamp and the 
        explicit random seed to guarantee that concurrent or repeated training runs do not overwrite each other's checkpoints or logs.

        Args:
            seed (int): The random seed used for this specific training run.

        Returns:
            str: A formatted relative path for the output directory (e.g., 'binaries/runs/microsoft-codebert-base_2026-10-12_09.17.00_seed42').
        """
        slug = re.sub(r"[^A-Za-z0-9]+", "-", self.model_name).strip("-")
        
        # %Y-%m-%d produces 2026-10-12
        # %H.%M.%S produces 09.17.00
        timestamp = time.strftime("%Y-%m-%d_%H.%M.%S")
        return os.path.join("binaries", "runs", f"{slug}_{timestamp}_seed{seed}")

    def _make_dataset(self, stories: List[str], points: Optional[List[float]]) -> StoryPointDataset:
        """Constructs a StoryPointDataset from raw text and optional point labels.

        Wraps the provided text strings and corresponding point values into a PyTorch-compatible Dataset. It automatically injects the class's pre-configured 
        tokenizer, the point-to-ordinal scale mapping, and the maximum sequence length constraints required by the transformer.

        Args:
            stories (List[str]): A list of raw text descriptions representing user stories or Jira tickets.
            points (Optional[List[float]]): A corresponding list of actual story points. This should be None if the dataset is being built strictly for inference.

        Returns:
            StoryPointDataset: A PyTorch dataset object fully configured for the training or evaluation data loaders.
        """
        return StoryPointDataset(stories, points, self.tokenizer, scale=self.scale, max_length=self.max_length)

    def _write_run_config(self, run_dir: str, hyperparams: Dict[str, Any], metrics: Dict[str, Any]) -> None:
        """Serializes the complete training configuration and final metrics to a JSON file.

        Captures the model architecture identifiers, preprocessing parameters, training hyperparameters, and the resulting validation metrics. It saves this data as 
        'run_config.json' within the specified run directory to ensure the experiment is fully documented, searchable, and mathematically reproducible.

        Args:
            run_dir (str): The directory path where the JSON configuration file will be saved.
            hyperparams (Dict[str, Any]): A dictionary containing the explicit hyperparameters used for the training run (e.g., learning rate, batch size, dropout, weight decay).
            metrics (Dict[str, Any]): A dictionary containing the final evaluation metrics achieved by the best epoch (e.g., validation loss, point MAE, rank MAE, and standard deviation).
        """
        record = {
            "model_name": self.model_name,
            "pooling": self.model.pooling,
            "max_length": self.max_length,
            "scale": self.scale,
            "hyperparameters": hyperparams,
            "metrics": metrics,
        }
        with open(os.path.join(run_dir, "run_config.json"), "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

    # ------------------------------------------------------------ pipeline runner
    def fit(
        self,
        # Data
        train_stories: List[str],
        train_points: List[float],
        val_stories: List[str],
        val_points: List[float],
        # Hyperparameters (defaults live in training.TRAINING_DEFAULTS)
        base_lr: float = TRAINING_DEFAULTS["base_lr"],
        head_lr: float = TRAINING_DEFAULTS["head_lr"],
        llrd_decay: float = TRAINING_DEFAULTS["llrd_decay"],
        weight_decay: float = TRAINING_DEFAULTS["weight_decay"],
        dropout: float = TRAINING_DEFAULTS["dropout"],
        mlp_head_width: int = TRAINING_DEFAULTS["mlp_head_width"],
        batch_size: int = TRAINING_DEFAULTS["batch_size"],
        epochs: int = TRAINING_DEFAULTS["epochs"],
        patience: int = TRAINING_DEFAULTS["patience"],
        freeze_epochs: int = TRAINING_DEFAULTS["freeze_epochs"],
        warmup_ratio: float = TRAINING_DEFAULTS["warmup_ratio"],
        seed: int = TRAINING_DEFAULTS["seed"],
        run_dir: Optional[str] = None,
        resume_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Executes the direct training process with explicit manual hyperparameters.

        Bypasses automated hyperparameter search to train the model using explicitly provided configuration values. It converts the raw text and point lists into PyTorch
        datasets, establishes the storage directory, invokes the core TransformerTrainer routine, locks the resulting model into evaluation mode, and serializes the 
        final configuration and metrics to disk.

        Args:
            Configuration
            Hyperparameters
        
        Returns: 
            Dict[str, Any]: A dictionary containing the validation metrics from the best epoch, along with the 'run_dir' key indicating where the artifacts were saved.
        """
        train_ds = self._make_dataset(train_stories, train_points)
        val_ds = self._make_dataset(val_stories, val_points)
        run_dir = run_dir or self._default_run_dir(seed)

        hyperparams = dict(
            base_lr=base_lr, head_lr=head_lr, llrd_decay=llrd_decay, weight_decay=weight_decay, dropout=dropout,
            mlp_head_width=mlp_head_width, batch_size=batch_size, epochs=epochs, patience=patience,
            freeze_epochs=freeze_epochs, warmup_ratio=warmup_ratio, seed=seed,
        )
        self.model, metrics = TransformerTrainer.run_training(
            model_name=self.model_name,
            num_classes=len(self.scale),
            pooling=self.pooling,
            train_ds=train_ds,
            val_ds=val_ds,
            checkpoint_dir=run_dir,
            resume_path=resume_path,
            **hyperparams,
        )
        self.model.eval()
        metrics["run_dir"] = run_dir
        self._write_run_config(run_dir, hyperparams, metrics)
        return metrics

    def tune_and_fit(
        self,
        train_stories: List[str],
        train_points: List[float],
        val_stories: List[str],
        val_points: List[float],
        run_tuning: bool = False,
        resume_path: Optional[str] = None,
        tune_trials: int = 6,
        tune_seeds: int = 2,
        **fit_kwargs: Any,   # any fit() hyperparameter; searched keys are overridden by the tuner
    ) -> Dict[str, Any]:
        """Orchestrates optional hyperparameter tuning followed by complete model training.

        Conditionally triggers a random-search hyperparameter optimization phase using the `HyperparameterTuner`. If enabled, it evaluates multiple parameter configurations
        across multiple random seeds, extracts the optimal configuration, overrides the baseline `fit_kwargs`, and executes the final training pass. If tuning is disabled, 
        it bypasses the search and directly executes standard training.

        Args:
            Configuration
            Hyperparameters

        Returns:
            Dict[str, Any]: A dictionary containing the validation metrics from the final trained model, alongside the execution telemetry (e.g., artifact directory).
        """
        if not run_tuning:
            return self.fit(train_stories, train_points, val_stories, val_points,
                            resume_path=resume_path, **fit_kwargs)
        if resume_path:
            raise ValueError("--resume cannot be combined with --tune (tuning trials are independent runs).")

        train_ds = self._make_dataset(train_stories, train_points)
        val_ds = self._make_dataset(val_stories, val_points)

        tuner = HyperparameterTuner(
            train_ds, val_ds, model_name=self.model_name, pooling=self.pooling, base_config=fit_kwargs,
        )
        tune_res = tuner.execute_tuning(num_random_trials=tune_trials, n_seeds=tune_seeds)
        best = tune_res["best_config"]
        print(f"[*] Tuning finished | best val loss {tune_res['best_val_loss']:.4f} | config: {best}")

        fit_kwargs.update(best) # Optimal Values from tuner overwrite baseline defaults
        return self.fit(train_stories, train_points, val_stories, val_points, **fit_kwargs)
    
    # ------------------------------------------------------------ persistence
    def save_weights(self, save_path: str):
        """Serializes the trained model weights and architectural configuration to disk.

        Saves a self-contained PyTorch checkpoint containing the model's state dictionary, structural initialization arguments, base Hugging Face identifier, ordinal point 
        scaling array, and maximum sequence length constraints. This ensures the complete neural network and text pipeline can be perfectly reconstructed for inference 
        without relying on external configuration files.

        Args:
            save_path (str): The destination file path where the PyTorch checkpoint will be saved (e.g., 'binaries/models/true_coral_final.pt').

        Raises:
            ValueError: If the internal model object is None, indicating that training has not yet occurred and there are no weights to save.
        """
        if self.model is None:
            raise ValueError("No model trained to save.")
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        checkpoint = {
            "version": CHECKPOINT_VERSION,
            "model_state_dict": self.model.state_dict(),
            "model_kwargs": self.model.get_init_config(),
            "model_name": self.model_name,
            "scale": self.scale,
            "max_length": self.max_length,
        }
        torch.save(checkpoint, save_path)
        print(f"[+] Model weights saved successfully to {save_path}")

    def load_weights(self, load_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """Restores the trained model architecture, learned weights, and configurations from disk.

        Reads the specified PyTorch checkpoint and validates its version to ensure compatibility with the current True CORAL codebase. It reconstructs the Transformer 
        architecture using the exact hyperparameters saved during training, populates the parameter matrices with the loaded state dictionary, and dynamically re-initializes
        the correct Hugging Face tokenizer to guarantee tokenization matches the embedded weights.

        Args:
            load_path (str): The file path to the saved PyTorch checkpoint.
            device (str, optional): The target hardware device for tensor allocation ('cpu', 'cuda', etc.). Defaults to 'cuda' if a compatible GPU is detected, otherwise 'cpu'.

        Raises:
            FileNotFoundError: If the specified checkpoint file does not exist at `load_path`.
            ValueError: If the checkpoint version is outdated (e.g., saved prior to the True CORAL classification head refactor) and structurally incompatible with the current architecture.
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Weight file not found at {load_path}")

        checkpoint = torch.load(load_path, map_location=device)
        if checkpoint.get("version") != CHECKPOINT_VERSION:
            raise ValueError(
                f"{load_path} was saved before the CORAL-head refactor (checkpoint version "
                f"{checkpoint.get('version', 1)}) and cannot be loaded. Retrain to produce a version "
                f"{CHECKPOINT_VERSION} checkpoint."
            )
        self.model_name = checkpoint["model_name"]
        self.scale = checkpoint["scale"]
        self.max_length = checkpoint["max_length"]
        self.tokenizer = self._build_tokenizer(self.model_name)   # tokenizer follows the checkpoint, not --model

        kwargs = dict(checkpoint["model_kwargs"])
        self.pooling = kwargs["pooling"]
        self.model = TransformerArchitecture(**kwargs, pretrained=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        self.model.eval()
        print(f"[+] Model weights loaded successfully from {load_path}")

    # -------------------------------------------------------------- inference
    def _logits(self, user_stories: List[str], batch_size: int = 32) -> torch.Tensor:
        """Executes a forward pass over the provided stories to generate raw prediction logits.

        Args:
            user_stories (List[str]): A list of raw text strings representing the user stories.
            batch_size (int, optional): The number of stories to process simultaneously during inference. Defaults to 32.

        Returns:
            torch.Tensor: A 2D tensor containing the raw, unnormalized True CORAL logits for each story across the ordinal class boundaries.
        """
        if self.model is None:
            raise ValueError("Model is not loaded or trained. Call fit() or load_weights() first.")
        self.model.eval()
        ds = self._make_dataset(user_stories, None)

        collate = OrdinalDataCollator(tokenizer=self.tokenizer)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
        
        device = next(self.model.parameters()).device
        chunks = []
        with torch.no_grad():
            for batch in loader:
                chunks.append(self.model(batch["input_ids"].to(device), batch["attention_mask"].to(device)).cpu())
                
        return torch.cat(chunks) if chunks else torch.empty(0, len(self.scale) - 1)

    def predict(self, user_stories: List[str], batch_size: int = 32) -> List[float]:
        """Predicts the story point values for a list of user stories.

        Executes the True CORAL inference logic by converting raw model logits into cumulative ordinal probabilities via a sigmoid activation. It sums the independent binary
        threshold predictions (>0.5) to determine the final ordinal rank. This rank is then mapped directly back to the project's specific numerical story point scale 
        (e.g., Fibonacci).

        Args:
            user_stories (List[str]): A list of raw text strings representing the user stories or Jira tickets to estimate.
            batch_size (int, optional): The number of stories to process simultaneously during inference execution. Defaults to 32.

        Returns:
            List[float]: A list of predicted story point values corresponding 
                to the input user stories.
        """
        ranks = (torch.sigmoid(self._logits(user_stories, batch_size)) > 0.5).sum(dim=1).tolist()
        return [self.scale[min(r, len(self.scale) - 1)] for r in ranks]

    def predict_proba(self, user_stories: List[str], batch_size: int = 32) -> List[List[float]]:
        """P(y > scale[k]) for k = 0..K-2, per story (for uncertainty / interpretability)."""
        return torch.sigmoid(self._logits(user_stories, batch_size)).tolist()

    def evaluate(
        self,
        stories: List[str],
        points: List[float],
        train_points: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """Metrics in raw story points; pass train_points to also get baselines, SA, improvement vs median."""
        return compute_metrics(points, self.predict(stories), reference_points=train_points)