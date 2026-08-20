import random
import itertools
import torch
from typing import Dict, List, Any
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.models.prediction_pipeline.architecture import TransformerStoryPointModel
from src.models.prediction_pipeline.trainer import StoryPointTrainer

class HyperparameterTuner:
    """
    Two-stage tuner:
    Phase 1: Coarse Random Search over broad search space.
    Phase 2: Focused Grid Search centered around top Phase 1 candidates.
    """
    def __init__(self, train_dataset, val_dataset, model_name: str = "gpt2"):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model_name = model_name

    def sample_random_config(self, space: Dict[str, List[Any]]) -> Dict[str, Any]:
        return {k: random.choice(v) for k, v in space.items()}

    def execute_tuning(self, num_random_trials: int = 5) -> Dict[str, Any]:
        # Coarse Space
        search_space = {
            "lr": [1e-5, 3e-5, 1e-4, 3e-4],
            "weight_decay": [0.01, 0.05, 0.1],
            "dropout": [0.1, 0.2, 0.3],
            "hidden_dim": [128, 256, 512],
            "batch_size": [8, 16],
            "freeze_strategy": ["full", "partial"]
        }

        best_score = float("inf")
        best_config = None

        print(f"--- PHASE 1: Random Search ({num_random_trials} Trials) ---")
        for i in range(num_random_trials):
            cfg = self.sample_random_config(search_space)
            score = self._evaluate_config(cfg)
            print(f"Trial {i+1}/{num_random_trials} | Config: {cfg} | Val MAE: {score:.4f}")
            if score < best_score:
                best_score = score
                best_config = cfg

        print(f"--- PHASE 2: Focused Grid Search around Best Params ---")
        # Refine around best found learning rate and dropout
        refined_lrs = [best_config["lr"] * 0.5, best_config["lr"], best_config["lr"] * 1.5]
        refined_dropouts = [max(0.0, best_config["dropout"] - 0.05), best_config["dropout"]]
        
        grid_space = {
            "lr": refined_lrs,
            "weight_decay": [best_config["weight_decay"]],
            "dropout": refined_dropouts,
            "hidden_dim": [best_config["hidden_dim"]],
            "batch_size": [best_config["batch_size"]],
            "freeze_strategy": [best_config["freeze_strategy"]]
        }

        keys, values = zip(*grid_space.items())
        grid_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        for idx, cfg in enumerate(grid_combinations):
            score = self._evaluate_config(cfg)
            print(f"Grid Step {idx+1}/{len(grid_combinations)} | Config: {cfg} | Val MAE: {score:.4f}")
            if score < best_score:
                best_score = score
                best_config = cfg

        return {"best_config": best_config, "best_val_mae": best_score}

    def _evaluate_config(self, cfg: Dict[str, Any]) -> float:
        train_loader = DataLoader(self.train_dataset, batch_size=cfg["batch_size"], shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=cfg["batch_size"], shuffle=False)

        model = TransformerStoryPointModel(
            model_name=self.model_name,
            dropout_prob=cfg["dropout"],
            hidden_dim=cfg["hidden_dim"]
        )
        model.freeze_backbone(strategy=cfg["freeze_strategy"])

        # Configure Optimizer with LLRD
        param_groups = model.build_llrd_param_groups(base_lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        optimizer = torch.optim.AdamW(param_groups)

        trainer = StoryPointTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            epochs=5,  # Shortened epochs for search
            patience=3
        )
        metrics = trainer.train()
        return metrics["val_mae"]