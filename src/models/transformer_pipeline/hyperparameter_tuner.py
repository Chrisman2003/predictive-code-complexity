import random
import itertools
import torch
from typing import Dict, List, Any, Optional
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.models.transformer_pipeline.trainer import TransformerTrainer


class HyperparameterTuner:
    """
    Two-stage tuner:
    Phase 1: Coarse Random Search over broad search space.
    Phase 2: Focused Grid Search centered around top Phase 1 candidates.
    """
    def __init__(self, train_dataset, val_dataset, model_name: str = "microsoft/codebert-base", base_config: Optional[Dict[str, any]] = None):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model_name = model_name
        self.base_config = base_config or {}


    def sample_random_config(self, space: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Samples a single random hyperparameter configuration from a search space.

        Iterates through the provided search space dictionary and selects exactly one candidate value for each hyperparameter key using the tuner's internal random number generator.

        Args:
            space (Dict[str, List[Any]]): A dictionary mapping hyperparameter names to lists of discrete permissible values.

        Returns:
            Dict[str, Any]: A finalized configuration dictionary containing exactly one randomly selected value for each hyperparameter key.
        """
        return {k: self.rng.choice(v) for k, v in space.items()}

    def execute_tuning(self, num_random_trials: int = 5, current_seed : int = 0) -> Dict[str, Any]:
        """Executes a two-phase hyperparameter optimization strategy.

        Performs an initial coarse random search over a predefined hyperparameter space, followed by a fine-grained grid search centered around the best configuration 
        discovered in the first phase. This hybrid approach efficiently explores the global parameter space before zeroing in on the optimal continuous values.

        Args:
            num_random_trials (int, optional): The number of distinct configurations to sample during the Phase 1 coarse random search.
            current_seed (int, optional): The deterministic seed for the random number generator to ensure reproducible configuration sampling.

        Returns:
            Dict[str, Any]: A dictionary containing two keys:
                - 'best_config' (Dict[str, Any]): The globally optimal hyperparameter configuration, merged with the fixed baseline parameters.
                - 'best_val_loss' (float): The corresponding lowest validation loss achieved.
        """
        self.rng = random.Random(current_seed)
        # Coarse Space
        search_space = {
            "head_lr": [3e-4, 1e-3, 3e-3],
            "base_lr": [1e-5, 3e-5, 5e-5],
            "weight_decay": [0.01, 0.05, 0.1],
            "dropout": [0.1, 0.2, 0.3],
            "mlp_head_width": [128, 256, 512],
            "batch_size": [16, 32],
        }
        fixed_space = {
            # Categorical Hyperparameters
            "activation_fn" : "gelu",
            "pooling" : "mean",
            "num_classes" : 7,            
            "unfreeze_layers_from" : 8,
            "tune_epochs" : 15, # Epochs for Hyperparameter Tuning Trial (Not Full Training)
            "tune_patience" : 5, # Patience for Hyperparameter Tuning Trial (Not Full Training)
            "strategy" : "fullfreeze",
            # Float Hyperparameters
            "layer_decay_factor" : 0.9,
        }

        best_score = float("inf")
        best_config = None

        # =========================================================
        # PHASE 1: COARSE RANDOM SEARCH
        # =========================================================
        print(f"--- PHASE 1: Random Search ({num_random_trials} trials x seed = {current_seed}) ---")
        for i in range(num_random_trials):
            cfg = self.sample_random_config(search_space)
            # PASS THE PREFIX "random_" AND THE INDEX
            score = self._evaluate_config({**fixed_space, **cfg}, f"random_{i+1}", current_seed)
            print(f"Trial {i+1}/{num_random_trials} | Config: {cfg} | Val MAE: {score:.4f}")
            if score < best_score:
                best_score = score
                best_config = cfg

        # =========================================================
        # PHASE 2: FINE-Tuning GRID SEARCH
        # =========================================================
        print(f"--- PHASE 2: Focused Grid Search around Best Params ---")
        # 1. Dynamically building a tight grid around the Phase 1 winner
        grid_space = {}
        for key, best_val in best_config.items():
            if isinstance(best_val, float):
                if "lr" in key or "decay" in key:
                    # Learning rates & decay: Test exactly at, 25% lower, and 25% higher
                    grid_space[key] = [best_val * 0.75, best_val, best_val * 1.25]
                else:
                    # Dropout/Probabilities: Test exactly at, -0.05, and +0.05 (clamped to valid bounds)
                    grid_space[key] = [max(0.0, best_val - 0.05), best_val, min(1.0, best_val + 0.05)]
            else:
                # Integers (batch_size, mlp_width) & Strings: Lock them to the Phase 1 winner.
                grid_space[key] = [best_val]

        # 2. Generate all mathematical combinations (Cartesian Product)
        keys, values = zip(*grid_space.items())
        grid_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        grid_combinations = [g for g in grid_combinations if g != best_config] # Remove the exact Phase 1 winner from the grid queue so we don't test it twice

        print(f"[*] Generated {len(grid_combinations)} new fine-tuned configurations.")

        # 3. Execute the Grid Search
        for i, grid_cfg in enumerate(grid_combinations):
            score = self._evaluate_config({**fixed_space, **grid_cfg}, f"grid_{i+1}", current_seed)
            print(f"Grid Trial {i+1}/{len(grid_combinations)} | Config: {grid_cfg} | Val Loss: {score:.4f}")
            if score < best_score:
                best_score = score
                best_config = grid_cfg

        print(f"\n[+] FINAL WINNER: {best_config} (Loss: {best_score:.4f})")
        # Return the absolute best configuration mixed with the fixed constraints
        return {"best_config": {**fixed_space, **best_config}, "best_val_loss": best_score}


    def _evaluate_config(self, cfg: Dict[str, Any], trial_id: str, current_seed: int) -> float:
        """Evaluates a single hyperparameter configuration via a short training run.

        Merges the candidate configuration with the baseline parameters, remaps the tuning-specific epoch and patience limits, and triggers a partial training execution. 
        Upon completion, it extracts the validation loss, deletes the model from memory, and manually clears the GPU cache to prevent cumulative out-of-memory errors across 
        successive trials.

        Args:
            cfg (Dict[str, Any]): The candidate hyperparameter configuration to evaluate.
            trial_id (str): A string identifier for the trial (e.g., 'random_1' or 'grid_3') used to segregate the trial's checkpoint directory.
            current_seed (int): The deterministic seed used to initialize the model and data loaders for this specific trial.

        Returns:
            float: The lowest validation loss achieved by the model during the trial.
        """

        params = {**self.base_config, **cfg}
        # If a hyperparameter is present in cfg, it overrides the value in base_config.
        # If a hyperparameter is omitted from cfg, it safely falls back to base_config.
        
        # the tuner names these tune_*; run_training names them epochs / patience
        params["epochs"] = params.pop("tune_epochs")
        params["patience"] = params.pop("tune_patience")

        model, metrics = TransformerTrainer.run_training(
            model_name=self.model_name,
            train_ds=self.train_dataset,
            val_ds=self.val_dataset,
            seed=current_seed,
            checkpoint_dir=f"binaries/tuning/trial_{trial_id}_seed{current_seed}",
            **params,
        )
        del model
        torch.cuda.empty_cache() # Release VRAM to GPU Allocator after a Hyperparameter Tuning Run
        return metrics["best_val_loss"]
    