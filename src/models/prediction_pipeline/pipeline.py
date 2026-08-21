import os
import torch
from typing import List, Dict, Any

from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from src.models.prediction_pipeline.dataset import StoryPointDataset
from src.models.prediction_pipeline.architecture import TransformerStoryPointModel
from src.models.prediction_pipeline.trainer import StoryPointTrainer
from src.models.prediction_pipeline.hyperparameter_tuner import HyperparameterTuner

class StoryPointPredictionPipeline:
    def __init__(
        self, 
        model_name: str = "gpt2",
        scale: List[float] = [1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 20.0]
    ):
        self.model_name = model_name
        self.scale = sorted(scale)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = None


    def fit(
        self,
        # Data
        train_stories: List[str],
        train_points: List[float],
        val_stories: List[str],
        val_points: List[float],
        # Hyperparameters
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        dropout: float = 0.1,
        hidden_dim: int = 256,
        batch_size: int = 16,
        freeze_strategy: str = "partial",
        unfreeze_layers_from: int = 7,
        epochs: int = 30,
        patience: int = 12
    ) -> Dict[str, Any]:
        """
        Direct training method allowing explicit manual hyperparameter tuning
        without triggering automated random/grid search.
        """
        train_ds = StoryPointDataset(train_stories, train_points, self.tokenizer, scale=self.scale)
        val_ds = StoryPointDataset(val_stories, val_points, self.tokenizer, scale=self.scale)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        self.model = TransformerStoryPointModel(
            model_name=self.model_name,
            num_classes=len(self.scale),
            dropout_prob=dropout,
            hidden_dim=hidden_dim
        )
        self.model.freeze_backbone(strategy=freeze_strategy, unfreeze_layers_from=unfreeze_layers_from)

        param_groups = self.model.build_llrd_param_groups(
            base_lr=lr,
            weight_decay=weight_decay
        )
        optimizer = torch.optim.AdamW(param_groups)

        trainer = StoryPointTrainer(
            model=self.model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            epochs=epochs,
            patience=patience
        )
        return trainer.train()


    def tune_and_fit(
        self, 
        train_stories: List[str], 
        train_points: List[float],
        val_stories: List[str], 
        val_points: List[float],
        run_tuning: bool = False
    ) -> Dict[str, Any]:
        """
        Runs hyperparameter tuning (if run_tuning=True) then executes training.
        """
        if not run_tuning:
            return self.fit(train_stories, train_points, val_stories, val_points)
        
        train_ds = StoryPointDataset(train_stories, train_points, self.tokenizer, scale=self.scale)
        val_ds = StoryPointDataset(val_stories, val_points, self.tokenizer, scale=self.scale)

        # Call fit with base hyperparameters
        tuner = HyperparameterTuner(train_ds, val_ds, model_name=self.model_name)
        tune_res = tuner.execute_tuning(num_random_trials=3)
        best = tune_res["best_config"]

        return self.fit(
            train_stories, train_points, val_stories, val_points,
            lr=best["lr"],
            weight_decay=best["weight_decay"],
            dropout=best["dropout"],
            hidden_dim=best["hidden_dim"],
            batch_size=best["batch_size"],
            freeze_strategy=best["freeze_strategy"]
        )
        

    def save_weights(self, save_path: str):
        """Saves model weights and configuration to disk."""
        if self.model is None:
            raise ValueError("No model trained to save.")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "model_name": self.model_name,
            "scale": self.scale
        }
        torch.save(checkpoint, save_path)
        print(f"[+] Model weights saved successfully to {save_path}")
        

    def load_weights(self, load_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """Loads trained model weights from disk."""
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Weight file not found at {load_path}")
        
        checkpoint = torch.load(load_path, map_location=device)
        self.model_name = checkpoint["model_name"]
        self.scale = checkpoint["scale"]

        self.model = TransformerStoryPointModel(
            model_name=self.model_name,
            num_classes=len(self.scale)
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        self.model.eval()
        print(f"[+] Model weights loaded successfully from {load_path}")
        

    def predict(self, user_stories: List[str]) -> List[float]:
        if self.model is None:
            raise ValueError("Model is not loaded or trained. Call fit() or load_weights() first.")
        
        self.model.eval()
        ds = StoryPointDataset(user_stories, None, self.tokenizer, scale=self.scale)
        loader = DataLoader(ds, batch_size=16, shuffle=False)
        
        predictions = []
        device = next(self.model.parameters()).device

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                logits = self.model(input_ids, mask)
                
                probs = torch.sigmoid(logits)
                predicted_ranks = (probs > 0.5).sum(dim=1).cpu().tolist()
                
                for rank in predicted_ranks:
                    predictions.append(self.scale[min(rank, len(self.scale) - 1)])

        return predictions