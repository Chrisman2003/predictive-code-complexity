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

    def tune_and_fit(
        self, 
        train_stories: List[str], 
        train_points: List[float],
        val_stories: List[str], 
        val_points: List[float],
        run_tuning: bool = False
    ):
        train_ds = StoryPointDataset(train_stories, train_points, self.tokenizer, scale=self.scale)
        val_ds = StoryPointDataset(val_stories, val_points, self.tokenizer, scale=self.scale)

        best_params = {
            "lr": 1e-4,
            "weight_decay": 0.01,
            "dropout": 0.1,
            "hidden_dim": 256,
            "batch_size": 16,
            "freeze_strategy": "partial"
        }

        if run_tuning:
            tuner = HyperparameterTuner(train_ds, val_ds, model_name=self.model_name)
            tune_res = tuner.execute_tuning(num_random_trials=3)
            best_params = tune_res["best_config"]

        # Final Training
        train_loader = DataLoader(train_ds, batch_size=best_params["batch_size"], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=best_params["batch_size"], shuffle=False)

        self.model = TransformerStoryPointModel(
            model_name=self.model_name,
            num_classes=len(self.scale),
            dropout_prob=best_params["dropout"],
            hidden_dim=best_params["hidden_dim"]
        )
        self.model.freeze_backbone(strategy=best_params["freeze_strategy"])

        param_groups = self.model.build_llrd_param_groups(
            base_lr=best_params["lr"],
            weight_decay=best_params["weight_decay"]
        )
        optimizer = torch.optim.AdamW(param_groups)

        trainer = StoryPointTrainer(
            model=self.model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            epochs=30,
            patience=10
        )
        return trainer.train()

    def predict(self, user_stories: List[str]) -> List[float]:
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