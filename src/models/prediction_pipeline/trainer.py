import math
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, Optional
from src.models.prediction_pipeline.loss import CoralOrdinalLoss

class EarlyStopping:
    def __init__(self, patience: int = 12, delta: float = 1e-4):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def check(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

class StoryPointTrainer:
    def __init__(
        self, 
        model: torch.nn.Module, 
        train_loader: DataLoader, 
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epochs: int = 50,
        warmup_steps: int = 100,
        patience: int = 12,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.epochs = epochs
        self.warmup_steps = warmup_steps
        self.device = device
        self.criterion = CoralOrdinalLoss()
        self.early_stopping = EarlyStopping(patience=patience)
        
        self.total_steps = len(train_loader) * epochs
        self.current_step = 0

    def _adjust_lr(self):
        """Cosine Schedule with Warmup"""
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            lr_factor = float(self.current_step) / float(max(1, self.warmup_steps))
        else:
            progress = float(self.current_step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            lr_factor = max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        for param_group in self.optimizer.param_groups:
            if "initial_lr" not in param_group:
                param_group["initial_lr"] = param_group["lr"]
            param_group["lr"] = param_group["initial_lr"] * lr_factor

    def _eval(self) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        count = 0

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                labels = batch["ordinal_labels"].to(self.device)
                ranks = batch["target_rank"].to(self.device)

                logits = self.model(input_ids, mask)
                loss = self.criterion(logits, labels)

                # Decode ordinal predictions
                probs = torch.sigmoid(logits)
                pred_ranks = (probs > 0.5).sum(dim=1)
                
                total_loss += loss.item() * input_ids.size(0)
                total_mae += torch.abs(pred_ranks - ranks).sum().item()
                count += input_ids.size(0)

        return total_loss / count, total_mae / count

    def train(self) -> Dict[str, Any]:
        best_val_loss = float("inf")

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0

            for batch in self.train_loader:
                input_ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                labels = batch["ordinal_labels"].to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(input_ids, mask)
                loss = self.criterion(logits, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                self._adjust_lr()
                train_loss += loss.item() * input_ids.size(0)

            train_loss /= len(self.train_loader.dataset)
            val_loss, val_mae = self._eval()

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            if self.early_stopping.check(val_loss):
                print(f"Early stopping triggered at Epoch {epoch + 1}")
                break

        return {"best_val_loss": best_val_loss, "val_mae": val_mae}