import math
import os
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, Optional
from src.models.prediction_pipeline.loss import CoralOrdinalLoss
from tqdm.auto import tqdm

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
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "binaries/checkpoints",
        freeze_epochs: int = 3
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
        self.checkpoint_dir = checkpoint_dir
        self.freeze_epochs = freeze_epochs
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
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

    def resume_from_checkpoint(self, checkpoint_path: str):
        if not os.path.exists(checkpoint_path):
            print(f"[!] Checkpoint not found at {checkpoint_path}")
            return 0
            
        print(f"[*] Resuming training from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_step = checkpoint['current_step']
        self.early_stopping.best_loss = checkpoint['best_val_loss']
        
        resumed_epoch = checkpoint['epoch'] + 1
        print(f"[+] Successfully restored. Resuming at epoch {resumed_epoch}.")
        return resumed_epoch

    def train(self, resume_path: Optional[str] = None) -> Dict[str, Any]:
        # Warmup Epochs: freeze fully model backbone and shrink learning rate
        best_val_loss = float("inf")
        best_val_mae = float("inf")
        
        # NEW: Handle resuming
        start_epoch = 0
        if resume_path:
            start_epoch = self.resume_from_checkpoint(resume_path)
            # Ensure early stopping best loss carries over
            best_val_loss = self.early_stopping.best_loss

        for epoch in range(start_epoch, self.epochs):
            # NEW: Two-Phase Training Logic
            if epoch < self.freeze_epochs:
                if epoch == 0 or (resume_path and epoch == start_epoch):
                    self.model.freeze_backbone(strategy="full")
            elif epoch == self.freeze_epochs:
                # Unfreeze at exactly this epoch
                self.model.freeze_backbone(strategy="unfreeze")

            self.model.train()
            train_loss = 0.0
            
            progress_bar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                leave=True,
                unit="batch"
            )

            for batch in progress_bar:
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
                
                batch_loss = loss.item()
                train_loss += loss.item() * input_ids.size(0)
                
                current_lr = self.optimizer.param_groups[0]["lr"]
                progress_bar.set_postfix({
                    "batch_loss": f"{batch_loss:.4f}",
                    "lr": f"{current_lr:.2e}"
                })

            train_loss /= len(self.train_loader.dataset)
            val_loss, val_mae = self._eval()
            
            tqdm.write(f"    └─ [Summary] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f}")

            # NEW: CHECKPOINT SAVING LOGIC
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'current_step': self.current_step,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss if val_loss >= best_val_loss else val_loss
            }
            
            # Save the latest epoch for crash recovery
            latest_path = os.path.join(self.checkpoint_dir, "checkpoint_latest.pt")
            torch.save(checkpoint, latest_path)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_mae = val_mae
                # Only save weights for the best model (for final inference)
                best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
                torch.save(self.model.state_dict(), best_path)
                tqdm.write(f"    └─ [*] New best model saved!")

            if self.early_stopping.check(val_loss):
                print(f"Early stopping triggered at Epoch {epoch + 1}")
                break

        return {"best_val_loss": best_val_loss, "val_mae": best_val_mae}