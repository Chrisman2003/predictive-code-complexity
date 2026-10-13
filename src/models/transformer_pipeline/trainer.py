import math
import os
import torch
import os
import random
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from torch.utils.data import DataLoader
from transformers import PretrainedConfig, AutoTokenizer
from src.models.transformer_pipeline.dataset import OrdinalDataCollator
from src.models.transformer_pipeline.architecture import TransformerArchitecture
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, Optional
from tqdm.auto import tqdm
from src.models.transformer_pipeline.architecture import CoralOrdinalLoss

class EarlyStopping:
    """Monitors validation loss to halt training when the model stops improving.

    Prevents overfitting by tracking the validation loss at each epoch. If the loss fails to decrease by a specified minimum amount over a defined number 
    of consecutive epochs, it triggers a flag to terminate training early.

    Attributes:
        patience (int): The number of consecutive epochs to wait for an improvement before stopping training.
        delta (float): The minimum absolute decrease in validation loss required to qualify as a valid improvement.
        counter (int): The current number of consecutive epochs without improvement.
        best_loss (float): The lowest validation loss observed so far.
        early_stop (bool): A flag indicating whether training should be halted.
    """
    def __init__(self, patience: int = 12, delta: float = 1e-4):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def check(self, val_loss: float) -> bool:
        """Evaluates the current validation loss against the historical best.

        If the current validation loss is lower than the best recorded loss (by at least `delta`), the best loss is updated and the patience counter resets. 
        Otherwise, the counter increments. If the counter reaches or exceeds `patience`, the early_stop flag is set to True.

        Args:
            val_loss (float): The calculated validation loss from the current epoch.

        Returns:
            bool: True if training should be halted immediately, False otherwise.
        """
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class TransformerTrainer:
    """
    Two-phase trainer.
      Phase A (epochs < freeze_epochs): backbone frozen (and in eval mode), only the head trains.
      Phase B (epochs >= freeze_epochs): everything POTENTIALLY trains with LLRD param groups.
    
    The LR schedule is phase-aware: each phase gets its own warmup (warmup_ratio of the phase's steps); phase B then decays with a cosine over the steps that are actually 
    left. The best epoch (lowest validation loss) is restored into the model when training ends.
    IMPORTANT: Phase B is only relevant given that one does not select: fully freezing the backbone
    """
    def __init__(
        self, 
        model: torch.nn.Module, 
        train_loader: DataLoader, 
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        checkpoint_dir: str = "binaries/checkpoints",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        epochs: int = 50,
        patience: int = 12,
        warmup_ratio: float = 0.1,
        freeze_epochs: int = 3
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.epochs = epochs
        self.criterion = CoralOrdinalLoss()
        self.early_stopping = EarlyStopping(patience=patience)
        self.warmup_ratio = warmup_ratio
        self.freeze_epochs = freeze_epochs 
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # For reporting MAE in story points, not ranks
        self.scale = list(getattr(val_loader.dataset, "scale", []) or [])
        
        # Defining Warmup steps: Transformers uniquely fragile at initialization
        self.steps_per_epoch = len(train_loader) # No. batches per epoch
        self.total_steps = self.steps_per_epoch * epochs
        self.frozen_steps = self.steps_per_epoch * min(freeze_epochs, epochs)
        self.current_step = 0

        for group in self.optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"]) 

        # Best-epoch bookkeeping (attributes, so they survive resume_from_checkpoint)
        self.best_val_loss = float("inf")
        self.best_val_mae_points = float("inf")
        self.best_val_mae_rank = float("inf")
        self.best_epoch = -1

    # -------------------------------------------------------------- schedule
    def _is_frozen_epoch(self, epoch: int) -> bool:
        """Determines whether the model's backbone should remain frozen for the given epoch.

        Args:
            epoch (int): The current 0-indexed training epoch.

        Returns:
            bool: True if the epoch falls within the initial freezing window, False otherwise.
        """
        return epoch < self.freeze_epochs # Boolean Flag: current epoch in window of temporarily frozen backbone

    def _lr_factor(self, step: int) -> float:
        """Calculates the learning rate multiplier for the specified training step.

        Computes a scaling factor between 0.0 and 1.0 based on a two-phase schedule. Phase A applies a linear warmup followed by a constant hold while the backbone is frozen. 
        Phase B initiates a secondary linear warmup upon unfreezing, followed by a cosine decay to 0.0 for the remainder of training.

        Args:
            step (int): The current 0-indexed global training step.

        Returns:
            float: The computed learning rate multiplier.
        """
        if step < self.frozen_steps:  # phase A: warmup, then hold
            warmup = max(1, int(self.warmup_ratio * self.frozen_steps))
            return min(1.0, (step + 1) / warmup)
        s = step - self.frozen_steps  # phase B: re-warmup, then cosine to 0
        phase_steps = max(1, self.total_steps - self.frozen_steps)
        warmup = max(1, int(self.warmup_ratio * phase_steps))
        if s < warmup:
            return (s + 1) / warmup
        progress = min(1.0, (s - warmup) / max(1, phase_steps - warmup))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    def _set_lr(self, step: int) -> None:
        """Applies the calculated learning rate multiplier to all optimizer parameter groups.

        Modifies the learning rate in-place for each parameter group by scaling its configured base rate. This must be called prior to the optimizer step to ensure the 
        initial update incorporates the exact warmup factor.

        Args:
            step (int): The current 0-indexed global training step.
        """        
        factor = self._lr_factor(step)
        for group in self.optimizer.param_groups:
            # Scale initial LR of every parameter in Network by "factor"
            group["lr"] = group["initial_lr"] * factor 

    def _lr_summary(self) -> Dict[str, float]:
        """Generates a snapshot of the current learning rates for telemetry logging.

        Extracts the active learning rate for the classification head and the highest active learning rate across all backbone parameter groups at the current time step.

        Returns:
            Dict[str, float]: A dictionary containing:
                - 'lr_head' (float): The current learning rate of the True CORAL classification head.
                - 'lr_bb' (float): The maximum current learning rate applied within the backbone layers.
        """
        head = [g["lr"] for g in self.optimizer.param_groups if str(g.get("name", "")).startswith("head")]
        backbone = [g["lr"] for g in self.optimizer.param_groups if not str(g.get("name", "")).startswith("head")]
        return {
            "lr_head": head[0] if head else self.optimizer.param_groups[0]["lr"],
            "lr_bb": max(backbone) if backbone else 0.0,
        }
    
    # ------------------------------------------------------------ evaluation
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Evaluates the model on a given dataset to compute loss and accuracy metrics.

        Executes a full forward pass over the provided DataLoader with gradient tracking disabled. It computes the True CORAL loss, converts predicted logits into ordinal 
        ranks, and calculates the Mean Absolute Error (MAE) for both the internal rank scale and the actual story points. It also computes the standard deviation of 
        predictions to detect potential mode collapse.

        Args:
            loader (DataLoader): The PyTorch DataLoader containing the evaluation 
                or test dataset batches.

        Returns:
            Dict[str, float]: A dictionary containing the evaluation metrics:
                - 'loss' (float): The average True CORAL loss across all samples.
                - 'mae_points' (float): The Mean Absolute Error in actual story points 
                  (or NaN if no scale is provided).
                - 'mae_rank' (float): The Mean Absolute Error in ordinal scale ranks.
                - 'pred_std' (float): The standard deviation of all predicted ranks 
                  (useful for identifying mode collapse where it approaches 0.0).
        """
        self.model.eval()
        scale_t = torch.tensor(self.scale, device=self.device) if self.scale else None
 
        loss_sum, ae_rank, ae_pts, count = 0.0, 0.0, 0.0, 0 # Accumulate data over running loop
        all_ranks = []
        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            mask = batch["attention_mask"].to(self.device)
            labels = batch["ordinal_labels"].to(self.device)
            ranks = batch["target_rank"].to(self.device)
 
            logits = self.model(input_ids, mask)
            loss = self.criterion(logits, labels)
            pred_ranks = (torch.sigmoid(logits) > 0.5).sum(dim=1)
 
            bs = input_ids.size(0)
            loss_sum += loss.item() * bs
            ae_rank += torch.abs(pred_ranks - ranks).sum().item()
            if scale_t is not None:
                raw = batch["raw_point"].to(self.device)
                ae_pts += torch.abs(scale_t[pred_ranks] - raw).sum().item()
            all_ranks.append(pred_ranks.cpu())
            count += bs
 
        pred = torch.cat(all_ranks).float() # Vector of predictions, by batch
        # Map predictions to real points for business variance tracking
        pred_pts_std = float("nan")
        if scale_t is not None and count > 1:
            # .long() converts the floats back to integer indices for the lookup
            pred_pts = scale_t[pred.long()]
            pred_pts_std = pred_pts.std().item()

        return {
            "loss": loss_sum / count,
            "mae_points": ae_pts / count if scale_t is not None else float("nan"),
            "mae_rank": ae_rank / count,
            "pred_rank_std": pred.std().item() if count > 1 else 0.0,
            "pred_pts_std": pred_pts_std
        }

    # ----------------------------------------------------------- checkpoints
    def resume_from_checkpoint(self, checkpoint_path: str) -> int:
        if not os.path.exists(checkpoint_path):
            print(f"[!] Checkpoint not found at {checkpoint_path}")
            return 0
 
        print(f"[*] Resuming training from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
 
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for group in self.optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
        self.current_step = checkpoint["current_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.best_val_mae_points = checkpoint.get("best_val_mae_points", float("inf"))
        self.best_val_mae_rank = checkpoint.get("best_val_mae_rank", float("inf"))
        self.best_epoch = checkpoint.get("best_epoch", -1)
        self.early_stopping.best_loss = self.best_val_loss
        self.early_stopping.counter = checkpoint.get("early_stopping_counter", 0)
 
        resumed_epoch = checkpoint["epoch"] + 1
        print(f"[+] Successfully restored. Resuming at epoch {resumed_epoch}.")
        return resumed_epoch
 
    def _save_latest(self, epoch: int, val: Dict[str, float]) -> None:
        get_cfg = getattr(self.model, "get_init_config", None)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "current_step": self.current_step,
            "val_loss": val["loss"],
            "best_val_loss": self.best_val_loss,
            "best_val_mae_points": self.best_val_mae_points,
            "best_val_mae_rank": self.best_val_mae_rank,
            "best_epoch": self.best_epoch,
            "early_stopping_counter": self.early_stopping.counter,
            "model_config": get_cfg() if get_cfg else None,
        }
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, "checkpoint_latest.pt"))

    # ----------------------------------------------------------------- train
    def train(self, resume_path: Optional[str] = None) -> Dict[str, Any]:
        """Executes the complete training loop for the True CORAL model.

        Manages the two-phase training protocol (freezing and unfreezing the transformer backbone), dynamic step-wise learning rate scheduling, batch processing, gradient 
        clipping, and evaluation. Tracks validation metrics to save the optimal model state, triggers early stopping if criteria are met, and restores the best weights into 
        memory upon completion.

        Args:
            resume_path (Optional[str], optional): Absolute or relative file path to a previously saved checkpoint file to resume training states, epoch count, 
                and optimizer parameters. Defaults to None.

        Returns:
            Dict[str, Any]: A telemetry dictionary summarizing the best epoch's performance, 
                containing:
                - 'best_val_loss' (float): The minimum validation True CORAL loss achieved.
                - 'val_mae' (float): Mean Absolute Error in actual story points for the best epoch.
                - 'val_mae_rank' (float): Mean Absolute Error in ordinal scale steps for the best epoch.
                - 'best_epoch' (int): The 1-based index of the epoch where the lowest validation loss occurred.
        """
        start_epoch = 0
        if resume_path:
            start_epoch = self.resume_from_checkpoint(resume_path)
 
        for epoch in range(start_epoch, self.epochs):
            # Two-phase protocol, derived from the epoch number every epoch so that
            # a resumed run always gets the right freeze state.
            frozen = self._is_frozen_epoch(epoch) # Checks whether in frozen state
            self.model.freeze_backbone()
            self.model.train()
            if frozen:
                self.model.backbone.eval()  # Only Transformer head is trained
 
            train_loss = 0.0
            progress_bar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{self.epochs}" + (" [frozen]" if frozen else ""),
                leave=True,
                unit="batch",
            )
 
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                labels = batch["ordinal_labels"].to(self.device)
 
                self._set_lr(self.current_step) # Throttle learning rate based on factor defined by warm-up state
                self.optimizer.zero_grad() # Old Gradients are cleared
                logits = self.model(input_ids, mask)
                loss = self.criterion(logits, labels)
 
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
 
                self.optimizer.step()
                self.current_step += 1
 
                batch_loss = loss.item()
                train_loss += batch_loss * input_ids.size(0)
 
                lrs = self._lr_summary()
                progress_bar.set_postfix({
                    "batch_loss": f"{batch_loss:.4f}",
                    "lr_head": f"{lrs['lr_head']:.2e}",
                    "lr_bb": f"{lrs['lr_bb']:.2e}",
                })
 
            train_loss /= len(self.train_loader.dataset)
            val = self.evaluate(self.val_loader)
            tqdm.write(
                f"    └─ [Summary] Train Loss: {train_loss:.4f} | Val Loss: {val['loss']:.4f}\n"
                f"    └─ [Points]  Val MAE: {val['mae_points']:.4f} | Pred std dev: {val['pred_pts_std']:.3f}\n"
                f"    └─ [Rank]    Val MAE: {val['mae_rank']:.4f} | Pred std dev: {val['pred_rank_std']:.3f}"
            )
        
            if val["loss"] < self.best_val_loss:
                self.best_val_loss = val["loss"]
                self.best_val_mae_points = val["mae_points"]
                self.best_val_mae_rank = val["mae_rank"]
                self.best_epoch = epoch
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, "best_model.pt"))
                tqdm.write("    └─ [*] New best model saved!")
 
            stop = self.early_stopping.check(val["loss"])
            self._save_latest(epoch, val)  # after early_stopping.check so the counter is current
 
            if stop:
                print(f"Early stopping triggered at Epoch {epoch + 1}")
                break
 
        # Ship the BEST epoch, not the last one.
        best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
        if os.path.exists(best_path):
            self.model.load_state_dict(torch.load(best_path, map_location=self.device))
            print(f"[+] Restored best weights (epoch {self.best_epoch + 1}) from {best_path}")
 
        return {
            "best_val_loss": self.best_val_loss,
            "val_mae": self.best_val_mae_points,   # NOTE: now story points, was ranks
            "val_mae_rank": self.best_val_mae_rank,
            "best_epoch": self.best_epoch + 1,     # 1-based
        }
    
    # ----------------------------------------------------------------- runner  
    @staticmethod
    def set_seed(seed: int) -> None:
        """Locks all pseudo-random number generators to ensure reproducible execution.

        Synchronizes the random seeds across Python's built-in random module, Numpy, and PyTorch (both CPU and CUDA operations). 
        It also disables Python's standard hash randomization.

        Args:
            seed (int): The integer value to use as the global random seed.
        """
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


    @staticmethod
    def run_training(
        model_name: str,
        num_classes: int,
        pooling: Optional[str],
        train_ds,
        val_ds,
        dropout: float,
        mlp_head_width: int,
        batch_size: int,
        epochs: int,
        patience: int,
        freeze_epochs: int,
        warmup_ratio: float,
        seed: int,
        checkpoint_dir: str,
        resume_path: Optional[str] = None,
        pretrained: bool = True,
        model_config: Optional[PretrainedConfig] = None,
    ) -> Tuple[TransformerArchitecture, Dict[str, Any]]:
        """Orchestrates the complete initialization and execution of the training pipeline.

        Constructs the data loaders with dynamic padding, instantiates the model architecture, configures the AdamW optimizer with Layer-wise Learning Rate 
        Decay (LLRD), and executes the training loop via the TransformerTrainer.

        Args:
            Configuration
            Hyperparameters
        Returns:
            Tuple[TransformerArchitecture, Dict[str, Any]]: A tuple containing:
                - The fully trained model architecture with the best-performing weights loaded.
                - A dictionary of validation metrics from the best epoch.
        """
        TransformerTrainer.set_seed(seed)

        collate = OrdinalDataCollator(tokenizer=train_ds.tokenizer) # Reproducibility
        generator = torch.Generator() # Random Number Generator
        generator.manual_seed(seed) # Generator Parametrized with seed
        pin = torch.cuda.is_available() # Automatic Device Detection for NVIDIA-CUDA Cluster
        
        train_loader = DataLoader( # Shuffling only for training
            train_ds, batch_size=batch_size, shuffle=True,
            collate_fn=collate, generator=generator, pin_memory=pin,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            collate_fn=collate, pin_memory=pin,
        )

        model = TransformerArchitecture(
            model_name=model_name,
            num_classes=num_classes,
            dropout_prob=dropout,
            mlp_head_width=mlp_head_width,
            pooling=pooling,
            pretrained=pretrained,
            config=model_config,
        )
        param_groups = model.build_llrd_param_groups()
        
        optimizer = torch.optim.AdamW(param_groups)

        trainer = TransformerTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            checkpoint_dir=checkpoint_dir,
            epochs=epochs,
            patience=patience,
            warmup_ratio=warmup_ratio,
            freeze_epochs=freeze_epochs,
        )
        metrics = trainer.train(resume_path=resume_path)
        return model, metrics
