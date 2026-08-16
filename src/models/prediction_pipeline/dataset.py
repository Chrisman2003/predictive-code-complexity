import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from src.dataloader.config import PROCESSED_DATA_DIR

class CodeComplexityDataset(Dataset):
    """PyTorch Dataset for Transformer-based Code Complexity Prediction."""

    def __init__(self, dataset_path: Path = None, target_metric: str = "halstead_effort"):
        if dataset_path is None:
            dataset_path = PROCESSED_DATA_DIR / "complexity_dataset.json"

        with open(dataset_path, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)

        self.target_metric = target_metric

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        item = self.raw_data[idx]

        # Input text combining Task Description + Paradigm context
        input_text = f"Task: {item['problem_description']} | Paradigm: {item['paradigm']}"
        
        # Target ground-truth metric (e.g., Halstead Effort)
        target_val = float(item["metrics"].get(self.target_metric, 0.0))

        return {
            "id": item["id"],
            "input_text": input_text,
            "source_code": item["source_code"],
            "target": torch.tensor(target_val, dtype=torch.float32),
        }