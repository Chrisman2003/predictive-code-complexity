import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional
from transformers import PreTrainedTokenizer
from transformers import DataCollatorWithPadding

class StoryPointDataset(Dataset):
    """
    Dataset wrapper converting User Story text and discrete Story Points
    into tokenized tensors and CORAL ordinal binary target vectors.
    """
    def __init__(
        self, 
        user_stories: List[str], 
        story_points: Optional[List[float]], 
        tokenizer: PreTrainedTokenizer, 
        scale: List[float] = [1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0],
        max_length: int = 512
    ):
        self.user_stories = user_stories
        self.story_points = story_points
        self.tokenizer = tokenizer
        self.scale = sorted(scale)
        self.max_length = max_length
        self.point_to_rank = {pt: idx for idx, pt in enumerate(self.scale)}

    def __len__(self) -> int:
        return len(self.user_stories)

    def _to_ordinal_target(self, rank: int) -> torch.Tensor:
        # Convert rank K to (num_classes - 1) binary targets
        # E.g., for 7 classes, rank 3 -> [1, 1, 1, 0, 0, 0]
        num_classes = len(self.scale)
        target = torch.zeros(num_classes - 1, dtype=torch.float32)
        if rank > 0:
            target[:rank] = 1.0
        return target

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.user_stories[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        
        item = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }

        if self.story_points is not None:
            raw_point = self.story_points[idx]
            closest_point = min(self.scale, key=lambda x: abs(x - raw_point)) # Match to nearest scale point if exact match not present
            rank = self.point_to_rank[closest_point]
            # Custom key
            item["ordinal_labels"] = self._to_ordinal_target(rank)
            item["target_rank"] = torch.tensor(rank, dtype=torch.long)
            item["raw_point"] = torch.tensor(raw_point, dtype=torch.float32)

        return item
    


class OrdinalDataCollator:
    def __init__(self, tokenizer):
        self.padding_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def __call__(self, examples):
        # 1. Extract only the model inputs.
        inputs = [
            {
                "input_ids": example["input_ids"],
                "attention_mask": example["attention_mask"],
            }
            for example in examples
        ]

        # 2. Batch the model inputs.
        batch = self.padding_collator(inputs)

        # 3. Batch the targets if they are available.
        if "ordinal_labels" in examples[0]:
            batch["ordinal_labels"] = torch.stack(
                [example["ordinal_labels"] for example in examples]
            )

            batch["target_rank"] = torch.stack(
                [example["target_rank"] for example in examples]
            )

            batch["raw_point"] = torch.stack(
                [example["raw_point"] for example in examples]
            )

        return batch