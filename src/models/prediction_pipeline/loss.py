import torch
import torch.nn as nn
import torch.nn.functional as F

class CoralOrdinalLoss(nn.Module):
    """
    Computes loss across K-1 binary classification heads for ordinal targets.
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, ordinal_labels: torch.Tensor) -> torch.Tensor:
        """
        logits: [batch_size, num_classes - 1]
        ordinal_labels: [batch_size, num_classes - 1]
        """
        loss = F.binary_cross_entropy_with_logits(logits, ordinal_labels, reduction="sum")
        return loss / logits.size(0)