import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import List, Dict, Any, Tuple

class TransformerStoryPointModel(nn.Module):
    """
    Transformer backbone with Pooling and Ordinal Head.
    Supports Partial Freezing and Layer-wise LR Decay.
    """
    def __init__(
        self, 
        model_name: str = "gpt2", 
        num_classes: int = 7,
        dropout_prob: float = 0.1,
        hidden_dim: int = 256,
        activation_fn: str = "gelu"
    ):
        super().__init__()
        self.num_classes = num_classes
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        
        # Adjust pad token handling for decoder-only architectures like GPT-2
        if self.config.pad_token_id is None:
            self.config.pad_token_id = self.config.eos_token_id

        d_model = self.config.hidden_size
        
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU()
        }
        
        self.head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            activations.get(activation_fn.lower(), nn.GELU()),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, num_classes - 1)  # K-1 ordinal thresholds
        )

    def freeze_backbone(self, strategy: str = "full", unfreeze_layers_from: int = 7):
        """
        Strategy 'full': Freeze entire transformer backbone.
        Strategy 'partial': Freeze bottom layers (0 to unfreeze_layers_from - 1), keep top layers trainable.
        """
        if strategy == "full":
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif strategy == "partial":
            # Freeze embeddings
            if hasattr(self.backbone, "wte"):
                for p in self.backbone.wte.parameters(): p.requires_grad = False
            if hasattr(self.backbone, "wpe"):
                for p in self.backbone.wpe.parameters(): p.requires_grad = False
                
            # Iterate transformer blocks (GPT2 uses 'h', BERT/DeBERTa use 'layer')
            blocks = getattr(self.backbone, "h", getattr(getattr(self.backbone, "encoder", None), "layer", []))
            for idx, block in enumerate(blocks):
                if idx < unfreeze_layers_from:
                    for p in block.parameters():
                        p.requires_grad = False
                else:
                    for p in block.parameters():
                        p.requires_grad = True

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        
        # Pool hidden states: last token for GPT models, EOS/mean for others
        if hasattr(outputs, "last_hidden_state"):
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = input_ids.shape[0]
            pooled_output = outputs.last_hidden_state[torch.arange(batch_size), sequence_lengths]
        else:
            pooled_output = outputs[0][:, 0, :]

        logits = self.head(pooled_output)
        return logits

    def build_llrd_param_groups(self, base_lr: float, decay_factor: float = 0.8, weight_decay: float = 0.01) -> List[Dict[str, Any]]:
        """
        Generates parameter groups with Layer-wise Learning Rate Decay (LLRD).
        Top layers get full LR; deeper layers decay exponentially by decay_factor.
        """
        param_groups = []
        blocks = getattr(self.backbone, "h", getattr(getattr(self.backbone, "encoder", None), "layer", []))
        num_layers = len(blocks)

        # 1. Prediction Head
        param_groups.append({
            "params": [p for p in self.head.parameters() if p.requires_grad],
            "lr": base_lr,
            "weight_decay": weight_decay
        })

        # 2. Transformer layers (decaying backwards)
        for idx in reversed(range(num_layers)):
            layer_lr = base_lr * (decay_factor ** (num_layers - idx))
            trainable_params = [p for p in blocks[idx].parameters() if p.requires_grad]
            if trainable_params:
                param_groups.append({
                    "params": trainable_params,
                    "lr": layer_lr,
                    "weight_decay": weight_decay
                })

        return param_groups