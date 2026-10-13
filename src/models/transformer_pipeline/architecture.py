import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, PretrainedConfig
from typing import List, Dict, Any, Optional

"""GENERAL INFORMATION
- Encoder-Only Backbones: All Pooling Strategies view entire sequence (due to bidirectional nature)
- Decoder-Only Backbones: Only "Last" Pooling views entire sequence (due to autoregressive nature)
---> Must incorporate last token as well for some view of the entire sequence
- MULTIPLE LOSS FUNCTIONS: Coral-Ordinal Loss, Cross-Entropy Loss, MSE Loss (Standard Regression)
"""

POOLING_CHOICES = ("cls", "eos", "mean", "max", "self-attentive")

GENERATIVE_MODEL_TYPES = {
    "gpt2", "gpt_neo", "gptj", "gpt_neox", "llama", "mistral",
    "qwen2", "qwen3", "phi", "phi3", "gemma", "gemma2",
}

class CoralHead(nn.Module):
    """
    CORAL output layer: ONE shared weight vector (a single scalar score per
    sample) plus K-1 threshold biases. Sharing the weights makes the K-1
    probabilities P(y > r_k) rank-consistent. Biases start in decreasing order.
    """
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.score = nn.Linear(in_dim, 1, bias=False)
        self.bias = nn.Parameter(torch.linspace(1.0, -1.0, num_classes - 1))
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.score(x) + self.bias  # (batch, K-1)
    
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

 

class TransformerArchitecture(nn.Module):
    """
    Transformer backbone with Pooling and Ordinal Head.
    Supports Partial Freezing and Layer-wise LR Decay.
    """
    def __init__(
        self, 
        model_name: str = "microsoft/codebert-base", 
        pretrained: bool = True, # Start Training on a fresh backbone or on a checkpoint run of the backbone
        num_classes: int = 7,
        pooling: Optional[str] = None,
        mlp_head_width: int = 256, # Width of the hidden layer in the head
        # mlp_head_depth: int = 2, # Depth of the hidden layers in the head
        config: Optional[PretrainedConfig] = None,
        activation_fn: str = "gelu", # Hyperparameter 
        dropout_prob: float = 0.1, # Hyperparameter
        weight_decay: float = 0.01, # Hyperparameter
        base_lr: float = 1e-5, # Hyperparameter
        head_lr: float = 1e-3, # Hyperparameter
        layer_decay_factor: float = 0.9, # Hyperparameter
        strategy: str = "fullfreeze", # Hyperparameter
        unfreeze_layers_from: int = 8, # Hyperparameter
    ):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.mlp_head_width = mlp_head_width # Width of the intermediate layer in the head
        self.activation_fn = activation_fn
        self.pooling = pooling if pooling in POOLING_CHOICES else None
        self.config = AutoConfig.from_pretrained(model_name)
        self.weight_decay = weight_decay
        self.base_lr = base_lr
        self.head_lr = head_lr
        self.layer_decay_factor = layer_decay_factor
        self.strategy = strategy
        self.unfreeze_layers_from = unfreeze_layers_from
        

        if pretrained and config is None:
            self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        else:  # random init: used when restoring a checkpoint and in offline tests
            self.backbone = AutoModel.from_config(self.config)
 
        d_model = self.config.hidden_size

        activations = {"relu": nn.ReLU(), "gelu": nn.GELU(), "silu": nn.SiLU()}
        key = activation_fn.lower()
        if key not in activations:
            raise ValueError(f"Unkown activation function {activation_fn}")
        mapped_activation_fn = activations[key]
        
        self.head = nn.Sequential(
            nn.Linear(d_model, mlp_head_width),
            nn.LayerNorm(mlp_head_width),
            mapped_activation_fn, # Lower Case version of parsed Activation Function name
            nn.Dropout(dropout_prob),
            CoralHead(mlp_head_width, num_classes) # Corrected to use CoralHead for ordinal regression
        )


    # ------------------------------------------------------------------ config
    def get_init_config(self) -> Dict[str, Any]:
        """Everything needed to rebuild this model's architecture from a checkpoint."""
        return {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "dropout_prob": self.dropout_prob,
            "mlp_head_width": self.mlp_head_width,
            "activation_fn": self.activation_fn,
            "pooling": self.pooling,
        }
    

    # ------------------------------------------------------------------ backbone map
    @staticmethod
    def _is_embedding_param(name: str) -> bool:
        # Identify if a parameter name corresponds to an embedding layer (token or positional)
        # Checks if "embed" is in the name or if the name starts with "wte" or "wpe"
        # E.g. "transformer.wte.weight" or "transformer.wpe.weight" or "embeddings.word_embeddings.weight"
        return "embed" in name.lower() or name.split(".")[0] in ("wte", "wpe")
    
    def _get_blocks(self) -> List[nn.Module]:
        # Locate the transformer blocks in the backbone model, supporting various architectures
        b = self.backbone
        encoder = getattr(b, "encoder", None)
        candidates = [
            getattr(b, "h", None),                 # GPT-2 family
            getattr(encoder, "layer", None),       # BERT / RoBERTa / DeBERTa
            getattr(b, "layers", None),            # Llama / Qwen / ModernBERT
            getattr(encoder, "layers", None), 
        ]
        for blocks in candidates:
            if blocks is not None and len(blocks) > 0:
                return list(blocks)
        raise ValueError(
            f"Could not locate transformer blocks on {type(b).__name__}; "
            "extend TransformerStoryPointModel._get_blocks()."
        )
    
    def _assert_param_coverage(self, groups: List[Dict[str, Any]]) -> None:
        """
        Ensure that every parameter in the model is included in exactly one optimizer group.
        Raises a RuntimeError if any parameter is missing or duplicated across groups.
        """
        grouped = [id(p) for g in groups for p in g["params"]] # Flatten the list of parameter IDs from all groups
        if len(grouped) != len(set(grouped)): # Check for Duplicates
            raise RuntimeError("A parameter appears in more than one optimizer group.")
        missing = {id(p) for p in self.parameters()} - set(grouped) # Output missing parameters
        if missing:
            raise RuntimeError(f"{len(missing)} model parameters are in no optimizer group.")


    def freeze_backbone(self):
        """
        Strategy 'fullfreeze': Freeze entire transformer backbone.
        Strategy 'partialfreeze': Freeze bottom layers (0 to unfreeze_layers_from - 1), keep top layers trainable.
        Strategy 'unfreeze': Unfreeze all layers.
        """
        if self.strategy == "fullfreeze":
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif self.strategy == "partialfreeze":
            blocks = self._get_blocks() # Get the transformer blocks
            block_ids = {id(p) for blk in blocks for p in blk.parameters()} # Collect parameter IDs of all blocks
            # Freeze/unfreeze transformer non-block parameters: Token Embeddings, Position Embeddings, Poolers.
            for name, p in self.backbone.named_parameters():
                if id(p) not in block_ids:
                    p.requires_grad = not self._is_embedding_param(name) # Freeze/unfreeze based on embedding status
            # Freeze/unfreeze transformer block parameters: Queries, Keys, Values, and MLPs inside the layers.
            for idx, blk in enumerate(blocks):
                for p in blk.parameters():
                    p.requires_grad = idx >= self.unfreeze_layers_from # Freeze bottom layers, unfreeze top layers
        if self.strategy == "unfreeze":
            for param in self.backbone.parameters():
                param.requires_grad = True

    def build_llrd_param_groups(self) -> List[Dict[str, Any]]:
        """
        Layer-wise LR decay. Every model parameter lands in exactly one group:
          embeddings            -> base_lr * layer_decay_factor ^ n_blocks             (lowest rate)
          transformer block_{i} -> base_lr * layer_decay_factor ^ (n_blocks - 1 - i)   (end block = base_lr)
          backbone_end          -> base_lr  (final norm, pooler, ...)
          head                  -> head_lr  (defaults to base_lr)
        Biases and LayerNorm weights (1-D tensors) get weight_decay = 0.
        Groups are built from ALL parameters regardless of requires_grad -> result does not depend on whether backbone is currently frozen.
        Backbone can be frozen in early epochs -> unfrozen in later epochs.
        """
        # GENERAL TRANSFORMER OUTLINE
        # Token IDS -> Token Embeddings -> Positional Embeddings -> Transformer Blocks (1..N) -> Pooling -> Head

        blocks = self._get_blocks()
        length = len(blocks)
        block_ids = {id(p) for blk in blocks for p in blk.parameters()}
        groups: List[Dict[str, Any]] = [] # List of parameter groups to be returned
 
        def add(name: str, params, lr: float):
            params = list(params)
            decay = [p for p in params if p.ndim > 1] # Weights (2D or more) get weight decay
            no_decay = [p for p in params if p.ndim <= 1] # Biases and LayerNorm weights (1D) get no weight decay
            if decay:
                groups.append({"name": f"{name}/decay", "params": decay, "lr": lr, "weight_decay": self.weight_decay})
            if no_decay:
                groups.append({"name": f"{name}/no_decay", "params": no_decay, "lr": lr, "weight_decay": 0.0})

        embed, other = [], [] # Separate embedding parameters from other backbone parameters
        for name, p in self.backbone.named_parameters():
            if id(p) in block_ids:
                continue
            (embed if self._is_embedding_param(name) else other).append(p)
            # Embedding Units include:
            # 1. Word/Token Embeddings (mapping vocab IDs to 768-dim vectors)
            # 2. Positional Embeddings (mapping absolute spatial positions to vectors)
            # 3. Token Type Embeddings (mapping sentence/segment IDs to vectors)
            # 4. Embedding LayerNorms (the learnable Gamma/Beta weights applied right before Block 0)

        add("embeddings", embed, self.base_lr * (self.layer_decay_factor ** (length))) # Add embedding parameters with the lowest learning rate
        for idx in reversed(range(length)): # Add transformer blocks in reverse order (top layers first) with decayed learning rates
            add(f"block_{idx}", blocks[idx].parameters(), self.base_lr * (self.layer_decay_factor ** (length - 1 - idx)))
        add("backbone_end", other, self.base_lr) # Add the end-layer backbone parameters with base learning rate --> includes pooling
        add("head", self.head.parameters(), self.head_lr) # Add the prediction head parameters with the specified head learning rate

        self._assert_param_coverage(groups)
        return groups


# ------------------------------------------------------------------ forward
    def _pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Example Attention Mask:
        attention_mask =
            [
              [1, 1, 1, 1, 0, 0],
              [1, 1, 1, 1, 1, 0],
              [1, 1, 1, 0, 0, 0]
            ]
        """
        if self.pooling == "self-attentive":
            raise NotImplementedError("Self-Attentive not implemented yet")
        elif self.pooling == "mean":
            m = attention_mask.unsqueeze(-1).to(hidden.dtype)
            return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        elif self.pooling == "max":
            m = attention_mask.unsqueeze(-1).bool()
            hidden_masked = hidden.masked_fill(~m, float("-inf")) # ~ = logical not --> where m is false replace hidden value with -1e9
            return hidden_masked.max(dim=1).values
        elif self.pooling == "eos":
            idx = (attention_mask.sum(dim=1) - 1).clamp(min=0) # Find the last non-padding token index per sequence
            positions = torch.arange(hidden.size(1), device=hidden.device).unsqueeze(0) # Create a tensor of positions for each token in the sequence
            bool_mask = (positions == idx.unsqueeze(1)) # Automatic broadcasting to create a boolean mask for the last token in each sequence
            return hidden[bool_mask] # Collapse tensor to 2D shape (batch, hidden_dim) by selecting only the last token for each sequence
        else:
            return hidden[:, 0, :] # "cls": first token ([CLS] | <TOKEN>)
 
    def extract_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pooled sentence representation (what the head sees)."""
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask) # Calls forward pass on the backbone model
        hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0] # Extracts model's last hidden states
        return self._pool(hidden, attention_mask) # Applies pooling to obtain 1 vector representation per sequence
 
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Call a composition of forward passes: extract features (backbone + pooling) followed by head (MLP + CoralHead)
        # MLP head = Function: (batch, hidden_dim) --> (batch, K-1)
        return self.head(self.extract_features(input_ids, attention_mask))
 
