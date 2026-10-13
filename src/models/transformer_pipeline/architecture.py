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
    Implements the Consistent Rank Logits (CORAL) classification head for ordinal classification. Utilizes a single shared weight vector to produce a scalar score for each 
    sample, which is then added to K-1 independent, monotonically decreasing bias thresholds.
    """
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.score = nn.Linear(in_dim, 1, bias=False)
        self.bias = nn.Parameter(torch.linspace(1.0, -1.0, num_classes - 1))
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executes the forward pass to compute ordinal logits.

        Adds the shared scalar score for each sample to the sequence of learnable threshold biases. 

        Args:
            x (torch.Tensor): The input feature tensor of shape (batch_size, in_dim).

        Returns:
            torch.Tensor: The rank-consistent logits of shape (batch_size, num_classes - 1). These are typically passed through a sigmoid activation downstream to represent 
                cumulative probabilities.
        """
        return self.score(x) + self.bias  # (batch, K-1)
    
class CoralOrdinalLoss(nn.Module):
    """
    Computes the loss for the CORAL ordinal regression framework. This loss function treats ordinal classification as a series of K-1 independent binary classification tasks.
    It computes the binary cross-entropy (BCE) between the rank-consistent logits and the extended binary ordinal labels, summing the errors across the K-1 thresholds for 
    each sample, and then averaging that total error across the batch.
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, ordinal_labels: torch.Tensor) -> torch.Tensor:
        """Calculates the batch-averaged CORAL loss.

        Args:
            logits (torch.Tensor): The raw, unnormalized predictions from the CORAL head of shape (batch_size, num_classes - 1).
            ordinal_labels (torch.Tensor): The extended binary representation of the true ranks, matching the shape (batch_size, num_classes - 1). 
                For example, a true rank of 2 in a 5-class system should be represented as [1, 1, 0, 0].

        Returns:
            torch.Tensor: A scalar tensor representing the computed loss for the batch.
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
        """Retrieves the architectural configuration required to reconstruct the model.

        Extracts the core hyperparameters and structural decisions used duringinstantiation. This dictionary is serialized directly into PyTorch checkpoints to ensure the 
        exact same architecture can be re-instantiated for evaluation or inference without relying on external configuration files.

        Returns:
            Dict[str, Any]: A dictionary mapping initialization parameter names ('model_name', 'num_classes', 'dropout_prob', 'mlp_head_width', 'activation_fn', 'pooling') 
            to their corresponding values.
        """
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
        """Determines if a given parameter name belongs to an embedding layer.

        Checks the parameter's string identifier against known naming conventions for token and positional embeddings across various transformer architectures 
        (e.g., 'wte' for GPT-2 token embeddings, or 'embed' for BERT variants).

        Args:
            name (str): The fully qualified name of the parameter as yielded by `named_parameters()` (e.g., 'transformer.wte.weight').

        Returns:
            bool: True if the parameter corresponds to an embedding layer, False otherwise.
        """
        return "embed" in name.lower() or name.split(".")[0] in ("wte", "wpe")
    
    def _get_blocks(self) -> List[nn.Module]:
        """Locates and extracts the sequential transformer blocks from the backbone.

        Dynamically inspects the underlying Hugging Face architecture to find the main stack of self-attention blocks, accommodating structural differences between 
        different models.

        Returns:
            List[nn.Module]: A list of PyTorch modules, where each module is a single transformer layer/block.
        """
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
        """Validates that every model parameter is assigned to exactly one optimizer group.

        Prevents silent optimization bugs by ensuring no parameters are accidentally dropped from training and no parameters are updated multiple times per 
        step due to overlapping group definitions.

        Args:
            groups (List[Dict[str, Any]]): The constructed list of optimizer parameter groups.

        Raises:
            RuntimeError: If any parameter appears in multiple groups.
            RuntimeError: If any parameters are entirely missing from the compiled groups.
        """
        grouped = [id(p) for g in groups for p in g["params"]] # Flatten the list of parameter IDs from all groups
        if len(grouped) != len(set(grouped)): # Check for Duplicates
            raise RuntimeError("A parameter appears in more than one optimizer group.")
        missing = {id(p) for p in self.parameters()} - set(grouped) # Output missing parameters
        if missing:
            raise RuntimeError(f"{len(missing)} model parameters are in no optimizer group.")


    def freeze_backbone(self):
        """Adjusts parameter gradient tracking based on the active freezing strategy.

        Dynamically modifies the `requires_grad` attribute of the model's parameters, based on the strategy:
        - 'fullfreeze': Locks the entire transformer backbone; only the head trains.
        - 'partialfreeze': Locks embeddings and the bottom N transformer layers (defined by `unfreeze_layers_from`), leaving upper layers and the head trainable.
        - 'unfreeze': Opens the entire network for end-to-end training.
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
        """Constructs optimizer parameter groups implementing Layer-wise Learning Rate Decay.

        Distributes all model parameters into specific groups to apply a decaying learning rate multiplier across the depth of the network. 
        Disables weight decay for specific parameters: 1-D tensors (biases and LayerNorm weights).
        Every model parameter lands in exactly one group:
          embeddings            -> base_lr * layer_decay_factor ^ n_blocks             (lowest rate)
          transformer block_{i} -> base_lr * layer_decay_factor ^ (n_blocks - 1 - i)   (end block = base_lr)
          backbone_end          -> base_lr  (final norm, pooler, ...)
          head                  -> head_lr  (defaults to base_lr)
        
        Returns:
            List[Dict[str, Any]]: A list of dictionaries compatible with PyTorch optimizers 
                (e.g., AdamW), where each dict contains the 'params', 'lr', and 
                'weight_decay' for that specific structural group.
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


# ------------------------------------------------------------------ End-To-End forward pass
    def _pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Aggregates token-level hidden states into a single sequence-level embedding.

        Applies the pooling strategy configured during model initialization to compress the sequence of token representations into a single fixed-size vector per 
        sequence. It strictly utilizes the attention mask to ensure padding tokens do not influence the aggregated representation.

        Supported strategies:
            - 'mean': Averages the embeddings of all non-padding tokens.
            - 'max': Takes the element-wise maximum across all non-padding tokens.
            - 'eos': Selects the embedding of the last non-padding token (useful for GPT models).
            - 'cls' (default): Selects the embedding of the very first token (e.g., [CLS] in BERT).

        Example Attention Mask:
            [
              [1, 1, 1, 1, 0, 0],
              [1, 1, 1, 1, 1, 0],
              [1, 1, 1, 0, 0, 0]
            ]

        Args:
            hidden (torch.Tensor): The output hidden states from the transformer backbone of shape (batch_size, sequence_length, hidden_dim).
            attention_mask (torch.Tensor): A binary tensor of shape (batch_size, sequence_length) indicating real tokens (1) and padding tokens (0).

        Returns:
            torch.Tensor: The pooled sequence embeddings of shape (batch_size, hidden_dim).
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
            bool_mask = (positions == idx.unsqueeze(1)) # Automatic broadcasting to create a boolean mask forj the last token in each sequence
            return hidden[bool_mask] # Collapse tensor to 2D shape (batch, hidden_dim) by selecting only the last token for each sequence
        else:
            return hidden[:, 0, :] # "cls": first token ([CLS] | <TOKEN>)
 
    def extract_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Generates pooled sequence-level feature representations from the transformer backbone.

        Passes the tokenized inputs through the underlying transformer model to obtain token-level hidden states, and then applies the configured pooling strategy 
        to compress them into a single fixed-size embedding vector per sequence. These are the exact embeddings fed directly into the classification head.

        Args:
            input_ids (torch.Tensor): A tensor of token indices of shape (batch_size, sequence_length).
            attention_mask (torch.Tensor): A binary tensor of shape (batch_size, sequence_length) indicating real tokens (1) and padding tokens (0).

        Returns:
            torch.Tensor: The pooled sequence embeddings of shape (batch_size, hidden_dim).
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask) # Calls forward pass on the backbone model
        hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0] # Extracts model's last hidden states
        return self._pool(hidden, attention_mask) # Applies pooling to obtain 1 vector representation per sequence
 
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Executes the complete end-to-end forward pass from token IDs to ordinal logits.

        Sequentially routes the inputs through the feature extractor (transformer backbone plus pooling layer) and finally through the custom prediction head 
        (MLP and True CORAL layer) to produce the rank-consistent threshold logits used for ordinal regression.

        Args:
            input_ids (torch.Tensor): A tensor of token indices of shape (batch_size, sequence_length).
            attention_mask (torch.Tensor): A binary tensor of shape (batch_size, sequence_length) indicating real tokens (1) and padding tokens (0).

        Returns:
            torch.Tensor: The raw, unnormalized ordinal logits of shape (batch_size, num_classes - 1).
        """
        return self.head(self.extract_features(input_ids, attention_mask))
    