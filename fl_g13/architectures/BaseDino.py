from typing import Callable

import torch
import torch.nn as nn

# === Default configuration constants ===
DEFAULT_VARIANT = "dino_vits16"
DEFAULT_HEAD_HIDDEN_SIZE = 512
DEFAULT_HEAD_LAYERS = 3
DEFAULT_UNFREEZE_BLOCKS = 0
DEFAULT_DROPOUT_RATE = 0.0
DEFAULT_ACTIVATION = nn.GELU
DEFAULT_PRETRAINED = True
DEFAULT_NUM_CLASSES = 100

class BaseDino(nn.Module):
    """
    Model class wrapping DINO Vision Transformer (e.g., S16, S8) with a configurable head and dropout.
    Includes utilities for configuration management.
    """
    def __init__(
        self,
        variant: str = DEFAULT_VARIANT,
        head_hidden_size: int = DEFAULT_HEAD_HIDDEN_SIZE,
        head_layers: int = DEFAULT_HEAD_LAYERS,
        unfreeze_blocks: int = DEFAULT_UNFREEZE_BLOCKS,
        dropout_rate: float = DEFAULT_DROPOUT_RATE,
        activation: Callable = DEFAULT_ACTIVATION,
        pretrained: bool = DEFAULT_PRETRAINED,
        num_classes: int = DEFAULT_NUM_CLASSES,
    ):
        super().__init__()
        # Load pretrained DINO model backbone (e.g., dino_vits16, dino_vits8)
        backbone = torch.hub.load('facebookresearch/dino:main', variant, pretrained=pretrained)
        
        # Add dropout to attention and MLP modules
        for block in backbone.blocks:
            block.attn.attn_drop = nn.Dropout(p=dropout_rate)
            block.attn.proj_drop = nn.Dropout(p=dropout_rate)
            block.mlp.drop = nn.Dropout(p=dropout_rate)

        # Build classification head
        # Get the output logits from the LayerNorm (as head is Identity() by now)
        input_dim = backbone.norm.normalized_shape[0]
        layers = []
        curr_dim = input_dim
        for _ in range(head_layers):
            layers.append(nn.Linear(curr_dim, head_hidden_size))
            layers.append(activation())
            layers.append(nn.Dropout(p=dropout_rate))
            curr_dim = head_hidden_size
        layers.append(nn.Linear(curr_dim, num_classes))
        head = nn.Sequential(*layers)

        # Initialize head weights
        self.init_weights(head)

        # Freeze all parameters in the backbone
        for param in backbone.parameters():
            param.requires_grad = False

        # Unfreeze the last `unfreeze_blocks` transformer blocks and LayerNorm
        if unfreeze_blocks > 0:
            for param in backbone.blocks[-unfreeze_blocks:].parameters():
                param.requires_grad = True
        
            # Make LayerNorm fine-tunable
            for param in backbone.norm.parameters():
                param.requires_grad = True

        # Make head fine-tunable
        for param in head.parameters():
            param.requires_grad = True

        # Save the backbone and head
        self.backbone = backbone
        self.head = head

        # Store configuration
        self._config = {
            'variant': variant,
            'dropout_rate': dropout_rate,
            'head_hidden_size': head_hidden_size,
            'head_layers': head_layers,
            'num_classes': num_classes,
            'unfreeze_blocks': unfreeze_blocks,
            'activation_fn': activation.__name__,
            'pretrained': pretrained
        }

    def init_weights(self, head):
        """Function to initialize weights for the head layers"""
        def init(layer):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        head.apply(init)

    def forward(self, x):
        features = self.backbone(x)
        out = self.head(features)
        return out

    def get_config(self) -> dict:
        """Return the config dict to reconstruct this model."""
        return self._config.copy()

    def get_name(self) -> str:
        """Return a unique name based on the network and head config."""
        cfg = self._config
        return (
            f"{cfg['variant']}_h{cfg['head_hidden_size']}_l{cfg['head_layers']}"
            f"_do{int(cfg['dropout_rate'] * 100)}_ub{cfg['unfreeze_blocks']}"
        )

    def unfreeze_blocks(self, num_blocks: int|str = 'all'):
        """Unfreeze the last `num_blocks` transformer blocks and LayerNorm."""
        if num_blocks == 'all':
            num_blocks = len(self.backbone.blocks)
        if num_blocks > 0:
            for param in self.backbone.blocks[-num_blocks:].parameters():
                param.requires_grad = True

            # Make LayerNorm fine-tunable
            for param in self.backbone.norm.parameters():
                param.requires_grad = True
                
    @classmethod
    def from_config(cls, config: dict) -> 'BaseDino':
        # Map string activation name back to class
        activation_map = {
            'ReLU': nn.ReLU,
            'GELU': nn.GELU,
            'SiLU': nn.SiLU,
            'ELU': nn.ELU,
        }
        act = activation_map.get(config.get('activation_fn', DEFAULT_ACTIVATION.__name__), DEFAULT_ACTIVATION)
        return cls(
            variant=config.get('variant', DEFAULT_VARIANT),
            dropout_rate=config.get('dropout_rate', DEFAULT_DROPOUT_RATE),
            head_hidden_size=config.get('head_hidden_size', DEFAULT_HEAD_HIDDEN_SIZE),
            head_layers=config.get('head_layers', DEFAULT_HEAD_LAYERS),
            num_classes=config.get('num_classes', DEFAULT_NUM_CLASSES),
            unfreeze_blocks=config.get('unfreeze_blocks', DEFAULT_UNFREEZE_BLOCKS),
            activation=act,
            pretrained=config.get('pretrained', DEFAULT_PRETRAINED),
        )