"""
src/pistn/network.py
===================
Fully Connected Residual MLP architecture for Phase 9 (PI-STN Framework).

Implements residual connections, LayerNorm, customizable activation, and dropout support.
"""

from __future__ import annotations

import logging
from typing import Any, List

# PyTorch import handling
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    class nn:  # type: ignore
        Module = object
    TORCH_AVAILABLE = False

from src.pistn.activations import ActivationFactory
from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


if TORCH_AVAILABLE:
    class ResidualBlock(nn.Module):
        """
        A single Linear layer with LayerNorm, activation, optional dropout, and residual shortcut.
        """
        def __init__(
            self,
            in_features: int,
            out_features: int,
            activation: str,
            use_norm: bool = True,
            dropout: float = 0.0
        ) -> None:
            super().__init__()
            self.linear = nn.Linear(in_features, out_features)
            self.norm = nn.LayerNorm(out_features) if use_norm else nn.Identity()
            self.act = ActivationFactory.create(activation)
            self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
            self.shortcut = nn.Identity()
            
            # If dimensions change, add a projection shortcut
            if in_features != out_features:
                self.shortcut = nn.Linear(in_features, out_features, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.linear(x)
            out = self.norm(out)
            out = self.act(out)
            out = self.drop(out)
            return out + self.shortcut(x)


    class ResidualMLP(nn.Module):
        """
        Fully Connected Residual MLP for State Transition learning.
        """

        def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_layers: List[int],
            activation: str,
            use_residual: bool = True,
            use_layer_norm: bool = True,
            dropout: float = 0.0,
            initializer: str = "xavier",
        ) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.hidden_layers = hidden_layers
            self.use_residual = use_residual
            self.use_layer_norm = use_layer_norm
            self.dropout = dropout
            self.initializer = initializer

            # Create layers
            blocks = []
            prev_dim = input_dim
            
            for h_dim in hidden_layers:
                if use_residual:
                    blocks.append(ResidualBlock(prev_dim, h_dim, activation, use_layer_norm, dropout))
                else:
                    layer_seq = [
                        nn.Linear(prev_dim, h_dim),
                        nn.LayerNorm(h_dim) if use_layer_norm else nn.Identity(),
                        ActivationFactory.create(activation)
                    ]
                    if dropout > 0.0:
                        layer_seq.append(nn.Dropout(dropout))
                    blocks.append(nn.Sequential(*layer_seq))
                prev_dim = h_dim

            self.hidden_net = nn.Sequential(*blocks)
            self.output_layer = nn.Linear(prev_dim, output_dim)
            self._init_weights()

            log.info(
                "ResidualMLP initialized: input=%d, hidden=%s, output=%d, residual=%s, layernorm=%s, dropout=%.2f, parameters=%d",
                input_dim,
                hidden_layers,
                output_dim,
                use_residual,
                use_layer_norm,
                dropout,
                sum(p.numel() for p in self.parameters() if p.requires_grad),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            features = self.hidden_net(x)
            return self.output_layer(features)

        def _init_weights(self) -> None:
            """Applies weight initialization to all linear layers."""
            init_name = self.initializer.lower()
            
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    if init_name == "xavier":
                        nn.init.xavier_normal_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    elif init_name == "kaiming":
                        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    elif init_name == "normal":
                        nn.init.normal_(m.weight, mean=0.0, std=0.02)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
else:
    class ResidualMLP:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None: pass
