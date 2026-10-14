"""
src/pistn/forward_model.py
==========================
Forward Model orchestrator for the PI-STN framework.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

# PyTorch import handling
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    class nn:  # type: ignore
        Module = object
    TORCH_AVAILABLE = False

from src.pistn.network import ResidualMLP
from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


if TORCH_AVAILABLE:
    class PISTNForwardModel(nn.Module):
        """
        Coordinates hidden residual MLP network for PI-STN forward passes.
        """

        def __init__(self, cfg: Dict[str, Any]) -> None:
            super().__init__()
            self._cfg = cfg
            net_cfg = cfg.get("pistn", {}).get("network", {})
            
            input_dim = int(net_cfg.get("input_dim", 8))
            output_dim = int(net_cfg.get("output_dim", 9))
            hidden_layers = net_cfg.get("hidden_layers", [128, 128, 256, 256, 128])
            activation = net_cfg.get("activation", "silu")
            use_residual = bool(net_cfg.get("residual_connections", True))
            use_layer_norm = bool(net_cfg.get("layer_norm", True))
            dropout = float(net_cfg.get("dropout", 0.0))

            self.input_dim = input_dim

            self.network = ResidualMLP(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_layers=hidden_layers,
                activation=activation,
                use_residual=use_residual,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass mapping input vector (8 features) to output vector (9 predictions).
            """
            return self.network(x)
else:
    class PISTNForwardModel:  # type: ignore
        def __init__(self, cfg: Dict[str, Any]) -> None: pass
