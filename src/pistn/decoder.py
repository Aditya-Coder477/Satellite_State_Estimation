"""
src/pistn/decoder.py
====================
Decoder module for the PI-STN model. Maps hidden states to output predictions.
"""

from __future__ import annotations

import logging
from typing import Any

# PyTorch import handling
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    class nn:  # type: ignore
        Module = object
    TORCH_AVAILABLE = False

from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


if TORCH_AVAILABLE:
    class FeatureDecoder(nn.Module):
        """
        Maps hidden state to 9-dimensional outputs (ΔS, a_res).
        """
        def __init__(self, hidden_dim: int, output_dim: int) -> None:
            super().__init__()
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            self.linear = nn.Linear(hidden_dim, output_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x)
else:
    class FeatureDecoder:  # type: ignore
        def __init__(self, hidden_dim: int, output_dim: int) -> None: pass
