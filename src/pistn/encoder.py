"""
src/pistn/encoder.py
====================
Encoder module for the PI-STN model. Maps raw input features to hidden space.
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
    class FeatureEncoder(nn.Module):
        """
        Maps 8-dimensional input (t, dt, x, y, z, vx, vy, vz) to hidden state.
        """
        def __init__(self, input_dim: int, hidden_dim: int) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.linear = nn.Linear(input_dim, hidden_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x)
else:
    class FeatureEncoder:  # type: ignore
        def __init__(self, input_dim: int, hidden_dim: int) -> None: pass
