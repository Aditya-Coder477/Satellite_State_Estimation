"""
src/pistn/activations.py
========================
Activation functions for Phase 9 (PI-STN Framework).
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
    class Sin(nn.Module):
        """Sine activation function."""
        def __init__(self, w0: float = 1.0) -> None:
            super().__init__()
            self.w0 = w0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.sin(self.w0 * x)
else:
    class Sin:  # type: ignore
        def __init__(self, w0: float = 1.0) -> None: pass


class ActivationFactory:
    """Factory to retrieve PyTorch activation function modules from config strings."""

    @classmethod
    def create(cls, name: str) -> Any:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to instantiate activation functions.")

        name_lower = name.lower()

        if name_lower == "tanh":
            return nn.Tanh()
        elif name_lower == "relu":
            return nn.ReLU()
        elif name_lower == "silu":
            return nn.SiLU()
        elif name_lower == "gelu":
            return nn.GELU()
        elif name_lower == "sin":
            return Sin(w0=1.0)
        elif name_lower == "swish":
            return nn.SiLU()  # PyTorch's SiLU is mathematically equivalent to Swish
        else:
            available = ["tanh", "relu", "silu", "gelu", "sin", "swish"]
            raise ValueError(
                f"Unknown activation function '{name}'. Available options: {available}"
            )
