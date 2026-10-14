"""
src/pinn/activations.py
=======================
Neural network activation functions for Phase 8 (PINN Framework).

Provides standard PyTorch activations and a custom Sin activation (SIREN-like)
through a unified ActivationFactory.
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
    # Stub classes for compilation safety
    class nn:  # type: ignore
        Module = object
    TORCH_AVAILABLE = False

from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


if TORCH_AVAILABLE:
    class Sin(nn.Module):
        """
        Sine activation function (SIREN-like).
        
        Applies elements-wise: sin(w0 * x)
        
        Parameters
        ----------
        w0 : float, optional
            Frequency scaling factor, by default 1.0.
        """
        def __init__(self, w0: float = 1.0) -> None:
            super().__init__()
            self.w0 = w0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.sin(self.w0 * x)

    class Swish(nn.Module):
        """
        Swish activation function (x * sigmoid(x)).
        
        Equivalent to SiLU. Included for backward compatibility.
        """
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * torch.sigmoid(x)
else:
    # Stubs if PyTorch is not available
    class Sin:  # type: ignore
        def __init__(self, w0: float = 1.0) -> None: pass
    class Swish:  # type: ignore
        pass


class ActivationFactory:
    """Factory to retrieve PyTorch activation function modules from config strings."""

    @classmethod
    def create(cls, name: str) -> Any:
        """
        Creates the requested activation module.

        Parameters
        ----------
        name : str
            Name of the activation function (e.g. "tanh", "relu", "silu", "gelu", "sin", "swish").

        Returns
        -------
        nn.Module
            Instantiated PyTorch activation module.

        Raises
        ------
        ImportError
            If PyTorch is not installed.
        ValueError
            If the activation name is unknown.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to instantiate activation functions.")

        name_lower = name.lower()

        # Mapping names to nn.Modules
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
            return Swish()
        else:
            available = ["tanh", "relu", "silu", "gelu", "sin", "swish"]
            raise ValueError(
                f"Unknown activation function '{name}'. Available options: {available}"
            )
