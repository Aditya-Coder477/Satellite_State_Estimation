"""
src/pistn/optimizer_factory.py
==============================
Optimizer factory for Phase 9 (PI-STN).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable

# PyTorch import handling
try:
    import torch
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class OptimizerFactory:
    """Creates optimizers for model training from config."""

    @classmethod
    def create(cls, name: str, params: Iterable[torch.Tensor], lr: float, weight_decay: float = 1e-5) -> Any:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to instantiate optimizers.")

        name_lower = name.lower()
        log.info("Creating optimizer: '%s' with lr=%.3e, weight_decay=%.3e", name_lower, lr, weight_decay)

        if name_lower == "adamw":
            return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        elif name_lower == "adam":
            return optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif name_lower == "sgd":
            return optim.SGD(params, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer '{name}'. Supported: ['adamw', 'adam', 'sgd'].")
