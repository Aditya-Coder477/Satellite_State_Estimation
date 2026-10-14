"""
src/pistn/scheduler_factory.py
==============================
Learning rate scheduler factory for Phase 9 (PI-STN).
"""

from __future__ import annotations

import logging
from typing import Any, Dict

# PyTorch import handling
try:
    import torch
    import torch.optim as optim
    from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class SchedulerFactory:
    """Creates learning rate schedulers for training."""

    @classmethod
    def create(cls, name: str, optimizer: Any, epochs: int) -> Any:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to instantiate schedulers.")

        name_lower = name.lower()
        log.info("Creating learning rate scheduler: '%s' over %d epochs", name_lower, epochs)

        if name_lower == "cosine":
            return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        elif name_lower == "step":
            return StepLR(optimizer, step_size=int(epochs / 5), gamma=0.5)
        elif name_lower == "plateau":
            return ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
        else:
            # Fallback to constant LR if scheduler is not recognized/defined
            log.warning("Unknown scheduler '%s'. No scheduler will be used.", name)
            return None
