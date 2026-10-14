"""
src/pistn/callbacks.py
======================
Callbacks for training Phase 9 (PI-STN).

Includes EarlyStopping with support for restore_best_weights.
"""

from __future__ import annotations

import logging
import copy
from typing import Any, Dict

# PyTorch import handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class EarlyStopping:
    """
    Stops training early if validation loss does not improve after a configured patience.
    Optionally restores the best weights.
    """

    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 1e-6,
        restore_best_weights: bool = True,
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_loss: float | None = None
        self.counter = 0
        self.early_stop = False
        self.best_weights: Dict[str, Any] | None = None

        log.info(
            "EarlyStopping initialized with patience=%d, min_delta=%.1e, restore_best_weights=%s",
            patience,
            min_delta,
            restore_best_weights,
        )

    def __call__(self, val_loss: float, model: Any) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights and TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
                self.best_weights = copy.deepcopy(model.state_dict())
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                log.info("Early stopping triggered. No improvement in val loss for %d epochs.", self.patience)
                self.early_stop = True
                if self.restore_best_weights and self.best_weights is not None and TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
                    log.info("Restoring best model weights...")
                    model.load_state_dict(self.best_weights)
        else:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights and TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
                self.best_weights = copy.deepcopy(model.state_dict())
            
        return self.early_stop
