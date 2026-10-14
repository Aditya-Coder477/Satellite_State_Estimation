"""
src/pinn/scheduler_factory.py
==============================
Creates PyTorch learning rate schedulers for Phase 8 (PINN Framework).

Supports CosineAnnealingLR, StepLR, ReduceLROnPlateau, and static learning rates (None).
"""

from __future__ import annotations

import logging
from typing import Any

# PyTorch import handling
try:
    import torch
    import torch.optim as optim
    import torch.optim.lr_scheduler as lr_scheduler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class SchedulerFactory:
    """Factory to instantiate learning rate schedulers for PyTorch optimizers."""

    @classmethod
    def create(
        cls,
        name: str,
        optimizer: torch.optim.Optimizer,
        **kwargs: Any,
    ) -> torch.optim.lr_scheduler._LRScheduler | None:
        """
        Creates the requested PyTorch learning rate scheduler.

        Parameters
        ----------
        name : str
            Name of the scheduler ("cosine", "step", "plateau", "none").
        optimizer : torch.optim.Optimizer
            The active PyTorch optimizer.
        **kwargs : Any
            Additional configuration settings (e.g. step_size, gamma, patience).

        Returns
        -------
        _LRScheduler | None
            Instantiated learning rate scheduler, or None if no decay is selected.

        Raises
        ------
        ImportError
            If PyTorch is not installed.
        ValueError
            If the scheduler name is unknown.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to instantiate schedulers.")

        name_lower = name.lower()
        if name_lower == "none":
            log.info("No learning rate scheduler configured.")
            return None

        log.info("Instantiating learning rate scheduler '%s'", name_lower)

        if name_lower == "cosine":
            # T_max is the number of iterations or epochs. Default to 1000 if not provided
            t_max = int(kwargs.get("step_size", 1000))
            return lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
            
        elif name_lower == "step":
            step_size = int(kwargs.get("step_size", 1000))
            gamma = float(kwargs.get("gamma", 0.5))
            return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            
        elif name_lower == "plateau":
            patience = int(kwargs.get("patience", 10))
            gamma = float(kwargs.get("gamma", 0.1))
            return lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=gamma,
                patience=patience,
                verbose=True,
            )
        else:
            available = ["cosine", "step", "plateau", "none"]
            raise ValueError(
                f"Unknown scheduler '{name}'. Available options: {available}"
            )
