"""
src/pinn/optimizer_factory.py
=============================
Creates PyTorch optimizers for Phase 8 (PINN Framework).

Supports Adam, AdamW, L-BFGS, and SGD. Integrates with the project configuration
for learning rates, weight decays, and L-BFGS line search settings.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

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
    """Factory to instantiate PyTorch optimizers based on config."""

    @classmethod
    def create(
        cls,
        name: str,
        params: Iterable[torch.Tensor],
        lr: float,
        weight_decay: float = 0.0,
        **kwargs: Any,
    ) -> torch.optim.Optimizer:
        """
        Creates the requested PyTorch optimizer.

        Parameters
        ----------
        name : str
            Name of the optimizer ("adam", "adamw", "lbfgs", "sgd").
        params : Iterable[torch.Tensor]
            Parameters of the network to optimize.
        lr : float
            Learning rate.
        weight_decay : float, optional
            L2 regularization strength, by default 0.0.
        **kwargs : Any
            Additional optimizer-specific settings (e.g. max_iter, history_size for L-BFGS).

        Returns
        -------
        torch.optim.Optimizer
            Instantiated optimizer.

        Raises
        ------
        ImportError
            If PyTorch is not installed.
        ValueError
            If the optimizer name is unknown.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to instantiate optimizers.")

        name_lower = name.lower()
        log.info("Instantiating optimizer '%s' with lr=%.6e, weight_decay=%.6e", name_lower, lr, weight_decay)

        if name_lower == "adam":
            return optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif name_lower == "adamw":
            return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        elif name_lower == "sgd":
            return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
        elif name_lower == "lbfgs":
            # Extract L-BFGS specific options with standard PINN defaults
            max_iter = int(kwargs.get("max_iter", 20))
            history_size = int(kwargs.get("history_size", 100))
            line_search_fn = kwargs.get("line_search_fn", "strong_wolfe")  # standard for L-BFGS in PINNs
            
            return optim.LBFGS(
                params,
                lr=lr,
                max_iter=max_iter,
                history_size=history_size,
                line_search_fn=line_search_fn,
            )
        else:
            available = ["adam", "adamw", "lbfgs", "sgd"]
            raise ValueError(
                f"Unknown optimizer '{name}'. Available options: {available}"
            )
