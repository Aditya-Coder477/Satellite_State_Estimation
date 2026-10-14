"""
src/physics/loss_components.py
==============================
Computes individual loss terms and their weighted sum for Phase 7 (Physics Engine).

This module prepares the loss components (physics residuals loss, boundary loss,
initial condition loss, and constraint penalties) used by the PINN trainer.
No optimization or backpropagation is performed here.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

# PyTorch import handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.physics.coordinate_dynamics import is_tensor
from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class PhysicsLossEvaluator:
    """
    Computes physics-informed loss components and weighted total loss.

    Parameters
    ----------
    cfg : Dict[str, Any]
        Project configuration.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self._cfg = cfg
        self._loss_cfg = cfg.get("physics", {}).get("loss", {})
        self._resid_cfg = cfg.get("physics", {}).get("residual", {})

        # Extract weights from configuration
        self.w_phys = float(self._loss_cfg.get("physics_weight", 1.0))
        self.w_bound = float(self._loss_cfg.get("boundary_weight", 1.0))
        self.w_init = float(self._loss_cfg.get("initial_weight", 1.0))
        self.w_const = float(self._loss_cfg.get("constraint_weight", 0.1))
        
        self.reduction = self._resid_cfg.get("reduction", "mean").lower()

        log.info(
            "PhysicsLossEvaluator initialised with weights: "
            "physics=%.3f, boundary=%.3f, initial=%.3f, constraint=%.3f",
            self.w_phys,
            self.w_bound,
            self.w_init,
            self.w_const,
        )

    def compute_loss(
        self,
        residuals: Any,
        boundary_loss: Any = 0.0,
        initial_loss: Any = 0.0,
        constraint_penalty: Any = 0.0,
    ) -> Dict[str, Any]:
        """
        Computes individual loss components and their weighted sum.

        Parameters
        ----------
        residuals : Any
            Residual tensor of shape (..., 6). Supports PyTorch & NumPy.
        boundary_loss : Any, optional
            Precomputed boundary loss value (MSE), by default 0.0.
        initial_loss : Any, optional
            Precomputed initial condition loss value (MSE), by default 0.0.
        constraint_penalty : Any, optional
            Precomputed constraint penalty sum, by default 0.0.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - physics_loss (unweighted)
            - boundary_loss (unweighted)
            - initial_loss (unweighted)
            - constraint_loss (unweighted)
            - total_physics_loss (weighted sum of all components)
        """
        is_tensor_input = is_tensor(residuals)

        # 1. Physics Residual Loss
        if is_tensor_input:
            squared_res = residuals ** 2
            if self.reduction == "sum":
                physics_loss = torch.sum(squared_res)
            else:
                physics_loss = torch.mean(squared_res)
        else:
            squared_res = residuals ** 2
            if self.reduction == "sum":
                physics_loss = float(np.sum(squared_res))
            else:
                physics_loss = float(np.mean(squared_res))

        # 2. Weighted Total Loss
        total_loss = (
            self.w_phys * physics_loss
            + self.w_bound * boundary_loss
            + self.w_init * initial_loss
            + self.w_const * constraint_penalty
        )

        return {
            "physics_loss": physics_loss,
            "boundary_loss": boundary_loss,
            "initial_loss": initial_loss,
            "constraint_loss": constraint_penalty,
            "total_physics_loss": total_loss,
        }
