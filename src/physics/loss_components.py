"""
src/physics/loss_components.py
==============================
Computes individual loss terms and their weighted sum for Phase 8 (Physics Engine).

Computes transition loss, physics loss, residual loss, constraint loss, and total loss.
Supports both PyTorch and NumPy.
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
    Computes physics-informed loss components and weighted total loss for PI-STN.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self._cfg = cfg
        self._loss_cfg = cfg.get("physics", {}).get("losses", {})

        # Extract weights from configuration
        self.w_trans = float(self._loss_cfg.get("transition_weight", self._loss_cfg.get("observation_weight", 1.0)))
        self.w_phys = float(self._loss_cfg.get("physics_weight", 10.0))
        self.w_resid = float(self._loss_cfg.get("residual_weight", 5.0))
        self.w_const = float(self._loss_cfg.get("constraint_weight", 1.0))

        log.info(
            "PhysicsLossEvaluator initialized with weights: "
            "transition=%.3f, physics=%.3f, residual=%.3f, constraint=%.3f",
            self.w_trans,
            self.w_phys,
            self.w_resid,
            self.w_const,
        )

    def compute_loss(
        self,
        pred_delta: Any,
        true_delta: Any,
        phys_delta: Any,
        pred_res: Any,
        true_res: Any,
        constraint_penalty: Any = 0.0,
    ) -> Dict[str, Any]:
        """
        Computes individual loss components and their weighted sum.

        Parameters
        ----------
        pred_delta : Any
            Predicted state transitions [Δx, Δy, Δz, Δvx, Δvy, Δvz] of shape (..., 6).
        true_delta : Any
            Ground-truth state transitions of shape (..., 6).
        phys_delta : Any
            Expected physics-based state transitions of shape (..., 6).
        pred_res : Any
            Predicted residual dynamics acceleration [ax_res, ay_res, az_res] of shape (..., 3).
        true_res : Any
            Ground-truth residual dynamics acceleration of shape (..., 3).
        constraint_penalty : Any, optional
            Precomputed constraint penalty sum, by default 0.0.

        Returns
        -------
        Dict[str, Any]
            Dictionary of unweighted and weighted loss components.
        """
        is_tensor_input = is_tensor(pred_delta)

        # 1. Transition Loss: MSE of predicted vs true transitions
        if is_tensor_input:
            transition_loss = torch.mean((pred_delta - true_delta) ** 2)
        else:
            transition_loss = float(np.mean((pred_delta - true_delta) ** 2))

        # 2. Physics Loss: MSE of predicted vs expected physics transitions
        if is_tensor_input:
            physics_loss = torch.mean((pred_delta - phys_delta) ** 2)
        else:
            physics_loss = float(np.mean((pred_delta - phys_delta) ** 2))

        # 3. Residual Loss: MSE of predicted vs true residual accelerations
        if is_tensor_input:
            residual_loss = torch.mean((pred_res - true_res) ** 2)
        else:
            residual_loss = float(np.mean((pred_res - true_res) ** 2))

        # 4. Total Loss (Weighted sum)
        total_loss = (
            self.w_trans * transition_loss
            + self.w_phys * physics_loss
            + self.w_resid * residual_loss
            + self.w_const * constraint_penalty
        )

        return {
            "transition_loss": transition_loss,
            "physics_loss": physics_loss,
            "residual_loss": residual_loss,
            "constraint_loss": constraint_penalty,
            "total_physics_loss": total_loss,
        }
