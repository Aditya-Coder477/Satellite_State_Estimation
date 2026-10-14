"""
src/physics/constraints.py
==========================
Evaluates physical state constraints for Phase 7 (Physics Engine).

Provides differentiable constraint penalties (e.g. radius above Earth surface,
velocity within escape bounds, and finiteness) to guide PINN optimization.
Supports both PyTorch and NumPy.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np

# PyTorch import handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.physics import constants
from src.physics.coordinate_dynamics import compute_norm, get_position, get_velocity, is_tensor
from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class ConstraintEvaluator:
    """
    Evaluates physical state violations and computes differentiable penalty losses.

    Parameters
    ----------
    cfg : Dict[str, Any]
        Project configuration.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self._cfg = cfg
        self._constr_cfg = cfg.get("physics", {}).get("constraints", {})

        # Resolve bounds
        self.min_radius = float(self._constr_cfg.get("min_radius_km", constants.R_E))
        self.max_velocity = float(self._constr_cfg.get("max_velocity_kms", 12.0))

        log.info(
            "ConstraintEvaluator initialised with min_radius=%.3f km, max_velocity=%.3f km/s",
            self.min_radius,
            self.max_velocity,
        )

    def evaluate(self, state_pred: Any, acc_pred: Any = None) -> Tuple[Dict[str, bool], Dict[str, Any]]:
        """
        Evaluates predictions against physical constraints and calculates penalties.

        Parameters
        ----------
        state_pred : Any
            Predicted state vector [x, y, z, vx, vy, vz] of shape (..., 6).
        acc_pred : Any, optional
            Predicted acceleration vector [ax, ay, az] of shape (..., 3), by default None.

        Returns
        -------
        Tuple[Dict[str, bool], Dict[str, Any]]
            1. Dictionary of boolean flags indicating if constraints are satisfied.
            2. Dictionary of differentiable penalty loss tensors/arrays.
        """
        pos = get_position(state_pred)
        vel = get_velocity(state_pred)

        r_norm = compute_norm(pos)
        v_norm = compute_norm(vel)

        # 1. Check basic constraints
        is_tensor_input = is_tensor(state_pred)

        # NaN / Inf Checks
        if is_tensor_input:
            has_nan = torch.isnan(state_pred).any().item()
            has_inf = torch.isinf(state_pred).any().item()
        else:
            has_nan = np.isnan(state_pred).any()
            has_inf = np.isinf(state_pred).any()

        no_nans = not has_nan
        no_infs = not has_inf

        # 2. Radius Positive Check (Above Earth surface)
        if is_tensor_input:
            radius_ok = (r_norm >= self.min_radius).all().item()
            # Quadratic penalty for violating radius lower bound: max(0, min_radius - r)^2
            r_viol = torch.clamp(self.min_radius - r_norm, min=0.0)
            radius_penalty = torch.mean(r_viol ** 2)
        else:
            radius_ok = bool(np.all(r_norm >= self.min_radius))
            r_viol = np.clip(self.min_radius - r_norm, 0.0, None)
            radius_penalty = float(np.mean(r_viol ** 2))

        # 3. Finite Velocity Check (Below Escape Velocity)
        if is_tensor_input:
            velocity_ok = (v_norm <= self.max_velocity).all().item()
            # Quadratic penalty for violating velocity upper bound: max(0, v - max_velocity)^2
            v_viol = torch.clamp(v_norm - self.max_velocity, min=0.0)
            velocity_penalty = torch.mean(v_viol ** 2)
        else:
            velocity_ok = bool(np.all(v_norm <= self.max_velocity))
            v_viol = np.clip(v_norm - self.max_velocity, 0.0, None)
            velocity_penalty = float(np.mean(v_viol ** 2))

        # 4. Finite Acceleration Check
        acceleration_ok = True
        acceleration_penalty = 0.0
        if acc_pred is not None:
            a_norm = compute_norm(acc_pred)
            max_acc = 0.5  # extreme upper limit: 0.5 km/s^2 (~50g)
            if is_tensor_input:
                acceleration_ok = (a_norm <= max_acc).all().item()
                a_viol = torch.clamp(a_norm - max_acc, min=0.0)
                acceleration_penalty = torch.mean(a_viol ** 2)
            else:
                acceleration_ok = bool(np.all(a_norm <= max_acc))
                a_viol = np.clip(a_norm - max_acc, 0.0, None)
                acceleration_penalty = float(np.mean(a_viol ** 2))

        # 5. Stubs for future conservation constraints
        energy_penalty = 0.0
        momentum_penalty = 0.0

        status = {
            "no_nans": no_nans,
            "no_infs": no_infs,
            "radius_satisfied": radius_ok,
            "velocity_satisfied": velocity_ok,
            "acceleration_satisfied": acceleration_ok,
            "overall_satisfied": no_nans and no_infs and radius_ok and velocity_ok and acceleration_ok,
        }

        penalties = {
            "radius_penalty": radius_penalty,
            "velocity_penalty": velocity_penalty,
            "acceleration_penalty": acceleration_penalty,
            "energy_conservation_penalty": energy_penalty,  # stub
            "angular_momentum_penalty": momentum_penalty,  # stub
            "total_constraint_penalty": radius_penalty + velocity_penalty + acceleration_penalty,
        }

        return status, penalties
