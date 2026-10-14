"""
src/physics/constraints.py
==========================
Evaluates physical state constraints for Phase 8/9 (Physics Engine).

Provides differentiable constraint penalties (using relu formulation) to guide optimization.
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
from src.physics.coordinate_dynamics import (
    compute_norm,
    get_position,
    get_velocity,
    is_tensor,
    orbital_energy,
    angular_momentum,
)
from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class ConstraintEvaluator:
    """
    Evaluates physical state violations and computes differentiable penalty losses.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self._cfg = cfg
        self._constr_cfg = cfg.get("physics", {}).get("constraints", {})

        # Resolve bounds
        self.min_radius = float(self._constr_cfg.get("minimum_radius_km", constants.R_E))
        self.max_velocity = float(self._constr_cfg.get("maximum_velocity_kms", 12.0))
        self.enforce_energy = bool(self._constr_cfg.get("enforce_energy", True))
        self.enforce_momentum = bool(self._constr_cfg.get("enforce_angular_momentum", True))

        log.info(
            "ConstraintEvaluator initialized with min_radius=%.3f km, max_velocity=%.3f km/s",
            self.min_radius,
            self.max_velocity,
        )

    def evaluate(self, state: Any, acc: Any = None) -> Tuple[Dict[str, bool], Dict[str, Any]]:
        """
        Evaluates state vector against physical constraints.

        Parameters
        ----------
        state : Any
            Cartesian state vector [x, y, z, vx, vy, vz] of shape (..., 6).
        acc : Any, optional
            Acceleration vector [ax, ay, az] of shape (..., 3), by default None.

        Returns
        -------
        Tuple[Dict[str, bool], Dict[str, Any]]
            1. Dictionary of constraint satisfaction boolean flags.
            2. Dictionary of differentiable penalty loss values.
        """
        pos = get_position(state)
        vel = get_velocity(state)

        r_norm = compute_norm(pos, keepdims=False)
        v_norm = compute_norm(vel, keepdims=False)

        is_tensor_input = is_tensor(state)

        # 1. NaN / Inf Checks
        if is_tensor_input:
            has_nan = torch.isnan(state).any().item()
            has_inf = torch.isinf(state).any().item()
        else:
            has_nan = np.isnan(state).any()
            has_inf = np.isinf(state).any()

        no_nans = not has_nan
        no_infs = not has_inf

        # 2. Radius Positive Check (Above Earth surface)
        # Correct formulation using relu: penalty = torch.relu(min_radius - radius)
        if is_tensor_input:
            radius_ok = (r_norm >= self.min_radius).all().item()
            radius_penalty = torch.mean(torch.relu(self.min_radius - r_norm))
        else:
            radius_ok = bool(np.all(r_norm >= self.min_radius))
            radius_penalty = float(np.mean(np.maximum(0.0, self.min_radius - r_norm)))

        # 3. Finite Velocity Check
        if is_tensor_input:
            velocity_ok = (v_norm <= self.max_velocity).all().item()
            velocity_penalty = torch.mean(torch.relu(v_norm - self.max_velocity))
        else:
            velocity_ok = bool(np.all(v_norm <= self.max_velocity))
            velocity_penalty = float(np.mean(np.maximum(0.0, v_norm - self.max_velocity)))

        # 4. Energy Finiteness / Conservation Check
        energy_ok = True
        energy_penalty = 0.0
        if self.enforce_energy:
            e = orbital_energy(state)
            if is_tensor_input:
                energy_penalty = torch.mean(torch.relu(e))  # GEO is bound, E should be < 0
                energy_ok = (e < 0.0).all().item()
            else:
                energy_penalty = float(np.mean(np.maximum(0.0, e)))
                energy_ok = bool(np.all(e < 0.0))

        # 5. Angular Momentum Finiteness Check
        momentum_ok = True
        momentum_penalty = 0.0
        if self.enforce_momentum:
            h = angular_momentum(state)
            h_norm = compute_norm(h, keepdims=False)
            min_h = 10000.0  # km^2/s minimum bound
            if is_tensor_input:
                momentum_penalty = torch.mean(torch.relu(min_h - h_norm))
                momentum_ok = (h_norm >= min_h).all().item()
            else:
                momentum_penalty = float(np.mean(np.maximum(0.0, min_h - h_norm)))
                momentum_ok = bool(np.all(h_norm >= min_h))

        # 6. Finite Acceleration Check (optional)
        acceleration_ok = True
        acceleration_penalty = 0.0
        if acc is not None:
            a_norm = compute_norm(acc, keepdims=False)
            max_acc = 0.5  # km/s^2 upper bound
            if is_tensor_input:
                acceleration_ok = (a_norm <= max_acc).all().item()
                acceleration_penalty = torch.mean(torch.relu(a_norm - max_acc))
            else:
                acceleration_ok = bool(np.all(a_norm <= max_acc))
                acceleration_penalty = float(np.mean(np.maximum(0.0, a_norm - max_acc)))

        status = {
            "no_nans": no_nans,
            "no_infs": no_infs,
            "radius_satisfied": radius_ok,
            "velocity_satisfied": velocity_ok,
            "energy_satisfied": energy_ok,
            "momentum_satisfied": momentum_ok,
            "acceleration_satisfied": acceleration_ok,
            "overall_satisfied": no_nans and no_infs and radius_ok and velocity_ok and energy_ok and momentum_ok and acceleration_ok,
        }

        penalties = {
            "radius_penalty": radius_penalty,
            "velocity_penalty": velocity_penalty,
            "energy_penalty": energy_penalty,
            "momentum_penalty": momentum_penalty,
            "acceleration_penalty": acceleration_penalty,
            "total_constraint_penalty": radius_penalty + velocity_penalty + energy_penalty + momentum_penalty + acceleration_penalty,
        }

        return status, penalties
