"""
src/physics/residual_dynamics.py
================================
Computes residual dynamics for Phase 8 (Physics Engine).

Calculates a_transition = Δv / Δt and a_physics = gravity(r).
Then computes the residual acceleration: a_res = a_transition - a_physics.
Operates purely on NumPy arrays and PyTorch tensors.
"""

from __future__ import annotations

import logging
from typing import Any, Tuple

import numpy as np

# PyTorch import handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.physics.coordinate_dynamics import get_position, is_tensor
from src.physics.gravity_models import GravityModel
from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class PhysicsResidualCalculator:
    """
    Computes governing physics residuals for satellite trajectories.

    For Version 2:
      - a_transition = Δv / Δt
      - a_physics = gravity(r)
      - a_res = a_transition - a_physics
    """

    def __init__(self, gravity_model: GravityModel) -> None:
        self.gravity_model = gravity_model

    def compute_residuals(self, state: Any, delta_v: Any, delta_t: Any) -> Tuple[Any, Any, Any]:
        """
        Computes transition, physics, and residual accelerations.

        Parameters
        ----------
        state : Any
            Cartesian state vector [x, y, z, vx, vy, vz] of shape (..., 6).
        delta_v : Any
            Velocity change vector [dvx, dvy, dvz] of shape (..., 3).
        delta_t : Any
            Time step [seconds] of shape (...) or float.

        Returns
        -------
        Tuple[Any, Any, Any]
            1. a_transition (..., 3)
            2. a_physics (..., 3)
            3. a_res (..., 3)
        """
        pos = get_position(state)

        # Align delta_t shape for element-wise division
        if isinstance(delta_t, (np.ndarray, torch.Tensor if TORCH_AVAILABLE else Any)):
            if len(delta_t.shape) < len(delta_v.shape):
                if is_tensor(delta_t):
                    dt_aligned = delta_t.unsqueeze(-1)
                else:
                    dt_aligned = np.expand_dims(delta_t, axis=-1)
            else:
                dt_aligned = delta_t
        else:
            dt_aligned = delta_t

        # Prevent division by zero
        eps = 1e-6
        if is_tensor(delta_v):
            dt_t = dt_aligned if is_tensor(dt_aligned) else torch.tensor(dt_aligned, device=delta_v.device, dtype=delta_v.dtype)
            dt_safe = torch.clamp(dt_t, min=eps)
            a_transition = delta_v / dt_safe
        else:
            dt_np = dt_aligned.cpu().numpy() if is_tensor(dt_aligned) else dt_aligned
            dt_safe = np.clip(dt_np, eps, None) if isinstance(dt_np, np.ndarray) else max(dt_np, eps)
            a_transition = delta_v / dt_safe

        # Compute physics gravity acceleration
        a_physics = self.gravity_model.compute_acceleration(pos)

        # Residual acceleration: a_res = a_transition - a_physics
        a_res = a_transition - a_physics

        return a_transition, a_physics, a_res
