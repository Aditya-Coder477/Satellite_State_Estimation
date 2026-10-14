"""
src/physics/orbital_equations.py
================================
Governing orbital dynamics equations for Phase 8 (Physics Engine).

Computes only derivatives: d(State)/dt for a Cartesian state using a configured gravity model.
Numerical integration belongs exclusively to state_transition.py.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

# PyTorch import handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.physics.coordinate_dynamics import get_position, get_velocity, is_tensor
from src.physics.gravity_models import GravityModel
from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


def compute_state_derivative(state: Any, gravity_model: GravityModel) -> Any:
    """
    Computes the time derivative of the satellite Cartesian state.

    Calculates:
        dx/dt = vx,  dy/dt = vy,  dz/dt = vz
        dvx/dt = ax, dvy/dt = ay, dvz/dt = az
    where [ax, ay, az] is computed via the selected GravityModel.

    Parameters
    ----------
    state : Any
        Cartesian state tensor of shape (..., 6) (km, km/s). Supports PyTorch & NumPy.
    gravity_model : GravityModel
        Fitted gravity model strategy instance.

    Returns
    -------
    Any
        State derivative tensor of shape (..., 6) (km/s, km/s^2).
    """
    pos = get_position(state)
    vel = get_velocity(state)

    # Compute acceleration using the gravity model strategy pattern
    acc = gravity_model.compute_acceleration(pos)

    # Concatenate velocity and acceleration
    if is_tensor(state):
        return torch.cat([vel, acc], dim=-1)
    else:
        return np.concatenate([vel, acc], axis=-1)
