"""
src/physics/coordinate_dynamics.py
==================================
Vector operations for satellite Cartesian coordinates in Phase 8 (Physics Engine).

Provides position/velocity parsing, norms, unit vectors, relative vectors, cross products,
angular momentum, and orbital energy calculations.
Supports both PyTorch tensors and NumPy arrays dynamically, including batched inputs.
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

from src.physics import constants
from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


def is_tensor(x: Any) -> bool:
    """Returns True if the object is a PyTorch Tensor."""
    return TORCH_AVAILABLE and isinstance(x, torch.Tensor)


def get_position(state: Any) -> Any:
    """
    Extracts the position vector [x, y, z] from a state vector [x, y, z, vx, vy, vz].
    """
    return state[..., :3]


def get_velocity(state: Any) -> Any:
    """
    Extracts the velocity vector [vx, vy, vz] from a state vector [x, y, z, vx, vy, vz].
    """
    return state[..., 3:6]


def compute_norm(vector: Any, axis: int = -1, keepdims: bool = True) -> Any:
    """
    Computes the vector magnitude (norm) with gradient safety at 0.
    """
    if is_tensor(vector):
        eps = 1e-16
        return torch.sqrt(torch.sum(vector ** 2, dim=axis, keepdim=keepdims) + eps)
    else:
        return np.linalg.norm(vector, axis=axis, keepdims=keepdims)


def compute_unit_vector(vector: Any) -> Any:
    """
    Computes the unit vector (direction) of a vector.
    """
    norm = compute_norm(vector)
    if is_tensor(vector):
        eps = 1e-3
        safe_norm = torch.clamp(norm, min=eps)
        return vector / safe_norm
    else:
        eps = 1e-12
        safe_norm = np.clip(norm, eps, None)
        return vector / safe_norm


def compute_relative_position(pos_sat: Any, pos_station: Any) -> Any:
    """Computes relative position vector from station to satellite: r_rel = r_sat - r_station."""
    return pos_sat - pos_station


def compute_relative_velocity(vel_sat: Any, vel_station: Any) -> Any:
    """Computes relative velocity vector: v_rel = v_sat - v_station."""
    return vel_sat - vel_station


def compute_dot_product(vec1: Any, vec2: Any, axis: int = -1, keepdims: bool = True) -> Any:
    """
    Computes the dot product between two vectors.
    """
    if is_tensor(vec1):
        return torch.sum(vec1 * vec2, dim=axis, keepdim=keepdims)
    else:
        return np.sum(vec1 * vec2, axis=axis, keepdims=keepdims)


def compute_cross_product(vec1: Any, vec2: Any, dim: int = -1) -> Any:
    """
    Computes the cross product between two vectors of shape (..., 3).
    """
    if is_tensor(vec1):
        return torch.linalg.cross(vec1, vec2, dim=dim)
    else:
        return np.cross(vec1, vec2, axis=dim)


def angular_momentum(state: Any) -> Any:
    """
    Computes specific angular momentum vector h = r x v.
    """
    pos = get_position(state)
    vel = get_velocity(state)
    return compute_cross_product(pos, vel)


def specific_mechanical_energy(state: Any, mu: float = constants.MU) -> Any:
    """
    Computes specific mechanical orbital energy: E = 0.5 * v^2 - mu / r.
    """
    pos = get_position(state)
    vel = get_velocity(state)
    
    r_norm = compute_norm(pos, keepdims=False)
    v_norm = compute_norm(vel, keepdims=False)
    
    if is_tensor(state):
        eps = 1e-3
        r_norm_safe = torch.clamp(r_norm, min=eps)
        return 0.5 * (v_norm ** 2) - mu / r_norm_safe
    else:
        eps = 1e-12
        r_norm_safe = np.clip(r_norm, eps, None)
        return 0.5 * (v_norm ** 2) - mu / r_norm_safe


# Aliases as requested by Phase 8 specification
norm = compute_norm
unit_vector = compute_unit_vector
dot = compute_dot_product
cross = compute_cross_product
orbital_energy = specific_mechanical_energy
