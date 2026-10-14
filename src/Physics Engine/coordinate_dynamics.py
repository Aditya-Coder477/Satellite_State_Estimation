"""
src/physics/coordinate_dynamics.py
==================================
Vector operations for satellite Cartesian coordinates in Phase 7 (Physics Engine).

Provides position/velocity parsing, norms, unit vectors, and relative vectors.
Supports both PyTorch tensors and NumPy arrays dynamically, including batched inputs.
"""

from __future__ import annotations

import logging
from typing import Any, Union

import numpy as np

# PyTorch import handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


def is_tensor(x: Any) -> bool:
    """Returns True if the object is a PyTorch Tensor."""
    return TORCH_AVAILABLE and isinstance(x, torch.Tensor)


def get_position(state: Any) -> Any:
    """
    Extracts the position vector [x, y, z] from a state vector [x, y, z, vx, vy, vz].

    Supports flat or batched arrays/tensors.
    """
    return state[..., :3]


def get_velocity(state: Any) -> Any:
    """
    Extracts the velocity vector [vx, vy, vz] from a state vector [x, y, z, vx, vy, vz].

    Supports flat or batched arrays/tensors.
    """
    return state[..., 3:6]


def compute_norm(vector: Any, axis: int = -1, keepdims: bool = True) -> Any:
    """
    Computes the vector magnitude (norm) with gradient safety at 0.

    Supports both NumPy arrays and PyTorch tensors.
    """
    if is_tensor(vector):
        # Add a small epsilon before sqrt to make it differentiable at 0.
        # This is critical for PINN backpropagation because freshly initialized networks
        # predict states close to 0, which makes standard torch.linalg.norm gradients NaN.
        eps = 1e-16
        return torch.sqrt(torch.sum(vector ** 2, dim=axis, keepdim=keepdims) + eps)
    else:
        return np.linalg.norm(vector, axis=axis, keepdims=keepdims)


def compute_unit_vector(vector: Any) -> Any:
    """
    Computes the unit vector (direction) of a vector.

    Avoids division by zero by adding a small epsilon to the norm.
    """
    norm = compute_norm(vector)
    if is_tensor(vector):
        # Larger epsilon for PyTorch to prevent float32 gradient underflow/overflow
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

    Supports both NumPy arrays and PyTorch tensors.
    """
    if is_tensor(vec1):
        return torch.sum(vec1 * vec2, dim=axis, keepdim=keepdims)
    else:
        return np.sum(vec1 * vec2, axis=axis, keepdims=keepdims)
