"""
src/physics/gravity_models.py
=============================
Defines Earth gravity model strategies for Phase 8 (Physics Engine).

Uses the Strategy Pattern to decouple gravitational acceleration math from
general orbital dynamics. Supports both PyTorch and NumPy.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np

# PyTorch import handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.physics import constants
from src.physics.coordinate_dynamics import compute_norm, is_tensor
from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class GravityModel(ABC):
    """Abstract Base Class representing a gravitational acceleration model."""

    @abstractmethod
    def compute_acceleration(self, position: Any) -> Any:
        """
        Computes gravitational acceleration vector [ax, ay, az] at given position(s).
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the gravity model."""
        pass


class TwoBodyGravity(GravityModel):
    """Classical two-body gravity model representing Earth as a point mass."""

    def compute_acceleration(self, position: Any) -> Any:
        r_norm = compute_norm(position)
        eps = 1e-3
        if is_tensor(position):
            r_norm_safe = torch.clamp(r_norm, min=eps)
            return -constants.MU * position / (r_norm_safe ** 3)
        else:
            r_norm_safe = np.clip(r_norm, eps, None)
            return -constants.MU * position / (r_norm_safe ** 3)

    @property
    def name(self) -> str:
        return "two_body"


class J2Gravity(GravityModel):
    """
    Two-body gravity model perturbed by Earth J2 oblateness effect.
    """

    def compute_acceleration(self, position: Any) -> Any:
        r_norm = compute_norm(position)
        eps = 1e-3
        
        # Check type
        if is_tensor(position):
            r_norm_safe = torch.clamp(r_norm, min=eps)
            a_2body = -constants.MU * position / (r_norm_safe ** 3)
        else:
            r_norm_safe = np.clip(r_norm, eps, None)
            a_2body = -constants.MU * position / (r_norm_safe ** 3)

        # Extract coordinates
        x = position[..., 0:1]
        y = position[..., 1:2]
        z = position[..., 2:3]

        # J2 factor: 1.5 * J2 * mu * Re^2 / r^5
        factor = 1.5 * constants.J2 * constants.MU * (constants.R_E ** 2) / (r_norm_safe ** 5)
        z2_r2 = (z / r_norm_safe) ** 2

        # J2 acceleration components
        if is_tensor(position):
            a_j2_x = -factor * x * (1.0 - 5.0 * z2_r2)
            a_j2_y = -factor * y * (1.0 - 5.0 * z2_r2)
            a_j2_z = -factor * z * (3.0 - 5.0 * z2_r2)
            a_j2 = torch.cat([a_j2_x, a_j2_y, a_j2_z], dim=-1)
        else:
            a_j2_x = -factor * x * (1.0 - 5.0 * z2_r2)
            a_j2_y = -factor * y * (1.0 - 5.0 * z2_r2)
            a_j2_z = -factor * z * (3.0 - 5.0 * z2_r2)
            a_j2 = np.concatenate([a_j2_x, a_j2_y, a_j2_z], axis=-1)

        return a_2body + a_j2

    @property
    def name(self) -> str:
        return "j2"


class J4Gravity(GravityModel):
    """J4 oblateness perturbation model (delegating to J2 for J4 dynamics)."""

    def compute_acceleration(self, position: Any) -> Any:
        log.warning("J4Gravity model is not yet implemented. Falling back to J2Gravity.")
        return J2Gravity().compute_acceleration(position)

    @property
    def name(self) -> str:
        return "j4"


class SRPGravity(GravityModel):
    """Stub representing Solar Radiation Pressure model."""

    def compute_acceleration(self, position: Any) -> Any:
        # Future ready stub
        if is_tensor(position):
            return torch.zeros_like(position)
        else:
            return np.zeros_like(position)

    @property
    def name(self) -> str:
        return "srp"


class ManeuverGravity(GravityModel):
    """Stub representing Maneuver / Active Thrust model."""

    def compute_acceleration(self, position: Any) -> Any:
        # Future ready stub
        if is_tensor(position):
            return torch.zeros_like(position)
        else:
            return np.zeros_like(position)

    @property
    def name(self) -> str:
        return "maneuver"


class GravityModelFactory:
    """Factory to instantiate the configured Earth gravity model strategy."""

    _MODELS: Dict[str, type[GravityModel]] = {
        "two_body": TwoBodyGravity,
        "j2": J2Gravity,
        "j4": J4Gravity,
        "srp": SRPGravity,
        "maneuver": ManeuverGravity,
    }

    @classmethod
    def create(cls, name: str) -> GravityModel:
        """
        Creates an instance of the requested gravity model.
        """
        model_class = cls._MODELS.get(name.lower())
        if not model_class:
            raise ValueError(
                f"Unknown gravity model '{name}'. Available: {list(cls._MODELS.keys())}"
            )
        return model_class()
