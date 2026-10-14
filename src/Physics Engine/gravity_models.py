"""
src/physics/gravity_models.py
=============================
Defines Earth gravity model strategies for Phase 7 (Physics Engine).

Uses the Strategy Pattern to decouple gravitational acceleration math from
general orbital dynamics. Provides:
- GravityModel (Abstract Base Class)
- TwoBodyGravity (Classical point-mass gravity)
- J2Gravity (Two-body + Earth oblateness perturbation)
- J4Gravity (stub)
- HighOrderGravity (stub)
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

        Parameters
        ----------
        position : Any
            Position vector of shape (..., 3) (km). Supports NumPy and PyTorch.

        Returns
        -------
        Any
            Acceleration vector of shape (..., 3) (km/s^2).
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
        # a = -mu * r / |r|^3
        r_norm = compute_norm(position)
        
        # Add epsilon to prevent division by zero and float32 gradient overflow/underflow (1e-3 = 1 meter)
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

    Computes analytical J2 acceleration perturbation in a geocentric equatorial frame.
    """

    def compute_acceleration(self, position: Any) -> Any:
        # Get base two-body acceleration
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

        # Common J2 factor: 1.5 * J2 * mu * Re^2 / r^5
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
    """Stub representing J4 oblateness perturbation."""

    def compute_acceleration(self, position: Any) -> Any:
        log.warning("J4Gravity model is not yet implemented. Falling back to J2Gravity.")
        # Instantiate and delegate to J2Gravity
        return J2Gravity().compute_acceleration(position)

    @property
    def name(self) -> str:
        return "j4"


class HighOrderGravity(GravityModel):
    """Stub representing spherical harmonics models (e.g. EGM96)."""

    def compute_acceleration(self, position: Any) -> Any:
        log.warning("HighOrderGravity model is not yet implemented. Falling back to J2Gravity.")
        return J2Gravity().compute_acceleration(position)

    @property
    def name(self) -> str:
        return "high_order"


class GravityModelFactory:
    """Factory to instantiate the configured Earth gravity model strategy."""

    _MODELS: Dict[str, type[GravityModel]] = {
        "two_body": TwoBodyGravity,
        "j2": J2Gravity,
        "j4": J4Gravity,
        "high_order": HighOrderGravity,
    }

    @classmethod
    def create(cls, name: str) -> GravityModel:
        """
        Creates an instance of the requested gravity model.

        Parameters
        ----------
        name : str
            Name of the model strategy (e.g. "two_body", "j2").

        Returns
        -------
        GravityModel
            Instantiated gravity model.

        Raises
        ------
        ValueError
            If the requested model name is unknown.
        """
        model_class = cls._MODELS.get(name.lower())
        if not model_class:
            raise ValueError(
                f"Unknown gravity model '{name}'. Available: {list(cls._MODELS.keys())}"
            )
        return model_class()
