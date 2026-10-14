"""
src/physics/validator.py
========================
Provides validation routines for Phase 7 (Physics Engine) outputs.

Verifies tensor shapes, checks for NaNs/Infs, validates that gravity and
acceleration magnitudes are physically realistic, and checks batch consistency.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

# PyTorch import handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.physics.coordinate_dynamics import compute_norm, get_position, get_velocity, is_tensor
from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class PhysicsValidator:
    """
    Validates physical parameters, tensor dimensions, and numerical correctness.
    """

    @classmethod
    def validate_physics_outputs(
        cls,
        state_pred: Any,
        state_deriv_pred: Any,
        residuals: Any,
        gravity_acc: Any,
    ) -> Tuple[bool, List[str]]:
        """
        Runs comprehensive validation checks on physics inputs and outputs.

        Parameters
        ----------
        state_pred : Any
            Predicted state tensor of shape (Batch, 6).
        state_deriv_pred : Any
            Predicted derivative tensor of shape (Batch, 6).
        residuals : Any
            Calculated physics residual tensor of shape (Batch, 6).
        gravity_acc : Any
            Calculated gravity acceleration tensor of shape (Batch, 3).

        Returns
        -------
        Tuple[bool, List[str]]
            1. Success flag (True if all checks pass, False otherwise).
            2. List of error messages.
        """
        errors: List[str] = []
        is_tensor_input = is_tensor(state_pred)

        # 1. Dimension and Shape Checks
        cls._check_shapes(state_pred, "state_pred", 6, errors)
        cls._check_shapes(state_deriv_pred, "state_deriv_pred", 6, errors)
        cls._check_shapes(residuals, "residuals", 6, errors)
        cls._check_shapes(gravity_acc, "gravity_acc", 3, errors)

        if errors:
            log.error("Dimension validation failed. Skipping numerical checks.")
            return False, errors

        # 2. Batch Consistency Check
        batch_size = state_pred.shape[0]
        for name, tensor in [
            ("state_deriv_pred", state_deriv_pred),
            ("residuals", residuals),
            ("gravity_acc", gravity_acc),
        ]:
            if tensor.shape[0] != batch_size:
                errors.append(
                    f"Batch size inconsistency: '{name}' has batch size {tensor.shape[0]}, "
                    f"expected {batch_size}."
                )

        # 3. Finiteness (NaN/Inf) Checks
        cls._check_finite(state_pred, "state_pred", errors)
        cls._check_finite(state_deriv_pred, "state_deriv_pred", errors)
        cls._check_finite(residuals, "residuals", errors)
        cls._check_finite(gravity_acc, "gravity_acc", errors)

        # 4. Realistic Gravity Magnitude Check
        # Earth surface gravity: ~9.8e-3 km/s^2. GEO gravity: ~2.2e-4 km/s^2.
        # We expect satellite gravity acceleration to lie in [1e-5, 1e-2] km/s^2.
        g_norm = compute_norm(gravity_acc)
        
        if is_tensor_input:
            min_g = torch.min(g_norm).item()
            max_g = torch.max(g_norm).item()
        else:
            min_g = float(np.min(g_norm))
            max_g = float(np.max(g_norm))

        # Tolerances
        g_lower_bound = 1e-5  # km/s^2 (corresponds to radius ~200,000 km)
        g_upper_bound = 1.5e-2  # km/s^2 (corresponds to radius ~5,150 km, sub-surface)

        if min_g < g_lower_bound:
            errors.append(
                f"Physically unrealistic low gravity: min gravity magnitude is {min_g:.6e} km/s^2 "
                f"(expected >= {g_lower_bound:.1e} km/s^2)."
            )
        if max_g > g_upper_bound:
            errors.append(
                f"Physically unrealistic high gravity: max gravity magnitude is {max_g:.6e} km/s^2 "
                f"(expected <= {g_upper_bound:.1e} km/s^2, satellite may have crashed)."
            )

        # 5. Position/Velocity Radius and Bounds Check
        pos = get_position(state_pred)
        r_norm = compute_norm(pos)
        
        if is_tensor_input:
            min_r = torch.min(r_norm).item()
        else:
            min_r = float(np.min(r_norm))

        earth_radius = 6378.137
        if min_r < earth_radius:
            errors.append(
                f"Satellite orbit is sub-surface: minimum distance from Earth center is {min_r:.3f} km "
                f"(Earth radius is {earth_radius:.3f} km)."
            )

        is_valid = len(errors) == 0
        if is_valid:
            log.info("Physics output validation passed successfully.")
        else:
            log.error("Physics validation failed with %d errors.", len(errors))
            for i, err in enumerate(errors, 1):
                log.error("  Error %d: %s", i, err)

        return is_valid, errors

    @staticmethod
    def _check_shapes(tensor: Any, name: str, expected_last_dim: int, errors: List[str]) -> None:
        """Verifies that the tensor/array is 2D and has the expected last dimension."""
        if len(tensor.shape) != 2:
            errors.append(
                f"Shape validation failed: '{name}' must be a 2D array/tensor, got shape {tensor.shape}."
            )
            return

        if tensor.shape[1] != expected_last_dim:
            errors.append(
                f"Shape validation failed: '{name}' must have last dimension = {expected_last_dim}, "
                f"got shape {tensor.shape}."
            )

    @staticmethod
    def _check_finite(tensor: Any, name: str, errors: List[str]) -> None:
        """Checks if there are NaNs or Infs present in the tensor."""
        if is_tensor(tensor):
            has_nan = torch.isnan(tensor).any().item()
            has_inf = torch.isinf(tensor).any().item()
        else:
            has_nan = np.isnan(tensor).any()
            has_inf = np.isinf(tensor).any()

        if has_nan:
            errors.append(f"NaN values detected in tensor '{name}'.")
        if has_inf:
            errors.append(f"Infinite values detected in tensor '{name}'.")
class PhysicsValidatorError(ValueError):
    """Exception raised when physical predictions violate basic physical parameters."""
    pass
