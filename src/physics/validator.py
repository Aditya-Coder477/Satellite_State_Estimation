"""
src/physics/validator.py
========================
Provides validation routines for Phase 8 (Physics Engine) outputs.

Verifies tensor shapes, checks for NaNs/Infs, validates gravity and transition scales,
and logs discrepancies.
"""

from __future__ import annotations

import logging
from typing import Any, List, Tuple

import numpy as np

# PyTorch import handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.physics.coordinate_dynamics import compute_norm, get_position, is_tensor
from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class PhysicsValidator:
    """
    Validates physical parameters, tensor dimensions, and numerical correctness.
    """

    @classmethod
    def validate_physics_outputs(
        cls,
        state: Any,
        gravity_acc: Any,
        expected_transition: Any,
        residual_acc: Any,
    ) -> Tuple[bool, List[str]]:
        """
        Runs comprehensive validation checks on physics outputs.

        Parameters
        ----------
        state : Any
            Cartesian state tensor/array of shape (Batch, 6).
        gravity_acc : Any
            Gravitational acceleration vector of shape (Batch, 3).
        expected_transition : Any
            Numerical physics state transition (expected ΔState) of shape (Batch, 6).
        residual_acc : Any
            Residual acceleration vector of shape (Batch, 3).

        Returns
        -------
        Tuple[bool, List[str]]
            1. Success flag (True if all checks pass, False otherwise).
            2. List of error messages.
        """
        errors: List[str] = []
        is_tensor_input = is_tensor(state)

        # 1. Dimension and Shape Checks
        cls._check_shapes(state, "state", 6, errors)
        cls._check_shapes(gravity_acc, "gravity_acc", 3, errors)
        cls._check_shapes(expected_transition, "expected_transition", 6, errors)
        cls._check_shapes(residual_acc, "residual_acc", 3, errors)

        if errors:
            log.error("Dimension validation failed. Skipping numerical checks.")
            return False, errors

        # 2. Batch Consistency Check
        batch_size = state.shape[0]
        for name, tensor in [
            ("gravity_acc", gravity_acc),
            ("expected_transition", expected_transition),
            ("residual_acc", residual_acc),
        ]:
            if tensor.shape[0] != batch_size:
                errors.append(
                    f"Batch size inconsistency: '{name}' has batch size {tensor.shape[0]}, "
                    f"expected {batch_size}."
                )

        # 3. Finiteness (NaN/Inf) Checks
        cls._check_finite(state, "state", errors)
        cls._check_finite(gravity_acc, "gravity_acc", errors)
        cls._check_finite(expected_transition, "expected_transition", errors)
        cls._check_finite(residual_acc, "residual_acc", errors)

        # 4. Realistic Gravity Magnitude Check
        g_norm = compute_norm(gravity_acc, keepdims=False)
        if is_tensor_input:
            min_g = torch.min(g_norm).item()
            max_g = torch.max(g_norm).item()
        else:
            min_g = float(np.min(g_norm))
            max_g = float(np.max(g_norm))

        g_lower_bound = 1e-5
        g_upper_bound = 1.5e-2

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

        # 5. Sub-surface Distance Check
        pos = get_position(state)
        r_norm = compute_norm(pos, keepdims=False)
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
            errors.append(f"NaN values detected in '{name}'.")
        if has_inf:
            errors.append(f"Infinite values detected in '{name}'.")
