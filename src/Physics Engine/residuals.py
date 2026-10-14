"""
src/physics/residuals.py
========================
Computes physics residuals for Phase 7 (Physics Engine).

The class compares the automatic differentiation derivatives of the predicted
state with the analytical orbital dynamics equations, producing:
- Position residuals: d(r_pred)/dt - v_pred
- Velocity residuals: d(v_pred)/dt - a_grav(r_pred)

Bug Fixes (Phase 9 — Physics Residual NaN elimination):
  * Added ``torch.isfinite`` validation immediately after computing residuals.
  * Added per-element NaN/Inf location logging (batch index, dimension, value).
  * Added input validation for state_pred and state_deriv_pred before computing.
"""

from __future__ import annotations

import logging
from typing import Any

from src.physics.gravity_models import GravityModel
from src.physics.orbital_equations import compute_state_derivative
from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class PhysicsResidualCalculator:
    """
    Computes governing physics residuals for satellite trajectories.

    Residual = StateDerivativePredicted - StateDerivativeAnalytical

    The residual encodes the physics violation:
      • Position part:  dr/dt_predicted  - v_predicted    (should be ≈ 0)
      • Velocity part:  dv/dt_predicted  - a_gravity(r)   (should be ≈ 0)
    """

    def __init__(self, gravity_model: GravityModel) -> None:
        self.gravity_model = gravity_model

    def compute_residuals(self, state_pred: Any, state_deriv_pred: Any) -> Any:
        """
        Calculates position and velocity physics residuals.

        Parameters
        ----------
        state_pred : Any
            Predicted state tensor [x, y, z, vx, vy, vz] of shape (B, 6).
            Supports PyTorch tensors and NumPy arrays.
        state_deriv_pred : Any
            Time derivative of predicted state of shape (B, 6), computed via autograd.

        Returns
        -------
        Any
            Residual tensor of shape (B, 6) representing the physics violations.

        Raises
        ------
        RuntimeError
            If ``state_pred`` or ``state_deriv_pred`` contain NaN/Inf values.
        """
        # ── Step 6: Validate inputs before computing ──────────────────────
        if TORCH_AVAILABLE and isinstance(state_pred, torch.Tensor):
            self._validate_finite_tensor("state_pred",     state_pred)
            self._validate_finite_tensor("state_deriv_pred", state_deriv_pred)

        # Compute analytical state derivatives based on orbital physics equations
        state_deriv_analytical = compute_state_derivative(state_pred, self.gravity_model)

        # Residual is the difference between predictions and equations
        residual = state_deriv_pred - state_deriv_analytical

        # ── Step 6: Validate output residual ─────────────────────────────
        if TORCH_AVAILABLE and isinstance(residual, torch.Tensor):
            if not torch.isfinite(residual).all():
                self._log_nan_locations("residual", residual)
                raise RuntimeError(
                    "Physics residual contains NaN/Inf. "
                    "This is caused by NaN inputs, broken autograd graph, or "
                    "extreme predicted state values (e.g. predicted radius ≈ 0). "
                    "Check the gradient diagnostic log for details."
                )

        return residual

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_finite_tensor(name: str, tensor: "torch.Tensor") -> None:
        """Raises RuntimeError with location info if tensor has non-finite values."""
        if torch.isfinite(tensor).all():
            return
        n_nan = torch.isnan(tensor).sum().item()
        n_inf = torch.isinf(tensor).sum().item()
        bad_mask = ~torch.isfinite(tensor)
        bad_indices = bad_mask.nonzero(as_tuple=False)
        # Log up to 10 locations
        for idx in bad_indices[:10]:
            row, col = idx[0].item(), idx[1].item()
            val = tensor[row, col].item()
            log.error(
                "NaN/Inf in '%s' | batch_idx=%d  dim=%d  value=%s",
                name, row, col, val,
            )
        raise RuntimeError(
            f"'{name}' contains {n_nan} NaN(s) and {n_inf} Inf(s). "
            "Physics residual cannot be computed on non-finite inputs. "
            "Common causes: (1) network outputs exploded, (2) t_colloc not "
            "attached to graph — use t.clone().detach().requires_grad_(True)."
        )

    @staticmethod
    def _log_nan_locations(name: str, tensor: "torch.Tensor") -> None:
        """Logs NaN/Inf locations in a tensor for debugging (Step 6)."""
        n_nan = torch.isnan(tensor).sum().item()
        n_inf = torch.isinf(tensor).sum().item()
        bad_indices = (~torch.isfinite(tensor)).nonzero(as_tuple=False)
        log.error(
            "Non-finite values in '%s': n_nan=%d  n_inf=%d  tensor_shape=%s",
            name, n_nan, n_inf, tuple(tensor.shape),
        )
        for idx in bad_indices[:10]:
            row, col = idx[0].item(), idx[1].item()
            val = tensor[row, col].item()
            log.error("  → batch_idx=%d  dim=%d  value=%s", row, col, val)
