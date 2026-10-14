"""
src/evaluation/metrics_calculator.py
=====================================
Regression and physics consistency metrics for Phase 9 evaluation.

Computes:
- Per-axis and total RMSE / MAE for position and velocity
- Max absolute error across the prediction horizon
- Cumulative drift (position error growth over time)
- Physics residual mean and max (via PhysicsEngine)

All computations are pure NumPy — no PyTorch dependency.
The physics residual section only runs if PyTorch is available.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)

# Optional PyTorch for physics residual computation
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class MetricsCalculator:
    """
    Computes evaluation metrics for the 2-day future prediction.

    Parameters
    ----------
    physics_engine : optional
        A ``PhysicsEngine`` instance (Phase 7). Required for physics
        consistency metrics. If ``None``, physics residuals are skipped.
    """

    def __init__(self, physics_engine: Optional[Any] = None) -> None:
        self._physics_engine = physics_engine
        log.info("MetricsCalculator initialised. Physics engine: %s",
                 "available" if physics_engine is not None else "not provided")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_all(
        self,
        times: np.ndarray,
        predicted: np.ndarray,
        truth: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Compute the full metrics suite.

        Parameters
        ----------
        times : np.ndarray, shape (N,)
            Elapsed seconds from epoch.
        predicted : np.ndarray, shape (N, 6)
            PINN-predicted states [x, y, z, vx, vy, vz].
        truth : np.ndarray, shape (N, 6)
            Ground-truth states [x, y, z, vx, vy, vz].

        Returns
        -------
        Dict[str, Any]
            Nested metrics dictionary.
        """
        pos_pred = predicted[:, :3]
        vel_pred = predicted[:, 3:]
        pos_true = truth[:, :3]
        vel_true = truth[:, 3:]

        pos_error = pos_pred - pos_true       # (N, 3)
        vel_error = vel_pred - vel_true       # (N, 3)
        pos_norm  = np.linalg.norm(pos_error, axis=1)   # (N,)
        vel_norm  = np.linalg.norm(vel_error, axis=1)   # (N,)

        metrics: Dict[str, Any] = {}

        # ── Position metrics ──────────────────────────────────────────
        metrics["position"] = {
            "rmse_km":              float(self._rmse(pos_norm)),
            "mae_km":               float(self._mae(pos_norm)),
            "max_error_km":         float(np.max(pos_norm)),
            "final_error_km":       float(pos_norm[-1]),
            "rmse_x_km":            float(self._rmse(pos_error[:, 0])),
            "rmse_y_km":            float(self._rmse(pos_error[:, 1])),
            "rmse_z_km":            float(self._rmse(pos_error[:, 2])),
            "mae_x_km":             float(self._mae(pos_error[:, 0])),
            "mae_y_km":             float(self._mae(pos_error[:, 1])),
            "mae_z_km":             float(self._mae(pos_error[:, 2])),
        }

        # ── Velocity metrics ──────────────────────────────────────────
        metrics["velocity"] = {
            "rmse_kms":             float(self._rmse(vel_norm)),
            "mae_kms":              float(self._mae(vel_norm)),
            "max_error_kms":        float(np.max(vel_norm)),
            "final_error_kms":      float(vel_norm[-1]),
            "rmse_vx_kms":          float(self._rmse(vel_error[:, 0])),
            "rmse_vy_kms":          float(self._rmse(vel_error[:, 1])),
            "rmse_vz_kms":          float(self._rmse(vel_error[:, 2])),
            "mae_vx_kms":           float(self._mae(vel_error[:, 0])),
            "mae_vy_kms":           float(self._mae(vel_error[:, 1])),
            "mae_vz_kms":           float(self._mae(vel_error[:, 2])),
        }

        # ── Trajectory drift (error evolution) ───────────────────────
        trapz_func = getattr(np, "trapezoid", getattr(np, "trapz", None))
        metrics["trajectory"] = {
            "cumulative_position_drift_km":  float(trapz_func(pos_norm, times)),
            "mean_position_error_km":        float(np.mean(pos_norm)),
            "std_position_error_km":         float(np.std(pos_norm)),
            "error_growth_rate_kmps":        self._linear_drift_rate(times, pos_norm),
        }

        # ── Physics residuals ─────────────────────────────────────────
        metrics["physics"] = self._compute_physics_residuals(times, predicted)

        log.info(
            "Metrics: position RMSE=%.4f km | velocity RMSE=%.6f km/s",
            metrics["position"]["rmse_km"],
            metrics["velocity"]["rmse_kms"],
        )
        return metrics

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rmse(errors: np.ndarray) -> float:
        return float(np.sqrt(np.mean(errors ** 2)))

    @staticmethod
    def _mae(errors: np.ndarray) -> float:
        return float(np.mean(np.abs(errors)))

    @staticmethod
    def _linear_drift_rate(times: np.ndarray, errors: np.ndarray) -> float:
        """
        Fits a linear trend to the error curve and returns the slope (km/s).
        A positive slope indicates error is growing over the prediction horizon.
        """
        if len(times) < 2:
            return 0.0
        t_norm = times - times[0]
        coeffs = np.polyfit(t_norm, errors, 1)
        return float(coeffs[0])  # slope in km / second

    def _compute_physics_residuals(
        self,
        times: np.ndarray,
        predicted: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Evaluates physics residuals on the predicted trajectory.

        Uses finite differences to approximate time derivatives of the
        predicted state, then passes them to the PhysicsEngine.

        Returns a dictionary with residual statistics (all float).
        Falls back to NaN if PyTorch or PhysicsEngine is unavailable.
        """
        nan_result = {
            "physics_residual_mean": float("nan"),
            "physics_residual_max":  float("nan"),
            "physics_residual_std":  float("nan"),
            "physics_available":     False,
        }

        if self._physics_engine is None:
            log.warning("PhysicsEngine not provided — physics residuals skipped.")
            return nan_result

        if not TORCH_AVAILABLE:
            log.warning("PyTorch not available — physics residuals skipped.")
            return nan_result

        try:
            # Use finite-differences to estimate dS/dt
            dt = np.diff(times)                           # (N-1,)
            dS = np.diff(predicted, axis=0)               # (N-1, 6)
            # Mid-point states
            state_mid = 0.5 * (predicted[:-1] + predicted[1:])  # (N-1, 6)
            deriv_fd = dS / dt[:, None]                   # (N-1, 6)

            # Filter out non-finite states/derivatives to avoid triggering RuntimeError in residuals
            finite_mask = np.isfinite(state_mid).all(axis=-1) & np.isfinite(deriv_fd).all(axis=-1)
            if not finite_mask.any():
                log.warning("No finite predicted states found for physics residual evaluation.")
                return nan_result

            state_mid = state_mid[finite_mask]
            deriv_fd = deriv_fd[finite_mask]

            state_t  = torch.tensor(state_mid,  dtype=torch.float32)
            deriv_t  = torch.tensor(deriv_fd,   dtype=torch.float32)

            residuals = self._physics_engine.compute_residuals(state_t, deriv_t)
            residual_norms = torch.norm(residuals, dim=-1).detach().cpu().numpy()

            # Ensure we only calculate stats on finite residual norms
            valid_norms = residual_norms[np.isfinite(residual_norms)]
            if len(valid_norms) == 0:
                return nan_result

            return {
                "physics_residual_mean": float(np.mean(valid_norms)),
                "physics_residual_max":  float(np.max(valid_norms)),
                "physics_residual_std":  float(np.std(valid_norms)),
                "physics_available":     True,
            }

        except Exception as exc:
            log.warning("Physics residual computation failed: %s", exc)
            return nan_result
