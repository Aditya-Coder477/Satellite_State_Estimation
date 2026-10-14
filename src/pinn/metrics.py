"""
src/pinn/metrics.py
===================
Defines evaluation metrics for Phase 8 (PINN Framework).

Computes RMSE, MAE, MAPE, physics residual norms, and constraint errors to
assess satellite state prediction accuracy.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


def compute_metrics(
    state_pred: np.ndarray,
    state_true: np.ndarray,
    residuals: np.ndarray | None = None,
    constraint_error: float | None = None,
) -> Dict[str, float]:
    """
    Computes statistical and physical trajectory error metrics.

    Parameters
    ----------
    state_pred : np.ndarray
        Predicted states [x,y,z,vx,vy,vz] of shape (Batch, 6).
    state_true : np.ndarray
        True GCRS/ITRS states of shape (Batch, 6).
    residuals : np.ndarray, optional
        Calculated physics residuals of shape (Batch, 6), by default None.
    constraint_error : float, optional
        Summed physical constraint penalties, by default None.

    Returns
    -------
    Dict[str, float]
        A dictionary containing:
        - position_rmse (km)
        - velocity_rmse (km/s)
        - position_mae (km)
        - velocity_mae (km/s)
        - position_mape (%)
        - velocity_mape (%)
        - physics_residual_norm (km/s, km/s^2)
        - constraint_error
    """
    # Parse position and velocity splits
    pos_pred = state_pred[:, :3]
    pos_true = state_true[:, :3]
    vel_pred = state_pred[:, 3:6]
    vel_true = state_true[:, 3:6]

    # RMSE
    pos_rmse = float(np.sqrt(np.mean((pos_pred - pos_true) ** 2)))
    vel_rmse = float(np.sqrt(np.mean((vel_pred - vel_true) ** 2)))

    # MAE
    pos_mae = float(np.mean(np.abs(pos_pred - pos_true)))
    vel_mae = float(np.mean(np.abs(vel_pred - vel_true)))

    # MAPE (Mean Absolute Percentage Error) - add eps to prevent div by zero
    eps = 1e-12
    pos_true_safe = np.where(np.abs(pos_true) < eps, eps, pos_true)
    vel_true_safe = np.where(np.abs(vel_true) < eps, eps, vel_true)

    pos_mape = float(np.mean(np.abs((pos_pred - pos_true) / pos_true_safe)) * 100)
    vel_mape = float(np.mean(np.abs((vel_pred - vel_true) / vel_true_safe)) * 100)

    # Physics residuals L2 norm
    res_norm = 0.0
    if residuals is not None:
        res_norm = float(np.mean(np.linalg.norm(residuals, axis=-1)))

    metrics = {
        "position_rmse_km": pos_rmse,
        "velocity_rmse_kms": vel_rmse,
        "position_mae_km": pos_mae,
        "velocity_mae_kms": vel_mae,
        "position_mape_pct": pos_mape,
        "velocity_mape_pct": vel_mape,
        "physics_residual_norm": res_norm,
        "constraint_error": constraint_error if constraint_error is not None else 0.0,
    }

    return metrics
