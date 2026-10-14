"""
src/evaluation/INSAT3D/metrics_calculator.py
============================================
Computes statistical evaluation metrics (RMSE, MAE, R² score) for Phase 10.
"""

from __future__ import annotations

import logging
from typing import Dict
import numpy as np

from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class PerformanceMetricsCalculator:
    """
    Computes statistical metrics (RMSE, MAE, R2) for predicted transitions.
    """

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Computes Root Mean Squared Error."""
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Computes Mean Absolute Error."""
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Computes Coefficient of Determination (R² Score)."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2)
        if ss_tot == 0.0:
            return 0.0
        return float(1.0 - (ss_res / ss_tot))

    @classmethod
    def compute_all_metrics(
        self,
        pred_delta: np.ndarray,
        true_delta: np.ndarray,
    ) -> Dict[str, float]:
        """
        Computes RMSE, MAE, and R² for Position, Velocity, and Overall transitions.
        """
        # Slices
        pred_pos = pred_delta[:, :3]
        true_pos = true_delta[:, :3]
        pred_vel = pred_delta[:, 3:6]
        true_vel = true_delta[:, 3:6]

        rmse_pos = self.rmse(true_pos, pred_pos)
        rmse_vel = self.rmse(true_vel, pred_vel)
        rmse_all = self.rmse(true_delta, pred_delta)

        mae_pos = self.mae(true_pos, pred_pos)
        mae_vel = self.mae(true_vel, pred_vel)
        mae_all = self.mae(true_delta, pred_delta)

        r2_pos = self.r2_score(true_pos, pred_pos)
        r2_vel = self.r2_score(true_vel, pred_vel)
        r2_all = self.r2_score(true_delta, pred_delta)

        metrics = {
            "rmse_position": rmse_pos,
            "rmse_velocity": rmse_vel,
            "rmse_overall": rmse_all,
            "mae_position": mae_pos,
            "mae_velocity": mae_vel,
            "mae_overall": mae_all,
            "r2_position": r2_pos,
            "r2_velocity": r2_vel,
            "r2_overall": r2_all,
        }

        log.info(
            "Transitions metrics calculated — RMSE Pos: %.4e km | Vel: %.4e km/s | R² Pos: %.4f | Vel: %.4f",
            rmse_pos,
            rmse_vel,
            r2_pos,
            r2_vel,
        )

        return metrics
