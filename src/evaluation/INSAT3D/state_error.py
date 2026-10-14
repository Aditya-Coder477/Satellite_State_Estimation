"""
src/evaluation/INSAT3D/state_error.py
======================================
Computes Euclidean error stats for position and velocity vectors.
"""

from __future__ import annotations

import logging
from typing import Dict
import numpy as np

from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class StateErrorCalculator:
    """
    Computes statistical position and velocity norms errors over prediction horizons.
    """

    @staticmethod
    def compute_errors(
        pred_delta: np.ndarray,
        true_delta: np.ndarray,
    ) -> Dict[str, float]:
        """
        Computes Euclidean errors between predictions and targets.
        """
        pred_pos = pred_delta[:, :3]
        true_pos = true_delta[:, :3]
        pred_vel = pred_delta[:, 3:6]
        true_vel = true_delta[:, 3:6]

        # 1. Euclidean Position Errors
        pos_errors = np.linalg.norm(true_pos - pred_pos, axis=1)
        mean_pos_err = float(np.mean(pos_errors))
        median_pos_err = float(np.median(pos_errors))
        max_pos_err = float(np.max(pos_errors))
        p95_pos_err = float(np.percentile(pos_errors, 95))

        # 2. Euclidean Velocity Errors
        vel_errors = np.linalg.norm(true_vel - pred_vel, axis=1)
        mean_vel_err = float(np.mean(vel_errors))
        max_vel_err = float(np.max(vel_errors))
        std_vel_err = float(np.std(vel_errors))

        errors = {
            "mean_position_error": mean_pos_err,
            "median_position_error": median_pos_err,
            "max_position_error": max_pos_err,
            "p95_position_error": p95_pos_err,
            "mean_velocity_error": mean_vel_err,
            "max_velocity_error": max_vel_err,
            "std_velocity_error": std_vel_err,
        }

        log.info(
            "State Errors — Mean Pos Error: %.4f km (Max: %.4f) | Mean Vel Error: %.6f km/s",
            mean_pos_err,
            max_pos_err,
            mean_vel_err,
        )

        return errors
