"""
src/evaluation/INSAT3D/validator.py
===================================
Validates evaluation data, model prediction shapes, values, and orbit consistency.
"""

from __future__ import annotations

import logging
from typing import Any, Dict
import numpy as np

from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class EvaluationValidator:
    """
    Validates shapes, missing values, infinite values, and mathematical consistency.
    """

    @staticmethod
    def validate_inputs_outputs(inputs: np.ndarray, targets: np.ndarray) -> bool:
        """Checks shapes, NaNs, and infinite values in raw data splits."""
        if inputs is None or targets is None:
            log.error("Inputs or targets are None.")
            return False

        # Check NaNs
        if np.isnan(inputs).any():
            log.error("Inputs contain NaN values.")
            return False
        if np.isnan(targets).any():
            log.error("Targets contain NaN values.")
            return False

        # Check Infs
        if np.isinf(inputs).any():
            log.error("Inputs contain Infinite values.")
            return False
        if np.isinf(targets).any():
            log.error("Targets contain Infinite values.")
            return False

        log.info("Inputs and targets data validation passed (No NaNs/Infs).")
        return True

    @staticmethod
    def validate_predictions(
        pred_delta: np.ndarray,
        true_delta: np.ndarray,
        pred_res: np.ndarray,
        true_res: np.ndarray,
    ) -> bool:
        """Validates prediction shapes and check for anomalies."""
        if pred_delta.shape != true_delta.shape:
            log.error("Shape mismatch: predicted delta %s vs true delta %s", pred_delta.shape, true_delta.shape)
            return False
        if pred_res.shape != true_res.shape:
            log.error("Shape mismatch: predicted residual %s vs true residual %s", pred_res.shape, true_res.shape)
            return False

        if np.isnan(pred_delta).any() or np.isnan(pred_res).any():
            log.error("Model predictions contain NaN values.")
            return False

        log.info("Model predictions shapes and values validated successfully.")
        return True

    @staticmethod
    def validate_metrics(metrics: Dict[str, float]) -> bool:
        """Checks if evaluated metrics are within physically meaningful ranges."""
        # R2 score should be <= 1.0
        for k, v in metrics.items():
            if "r2" in k.lower() and v > 1.0:
                log.error("Invalid R² score value: %s = %.4f (must be <= 1.0)", k, v)
                return False
            # RMSE/MAE must be non-negative
            if ("rmse" in k.lower() or "mae" in k.lower()) and v < 0.0:
                log.error("Negative error metric value: %s = %.4f", k, v)
                return False

        log.info("Evaluation metrics ranges validated successfully.")
        return True

    @staticmethod
    def validate_orbit_consistency(
        true_states: np.ndarray,
        pred_states: np.ndarray,
        max_allowed_deviation: float = 1e5,
    ) -> bool:
        """Checks that reconstructed states are physically realistic."""
        # Ensure predicted state coordinates do not blow up to infinity
        pos_err = np.linalg.norm(true_states[:, :3] - pred_states[:, :3], axis=1)
        mean_err = np.mean(pos_err)
        
        if mean_err > max_allowed_deviation:
            log.warning("High average orbit reconstruction error: %.2f km. Verify coordinates scaling.", mean_err)
            return False
            
        log.info("Orbit reconstruction consistency validated successfully.")
        return True
