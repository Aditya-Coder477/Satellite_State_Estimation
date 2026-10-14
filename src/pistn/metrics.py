"""
src/pistn/metrics.py
====================
Calculates regression metrics (RMSE, MAE, R2) for Phase 9 (PI-STN) evaluation.
"""

from __future__ import annotations

import logging
from typing import Dict, Union

import numpy as np

# PyTorch import handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class MetricsCalculator:
    """
    Computes statistical regression metrics (RMSE, MAE, R2) for PI-STN model evaluations.
    """

    @staticmethod
    def compute_all(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
        """
        Computes RMSE, MAE, and R2.

        Parameters
        ----------
        pred : np.ndarray
            Predicted states/transitions.
        true : np.ndarray
            Target states/transitions.

        Returns
        -------
        Dict[str, float]
            Dictionary of metrics.
        """
        pred_flat = pred.flatten()
        true_flat = true.flatten()

        # MAE
        mae = float(np.mean(np.abs(pred_flat - true_flat)))

        # RMSE
        rmse = float(np.sqrt(np.mean((pred_flat - true_flat) ** 2)))

        # R2
        ss_res = np.sum((true_flat - pred_flat) ** 2)
        ss_tot = np.sum((true_flat - np.mean(true_flat)) ** 2)
        if ss_tot == 0.0:
            r2 = 1.0 if ss_res == 0.0 else 0.0
        else:
            r2 = float(1.0 - (ss_res / ss_tot))

        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        }
