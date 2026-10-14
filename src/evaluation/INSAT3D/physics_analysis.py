"""
src/evaluation/INSAT3D/physics_analysis.py
===========================================
Analyzes physical residuals and gravity acceleration discrepancies for Phase 10.
"""

from __future__ import annotations

import logging
from typing import Any, Dict
import numpy as np

# PyTorch import handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class PhysicsResidualAnalyzer:
    """
    Evaluates acceleration residuals against ground truth and physics integration.
    """

    def __init__(self, physics_engine: Any) -> None:
        self.physics_engine = physics_engine

    def evaluate_residuals(
        self,
        states_raw: np.ndarray,
        true_deltas_raw: np.ndarray,
        pred_res_raw: np.ndarray,
        dt_raw: np.ndarray,
    ) -> Dict[str, float]:
        """
        Computes observed minus predicted acceleration residuals.
        """
        # Convert inputs to PyTorch Tensors if physics_engine requires them
        if TORCH_AVAILABLE:
            s_t = torch.tensor(states_raw, dtype=torch.float32)
            dv_t = torch.tensor(true_deltas_raw[:, 3:6], dtype=torch.float32)
            dt_t = torch.tensor(dt_raw, dtype=torch.float32)
            
            with torch.no_grad():
                a_trans_t, a_phys_t, a_res_true_t = self.physics_engine.compute_residuals(s_t, dv_t, dt_t)
                a_res_true = a_res_true_t.cpu().numpy()
        else:
            # Fallback to NumPy if PyTorch not available
            pos = states_raw[:, :3]
            a_phys = self.physics_engine.gravity_model.compute_acceleration(pos)
            dt_clip = np.clip(dt_raw[:, None], 1e-6, None)
            a_trans = true_deltas_raw[:, 3:6] / dt_clip
            a_res_true = a_trans - a_phys

        # Physics residual = true residual - predicted residual (a_observed - a_predicted)
        res_discrepancy = a_res_true - pred_res_raw
        res_norms = np.linalg.norm(res_discrepancy, axis=1)

        mean_res = float(np.mean(res_norms))
        max_res = float(np.max(res_norms))
        rmse_res = float(np.sqrt(np.mean(res_norms ** 2)))

        summary = {
            "mean_physics_residual": mean_res,
            "max_physics_residual": max_res,
            "rmse_physics_residual": rmse_res,
        }

        log.info(
            "Physics Residuals — Mean: %.4e km/s² | Max: %.4e | RMSE: %.4e",
            mean_res,
            max_res,
            rmse_res,
        )

        return summary
