"""
src/physics/statistics.py
==========================
Computes comprehensive statistical metrics for Phase 8 (Physics Engine).

Calculates statistics on gravity, residuals, constraints, energy, and angular momentum.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class DatasetStatistics:
    """
    Computes physics execution metrics and formatting.
    """

    @staticmethod
    def compute_physics_statistics(
        gravity_acc: np.ndarray,
        residual_acc: np.ndarray,
        constraints: Dict[str, Any],
        orbital_energy: np.ndarray,
        angular_momentum: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Computes mean, std, min, max for various physics metrics.
        """
        log.info("Computing Phase 8 physics statistics...")

        g_norms = np.linalg.norm(gravity_acc, axis=-1)
        res_norms = np.linalg.norm(residual_acc, axis=-1)
        h_norms = np.linalg.norm(angular_momentum, axis=-1)

        # Basic statistics helper
        def get_stats(arr: np.ndarray) -> Dict[str, float]:
            return {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }

        stats = {
            "gravity": {
                "acceleration_magnitude": get_stats(g_norms),
                "components": {
                    "ax": get_stats(gravity_acc[:, 0]),
                    "ay": get_stats(gravity_acc[:, 1]),
                    "az": get_stats(gravity_acc[:, 2]),
                }
            },
            "residuals": {
                "residual_magnitude": get_stats(res_norms),
                "components": {
                    "ax_res": get_stats(residual_acc[:, 0]),
                    "ay_res": get_stats(residual_acc[:, 1]),
                    "az_res": get_stats(residual_acc[:, 2]),
                }
            },
            "energy": get_stats(orbital_energy),
            "angular_momentum": {
                "magnitude": get_stats(h_norms),
                "components": {
                    "hx": get_stats(angular_momentum[:, 0]),
                    "hy": get_stats(angular_momentum[:, 1]),
                    "hz": get_stats(angular_momentum[:, 2]),
                }
            },
            "constraints": {
                "radius_penalty": float(np.mean(constraints.get("radius_penalty", 0.0))),
                "velocity_penalty": float(np.mean(constraints.get("velocity_penalty", 0.0))),
                "energy_penalty": float(np.mean(constraints.get("energy_penalty", 0.0))),
                "momentum_penalty": float(np.mean(constraints.get("momentum_penalty", 0.0))),
                "total_penalty": float(np.mean(constraints.get("total_constraint_penalty", 0.0))),
            }
        }

        return stats
