"""
dashboard/backend/services/evaluation_service.py
=================================================
Evaluation service for the Satellite Orbit Lab.

Wraps existing scientific modules — no duplicate metric implementations:
  - src.evaluation.metrics_calculator.MetricsCalculator
  - src.evaluation.orbital_elements.OrbitalElementsCalculator

The service accepts pre-computed GCRS trajectory arrays (from SGP4 and PI-STN
services) and produces comparison metrics + error time-series for the dashboard.

All metric definitions are numerically identical to what the research pipeline
produces. Labels:
  REFERENCE — SGP4 trajectory
  PREDICTION — PI-STN rollout
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)


class EvaluationService:
    """
    Computes comparison metrics between SGP4 reference and PI-STN prediction.

    All metric computations are delegated to the existing
    MetricsCalculator and OrbitalElementsCalculator modules in src/.
    """

    def __init__(self, project_root: Path) -> None:
        self._root = Path(project_root).resolve()
        if str(self._root) not in sys.path:
            sys.path.insert(0, str(self._root))

    def compare(
        self,
        reference_gcrs: np.ndarray,   # (N, 6) GCRS [x,y,z,vx,vy,vz] km/km/s  (SGP4)
        prediction_gcrs: np.ndarray,   # (N, 6) GCRS [x,y,z,vx,vy,vz] km/km/s  (PI-STN)
        elapsed_seconds: np.ndarray,   # (N,) seconds from epoch
    ) -> Dict[str, Any]:
        """
        Compute full comparison metrics.

        Returns a dict compatible with ComparisonData and MetricsSummary schemas.

        Parameters
        ----------
        reference_gcrs : np.ndarray (N, 6)
            SGP4 reference trajectory in GCRS frame [km, km/s].
        prediction_gcrs : np.ndarray (N, 6)
            PI-STN predicted trajectory in GCRS frame [km, km/s].
        elapsed_seconds : np.ndarray (N,)
            Elapsed seconds from epoch.

        Returns
        -------
        dict with keys:
          position_error_km, velocity_error_kms,
          err_x_km, err_y_km, err_z_km,
          err_vx_kms, err_vy_kms, err_vz_kms,
          metrics (dict)
        """
        from src.evaluation.metrics_calculator import MetricsCalculator
        from src.evaluation.orbital_elements import OrbitalElementsCalculator

        log.info(
            "EvaluationService.compare | N=%d steps | elapsed=[%.0f, %.0f]s",
            len(elapsed_seconds), elapsed_seconds[0], elapsed_seconds[-1],
        )

        # ── Per-step error arrays ──────────────────────────────────────
        pos_error_vec = prediction_gcrs[:, :3] - reference_gcrs[:, :3]
        vel_error_vec = prediction_gcrs[:, 3:] - reference_gcrs[:, 3:]

        position_error_km = np.linalg.norm(pos_error_vec, axis=1)
        velocity_error_kms = np.linalg.norm(vel_error_vec, axis=1)

        # ── Summary metrics (via existing MetricsCalculator) ──────────
        calc = MetricsCalculator(physics_engine=None)  # no physics residuals in dashboard
        metrics_dict = calc.compute_all(
            times=elapsed_seconds,
            predicted=prediction_gcrs,
            truth=reference_gcrs,
        )

        # ── Orbital element drift ──────────────────────────────────────
        oe_calc = OrbitalElementsCalculator()
        try:
            oe_drift = oe_calc.compute_drift(prediction_gcrs, reference_gcrs)
        except Exception as exc:
            log.warning("Orbital element drift computation failed: %s", exc)
            oe_drift = {}

        # ── Build unified result ───────────────────────────────────────
        pos_m = metrics_dict.get("position", {})
        vel_m = metrics_dict.get("velocity", {})
        traj_m = metrics_dict.get("trajectory", {})
        phys_m = metrics_dict.get("physics", {})

        return {
            # Per-step arrays
            "position_error_km": position_error_km.tolist(),
            "velocity_error_kms": velocity_error_kms.tolist(),
            "err_x_km": pos_error_vec[:, 0].tolist(),
            "err_y_km": pos_error_vec[:, 1].tolist(),
            "err_z_km": pos_error_vec[:, 2].tolist(),
            "err_vx_kms": vel_error_vec[:, 0].tolist(),
            "err_vy_kms": vel_error_vec[:, 1].tolist(),
            "err_vz_kms": vel_error_vec[:, 2].tolist(),
            # Summary metrics
            "metrics": {
                "position_rmse_km": pos_m.get("rmse_km"),
                "position_mae_km": pos_m.get("mae_km"),
                "position_max_error_km": pos_m.get("max_error_km"),
                "velocity_rmse_kms": vel_m.get("rmse_kms"),
                "velocity_mae_kms": vel_m.get("mae_kms"),
                "cumulative_drift_km": traj_m.get("cumulative_position_drift_km"),
                "error_growth_rate_kmps": traj_m.get("error_growth_rate_kmps"),
                "physics_residual_mean": phys_m.get("physics_residual_mean"),
                "physics_residual_max": phys_m.get("physics_residual_max"),
                "physics_available": phys_m.get("physics_available", False),
                # Orbital element drift
                "a_km_mae": oe_drift.get("a_km_mae"),
                "a_km_rmse": oe_drift.get("a_km_rmse"),
                "e_mae": oe_drift.get("e_mae"),
                "e_rmse": oe_drift.get("e_rmse"),
                "i_deg_mae": oe_drift.get("i_deg_mae"),
                "i_deg_rmse": oe_drift.get("i_deg_rmse"),
            },
        }
