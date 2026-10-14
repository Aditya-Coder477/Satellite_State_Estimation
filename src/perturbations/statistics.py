"""
src/perturbations/statistics.py
=================================
PerturbationStatistics — Computes and persists statistics for the
perturbation simulation.

Generates three JSON outputs:
  1. ``perturbation_summary.json``   — high-level mission statistics
  2. ``disturbance_statistics.json`` — per-model percentile / RMS stats
  3. ``perturbation_config_used.json`` — exact config snapshot for reproducibility

Also generates ``disturbance_log.csv`` — per-model tabular summary.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.perturbations.disturbance_generator import PerturbationResult
from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class PerturbationStatistics:
    """
    Computes and persists statistics for the perturbation simulation.

    Parameters
    ----------
    cfg : dict[str, Any]
        Project configuration.
    output_dir : Path
        Directory to write JSON and CSV statistics files.

    Examples
    --------
    >>> stats = PerturbationStatistics(cfg, output_dir=processed_dir)
    >>> stats.compute_and_save(df_nominal, df_disturbed, pert_result)
    """

    def __init__(self, cfg: dict[str, Any], output_dir: Path) -> None:
        self._cfg = cfg
        self._out = Path(output_dir)
        self._out.mkdir(parents=True, exist_ok=True)
        self._pert_cfg = cfg.get("perturbation", {})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_and_save(
        self,
        df_nominal: pd.DataFrame,
        df_disturbed: pd.DataFrame,
        result: PerturbationResult,
        val_stats: dict[str, Any] | None = None,
    ) -> None:
        """
        Compute all statistics and write output files.

        Parameters
        ----------
        df_nominal : pd.DataFrame
            Nominal orbit.
        df_disturbed : pd.DataFrame
            Disturbed orbit.
        result : PerturbationResult
            Per-model acceleration arrays.
        val_stats : dict | None
            Validation statistics from PerturbationValidator.
        """
        self._write_summary(df_nominal, df_disturbed, result, val_stats)
        self._write_model_statistics(result)
        self._write_config_snapshot()
        self._write_disturbance_log(result)
        log.info("Statistics written to: %s", str(self._out))

    # ------------------------------------------------------------------
    # Individual writers
    # ------------------------------------------------------------------

    def _write_summary(
        self,
        df_nominal: pd.DataFrame,
        df_disturbed: pd.DataFrame,
        result: PerturbationResult,
        val_stats: dict[str, Any] | None,
    ) -> None:
        """Write perturbation_summary.json."""
        r_nom = df_nominal[["x_teme_km", "y_teme_km", "z_teme_km"]].to_numpy()
        r_dist = df_disturbed[["x_teme_km", "y_teme_km", "z_teme_km"]].to_numpy()
        v_nom = df_nominal[["vx_teme_kms", "vy_teme_kms", "vz_teme_kms"]].to_numpy()
        v_dist = df_disturbed[["vx_teme_kms", "vy_teme_kms", "vz_teme_kms"]].to_numpy()

        dr_mag = np.linalg.norm(r_dist - r_nom, axis=-1)
        dv_mag = np.linalg.norm(v_dist - v_nom, axis=-1)
        acc_mag = result.perturbation_magnitude

        models_cfg = self._pert_cfg.get("models", {})
        summary = {
            "phase": "Phase 3 — Synthetic Perturbation Simulation",
            "satellite": "INSAT-3D",
            "models_enabled": {
                "solar_radiation_pressure": models_cfg.get("solar_radiation_pressure", True),
                "maneuver": models_cfg.get("maneuver", True),
                "random": models_cfg.get("random", True),
                "constant_bias": models_cfg.get("constant_bias", True),
            },
            "simulation_duration_days": 7,
            "n_samples": int(len(df_disturbed)),
            "step_seconds": float(self._cfg.get("propagation", {}).get("step_seconds", 60)),
            "residual_acceleration": {
                "max_km_s2": float(acc_mag.max()),
                "mean_km_s2": float(acc_mag.mean()),
                "rms_km_s2": float(np.sqrt(np.mean(acc_mag**2))),
                "min_km_s2": float(acc_mag.min()),
            },
            "position_deviation": {
                "max_km": float(dr_mag.max()),
                "mean_km": float(dr_mag.mean()),
                "rms_km": float(np.sqrt(np.mean(dr_mag**2))),
            },
            "velocity_deviation": {
                "max_km_s": float(dv_mag.max()),
                "mean_km_s": float(dv_mag.mean()),
                "rms_km_s": float(np.sqrt(np.mean(dv_mag**2))),
            },
        }
        if val_stats:
            summary["validation"] = val_stats

        path = self._out / "perturbation_summary.json"
        self._write_json(summary, path)

    def _write_model_statistics(self, result: PerturbationResult) -> None:
        """Write disturbance_statistics.json with per-model RMS/percentile stats."""
        models = {
            "srp": result.acc_srp,
            "maneuver": result.acc_maneuver,
            "random": result.acc_random,
            "bias": result.acc_bias,
            "total": result.acc_total,
        }
        out: dict[str, Any] = {}
        for name, acc in models.items():
            mag = np.linalg.norm(acc, axis=-1)
            out[name] = {
                "mean_km_s2": float(mag.mean()),
                "std_km_s2": float(mag.std()),
                "rms_km_s2": float(np.sqrt(np.mean(mag**2))),
                "max_km_s2": float(mag.max()),
                "min_km_s2": float(mag.min()),
                "p50_km_s2": float(np.percentile(mag, 50)),
                "p95_km_s2": float(np.percentile(mag, 95)),
                "p99_km_s2": float(np.percentile(mag, 99)),
                "n_nonzero": int(np.sum(mag > 0)),
                "ax_mean": float(acc[:, 0].mean()),
                "ay_mean": float(acc[:, 1].mean()),
                "az_mean": float(acc[:, 2].mean()),
            }

        path = self._out / "disturbance_statistics.json"
        self._write_json(out, path)

    def _write_config_snapshot(self) -> None:
        """Write perturbation_config_used.json — exact config snapshot."""
        snapshot = self._pert_cfg.copy()
        path = self._out / "perturbation_config_used.json"
        self._write_json(snapshot, path)

    def _write_disturbance_log(self, result: PerturbationResult) -> None:
        """Write disturbance_log.csv — per-model tabular summary."""
        models_data = {
            "srp": result.acc_srp,
            "maneuver": result.acc_maneuver,
            "random": result.acc_random,
            "bias": result.acc_bias,
        }

        rows = []
        t = result.times
        t_start_str = str(float(t.min()))
        t_end_str = str(float(t.max()))

        for model_name, acc in models_data.items():
            mag = np.linalg.norm(acc, axis=-1)
            active_mask = mag > 0
            n_active = int(active_mask.sum())

            t_active = t[active_mask] if n_active > 0 else np.array([])
            rows.append({
                "Model": model_name,
                "Start Time": float(t_active.min()) if n_active > 0 else float("nan"),
                "End Time": float(t_active.max()) if n_active > 0 else float("nan"),
                "Maximum Acceleration": float(mag.max()),
                "Mean Acceleration": float(mag[active_mask].mean()) if n_active > 0 else 0.0,
                "Affected Samples": n_active,
            })

        df_log = pd.DataFrame(rows)
        path = self._out / "disturbance_log.csv"
        df_log.to_csv(path, index=False, float_format="%.10e")
        log.info("Saved: %s", path)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _write_json(data: dict, path: Path) -> None:
        """Write dict to JSON with indentation."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
        log.info("Saved: %s", path)
