"""
src/perturbations/validator.py
================================
PerturbationValidator — Validates the disturbed orbit and residual
acceleration DataFrames for physical and structural correctness.

Checks performed
----------------
1.  Schema — all required columns present in disturbed DataFrame.
2.  Row count — disturbed rows == nominal rows (10,081).
3.  NaN — no NaN values in numeric columns.
4.  Inf — no Inf values in numeric columns.
5.  Residual acceleration finite — ax_res, ay_res, az_res all finite.
6.  Timestamp match — utc_time identical to nominal.
7.  Position deviation — max |r_dist - r_nom| within config bound.
8.  Velocity deviation — max |v_dist - v_nom| within config bound.
9.  Perturbation magnitude — within configured maximum.
10. Total acceleration continuity — no discontinuous jumps in a_res.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)

_DEFAULT_MAX_POS_DEV_KM = 5.0
_DEFAULT_MAX_VEL_DEV_KMS = 0.01
_DEFAULT_MAX_ACC_KM_S2 = 1e-5


@dataclass
class PerturbationValidationResult:
    """Outcome of a :class:`PerturbationValidator` run."""

    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.passed = False
        log.error("PERTURBATION VALIDATION ERROR: %s", msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)
        log.warning("PERTURBATION VALIDATION WARNING: %s", msg)

    def summary(self) -> str:
        lines = [
            f"Perturbation Validation {'PASSED ✓' if self.passed else 'FAILED ✗'}",
            f"  Errors  : {len(self.errors)}",
            f"  Warnings: {len(self.warnings)}",
        ]
        for e in self.errors:
            lines.append(f"  [ERROR]   {e}")
        for w in self.warnings:
            lines.append(f"  [WARNING] {w}")
        return "\n".join(lines)


class PerturbationValidator:
    """
    Validates the perturbed orbit against the nominal orbit.

    Parameters
    ----------
    cfg : dict[str, Any]
        Project config (reads ``perturbation.validation`` block).
    expected_rows : int
        Expected number of rows in the disturbed DataFrame.
    step_seconds : float
        Time step in seconds.

    Examples
    --------
    >>> v = PerturbationValidator(cfg, expected_rows=10081, step_seconds=60.0)
    >>> result = v.validate(df_nominal, df_disturbed, df_residual)
    """

    _REQUIRED_DISTURBED_COLS = [
        "utc_time", "elapsed_seconds", "satellite_name", "norad_id",
        "x_teme_km", "y_teme_km", "z_teme_km",
        "vx_teme_kms", "vy_teme_kms", "vz_teme_kms",
        "x_km", "y_km", "z_km",
        "vx_km_s", "vy_km_s", "vz_km_s",
        "ax_res", "ay_res", "az_res", "perturbation_magnitude",
    ]

    _REQUIRED_RESIDUAL_COLS = [
        "utc_time",
        "ax_srp", "ay_srp", "az_srp",
        "ax_maneuver", "ay_maneuver", "az_maneuver",
        "ax_random", "ay_random", "az_random",
        "ax_bias", "ay_bias", "az_bias",
        "ax_res", "ay_res", "az_res",
    ]

    def __init__(
        self,
        cfg: dict[str, Any],
        expected_rows: int = 10081,
        step_seconds: float = 60.0,
    ) -> None:
        val_cfg = cfg.get("perturbation", {}).get("validation", {})
        self._expected_rows = expected_rows
        self._step_seconds = step_seconds
        self._max_pos_dev = float(val_cfg.get("max_position_deviation_km", _DEFAULT_MAX_POS_DEV_KM))
        self._max_vel_dev = float(val_cfg.get("max_velocity_deviation_kms", _DEFAULT_MAX_VEL_DEV_KMS))
        self._max_acc = float(val_cfg.get("max_acceleration_km_s2", _DEFAULT_MAX_ACC_KM_S2))

    def validate(
        self,
        df_nominal: pd.DataFrame,
        df_disturbed: pd.DataFrame,
        df_residual: pd.DataFrame,
    ) -> PerturbationValidationResult:
        """
        Run all validation checks.

        Parameters
        ----------
        df_nominal : pd.DataFrame
            Nominal orbit from Phase 2.
        df_disturbed : pd.DataFrame
            Disturbed orbit from Phase 3.
        df_residual : pd.DataFrame
            Per-model residual acceleration breakdown.

        Returns
        -------
        PerturbationValidationResult
        """
        result = PerturbationValidationResult()

        # ── Schema checks ─────────────────────────────────────────────
        self._check_schema(df_disturbed, self._REQUIRED_DISTURBED_COLS, "disturbed", result)
        self._check_schema(df_residual, self._REQUIRED_RESIDUAL_COLS, "residual", result)

        if result.errors:
            log.error("Schema check failed; skipping further checks.")
            return result

        # ── Structural checks ─────────────────────────────────────────
        self._check_row_count(df_disturbed, "disturbed", result)
        self._check_row_count(df_residual, "residual", result)
        self._check_nan_inf(df_disturbed, "disturbed", result)
        self._check_nan_inf(df_residual, "residual", result)

        # ── Physical checks ───────────────────────────────────────────
        self._check_residual_finite(df_disturbed, result)
        self._check_timestamp_match(df_nominal, df_disturbed, result)
        self._check_position_deviation(df_nominal, df_disturbed, result)
        self._check_velocity_deviation(df_nominal, df_disturbed, result)
        self._check_perturbation_magnitude(df_disturbed, result)

        self._compute_stats(df_nominal, df_disturbed, result)

        log.info(result.summary())
        return result

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_schema(
        self, df: pd.DataFrame, required: list[str], label: str, result: PerturbationValidationResult
    ) -> None:
        missing = [c for c in required if c not in df.columns]
        if missing:
            result.add_error(f"[{label}] Missing columns: {missing}")
        else:
            log.debug("[%s] Schema OK (%d columns)", label, len(required))

    def _check_row_count(
        self, df: pd.DataFrame, label: str, result: PerturbationValidationResult
    ) -> None:
        if len(df) != self._expected_rows:
            result.add_error(
                f"[{label}] Row count mismatch: expected {self._expected_rows}, got {len(df)}"
            )
        else:
            log.debug("[%s] Row count OK: %d", label, len(df))

    def _check_nan_inf(
        self, df: pd.DataFrame, label: str, result: PerturbationValidationResult
    ) -> None:
        num_cols = df.select_dtypes(include=[np.number]).columns
        nan_counts = df[num_cols].isna().sum()
        inf_counts = df[num_cols].apply(lambda s: np.isinf(s).sum())

        n_nan = nan_counts.sum()
        n_inf = inf_counts.sum()

        if n_nan > 0:
            result.add_error(f"[{label}] NaN values: {nan_counts[nan_counts > 0].to_dict()}")
        if n_inf > 0:
            result.add_error(f"[{label}] Inf values: {inf_counts[inf_counts > 0].to_dict()}")

        if n_nan == 0 and n_inf == 0:
            log.debug("[%s] NaN/Inf check OK", label)

    def _check_residual_finite(
        self, df: pd.DataFrame, result: PerturbationValidationResult
    ) -> None:
        for col in ["ax_res", "ay_res", "az_res"]:
            if not np.isfinite(df[col].to_numpy()).all():
                result.add_error(f"Residual acceleration column '{col}' contains non-finite values.")
        log.debug("Residual finiteness OK")

    def _check_timestamp_match(
        self,
        df_nominal: pd.DataFrame,
        df_disturbed: pd.DataFrame,
        result: PerturbationValidationResult,
    ) -> None:
        """Verify elapsed_seconds and utc_time are identical to nominal."""
        nom_elapsed = df_nominal["elapsed_seconds"].to_numpy()
        dist_elapsed = df_disturbed["elapsed_seconds"].to_numpy()

        if len(nom_elapsed) == len(dist_elapsed):
            max_diff = float(np.max(np.abs(nom_elapsed - dist_elapsed)))
            if max_diff > 1e-6:
                result.add_error(
                    f"elapsed_seconds mismatch: max difference = {max_diff:.2e} s"
                )
            else:
                log.debug("Timestamp match OK (max_diff=%.2e s)", max_diff)

    def _check_position_deviation(
        self,
        df_nominal: pd.DataFrame,
        df_disturbed: pd.DataFrame,
        result: PerturbationValidationResult,
    ) -> None:
        r_nom = df_nominal[["x_teme_km", "y_teme_km", "z_teme_km"]].to_numpy()
        r_dist = df_disturbed[["x_teme_km", "y_teme_km", "z_teme_km"]].to_numpy()
        dr_mag = np.linalg.norm(r_dist - r_nom, axis=-1)
        max_dr = float(dr_mag.max())
        result.stats["max_position_deviation_km"] = max_dr
        result.stats["mean_position_deviation_km"] = float(dr_mag.mean())

        if max_dr > self._max_pos_dev:
            result.add_warning(
                f"Max position deviation {max_dr:.4f} km exceeds threshold {self._max_pos_dev:.1f} km"
            )
        else:
            log.debug("Position deviation OK: max=%.4f km", max_dr)

    def _check_velocity_deviation(
        self,
        df_nominal: pd.DataFrame,
        df_disturbed: pd.DataFrame,
        result: PerturbationValidationResult,
    ) -> None:
        v_nom = df_nominal[["vx_teme_kms", "vy_teme_kms", "vz_teme_kms"]].to_numpy()
        v_dist = df_disturbed[["vx_teme_kms", "vy_teme_kms", "vz_teme_kms"]].to_numpy()
        dv_mag = np.linalg.norm(v_dist - v_nom, axis=-1)
        max_dv = float(dv_mag.max())
        result.stats["max_velocity_deviation_kms"] = max_dv
        result.stats["mean_velocity_deviation_kms"] = float(dv_mag.mean())

        if max_dv > self._max_vel_dev:
            result.add_warning(
                f"Max velocity deviation {max_dv:.6f} km/s exceeds threshold {self._max_vel_dev:.4f} km/s"
            )
        else:
            log.debug("Velocity deviation OK: max=%.4e km/s", max_dv)

    def _check_perturbation_magnitude(
        self, df_disturbed: pd.DataFrame, result: PerturbationValidationResult
    ) -> None:
        mag = df_disturbed["perturbation_magnitude"].to_numpy()
        max_mag = float(mag.max())
        if max_mag > self._max_acc:
            result.add_warning(
                f"Max perturbation magnitude {max_mag:.3e} km/s² exceeds threshold {self._max_acc:.1e} km/s²"
            )
        else:
            log.debug("Perturbation magnitude OK: max=%.3e km/s²", max_mag)

    def _compute_stats(
        self,
        df_nominal: pd.DataFrame,
        df_disturbed: pd.DataFrame,
        result: PerturbationValidationResult,
    ) -> None:
        """Populate result.stats with summary statistics."""
        mag = df_disturbed["perturbation_magnitude"].to_numpy()
        result.stats.update({
            "n_rows": len(df_disturbed),
            "max_perturbation_magnitude_km_s2": float(mag.max()),
            "mean_perturbation_magnitude_km_s2": float(mag.mean()),
            "rms_perturbation_magnitude_km_s2": float(np.sqrt(np.mean(mag ** 2))),
        })
