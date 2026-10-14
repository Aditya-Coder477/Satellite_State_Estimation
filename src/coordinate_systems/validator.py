"""
src/coordinate_systems/validator.py
===================================
Validator for Coordinate Transformations.

Performs schema, time-monotonicity, NaN/Inf, physical range, norm-preservation,
and continuity check validations on the transformed trajectory DataFrame.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class ValidationResult:
    """Container for the validation outcomes."""
    passed: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.passed = False
        log.error("VALIDATION ERROR: %s", msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)
        log.warning("VALIDATION WARNING: %s", msg)

    def summary(self) -> str:
        lines = [
            f"Coordinate Transformation Validation {'PASSED ✓' if self.passed else 'FAILED ✗'}",
            f"  Errors  : {len(self.errors)}",
            f"  Warnings: {len(self.warnings)}",
        ]
        for e in self.errors:
            lines.append(f"  [ERROR]   {e}")
        for w in self.warnings:
            lines.append(f"  [WARNING] {w}")
        return "\n".join(lines)


class CoordinateTransformationValidator:
    """
    Validates a transformed satellite trajectory DataFrame.

    Parameters
    ----------
    cfg : dict[str, Any]
        Full project config.
    """

    _REQUIRED_INPUTS = [
        "utc_time",
        "elapsed_seconds",
        "x_teme_km",
        "y_teme_km",
        "z_teme_km",
        "vx_teme_kms",
        "vy_teme_kms",
        "vz_teme_kms",
    ]

    _REQUIRED_OUTPUTS = [
        "julian_date",
        "modified_julian_date",
        "x_gcrs_km",
        "y_gcrs_km",
        "z_gcrs_km",
        "vx_gcrs_kms",
        "vy_gcrs_kms",
        "vz_gcrs_kms",
        "x_itrs_km",
        "y_itrs_km",
        "z_itrs_km",
        "vx_itrs_kms",
        "vy_itrs_kms",
        "vz_itrs_kms",
    ]

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        # Extract validation config parameters
        self.val_cfg = cfg.get("coordinate_transforms", {}).get("validation", {})
        
        self.norm_tolerance_km = float(self.val_cfg.get("norm_tolerance_m", 10.0)) / 1000.0
        self.max_pos_jump_kms = float(self.val_cfg.get("max_position_jump_kms", 5.0))
        self.max_vel_jump_kms2 = float(self.val_cfg.get("max_velocity_jump_kms2", 0.1))
        
        r_range = self.val_cfg.get("position_magnitude_range_km", [35000.0, 45000.0])
        v_range = self.val_cfg.get("velocity_magnitude_range_kms", [2.0, 5.0])
        self.r_min, self.r_max = float(r_range[0]), float(r_range[1])
        self.v_min, self.v_max = float(v_range[0]), float(v_range[1])

    def validate(self, df: pd.DataFrame, df_in: pd.DataFrame | None = None) -> ValidationResult:
        """
        Run all coordinate transformation validation checks.

        Parameters
        ----------
        df : pd.DataFrame
            The transformed trajectory DataFrame.
        df_in : pd.DataFrame | None
            The input trajectory DataFrame before transformation, for checking preservation.

        Returns
        -------
        ValidationResult
        """
        result = ValidationResult()

        # 1. Schema check
        self._check_schema(df, result)
        if not result.passed:
            log.error("Schema check failed. Skipping further checks.")
            return result

        # 2. NaN / Inf check
        self._check_no_nan_inf(df, result)

        # 3. Monotonic time check
        self._check_monotonic_time(df, result)

        # 4. Position and velocity range checks
        self._check_physical_ranges(df, result)

        # 5. Position norm preservation check (TEME vs GCRS vs ITRS)
        self._check_norm_preservation(df, result)

        # 6. State continuity check (detect unrealistic jumps)
        self._check_continuity(df, result)

        # 7. Residual acceleration preservation check
        if df_in is not None:
            self._check_residual_preservation(df, df_in, result)

        # Compute summary stats
        self._compute_stats(df, result)

        log.info(result.summary())
        return result

    def _check_residual_preservation(
        self, df_out: pd.DataFrame, df_in: pd.DataFrame, result: ValidationResult
    ) -> None:
        """Verify residual acceleration columns are unchanged."""
        res_cols = ["ax_res", "ay_res", "az_res", "perturbation_magnitude"]
        cols_to_check = [c for c in res_cols if c in df_in.columns]

        for col in cols_to_check:
            if col not in df_out.columns:
                result.add_error(f"Residual column '{col}' is missing in the output DataFrame.")
                continue

            val_in = df_in[col].to_numpy()
            val_out = df_out[col].to_numpy()

            # Check if identical (within floating point precision)
            if not np.allclose(val_in, val_out, rtol=1e-12, atol=1e-12, equal_nan=True):
                max_diff = float(np.max(np.abs(val_in - val_out)))
                result.add_error(
                    f"Residual column '{col}' values were modified during transformation (max diff: {max_diff:.2e})."
                )
            else:
                log.debug("Residual preservation check for '%s': OK", col)

    def _check_schema(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Verify all expected input and output columns exist."""
        all_cols = self._REQUIRED_INPUTS + self._REQUIRED_OUTPUTS
        missing = [c for c in all_cols if c not in df.columns]
        if missing:
            result.add_error(f"Missing required columns in DataFrame: {missing}")

    def _check_no_nan_inf(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Check for NaN and Inf values in the output columns."""
        cols = self._REQUIRED_OUTPUTS
        for col in cols:
            n_nan = df[col].isna().sum()
            n_inf = np.isinf(df[col]).sum()
            if n_nan > 0:
                result.add_error(f"Column '{col}' has {n_nan} NaN values.")
            if n_inf > 0:
                result.add_error(f"Column '{col}' has {n_inf} infinite values.")

    def _check_monotonic_time(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Verify that Julian dates and elapsed seconds are strictly monotonic."""
        jd = df["julian_date"].to_numpy()
        elapsed = df["elapsed_seconds"].to_numpy()

        if not np.all(np.diff(jd) > 0):
            result.add_error("julian_date column is not strictly monotonic.")
        if not np.all(np.diff(elapsed) > 0):
            result.add_error("elapsed_seconds column is not strictly monotonic.")

    def _check_physical_ranges(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Verify position and velocity magnitudes are within reasonable bounds."""
        for frame in ["gcrs", "itrs"]:
            r = np.sqrt(
                df[f"x_{frame}_km"].to_numpy() ** 2
                + df[f"y_{frame}_km"].to_numpy() ** 2
                + df[f"z_{frame}_km"].to_numpy() ** 2
            )
            v = np.sqrt(
                df[f"vx_{frame}_kms"].to_numpy() ** 2
                + df[f"vy_{frame}_kms"].to_numpy() ** 2
                + df[f"vz_{frame}_kms"].to_numpy() ** 2
            )

            r_min_val, r_max_val = r.min(), r.max()
            v_min_val, v_max_val = v.min(), v.max()

            # Positions
            if r_min_val < self.r_min or r_max_val > self.r_max:
                result.add_warning(
                    f"Position magnitude in {frame.upper()} frame ({r_min_val:.1f} to {r_max_val:.1f} km) "
                    f"exceeds range [{self.r_min:.1f}, {self.r_max:.1f}] km."
                )
            # Velocities
            if v_min_val < self.v_min or v_max_val > self.v_max:
                result.add_warning(
                    f"Velocity magnitude in {frame.upper()} frame ({v_min_val:.4f} to {v_max_val:.4f} km/s) "
                    f"exceeds range [{self.v_min:.4f}, {self.v_max:.4f}] km/s."
                )

    def _check_norm_preservation(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Verify position magnitude is preserved between geocentric frames TEME and GCRS."""
        r_teme = np.sqrt(
            df["x_teme_km"].to_numpy() ** 2
            + df["y_teme_km"].to_numpy() ** 2
            + df["z_teme_km"].to_numpy() ** 2
        )
        r_gcrs = np.sqrt(
            df["x_gcrs_km"].to_numpy() ** 2
            + df["y_gcrs_km"].to_numpy() ** 2
            + df["z_gcrs_km"].to_numpy() ** 2
        )
        r_itrs = np.sqrt(
            df["x_itrs_km"].to_numpy() ** 2
            + df["y_itrs_km"].to_numpy() ** 2
            + df["z_itrs_km"].to_numpy() ** 2
        )

        diff_gcrs = np.abs(r_teme - r_gcrs)
        diff_itrs = np.abs(r_teme - r_itrs)

        max_diff_gcrs = float(diff_gcrs.max())
        max_diff_itrs = float(diff_itrs.max())

        log.debug(
            "Max position norm difference TEME vs GCRS: %.6f km, TEME vs ITRS: %.6f km",
            max_diff_gcrs,
            max_diff_itrs,
        )

        if max_diff_gcrs > self.norm_tolerance_km:
            result.add_error(
                f"Position norm not preserved TEME -> GCRS. Max diff: {max_diff_gcrs * 1000.0:.3f} m "
                f"(tolerance: {self.norm_tolerance_km * 1000.0:.1f} m)."
            )
        if max_diff_itrs > self.norm_tolerance_km:
            result.add_error(
                f"Position norm not preserved TEME -> ITRS. Max diff: {max_diff_itrs * 1000.0:.3f} m "
                f"(tolerance: {self.norm_tolerance_km * 1000.0:.1f} m)."
            )

    def _check_continuity(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Add continuity checks between consecutive transformed states to detect unrealistic jumps."""
        t = df["elapsed_seconds"].to_numpy()
        dt = np.diff(t)

        if len(dt) == 0:
            return

        # Ensure no dt <= 0 to avoid division by zero
        if np.any(dt <= 0):
            result.add_error("Non-monotonic time steps encountered; skipping continuity checks.")
            return

        for frame in ["gcrs", "itrs"]:
            x = df[f"x_{frame}_km"].to_numpy()
            y = df[f"y_{frame}_km"].to_numpy()
            z = df[f"z_{frame}_km"].to_numpy()
            vx = df[f"vx_{frame}_kms"].to_numpy()
            vy = df[f"vy_{frame}_kms"].to_numpy()
            vz = df[f"vz_{frame}_kms"].to_numpy()

            # Position continuity: |r_{i+1} - r_i| / dt <= max_pos_jump_kms
            dx = np.diff(x)
            dy = np.diff(y)
            dz = np.diff(z)
            pos_jump = np.sqrt(dx**2 + dy**2 + dz**2) / dt
            max_pos_jump = float(pos_jump.max())

            # Velocity continuity: |v_{i+1} - v_i| / dt <= max_vel_jump_kms2
            dvx = np.diff(vx)
            dvy = np.diff(vy)
            dvz = np.diff(vz)
            vel_jump = np.sqrt(dvx**2 + dvy**2 + dvz**2) / dt
            max_vel_jump = float(vel_jump.max())

            log.debug(
                "[%s] Max state jump: Pos jump = %.6f km/s, Vel jump = %.6f km/s^2",
                frame.upper(),
                max_pos_jump,
                max_vel_jump,
            )

            if max_pos_jump > self.max_pos_jump_kms:
                result.add_error(
                    f"Position jump discontinuity in {frame.upper()} frame: {max_pos_jump:.4f} km/s "
                    f"(limit: {self.max_pos_jump_kms:.4f} km/s)."
                )

            if max_vel_jump > self.max_vel_jump_kms2:
                result.add_error(
                    f"Velocity jump discontinuity in {frame.upper()} frame: {max_vel_jump:.4f} km/s^2 "
                    f"(limit: {self.max_vel_jump_kms2:.4f} km/s^2)."
                )

    def _compute_stats(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Compute helpful stats for reporting."""
        r_gcrs = np.sqrt(df["x_gcrs_km"]**2 + df["y_gcrs_km"]**2 + df["z_gcrs_km"]**2)
        r_itrs = np.sqrt(df["x_itrs_km"]**2 + df["y_itrs_km"]**2 + df["z_itrs_km"]**2)
        
        result.stats.update({
            "n_samples": len(df),
            "julian_date_start": float(df["julian_date"].iloc[0]),
            "julian_date_end": float(df["julian_date"].iloc[-1]),
            "mean_r_gcrs": float(r_gcrs.mean()),
            "mean_r_itrs": float(r_itrs.mean()),
            "max_norm_diff_gcrs_m": float(np.abs(df["x_teme_km"]**2 + df["y_teme_km"]**2 + df["z_teme_km"]**2 - r_gcrs**2).max() * 1000.0),
        })
