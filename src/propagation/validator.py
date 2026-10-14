"""
src/propagation/validator.py
============================
Post-propagation data-quality validation.

:class:`TrajectoryValidator` checks the propagated DataFrame for physical and
structural correctness before it is persisted to disk.  All checks are
collected — the validator does not short-circuit on first failure — so the
caller receives a complete list of issues.

Checks performed
----------------
1. **Schema** — all required columns present.
2. **Row count** — matches the expected number from config.
3. **NaN / Inf** — no missing or non-finite values.
4. **Monotonic timestamps** — ``elapsed_seconds`` is strictly increasing.
5. **Step uniformity** — all time gaps equal ``step_seconds`` within tolerance.
6. **Physical sanity**
   * GEO-class altitude: position magnitude in [35 000, 37 000] km.
   * Velocity magnitude in [2.5, 4.0] km/s (GEO nominal ≈ 3.07 km/s).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)

# Physical thresholds for GEO-belt satellites
_GEO_R_MIN_KM = 35_000.0   # conservative lower bound
_GEO_R_MAX_KM = 37_000.0   # conservative upper bound
_GEO_V_MIN_KMS = 2.5
_GEO_V_MAX_KMS = 4.0


@dataclass
class ValidationResult:
    """
    Container for the outcome of a :class:`TrajectoryValidator` run.

    Attributes
    ----------
    passed : bool
        ``True`` if all checks passed.
    warnings : list[str]
        Non-fatal issues (physical checks outside typical GEO range).
    errors : list[str]
        Fatal issues (schema, NaN, row-count violations).
    stats : dict[str, Any]
        Summary statistics computed during validation.
    """

    passed: bool = True
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.passed = False
        log.error("VALIDATION ERROR: %s", msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)
        log.warning("VALIDATION WARNING: %s", msg)

    def summary(self) -> str:
        lines = [
            f"Validation {'PASSED ✓' if self.passed else 'FAILED ✗'}",
            f"  Errors  : {len(self.errors)}",
            f"  Warnings: {len(self.warnings)}",
        ]
        for e in self.errors:
            lines.append(f"  [ERROR]   {e}")
        for w in self.warnings:
            lines.append(f"  [WARNING] {w}")
        return "\n".join(lines)


class TrajectoryValidator:
    """
    Validates a propagated trajectory DataFrame.

    Parameters
    ----------
    expected_rows : int
        Expected row count from ``cfg["propagation"]["expected_rows"]``.
    step_seconds : float
        Expected time step in seconds (used for uniformity check).
    required_columns : list[str]
        List of column names that must be present.
    step_tolerance : float
        Allowed fractional deviation from ``step_seconds`` for uniformity check
        (default: 1e-6, i.e. < 1 microsecond drift).

    Examples
    --------
    >>> validator = TrajectoryValidator(expected_rows=10080, step_seconds=60)
    >>> result    = validator.validate(df)
    >>> print(result.summary())
    """

    _DEFAULT_COLUMNS = [
        "utc_time",
        "elapsed_seconds",
        "satellite_name",
        "norad_id",
        "x_teme_km",
        "y_teme_km",
        "z_teme_km",
        "vx_teme_kms",
        "vy_teme_kms",
        "vz_teme_kms",
    ]

    def __init__(
        self,
        expected_rows: int,
        step_seconds: float,
        required_columns: list[str] | None = None,
        step_tolerance: float = 1e-6,
    ) -> None:
        self.expected_rows = expected_rows
        self.step_seconds = step_seconds
        self.required_columns = required_columns or self._DEFAULT_COLUMNS
        self.step_tolerance = step_tolerance

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """
        Run all validation checks on *df* and return a :class:`ValidationResult`.

        Parameters
        ----------
        df : pd.DataFrame
            The propagated trajectory DataFrame.

        Returns
        -------
        ValidationResult
        """
        result = ValidationResult()

        self._check_schema(df, result)

        if result.errors:
            # Cannot continue without the required columns
            log.error("Schema check failed; skipping further checks.")
            return result

        self._check_row_count(df, result)
        self._check_no_nan_inf(df, result)
        self._check_monotonic_timestamps(df, result)
        self._check_step_uniformity(df, result)
        self._check_physical_range(df, result)
        self._compute_stats(df, result)

        log.info(result.summary())
        return result

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_schema(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Verify all required columns are present."""
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            result.add_error(f"Missing required columns: {missing}")
        else:
            log.debug("Schema check: OK  (all %d columns present)", len(self.required_columns))

    def _check_row_count(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Verify row count equals expected."""
        actual = len(df)
        if actual != self.expected_rows:
            result.add_error(
                f"Row count mismatch: expected {self.expected_rows}, got {actual}"
            )
        else:
            log.debug("Row count check: OK  (%d rows)", actual)

    def _check_no_nan_inf(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Check for NaN or Inf in numeric columns."""
        numeric_cols = [
            c for c in self.required_columns
            if df[c].dtype in (np.float64, np.float32, float)
        ]
        numeric_df = df[numeric_cols]

        nan_counts = numeric_df.isna().sum()
        inf_counts = numeric_df.apply(lambda s: np.isinf(s).sum())

        total_nan = nan_counts.sum()
        total_inf = inf_counts.sum()

        if total_nan > 0:
            bad_cols = nan_counts[nan_counts > 0].to_dict()
            result.add_error(f"NaN values detected: {bad_cols}")
        else:
            log.debug("NaN check: OK")

        if total_inf > 0:
            bad_cols = inf_counts[inf_counts > 0].to_dict()
            result.add_error(f"Inf values detected: {bad_cols}")
        else:
            log.debug("Inf check: OK")

    def _check_monotonic_timestamps(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Verify elapsed_seconds is strictly increasing."""
        elapsed = df["elapsed_seconds"].to_numpy()
        diffs = np.diff(elapsed)
        if not np.all(diffs > 0):
            n_bad = int(np.sum(diffs <= 0))
            result.add_error(
                f"elapsed_seconds is not strictly monotonic ({n_bad} violations)"
            )
        else:
            log.debug("Monotonic timestamp check: OK")

    def _check_step_uniformity(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Verify all time gaps equal step_seconds within floating-point tolerance."""
        elapsed = df["elapsed_seconds"].to_numpy()
        diffs = np.diff(elapsed)
        max_dev = float(np.max(np.abs(diffs - self.step_seconds)))
        tol = self.step_tolerance * self.step_seconds
        if max_dev > tol:
            result.add_warning(
                f"Time step non-uniformity: max deviation = {max_dev:.2e} s "
                f"(tolerance = {tol:.2e} s)"
            )
        else:
            log.debug("Step uniformity check: OK  (max_dev=%.2e s)", max_dev)

    def _check_physical_range(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Check position and velocity magnitudes are within GEO-class bounds."""
        r = np.sqrt(
            df["x_teme_km"].to_numpy() ** 2
            + df["y_teme_km"].to_numpy() ** 2
            + df["z_teme_km"].to_numpy() ** 2
        )
        v = np.sqrt(
            df["vx_teme_kms"].to_numpy() ** 2
            + df["vy_teme_kms"].to_numpy() ** 2
            + df["vz_teme_kms"].to_numpy() ** 2
        )

        r_min, r_max = float(r.min()), float(r.max())
        v_min, v_max = float(v.min()), float(v.max())

        log.debug(
            "Position magnitude: min=%.1f km, max=%.1f km | "
            "Velocity magnitude: min=%.4f km/s, max=%.4f km/s",
            r_min, r_max, v_min, v_max,
        )

        if r_min < _GEO_R_MIN_KM or r_max > _GEO_R_MAX_KM:
            result.add_warning(
                f"Position magnitude out of typical GEO range "
                f"[{_GEO_R_MIN_KM}, {_GEO_R_MAX_KM}] km: "
                f"min={r_min:.1f}, max={r_max:.1f}"
            )
        if v_min < _GEO_V_MIN_KMS or v_max > _GEO_V_MAX_KMS:
            result.add_warning(
                f"Velocity magnitude out of typical GEO range "
                f"[{_GEO_V_MIN_KMS}, {_GEO_V_MAX_KMS}] km/s: "
                f"min={v_min:.4f}, max={v_max:.4f}"
            )

    def _compute_stats(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Populate result.stats with summary statistics."""
        r = np.sqrt(
            df["x_teme_km"].to_numpy() ** 2
            + df["y_teme_km"].to_numpy() ** 2
            + df["z_teme_km"].to_numpy() ** 2
        )
        v = np.sqrt(
            df["vx_teme_kms"].to_numpy() ** 2
            + df["vy_teme_kms"].to_numpy() ** 2
            + df["vz_teme_kms"].to_numpy() ** 2
        )
        result.stats.update(
            {
                "n_rows": len(df),
                "satellite_name": str(df["satellite_name"].iloc[0]),
                "norad_id": int(df["norad_id"].iloc[0]),
                "t_start": str(df["utc_time"].iloc[0]),
                "t_end": str(df["utc_time"].iloc[-1]),
                "r_mean_km": float(r.mean()),
                "r_std_km": float(r.std()),
                "v_mean_kms": float(v.mean()),
                "v_std_kms": float(v.std()),
            }
        )
