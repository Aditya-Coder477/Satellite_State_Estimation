"""
src/ground_station/validator.py
================================
GroundStationValidator.

Performs data-quality and geometric correctness checks on the ground station
configuration and computed geometry DataFrame.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.ground_station.ground_station import GroundStation
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class GroundStationValidationResult:
    """Outcome of validation checks."""

    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
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
            f"Ground Station Validation {'PASSED ✓' if self.passed else 'FAILED ✗'}",
            f"  Errors  : {len(self.errors)}",
            f"  Warnings: {len(self.warnings)}",
        ]
        for e in self.errors:
            lines.append(f"  [ERROR]   {e}")
        for w in self.warnings:
            lines.append(f"  [WARNING] {w}")
        return "\n".join(lines)


class GroundStationValidator:
    """Validates ground station network metadata and geometry output."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.gs_cfg = cfg.get("ground_station", {})
        self.enforce_unique_ids = self.gs_cfg.get("validation", {}).get("enforce_unique_station_ids", True)

    def validate_metadata(self, stations: list[GroundStation]) -> GroundStationValidationResult:
        """Verify ground station properties and coordinate bounds."""
        res = GroundStationValidationResult()

        # Unique IDs check
        if self.enforce_unique_ids:
            ids = [s.id for s in stations]
            if len(ids) != len(set(ids)):
                res.add_error(f"Duplicate Ground Station IDs detected: {ids}")

        # Coordinates and ENU orthogonality checks
        for s in stations:
            # Lat/Lon ranges
            if not (-90.0 <= s.lat_deg <= 90.0):
                res.add_error(f"Station '{s.name}' latitude out of range [-90, 90]: {s.lat_deg}")
            if not (-180.0 <= s.lon_deg <= 180.0):
                res.add_error(f"Station '{s.name}' longitude out of range [-180, 180]: {s.lon_deg}")
            if s.height_m < -200.0:
                res.add_warning(f"Station '{s.name}' has unusually low height: {s.height_m} m")

            # Check ENU rotation matrix orthogonality
            r = s.r_itrs_to_enu
            # 1. Norm of rows must be 1.0
            row_norms = np.linalg.norm(r, axis=1)
            if not np.allclose(row_norms, 1.0, atol=1e-8):
                res.add_error(f"Station '{s.name}' ENU matrix rows are not unit vectors: {row_norms}")

            # 2. Dot products between rows must be 0 (orthogonal)
            dp01 = np.dot(r[0], r[1])
            dp02 = np.dot(r[0], r[2])
            dp12 = np.dot(r[1], r[2])
            if not (np.abs(dp01) < 1e-8 and np.abs(dp02) < 1e-8 and np.abs(dp12) < 1e-8):
                res.add_error(
                    f"Station '{s.name}' ENU matrix is not orthogonal. "
                    f"Row dot products: e.n={dp01:.2e}, e.u={dp02:.2e}, n.u={dp12:.2e}"
                )

        if res.passed:
            log.debug("Ground Station metadata checks: OK")
        return res

    def validate_geometry(
        self,
        df: pd.DataFrame,
        stations: list[GroundStation],
    ) -> GroundStationValidationResult:
        """Validate computed satellite-to-station geometry DataFrame."""
        res = GroundStationValidationResult()

        if len(df) == 0:
            res.add_error("Empty geometry DataFrame.")
            return res

        # Column schema validation
        required_cols = [
            "utc_time", "elapsed_seconds", "station_id", "station_name", "visible",
            "azimuth_deg", "elevation_deg", "range_km", "range_rate_km_s",
            "enu_east_km", "enu_north_km", "enu_up_km", "los_x", "los_y", "los_z",
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            res.add_error(f"Missing required columns in geometry CSV: {missing}")
            return res

        # Check NaN and Inf
        for col in df.select_dtypes(include=[np.number]).columns:
            n_nan = df[col].isna().sum()
            n_inf = np.isinf(df[col]).sum()
            if n_nan > 0:
                res.add_error(f"Column '{col}' contains {n_nan} NaN values.")
            if n_inf > 0:
                res.add_error(f"Column '{col}' contains {n_inf} infinite values.")

        # Vectorized range validations
        az = df["azimuth_deg"].to_numpy()
        el = df["elevation_deg"].to_numpy()
        r = df["range_km"].to_numpy()
        rr = df["range_rate_km_s"].to_numpy()
        vis = df["visible"].to_numpy()
        stat_ids = df["station_id"].to_numpy()

        # Bounds checks
        if np.any((az < 0.0) | (az >= 360.0)):
            res.add_error("Azimuth contains values outside [0, 360) degrees.")
        if np.any((el < -90.0) | (el > 90.0)):
            res.add_error("Elevation contains values outside [-90, 90] degrees.")
        if np.any(r <= 0.0):
            res.add_error("Slant range contains non-positive values.")
        if np.any(~np.isfinite(rr)):
            res.add_error("Slant range rate contains non-finite values.")

        # Vectorized consistency check: visibility matches elevation threshold
        for s in stations:
            mask = stat_ids == s.id
            if not np.any(mask):
                continue
            
            # For this station, visible should be True iff elevation >= min_elevation
            s_el = el[mask]
            s_vis = vis[mask]
            expected_vis = s_el >= s.min_elevation_deg

            violations = np.sum(s_vis != expected_vis)
            if violations > 0:
                res.add_error(
                    f"Visibility flag inconsistency for station '{s.name}': "
                    f"{violations} samples violate threshold of >= {s.min_elevation_deg}°"
                )

        log.info(res.summary())
        return res
