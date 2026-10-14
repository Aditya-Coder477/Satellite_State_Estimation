"""
dashboard/backend/services/sgp4_service.py
==========================================
SGP4 simulation service for the Satellite Orbit Lab.

Wraps:
  - src.propagation.omm_parser.OMMParser
  - src.propagation.sgp4_propagator.SGP4Propagator
  - src.coordinate_systems.coordinate_transformer.CoordinateTransformer
  - src.evaluation.orbital_elements.OrbitalElementsCalculator

Pipeline:
  OMM file → SGP4Propagator → TEME DataFrame
    → CoordinateTransformer → GCRS + ITRS columns
    → ITRS Cartesian → lat/lon/alt (WGS-84 via astropy)
    → OrbitalElementsCalculator → Keplerian elements per step

Returns structured dicts matching the dashboard's schema.

Unit conventions (enforced explicitly):
  position  : km
  velocity  : km/s
  angles    : degrees
  time      : UTC ISO-8601 strings + elapsed seconds
  frame     : GCRS for PI-STN inputs; ITRS for geographic output
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# WGS-84 constants for ITRS → geodetic conversion
# ---------------------------------------------------------------------------
_WGS84_A = 6378.137        # km (equatorial radius)
_WGS84_F = 1.0 / 298.257223563
_WGS84_B = _WGS84_A * (1.0 - _WGS84_F)          # km (polar radius)
_WGS84_E2 = 2.0 * _WGS84_F - _WGS84_F ** 2       # first eccentricity squared


def itrs_to_geodetic(
    x_km: np.ndarray,
    y_km: np.ndarray,
    z_km: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert ITRS Cartesian (km) to geodetic (WGS-84) latitude, longitude, altitude.

    Parameters
    ----------
    x_km, y_km, z_km : np.ndarray
        ITRS Cartesian coordinates [km].

    Returns
    -------
    lat_deg, lon_deg, alt_km : np.ndarray
        Geodetic latitude [deg], longitude [deg], altitude above ellipsoid [km].
    """
    # Longitude is straightforward
    lon_rad = np.arctan2(y_km, x_km)
    lon_deg = np.degrees(lon_rad)

    # Latitude — Bowring's iterative method (fast, accurate)
    p = np.sqrt(x_km ** 2 + y_km ** 2)  # distance from z-axis

    # Initial estimate (geocentric)
    lat_rad = np.arctan2(z_km, p * (1.0 - _WGS84_E2))

    for _ in range(5):  # 5 iterations is more than sufficient
        sin_lat = np.sin(lat_rad)
        N = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * sin_lat ** 2)
        lat_rad = np.arctan2(z_km + _WGS84_E2 * N * sin_lat, p)

    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    N = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * sin_lat ** 2)

    # Altitude above ellipsoid
    alt_km = np.where(
        np.abs(cos_lat) > 1e-10,
        p / cos_lat - N,
        np.abs(z_km) / np.abs(sin_lat) - N * (1.0 - _WGS84_E2),
    )

    lat_deg = np.degrees(lat_rad)
    return lat_deg, lon_deg, alt_km


class SGP4Service:
    """
    Service class that runs the SGP4 propagation pipeline and returns
    structured trajectory data in GCRS and geographic coordinates.

    All scientific computation is delegated to the existing src/ modules.
    This service is purely an integration/orchestration layer.
    """

    def __init__(self, project_root: Path) -> None:
        self._root = Path(project_root).resolve()
        self._orbital_calc = None  # lazy import

    def _get_orbital_calculator(self) -> Any:
        if self._orbital_calc is None:
            import sys
            if str(self._root) not in sys.path:
                sys.path.insert(0, str(self._root))
            from src.evaluation.orbital_elements import OrbitalElementsCalculator
            self._orbital_calc = OrbitalElementsCalculator()
        return self._orbital_calc

    def run(
        self,
        omm_path: Path,
        duration_hours: float,
        step_seconds: float,
    ) -> Dict[str, Any]:
        """
        Run the full SGP4 propagation pipeline.

        Parameters
        ----------
        omm_path : Path
            Path to the OMM CSV file.
        duration_hours : float
            Simulation duration [hours].
        step_seconds : float
            Propagation time step [seconds].

        Returns
        -------
        dict with keys:
          timestamps, elapsed_seconds,
          x_gcrs_km, y_gcrs_km, z_gcrs_km,
          vx_gcrs_kms, vy_gcrs_kms, vz_gcrs_kms,
          lat_deg, lon_deg, alt_km,
          orbital_elements,
          epoch_utc, initial_state_gcrs
        """
        import sys
        if str(self._root) not in sys.path:
            sys.path.insert(0, str(self._root))

        from src.propagation.omm_parser import OMMParser
        from src.propagation.sgp4_propagator import SGP4Propagator
        from src.coordinate_systems.coordinate_transformer import CoordinateTransformer

        log.info("SGP4Service.run | omm=%s | duration=%.1fh | step=%gs",
                 omm_path, duration_hours, step_seconds)

        duration_days = duration_hours / 24.0

        # ── Step 1: Parse OMM ─────────────────────────────────────────
        if not omm_path.is_file():
            raise FileNotFoundError(f"OMM file not found: {omm_path}")
        parser = OMMParser(str(omm_path))
        epoch_dt = parser.metadata["epoch"]
        epoch_utc = epoch_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
        satellite_name = parser.metadata["satellite_name"]
        log.info("OMM parsed | satellite=%s | epoch=%s", satellite_name, epoch_utc)

        # ── Step 2: SGP4 propagation → TEME DataFrame ─────────────────
        propagator = SGP4Propagator(parser, duration_days=duration_days, step_seconds=step_seconds)
        df_teme = propagator.propagate()
        log.info("SGP4 propagation complete | rows=%d", len(df_teme))

        # ── Step 3: TEME → GCRS → ITRS ────────────────────────────────
        df_full = CoordinateTransformer.transform_dataframe(df_teme)
        log.info("Coordinate transformation complete | frames: TEME→GCRS→ITRS")

        # ── Step 4: ITRS Cartesian → lat/lon/alt ──────────────────────
        lat_deg, lon_deg, alt_km = itrs_to_geodetic(
            df_full["x_itrs_km"].to_numpy(),
            df_full["y_itrs_km"].to_numpy(),
            df_full["z_itrs_km"].to_numpy(),
        )
        log.info("Geographic conversion complete | lat range: [%.2f, %.2f] deg",
                 lat_deg.min(), lat_deg.max())

        # ── Step 5: Orbital elements per timestep ────────────────────
        gcrs_states = df_full[[
            "x_gcrs_km", "y_gcrs_km", "z_gcrs_km",
            "vx_gcrs_kms", "vy_gcrs_kms", "vz_gcrs_kms",
        ]].to_numpy()  # (N, 6)

        oe_calc = self._get_orbital_calculator()
        orbital_elements = oe_calc.batch_to_elements(gcrs_states)  # dict of (N,) arrays
        log.info("Orbital elements computed | N=%d steps", len(gcrs_states))

        # ── Step 6: Record initial state (GCRS at epoch) ──────────────
        initial_state_gcrs = gcrs_states[0].tolist()  # [x,y,z,vx,vy,vz]

        log.info(
            "Initial state (GCRS) | x=%.1fkm, y=%.1fkm, z=%.1fkm | "
            "vx=%.4fkm/s, vy=%.4fkm/s, vz=%.4fkm/s",
            *initial_state_gcrs,
        )

        return {
            "timestamps": df_full["utc_time"].tolist(),
            "elapsed_seconds": df_full["elapsed_seconds"].tolist(),
            # GCRS position [km]
            "x_gcrs_km": df_full["x_gcrs_km"].tolist(),
            "y_gcrs_km": df_full["y_gcrs_km"].tolist(),
            "z_gcrs_km": df_full["z_gcrs_km"].tolist(),
            # GCRS velocity [km/s]
            "vx_gcrs_kms": df_full["vx_gcrs_kms"].tolist(),
            "vy_gcrs_kms": df_full["vy_gcrs_kms"].tolist(),
            "vz_gcrs_kms": df_full["vz_gcrs_kms"].tolist(),
            # Geographic (from ITRS WGS-84)
            "lat_deg": lat_deg.tolist(),
            "lon_deg": lon_deg.tolist(),
            "alt_km": alt_km.tolist(),
            # Keplerian elements (GCRS-based)
            "orbital_elements": {k: v.tolist() for k, v in orbital_elements.items()},
            # Metadata
            "epoch_utc": epoch_utc,
            "initial_state_gcrs": initial_state_gcrs,
            "satellite_name": satellite_name,
        }
