"""
src/ground_station/station_geometry.py
=======================================
StationGeometryEngine.

Executes the vectorized satellite-station geometry pipeline for a single station.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from src.ground_station.ground_station import GroundStation
from src.ground_station.enu_frame import ITRSToENUConverter
from src.ground_station.line_of_sight import LineOfSightCalculator
from src.ground_station.visibility import VisibilityTester
from src.ground_station.range_metrics import RangeMetricsCalculator
from src.utils.logger import get_logger

log = get_logger(__name__)


class StationGeometryEngine:
    """Coordinates vectorized geometry computations for a ground station."""

    @staticmethod
    def compute_geometry(
        station: GroundStation,
        sat_pos_itrs: np.ndarray,
        sat_vel_itrs: np.ndarray,
        times: np.ndarray,
        utc_times: np.ndarray,
    ) -> pd.DataFrame:
        """
        Compute line-of-sight, ENU coordinates, Azimuth, Elevation, Slant Range,
        Range-rate, and visibility vectorially for a station.

        Parameters
        ----------
        station : GroundStation
            The ground station instance.
        sat_pos_itrs : np.ndarray, shape (N, 3)
            Satellite positions in ITRS [km].
        sat_vel_itrs : np.ndarray, shape (N, 3)
            Satellite velocities in ITRS [km/s].
        times : np.ndarray, shape (N,)
            Elapsed seconds.
        utc_times : np.ndarray, shape (N,)
            UTC timestamps.

        Returns
        -------
        pd.DataFrame
        """
        n = len(times)

        # ── 1. Line of Sight ───────────────────────────────────────────
        los_itrs = LineOfSightCalculator.compute_vectorized(sat_pos_itrs, station.pos_itrs_km)

        # ── 2. local ENU coordinates ───────────────────────────────────
        enu_coords = ITRSToENUConverter.convert_vectorized(los_itrs, station.r_itrs_to_enu)

        # ── 3. Range & Range-rate ──────────────────────────────────────
        range_km = RangeMetricsCalculator.compute_range_vectorized(los_itrs)
        range_rate = RangeMetricsCalculator.compute_range_rate_vectorized(los_itrs, sat_vel_itrs)

        # ── 4. Azimuth & Elevation ─────────────────────────────────────
        azimuth_deg, elevation_deg = VisibilityTester.compute_az_el_vectorized(enu_coords)

        # ── 5. Visibility ──────────────────────────────────────────────
        visible = VisibilityTester.test_visibility_vectorized(elevation_deg, station.min_elevation_deg)

        # Normalize/norm values of LOS vector for saving (unit components)
        r_slant = np.maximum(range_km, 1e-12)
        los_unit = los_itrs / r_slant[:, np.newaxis]

        # Assemble DataFrame
        df = pd.DataFrame({
            "utc_time": utc_times,
            "elapsed_seconds": times,
            "station_id": station.id,
            "station_name": station.name,
            "visible": visible,
            "azimuth_deg": azimuth_deg,
            "elevation_deg": elevation_deg,
            "range_km": range_km,
            "range_rate_km_s": range_rate,
            "enu_east_km": enu_coords[:, 0],
            "enu_north_km": enu_coords[:, 1],
            "enu_up_km": enu_coords[:, 2],
            "los_x": los_unit[:, 0],
            "los_y": los_unit[:, 1],
            "los_z": los_unit[:, 2],
        })

        log.debug(
            "Station [%s] geometry: computed %d samples | visible=%d (%.1f%%)",
            station.name, n, int(visible.sum()), float(visible.sum() / n * 100.0)
        )
        return df
