"""
src/ground_station/ground_station.py
=====================================
GroundStation Class.

Represents a single Earth-based ground station. Computes geocentric Cartesian
coordinates using Astropy's EarthLocation, and builds the rotation matrix
to convert vectors from the ITRS frame to the local East-North-Up (ENU) frame.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import astropy.units as u
from astropy.coordinates import EarthLocation

from src.utils.logger import get_logger

log = get_logger(__name__)


class GroundStation:
    """
    Representation of an Earth ground station.

    Parameters
    ----------
    station_id : int
        Unique identifier.
    name : str
        Human-readable name.
    lat_deg : float
        Geodetic latitude in degrees.
    lon_deg : float
        Geodetic longitude in degrees.
    height_m : float
        Height above the reference ellipsoid in meters.
    min_elevation_deg : float
        Minimum visibility elevation angle in degrees. Default: 10.0.
    """

    def __init__(
        self,
        station_id: int,
        name: str,
        lat_deg: float,
        lon_deg: float,
        height_m: float,
        min_elevation_deg: float = 10.0,
    ) -> None:
        self.id = int(station_id)
        self.name = str(name)
        self.lat_deg = float(lat_deg)
        self.lon_deg = float(lon_deg)
        self.height_m = float(height_m)
        self.min_elevation_deg = float(min_elevation_deg)

        # ── 1. Geocentric Cartesian Coordinates via EarthLocation ──────
        self.location = EarthLocation.from_geodetic(
            lon=self.lon_deg * u.deg,
            lat=self.lat_deg * u.deg,
            height=self.height_m * u.m,
        )

        # Convert to km in ECEF/ITRS
        self.pos_itrs_km = np.array([
            self.location.x.to_value(u.km),
            self.location.y.to_value(u.km),
            self.location.z.to_value(u.km),
        ], dtype=np.float64)

        # ── 2. local ENU Rotation Matrix ───────────────────────────────
        self.r_itrs_to_enu = self._build_enu_rotation_matrix()

        log.debug(
            "GroundStation %d [%s] registered: lat=%.4f°, lon=%.4f°, H=%.1f m | "
            "ITRS=[%.3f, %.3f, %.3f] km",
            self.id, self.name, self.lat_deg, self.lon_deg, self.height_m,
            *self.pos_itrs_km
        )

    def _build_enu_rotation_matrix(self) -> np.ndarray:
        """
        Build the rotation matrix from geocentric ITRS to local ENU.

        Returns
        -------
        np.ndarray, shape (3, 3)
        """
        lat_rad = np.radians(self.lat_deg)
        lon_rad = np.radians(self.lon_deg)

        # Trig terms
        sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)
        sin_lon, cos_lon = np.sin(lon_rad), np.cos(lon_rad)

        # Rotation matrix rows: East, North, Up
        # East:  [-sin_lon, cos_lon, 0]
        # North: [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat]
        # Up:    [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]
        rot = np.array([
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]
        ], dtype=np.float64)

        return rot

    def __repr__(self) -> str:
        return (
            f"GroundStation(id={self.id}, name={self.name!r}, "
            f"lat={self.lat_deg:.4f}, lon={self.lon_deg:.4f})"
        )
