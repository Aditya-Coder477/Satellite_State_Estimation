"""
src/observation_generation/ground_station.py
============================================
Domain model for a ground-based optical tracking station.

This module is the authoritative data model for ground stations consumed by
the observation generation pipeline (Phase 4 & 5).  It is intentionally
separate from ``src/coordinate_systems/station_frame.py`` which exists solely
for ENU rotation math within the Phase-3 coordinate transformation pipeline.

Classes
-------
GroundStation
    Dataclass representing one station with geodetic coordinates and helpers
    for ITRS Cartesian position and local ENU frame construction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, Tuple

import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation

from src.utils.logger import get_logger

log = get_logger(__name__)

# WGS-84 semi-major axis (km) and inverse flattening
_WGS84_A_KM = 6378.137
_WGS84_F = 1.0 / 298.257223563
_WGS84_B_KM = _WGS84_A_KM * (1.0 - _WGS84_F)
_WGS84_E2 = 2.0 * _WGS84_F - _WGS84_F ** 2   # first eccentricity squared


@dataclass
class GroundStation:
    """
    Represents a single optical tracking ground station.

    Parameters
    ----------
    station_id : int
        Unique integer identifier (from ``ground_stations.json``).
    name : str
        Human-readable station name, e.g. "Cape Canaveral".
    country : str
        Country of the station.
    lat_deg : float
        WGS-84 geodetic latitude in degrees (north positive).
    lon_deg : float
        WGS-84 geodetic longitude in degrees (east positive).
    height_m : float
        Height above the WGS-84 ellipsoid in metres.
    station_code : str, optional
        Short ICAO/IATA-style code, e.g. "CCAFS".
    timezone : str, optional
        IANA timezone string, e.g. "America/New_York".

    Attributes
    ----------
    location : EarthLocation
        Astropy ``EarthLocation`` built from geodetic coordinates.
    """

    station_id:   int
    name:         str
    country:      str
    lat_deg:      float
    lon_deg:      float
    height_m:     float
    station_code: str = field(default="")
    timezone:     str = field(default="")

    def __post_init__(self) -> None:
        self.lat_deg   = float(self.lat_deg)
        self.lon_deg   = float(self.lon_deg)
        self.height_m  = float(self.height_m)
        log.debug(
            "GroundStation '%s' [%s] created — lat=%.4f°, lon=%.4f°, h=%.1fm",
            self.name, self.station_code,
            self.lat_deg, self.lon_deg, self.height_m,
        )

    # ------------------------------------------------------------------
    # Astropy EarthLocation
    # ------------------------------------------------------------------

    @cached_property
    def location(self) -> EarthLocation:
        """Astropy EarthLocation for this station."""
        return EarthLocation.from_geodetic(
            lon=self.lon_deg  * u.deg,
            lat=self.lat_deg  * u.deg,
            height=self.height_m * u.m,
        )

    # ------------------------------------------------------------------
    # ITRS Cartesian position
    # ------------------------------------------------------------------

    @cached_property
    def itrs_position_km(self) -> np.ndarray:
        """
        Geocentric ITRS Cartesian position of the station in km.

        Returns
        -------
        np.ndarray
            Shape (3,) — [x, y, z] in km.
        """
        return np.array([
            self.location.x.to_value(u.km),
            self.location.y.to_value(u.km),
            self.location.z.to_value(u.km),
        ], dtype=np.float64)

    # ------------------------------------------------------------------
    # ENU rotation matrix (ITRS → local East-North-Up)
    # ------------------------------------------------------------------

    @cached_property
    def enu_rotation_matrix(self) -> np.ndarray:
        """
        3×3 rotation matrix that projects an ITRS vector into local ENU.

        Rows are [East, North, Up] unit vectors in ITRS coordinates.

        Returns
        -------
        np.ndarray
            Shape (3, 3).
        """
        lat = math.radians(self.lat_deg)
        lon = math.radians(self.lon_deg)

        sl, cl = math.sin(lat), math.cos(lat)
        so, co = math.sin(lon), math.cos(lon)

        # East  = [-sin(lon),           cos(lon),        0      ]
        # North = [-sin(lat)*cos(lon), -sin(lat)*sin(lon), cos(lat)]
        # Up    = [ cos(lat)*cos(lon),  cos(lat)*sin(lon), sin(lat)]
        east  = np.array([-so,       co,       0.0], dtype=np.float64)
        north = np.array([-sl * co, -sl * so,  cl],  dtype=np.float64)
        up    = np.array([ cl * co,  cl * so,  sl],  dtype=np.float64)

        return np.vstack([east, north, up])  # (3, 3)

    # ------------------------------------------------------------------
    # Summary / repr
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a one-line formatted summary for logging."""
        return (
            f"[{self.station_id:>2}] {self.name:<20} ({self.country:<14}) "
            f"lat={self.lat_deg:+.4f}°  lon={self.lon_deg:+.4f}°  "
            f"h={self.height_m:.0f}m  code={self.station_code}"
        )

    def __repr__(self) -> str:
        return (
            f"GroundStation(id={self.station_id}, name={self.name!r}, "
            f"lat={self.lat_deg:.4f}, lon={self.lon_deg:.4f})"
        )
