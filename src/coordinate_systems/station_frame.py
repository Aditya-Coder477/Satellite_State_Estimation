"""
src/coordinate_systems/station_frame.py
=======================================
Ground Station Local Frame (ENU) coordinates.

Defines classes to represent ground stations and transform coordinates from
geocentric Earth-fixed frames (ITRS) to topocentric local horizontal frames:
East-North-Up (ENU).
"""

from __future__ import annotations

import logging
from typing import Tuple, Union, Optional

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import EarthLocation

from src.utils.logger import get_logger

log = get_logger(__name__)


class GroundStation:
    """
    Represents a ground station on Earth's surface.

    Parameters
    ----------
    lat_deg : float
        Geodetic latitude in degrees.
    lon_deg : float
        Geodetic longitude in degrees.
    height_m : float
        Height above the ellipsoid in meters.
    name : str, optional
        Human-readable name of the station, by default ""
    """

    def __init__(self, lat_deg: float, lon_deg: float, height_m: float, name: str = "") -> None:
        self.lat_deg = float(lat_deg)
        self.lon_deg = float(lon_deg)
        self.height_m = float(height_m)
        self.name = name

        # Astropy EarthLocation object
        self._location = EarthLocation.from_geodetic(
            lon=self.lon_deg * u.deg,
            lat=self.lat_deg * u.deg,
            height=self.height_m * u.m,
        )

        log.debug(
            "Created GroundStation '%s' at Lat=%.4f, Lon=%.4f, H=%.1fm",
            self.name,
            self.lat_deg,
            self.lon_deg,
            self.height_m,
        )

    @property
    def location(self) -> EarthLocation:
        """Get the Astropy EarthLocation object."""
        return self._location

    @property
    def itrs_position_km(self) -> np.ndarray:
        """
        Get the Cartesian coordinates of the station in ITRS frame (km).

        Returns
        -------
        np.ndarray
            Shape (3,) containing [x, y, z] in km.
        """
        return np.array(
            [
                self._location.x.to_value(u.km),
                self._location.y.to_value(u.km),
                self._location.z.to_value(u.km),
            ],
            dtype=np.float64,
        )

    def __repr__(self) -> str:
        return (
            f"GroundStation(name={self.name!r}, lat={self.lat_deg:.4f}, "
            f"lon={self.lon_deg:.4f}, height={self.height_m:.1f})"
        )


class GroundStationFrame:
    """
    Computes local horizontal reference directions (East, North, Up unit vectors)
    for a given GroundStation.
    """

    def __init__(self, station: GroundStation) -> None:
        self.station = station
        self._compute_unit_vectors()

    def _compute_unit_vectors(self) -> None:
        """
        Compute the local East, North, and Up unit vectors in ITRS coordinates.
        Uses WGS-84 ellipsoidal normal directions.
        """
        lat_rad = np.radians(self.station.lat_deg)
        lon_rad = np.radians(self.station.lon_deg)

        cos_lat = np.cos(lat_rad)
        sin_lat = np.sin(lat_rad)
        cos_lon = np.cos(lon_rad)
        sin_lon = np.sin(lon_rad)

        # local Up (normal to ellipsoid)
        self.u_up = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat])

        # local East (perpendicular to meridian, horizontal plane)
        self.u_east = np.array([-sin_lon, cos_lon, 0.0])

        # local North (perpendicular to local East and Up)
        self.u_north = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat])

        # Rotation matrix from ITRS to ENU: rows are local East, North, Up directions
        self.rotation_matrix = np.vstack([self.u_east, self.u_north, self.u_up])

    def __repr__(self) -> str:
        return f"GroundStationFrame(station={self.station.name!r})"


class StationTransformer:
    """
    Orchestrates transformations between ITRS and local station ENU frames.

    Parameters
    ----------
    station : GroundStation
        The GroundStation defining the target ENU coordinate frame.
    """

    def __init__(self, station: GroundStation) -> None:
        self.station = station
        self.frame = GroundStationFrame(station)

    def itrs_to_enu(
        self,
        x_itrs: np.ndarray | float,
        y_itrs: np.ndarray | float,
        z_itrs: np.ndarray | float,
        vx_itrs: Optional[np.ndarray | float] = None,
        vy_itrs: Optional[np.ndarray | float] = None,
        vz_itrs: Optional[np.ndarray | float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Transform position and velocity from geocentric ITRS frame to topocentric ENU frame.

        Parameters
        ----------
        x_itrs, y_itrs, z_itrs : np.ndarray or float
            ITRS satellite positions in km.
        vx_itrs, vy_itrs, vz_itrs : np.ndarray or float, optional
            ITRS satellite velocities in km/s.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]
            Topocentric position (East, North, Up) in km, and optional velocities (v_east, v_north, v_up) in km/s.
        """
        # Convert to numpy arrays for calculation
        x = np.atleast_1d(x_itrs)
        y = np.atleast_1d(y_itrs)
        z = np.atleast_1d(z_itrs)

        # Get station position in km
        x_st, y_st, z_st = self.station.itrs_position_km

        # relative position
        dx = x - x_st
        dy = y - y_st
        dz = z - z_st

        # Project onto unit vectors
        # East
        e = dx * self.frame.u_east[0] + dy * self.frame.u_east[1] + dz * self.frame.u_east[2]
        # North
        n = dx * self.frame.u_north[0] + dy * self.frame.u_north[1] + dz * self.frame.u_north[2]
        # Up
        u_val = dx * self.frame.u_up[0] + dy * self.frame.u_up[1] + dz * self.frame.u_up[2]

        ve, vn, vu = None, None, None
        if vx_itrs is not None and vy_itrs is not None and vz_itrs is not None:
            vx = np.atleast_1d(vx_itrs)
            vy = np.atleast_1d(vy_itrs)
            vz = np.atleast_1d(vz_itrs)

            # Project velocities onto unit vectors (rigid rotation relative velocity)
            ve = vx * self.frame.u_east[0] + vy * self.frame.u_east[1] + vz * self.frame.u_east[2]
            vn = vx * self.frame.u_north[0] + vy * self.frame.u_north[1] + vz * self.frame.u_north[2]
            vu = vx * self.frame.u_up[0] + vy * self.frame.u_up[1] + vz * self.frame.u_up[2]

            # If input was scalar, return scalar
            if isinstance(vx_itrs, (float, int)) and ve.size == 1:
                ve, vn, vu = float(ve[0]), float(vn[0]), float(vu[0])

        if isinstance(x_itrs, (float, int)) and e.size == 1:
            e, n, u_val = float(e[0]), float(n[0]), float(u_val[0])

        return e, n, u_val, ve, vn, vu
