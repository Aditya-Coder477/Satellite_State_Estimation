"""
src/ground_station/visibility.py
=================================
VisibilityTester.

Computes Azimuth, Elevation, and visibility flags in a vectorized manner
using local East-North-Up (ENU) coordinates.
"""

from __future__ import annotations

import logging
import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)


class VisibilityTester:
    """Computes vectorized Azimuth, Elevation, and visibility flags."""

    @staticmethod
    def compute_az_el_vectorized(
        enu_coords: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute Azimuth and Elevation from ENU coordinates.

        Parameters
        ----------
        enu_coords : np.ndarray, shape (N, 3)
            Local ENU coordinates [km]: [East, North, Up].

        Returns
        -------
        azimuth_deg : np.ndarray, shape (N,)
            Azimuth in degrees [0, 360).
        elevation_deg : np.ndarray, shape (N,)
            Elevation in degrees [-90, 90].
        """
        east = enu_coords[:, 0]
        north = enu_coords[:, 1]
        up = enu_coords[:, 2]

        # Slant range in ENU
        r_slant = np.linalg.norm(enu_coords, axis=1)
        r_slant = np.maximum(r_slant, 1e-12)

        # Elevation: arcsin(Up / Range)
        elevation_rad = np.arcsin(up / r_slant)
        elevation_deg = np.degrees(elevation_rad)

        # Azimuth: arctan2(East, North) modulo 360
        azimuth_rad = np.arctan2(east, north)
        azimuth_deg = np.degrees(azimuth_rad) % 360.0

        return azimuth_deg, elevation_deg

    @staticmethod
    def test_visibility_vectorized(
        elevation_deg: np.ndarray,
        min_elevation_deg: float,
    ) -> np.ndarray:
        """
        Check if the satellite is visible at each step.

        Parameters
        ----------
        elevation_deg : np.ndarray, shape (N,)
            Elevation angle in degrees.
        min_elevation_deg : float
            Minimum elevation angle in degrees.

        Returns
        -------
        visible : np.ndarray, shape (N,)
            Boolean mask where True means visible.
        """
        return elevation_deg >= min_elevation_deg
