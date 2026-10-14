"""
src/ground_station/line_of_sight.py
====================================
LineOfSightCalculator.

Computes vectorized line-of-sight (LOS) vectors from ground station to satellite
in the geocentric ITRS reference frame.
"""

from __future__ import annotations

import logging
import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)


class LineOfSightCalculator:
    """Computes vectorized line-of-sight vectors from station to satellite."""

    @staticmethod
    def compute_vectorized(
        sat_pos_itrs: np.ndarray,
        station_pos_itrs: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the line-of-sight (LOS) vectors from the station to the satellite.

        Parameters
        ----------
        sat_pos_itrs : np.ndarray, shape (N, 3)
            Satellite position vectors in ITRS frame [km].
        station_pos_itrs : np.ndarray, shape (3,)
            Ground station geocentric position vector in ITRS [km].

        Returns
        -------
        np.ndarray, shape (N, 3)
            ITRS Line-of-sight vectors pointing from station to satellite [km].
        """
        # Broadcasting: subtracts station coordinates from all satellite coordinates (N, 3)
        return sat_pos_itrs - station_pos_itrs
