"""
src/ground_station/range_metrics.py
====================================
RangeMetricsCalculator.

Performs vectorized calculations of slant range and range rate.
"""

from __future__ import annotations

import logging
import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)


class RangeMetricsCalculator:
    """Computes vectorized slant range and range-rate metrics."""

    @staticmethod
    def compute_range_vectorized(los_itrs: np.ndarray) -> np.ndarray:
        """
        Compute slant range (magnitude of the line-of-sight vector).

        Parameters
        ----------
        los_itrs : np.ndarray, shape (N, 3)
            Line-of-sight vectors in ITRS frame [km].

        Returns
        -------
        range_km : np.ndarray, shape (N,)
            Slant range in km.
        """
        return np.linalg.norm(los_itrs, axis=1)

    @staticmethod
    def compute_range_rate_vectorized(
        los_itrs: np.ndarray,
        sat_vel_itrs: np.ndarray,
    ) -> np.ndarray:
        """
        Compute range-rate (radial velocity) in the ITRS frame.

        Parameters
        ----------
        los_itrs : np.ndarray, shape (N, 3)
            Line-of-sight vectors in ITRS frame [km].
        sat_vel_itrs : np.ndarray, shape (N, 3)
            Satellite velocity vectors in ITRS frame [km/s].

        Returns
        -------
        range_rate_km_s : np.ndarray, shape (N,)
            Slant range-rate in km/s.
        """
        # Calculate range
        r_slant = np.linalg.norm(los_itrs, axis=1, keepdims=True)
        r_slant = np.maximum(r_slant, 1e-12)

        # Unit line-of-sight vector
        los_unit = los_itrs / r_slant

        # Project satellite velocity onto the line-of-sight unit vector
        # Range rate = v_sat . los_unit
        # Row-wise dot product: sum(v_sat * los_unit, axis=1)
        return np.sum(sat_vel_itrs * los_unit, axis=1)
