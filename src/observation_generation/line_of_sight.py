"""
src/observation_generation/line_of_sight.py
===========================================
Pure-geometry line-of-sight (LOS) computations between a satellite and a
ground station in the ITRS reference frame.

This module contains **no** observation logic or visibility thresholds.
It is intentionally minimal so that it can be reused by any future module
(e.g. radar cross-section, shadow analysis) without carrying observation
semantics.

Functions
---------
compute_los
    Compute topocentric range vector, scalar range, and unit vector for
    N satellite epochs simultaneously (fully vectorized).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from src.observation_generation.ground_station import GroundStation


def compute_los(
    x_itrs: np.ndarray,
    y_itrs: np.ndarray,
    z_itrs: np.ndarray,
    station: GroundStation,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the topocentric line-of-sight from a station to the satellite.

    All computation is vectorized over N epochs.  The station is Earth-fixed,
    so its ITRS velocity is zero by definition.

    Parameters
    ----------
    x_itrs, y_itrs, z_itrs : np.ndarray
        Satellite ITRS positions (km), shape (N,).
    station : GroundStation
        Observer station.

    Returns
    -------
    rho_vec : np.ndarray
        Topocentric range vectors in km, shape (N, 3).
    range_km : np.ndarray
        Scalar slant range in km, shape (N,).
    unit_vec : np.ndarray
        Unit LOS vectors (dimensionless), shape (N, 3).

    Notes
    -----
    The topocentric range vector is:

    .. math::

        \\vec{\\rho} = \\vec{r}_{sat} - \\vec{r}_{station}

    where both vectors are expressed in ITRS (Earth-centred, Earth-fixed).
    """
    x = np.asarray(x_itrs, dtype=np.float64)
    y = np.asarray(y_itrs, dtype=np.float64)
    z = np.asarray(z_itrs, dtype=np.float64)

    xs, ys, zs = station.itrs_position_km  # shape (3,) constant

    # Topocentric range vector: (N, 3)
    rho_vec = np.column_stack([x - xs, y - ys, z - zs])

    # Scalar range: (N,)
    range_km = np.linalg.norm(rho_vec, axis=1)

    # Unit vector: (N, 3)
    # Guard against division by zero (should never happen for an orbiting sat)
    safe_range = np.where(range_km > 0.0, range_km, 1.0)
    unit_vec = rho_vec / safe_range[:, np.newaxis]

    return rho_vec, range_km, unit_vec
