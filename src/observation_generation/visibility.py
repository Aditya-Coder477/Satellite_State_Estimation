"""
src/observation_generation/visibility.py
========================================
Topocentric visibility computation — Az, El, Range, and Range-Rate.

This module applies the ENU rotation matrix from a
:class:`~src.observation_generation.ground_station.GroundStation` to the
topocentric range vector and computes all geometric observation quantities.

Classes
-------
VisibilityCalculator
    Stateless worker that computes visibility geometry for one station
    against the full propagated state DataFrame.  All operations are
    vectorized via NumPy matrix multiply.

Notes
-----
**ITRS velocity of an Earth-fixed station is zero** because ITRS rotates
with the Earth.  Therefore range-rate simplifies to:

.. math::

    \\dot{\\rho} = \\vec{v}_{sat,ITRS} \\cdot \\hat{\\rho}

where :math:`\\hat{\\rho}` is the topocentric unit vector.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.observation_generation.ground_station import GroundStation
from src.observation_generation.line_of_sight import compute_los
from src.utils.logger import get_logger

log = get_logger(__name__)


class VisibilityCalculator:
    """
    Compute topocentric observation geometry for a single ground station.

    Parameters
    ----------
    min_elevation_deg : float
        Minimum elevation angle (degrees) for a satellite to be considered
        visible.  Standard optical tracking uses ≥ 10°.
    """

    def __init__(self, min_elevation_deg: float = 10.0) -> None:
        self.min_elevation_deg = float(min_elevation_deg)

    def compute(
        self,
        df: pd.DataFrame,
        station: GroundStation,
    ) -> pd.DataFrame:
        """
        Compute full topocentric geometry for every epoch in *df*.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: ``x_itrs_km``, ``y_itrs_km``,
            ``z_itrs_km``, ``vx_itrs_kms``, ``vy_itrs_kms``,
            ``vz_itrs_kms``.
        station : GroundStation
            Observer location.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with added columns:

            * ``elevation_deg``  — topocentric elevation in degrees
            * ``azimuth_deg``    — topocentric azimuth [0, 360) in degrees
            * ``range_km``       — slant range in km
            * ``range_rate_kms`` — radial range rate in km/s
            * ``visible``        — bool, True if elevation ≥ min_elevation_deg
        """
        x = df["x_itrs_km"].to_numpy()
        y = df["y_itrs_km"].to_numpy()
        z = df["z_itrs_km"].to_numpy()
        vx = df["vx_itrs_kms"].to_numpy()
        vy = df["vy_itrs_kms"].to_numpy()
        vz = df["vz_itrs_kms"].to_numpy()

        # --- Line-of-sight geometry ---
        rho_vec, range_km, unit_vec = compute_los(x, y, z, station)

        # --- ENU projection ---
        # R is (3, 3); rho_vec is (N, 3) → result is (N, 3)
        R = station.enu_rotation_matrix  # rows: [East, North, Up]
        enu = rho_vec @ R.T  # (N, 3)

        e_comp = enu[:, 0]
        n_comp = enu[:, 1]
        u_comp = enu[:, 2]

        # --- Elevation and Azimuth ---
        elevation_deg = np.degrees(np.arcsin(np.clip(u_comp / range_km, -1.0, 1.0)))
        azimuth_deg   = np.degrees(np.arctan2(e_comp, n_comp)) % 360.0

        # --- Range rate ---
        # dot product of satellite velocity with unit LOS vector
        v_sat = np.column_stack([vx, vy, vz])  # (N, 3)
        range_rate_kms = np.einsum("ij,ij->i", v_sat, unit_vec)

        # --- Visibility flag ---
        visible = elevation_deg >= self.min_elevation_deg

        log.debug(
            "Station '%s': %d/%d epochs visible (el ≥ %.1f°)",
            station.name,
            int(visible.sum()),
            len(visible),
            self.min_elevation_deg,
        )

        result = df.copy()
        result["elevation_deg"]  = elevation_deg
        result["azimuth_deg"]    = azimuth_deg
        result["range_km"]       = range_km
        result["range_rate_kms"] = range_rate_kms
        result["visible"]        = visible

        return result
