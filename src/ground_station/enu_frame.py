"""
src/ground_station/enu_frame.py
================================
ITRSToENUConverter.

Performs vectorized conversion of position vectors from geocentric ECEF/ITRS
reference frame into the local East-North-Up (ENU) coordinate frame at a ground station.
"""

from __future__ import annotations

import logging
import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)


class ITRSToENUConverter:
    """Performs vectorized coordinate transformations from ITRS/ECEF to local ENU."""

    @staticmethod
    def convert_vectorized(
        los_itrs: np.ndarray,
        r_itrs_to_enu: np.ndarray,
    ) -> np.ndarray:
        """
        Transform a set of ITRS line-of-sight vectors to the local ENU frame.

        Parameters
        ----------
        los_itrs : np.ndarray, shape (N, 3)
            Line-of-sight vectors in ITRS [km].
        r_itrs_to_enu : np.ndarray, shape (3, 3)
            ITRS-to-ENU rotation matrix of the ground station.

        Returns
        -------
        np.ndarray, shape (N, 3)
            Vectors in local ENU coordinates [km] representing [East, North, Up].
        """
        # Multiplies (N, 3) by (3, 3) transpose to compute: R_itrs_to_enu @ v_itrs
        # fully vectorized with zero Python loops
        return los_itrs @ r_itrs_to_enu.T
