"""
src/coordinate_systems/astronomy_utils.py
=========================================
Astronomical and time conversion utilities.

This module provides clean wrappers around Astropy's Time classes to parse UTC
timestamps into Astropy Time objects, and compute Julian Date (JD) and
Modified Julian Date (MJD).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Sequence, Union

import numpy as np
import pandas as pd
from astropy.time import Time

from src.utils.logger import get_logger

log = get_logger(__name__)


class AstronomyTimeConverter:
    """
    Utility class for time representation conversions (UTC, JD, MJD).
    Wraps Astropy Time functionalities cleanly.
    """

    @staticmethod
    def parse_utc_to_time(
        utc_inputs: Union[str, datetime, Sequence[Union[str, datetime]], pd.Series, np.ndarray]
    ) -> Time:
        """
        Parse ISO-8601 UTC date-times into an astropy.time.Time object.

        Parameters
        ----------
        utc_inputs : str, datetime, Sequence, or array-like
            UTC time input(s) to convert.

        Returns
        -------
        Time
            Astropy Time object (in UTC scale).
        """
        try:
            is_scalar = isinstance(utc_inputs, (str, datetime)) or not hasattr(utc_inputs, "__len__")
            
            if is_scalar:
                inputs = str(utc_inputs)
            else:
                if isinstance(utc_inputs, pd.Series):
                    inputs = utc_inputs.astype(str).to_numpy(dtype='U')
                elif isinstance(utc_inputs, np.ndarray):
                    inputs = utc_inputs.astype('U')
                else:
                    inputs = np.array([str(x) for x in utc_inputs], dtype='U')

            return Time(inputs, scale="utc", format="isot")
        except Exception as exc:
            log.error("Failed to parse UTC inputs to Astropy Time: %s", exc)
            raise ValueError(f"Error parsing UTC time inputs: {exc}") from exc

    @classmethod
    def get_julian_date(
        self,
        utc_inputs: Union[str, datetime, Sequence[Union[str, datetime]], pd.Series, np.ndarray]
    ) -> np.ndarray:
        """
        Compute Julian Dates (JD) for the given UTC inputs.

        Parameters
        ----------
        utc_inputs : Union[str, datetime, Sequence, array-like]
            UTC time inputs.

        Returns
        -------
        np.ndarray
            Julian Date values as floats.
        """
        t = self.parse_utc_to_time(utc_inputs)
        # Handle scalar vs array-like return
        if t.isscalar:
            return np.array([t.jd], dtype=np.float64)
        return t.jd

    @classmethod
    def get_modified_julian_date(
        self,
        utc_inputs: Union[str, datetime, Sequence[Union[str, datetime]], pd.Series, np.ndarray]
    ) -> np.ndarray:
        """
        Compute Modified Julian Dates (MJD) for the given UTC inputs.

        Parameters
        ----------
        utc_inputs : Union[str, datetime, Sequence, array-like]
            UTC time inputs.

        Returns
        -------
        np.ndarray
            Modified Julian Date values as floats.
        """
        t = self.parse_utc_to_time(utc_inputs)
        if t.isscalar:
            return np.array([t.mjd], dtype=np.float64)
        return t.mjd
