"""
src/coordinate_systems/coordinate_transformer.py
=================================================
Unified Coordinate Transformation API.

This module is the entry point for other modules to transform coordinates.
It provides functions to transform single states, whole DataFrames, or run
transformations for a specific satellite configuration.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Dict, Union

import pandas as pd
from astropy.time import Time

from src.coordinate_systems.astronomy_utils import AstronomyTimeConverter
from src.coordinate_systems.teme_to_gcrs import teme_to_gcrs_dataframe, teme_to_gcrs_vector
from src.coordinate_systems.gcrs_to_itrs import gcrs_to_itrs_dataframe, gcrs_to_itrs_vector
from src.utils.logger import get_logger

log = get_logger(__name__)


class CoordinateTransformer:
    """
    Unified coordinator class for frame transformations.
    Ensures zero direct Astropy coordinate calls outside this package.
    """

    @staticmethod
    def transform_single_state(
        x_teme: float,
        y_teme: float,
        z_teme: float,
        vx_teme: float,
        vy_teme: float,
        vz_teme: float,
        utc_time: Union[str, datetime],
    ) -> Dict[str, Any]:
        """
        Transform a single TEME state to GCRS and ITRS frames.

        Parameters
        ----------
        x_teme, y_teme, z_teme : float
            Position coordinates in TEME (km).
        vx_teme, vy_teme, vz_teme : float
            Velocity coordinates in TEME (km/s).
        utc_time : Union[str, datetime]
            UTC time of the state.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing original TEME states, transformed GCRS/ITRS states,
            and Julian / Modified Julian Dates.
        """
        # Parse time
        obstime = AstronomyTimeConverter.parse_utc_to_time(utc_time)
        jd = obstime.jd
        mjd = obstime.mjd

        # Convert TEME to GCRS
        x_g, y_g, z_g, vx_g, vy_g, vz_g = teme_to_gcrs_vector(
            x_teme, y_teme, z_teme, vx_teme, vy_teme, vz_teme, obstime
        )

        # Convert GCRS to ITRS
        x_i, y_i, z_i, vx_i, vy_i, vz_i = gcrs_to_itrs_vector(
            x_g, y_g, z_g, vx_g, vy_g, vz_g, obstime
        )

        return {
            "utc_time": str(utc_time) if isinstance(utc_time, datetime) else utc_time,
            "julian_date": float(jd),
            "modified_julian_date": float(mjd),
            "x_teme_km": x_teme,
            "y_teme_km": y_teme,
            "z_teme_km": z_teme,
            "vx_teme_kms": vx_teme,
            "vy_teme_kms": vy_teme,
            "vz_teme_kms": vz_teme,
            "x_gcrs_km": float(x_g),
            "y_gcrs_km": float(y_g),
            "z_gcrs_km": float(z_g),
            "vx_gcrs_kms": float(vx_g),
            "vy_gcrs_kms": float(vy_g),
            "vz_gcrs_kms": float(vz_g),
            "x_itrs_km": float(x_i),
            "y_itrs_km": float(y_i),
            "z_itrs_km": float(z_i),
            "vx_itrs_kms": float(vx_i),
            "vy_itrs_kms": float(vy_i),
            "vz_itrs_kms": float(vz_i),
        }

    @classmethod
    def transform_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform a whole DataFrame of TEME states to GCRS and ITRS.
        Also appends `julian_date` and `modified_julian_date`.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with TEME coordinates.

        Returns
        -------
        pd.DataFrame
            Enriched DataFrame with GCRS and ITRS coordinates and time columns.
        """
        # Enforce copy
        df_out = df.copy()

        # Parse times & compute Julian Dates
        obstime = AstronomyTimeConverter.parse_utc_to_time(df_out["utc_time"])
        df_out["julian_date"] = obstime.jd
        df_out["modified_julian_date"] = obstime.mjd

        # TEME -> GCRS
        df_out = teme_to_gcrs_dataframe(df_out)

        # GCRS -> ITRS
        df_out = gcrs_to_itrs_dataframe(df_out)

        return df_out


def transform_single_state(
    x_teme: float,
    y_teme: float,
    z_teme: float,
    vx_teme: float,
    vy_teme: float,
    vz_teme: float,
    utc_time: Union[str, datetime],
) -> Dict[str, Any]:
    """Exposes transform_single_state as a top-level function."""
    return CoordinateTransformer.transform_single_state(
        x_teme, y_teme, z_teme, vx_teme, vy_teme, vz_teme, utc_time
    )


def transform_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Exposes transform_dataframe as a top-level function."""
    return CoordinateTransformer.transform_dataframe(df)
