"""
src/coordinate_systems/teme_to_gcrs.py
=======================================
TEME to GCRS coordinate transformation.

Provides functions to transform position and velocity from the TEME (True Equator
Mean Equinox) frame to the GCRS (Geocentric Celestial Reference System) frame.
Supports both single state vectors and vectorized datasets.
"""

from __future__ import annotations

import logging
from typing import Tuple

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import TEME, GCRS, CartesianRepresentation, CartesianDifferential
from astropy.time import Time

from src.coordinate_systems.astronomy_utils import AstronomyTimeConverter
from src.utils.logger import get_logger

log = get_logger(__name__)


def teme_to_gcrs_vector(
    x_teme: np.ndarray,
    y_teme: np.ndarray,
    z_teme: np.ndarray,
    vx_teme: np.ndarray,
    vy_teme: np.ndarray,
    vz_teme: np.ndarray,
    obstime: Time,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Transform TEME Cartesian states (position & velocity) to GCRS.

    Parameters
    ----------
    x_teme, y_teme, z_teme : np.ndarray
        TEME positions in km.
    vx_teme, vy_teme, vz_teme : np.ndarray
        TEME velocities in km/s.
    obstime : astropy.time.Time
        Observation times (same length as state arrays).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Position (x, y, z) in km and velocity (vx, vy, vz) in km/s in GCRS frame.
    """
    try:
        # Construct CartesianRepresentation and CartesianDifferential
        pos_repr = CartesianRepresentation(x_teme, y_teme, z_teme, unit=u.km)
        vel_diff = CartesianDifferential(vx_teme, vy_teme, vz_teme, unit=u.km / u.s)
        
        # Define coordinates in TEME
        teme_coord = TEME(pos_repr.with_differentials(vel_diff), obstime=obstime)
        
        # Transform to GCRS
        # GCRS obstime matches the input times
        gcrs_coord = teme_coord.transform_to(GCRS(obstime=obstime))
        
        # Extract components
        x_gcrs = gcrs_coord.cartesian.x.to_value(u.km)
        y_gcrs = gcrs_coord.cartesian.y.to_value(u.km)
        z_gcrs = gcrs_coord.cartesian.z.to_value(u.km)
        
        vx_gcrs = gcrs_coord.velocity.d_x.to_value(u.km / u.s)
        vy_gcrs = gcrs_coord.velocity.d_y.to_value(u.km / u.s)
        vz_gcrs = gcrs_coord.velocity.d_z.to_value(u.km / u.s)
        
        return x_gcrs, y_gcrs, z_gcrs, vx_gcrs, vy_gcrs, vz_gcrs
    except Exception as exc:
        log.error("Failed to convert TEME to GCRS vector: %s", exc)
        raise exc


def teme_to_gcrs_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform TEME coordinates to GCRS from a Pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing TEME state columns and `utc_time`.

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with GCRS coordinates.
    """
    log.info("Transforming TEME dataframe to GCRS (vectorized)...")
    
    # Extract arrays
    x = df["x_teme_km"].to_numpy()
    y = df["y_teme_km"].to_numpy()
    z = df["z_teme_km"].to_numpy()
    vx = df["vx_teme_kms"].to_numpy()
    vy = df["vy_teme_kms"].to_numpy()
    vz = df["vz_teme_kms"].to_numpy()
    
    # Parse times
    obstime = AstronomyTimeConverter.parse_utc_to_time(df["utc_time"])
    
    # Perform transformation
    x_gcrs, y_gcrs, z_gcrs, vx_gcrs, vy_gcrs, vz_gcrs = teme_to_gcrs_vector(
        x, y, z, vx, vy, vz, obstime
    )
    
    # Add to DataFrame
    df_out = df.copy()
    df_out["x_gcrs_km"] = x_gcrs
    df_out["y_gcrs_km"] = y_gcrs
    df_out["z_gcrs_km"] = z_gcrs
    df_out["vx_gcrs_kms"] = vx_gcrs
    df_out["vy_gcrs_kms"] = vy_gcrs
    df_out["vz_gcrs_kms"] = vz_gcrs
    
    log.debug("TEME -> GCRS transformation complete.")
    return df_out
