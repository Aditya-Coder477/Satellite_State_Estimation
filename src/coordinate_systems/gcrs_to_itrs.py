"""
src/coordinate_systems/gcrs_to_itrs.py
=======================================
GCRS to ITRS coordinate transformation.

Provides functions to transform position and velocity from the GCRS (Geocentric
Celestial Reference System) frame to the ITRS (International Terrestrial Reference
System) frame. Supports both single state vectors and vectorized datasets.
"""

from __future__ import annotations

import logging
from typing import Tuple

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation, CartesianDifferential
from astropy.time import Time

from src.coordinate_systems.astronomy_utils import AstronomyTimeConverter
from src.utils.logger import get_logger

log = get_logger(__name__)


def gcrs_to_itrs_vector(
    x_gcrs: np.ndarray,
    y_gcrs: np.ndarray,
    z_gcrs: np.ndarray,
    vx_gcrs: np.ndarray,
    vy_gcrs: np.ndarray,
    vz_gcrs: np.ndarray,
    obstime: Time,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Transform GCRS Cartesian states (position & velocity) to ITRS.

    This handles Earth rotation automatically, adding Coriolis terms to the velocities.

    Parameters
    ----------
    x_gcrs, y_gcrs, z_gcrs : np.ndarray
        GCRS positions in km.
    vx_gcrs, vy_gcrs, vz_gcrs : np.ndarray
        GCRS velocities in km/s.
    obstime : astropy.time.Time
        Observation times (same length as state arrays).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Position (x, y, z) in km and velocity (vx, vy, vz) in km/s in ITRS frame.
    """
    try:
        # Construct CartesianRepresentation and CartesianDifferential
        pos_repr = CartesianRepresentation(x_gcrs, y_gcrs, z_gcrs, unit=u.km)
        vel_diff = CartesianDifferential(vx_gcrs, vy_gcrs, vz_gcrs, unit=u.km / u.s)
        
        # Define coordinates in GCRS
        gcrs_coord = GCRS(pos_repr.with_differentials(vel_diff), obstime=obstime)
        
        # Transform to ITRS (rotating frame)
        itrs_coord = gcrs_coord.transform_to(ITRS(obstime=obstime))
        
        # Extract components
        x_itrs = itrs_coord.cartesian.x.to_value(u.km)
        y_itrs = itrs_coord.cartesian.y.to_value(u.km)
        z_itrs = itrs_coord.cartesian.z.to_value(u.km)
        
        vx_itrs = itrs_coord.velocity.d_x.to_value(u.km / u.s)
        vy_itrs = itrs_coord.velocity.d_y.to_value(u.km / u.s)
        vz_itrs = itrs_coord.velocity.d_z.to_value(u.km / u.s)
        
        return x_itrs, y_itrs, z_itrs, vx_itrs, vy_itrs, vz_itrs
    except Exception as exc:
        log.error("Failed to convert GCRS to ITRS vector: %s", exc)
        raise exc


def gcrs_to_itrs_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform GCRS coordinates to ITRS from a Pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing GCRS state columns and `utc_time`.

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with ITRS coordinates.
    """
    log.info("Transforming GCRS dataframe to ITRS (vectorized)...")
    
    # Extract arrays
    x = df["x_gcrs_km"].to_numpy()
    y = df["y_gcrs_km"].to_numpy()
    z = df["z_gcrs_km"].to_numpy()
    vx = df["vx_gcrs_kms"].to_numpy()
    vy = df["vy_gcrs_kms"].to_numpy()
    vz = df["vz_gcrs_kms"].to_numpy()
    
    # Parse times
    obstime = AstronomyTimeConverter.parse_utc_to_time(df["utc_time"])
    
    # Perform transformation
    x_itrs, y_itrs, z_itrs, vx_itrs, vy_itrs, vz_itrs = gcrs_to_itrs_vector(
        x, y, z, vx, vy, vz, obstime
    )
    
    # Add to DataFrame
    df_out = df.copy()
    df_out["x_itrs_km"] = x_itrs
    df_out["y_itrs_km"] = y_itrs
    df_out["z_itrs_km"] = z_itrs
    df_out["vx_itrs_kms"] = vx_itrs
    df_out["vy_itrs_kms"] = vy_itrs
    df_out["vz_itrs_kms"] = vz_itrs
    
    log.debug("GCRS -> ITRS transformation complete.")
    return df_out
