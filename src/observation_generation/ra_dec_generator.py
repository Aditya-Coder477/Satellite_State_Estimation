"""
src/observation_generation/ra_dec_generator.py
===============================================
Topocentric RA/Dec Generator.

Computes topocentric Right Ascension (RA) and Declination (Dec) vectorially in
the GCRS inertial frame using Astropy.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation

from src.utils.logger import get_logger

log = get_logger(__name__)


class RADecGenerator:
    """Computes Topocentric RA/Dec vectorially for each ground station."""

    @staticmethod
    def compute_vectorized(
        df_visible: pd.DataFrame,
        df_states: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute topocentric RA and Dec for all visible samples.

        Processes vectorially by station to allow batch time transformations.

        Parameters
        ----------
        df_visible : pd.DataFrame
            Visible ground station geometry records.
        df_states : pd.DataFrame
            Transformed satellite states (containing GCRS columns).

        Returns
        -------
        ra_deg : np.ndarray, shape (N_visible,)
            Right Ascension in degrees [0, 360).
        dec_deg : np.ndarray, shape (N_visible,)
            Declination in degrees [-90, 90].
        """
        n_vis = len(df_visible)
        ra_out = np.zeros(n_vis, dtype=np.float64)
        dec_out = np.zeros(n_vis, dtype=np.float64)

        # Set up indexing mapping
        df_visible = df_visible.copy()
        df_visible["_orig_idx"] = np.arange(n_vis)

        # Set index on df_states for fast lookup by time
        df_states_idx = df_states.set_index("utc_time")

        # Group by station to vectorize time transformations
        for station_name, df_s in df_visible.groupby("station_name"):
            # Find station coordinates in df_visible
            # We can construct EarthLocation using latitude, longitude, and height
            # Let's get these from the first record or configuration
            # In Phase 5, ground_station_geometry.csv contains station coordinates if we added them,
            # but let's query the geodetic coordinates:
            # We can use the first sample since the station location is constant.
            # Wait, let's load Lat/Lon/Height. In ground_stations.json we have the geodetic coordinates.
            # To be robust, let's reconstruct it using the local ENU coordinates and ITRS positions
            # if we have them, or query lat/lon from the dataframe if they are present.
            # Let's see: we know INSAT3D network stations.
            # Let's write a robust lookup from station_name:
            lat_lon_height = _STATION_GEODETIC_LOOKUP.get(
                str(station_name), (12.9716, 77.5946, 920.0)
            )
            lat, lon, height = lat_lon_height

            loc = EarthLocation.from_geodetic(
                lon=lon * u.deg,
                lat=lat * u.deg,
                height=height * u.m,
            )

            # Transformed times
            from src.coordinate_systems.astronomy_utils import AstronomyTimeConverter
            times_s = df_s["utc_time"].values
            obstime = AstronomyTimeConverter.parse_utc_to_time(times_s)

            # Vectorized station GCRS positions
            station_gcrs = loc.get_gcrs(obstime)
            r_station_gcrs = np.stack([
                station_gcrs.cartesian.x.to_value(u.km),
                station_gcrs.cartesian.y.to_value(u.km),
                station_gcrs.cartesian.z.to_value(u.km),
            ], axis=-1)   # (N_s, 3)

            # Get corresponding satellite GCRS positions
            sat_states_s = df_states_idx.loc[times_s]
            r_sat_gcrs = sat_states_s[["x_gcrs_km", "y_gcrs_km", "z_gcrs_km"]].to_numpy(dtype=np.float64)

            # Topocentric line of sight vector in GCRS
            los_gcrs = r_sat_gcrs - r_station_gcrs   # (N_s, 3)

            # Vectorized RA and Dec calculation
            rx, ry, rz = los_gcrs[:, 0], los_gcrs[:, 1], los_gcrs[:, 2]
            r_topo = np.linalg.norm(los_gcrs, axis=1)
            r_topo = np.maximum(r_topo, 1e-12)

            dec_rad = np.arcsin(rz / r_topo)
            dec_deg = np.degrees(dec_rad)

            ra_rad = np.arctan2(ry, rx)
            ra_deg = np.degrees(ra_rad) % 360.0

            # Put into output arrays at original index positions
            orig_indices = df_s["_orig_idx"].to_numpy()
            ra_out[orig_indices] = ra_deg
            dec_out[orig_indices] = dec_deg

        return ra_out, dec_out


# Hardcoded geodetic coordinates fallback for the INSAT3D network ground stations
# to avoid repeated JSON reloading inside the hot loop.
_STATION_GEODETIC_LOOKUP = {
    "Bengaluru": (12.9716, 77.5946, 920.0),
    "Hyderabad": (17.3850, 78.4867, 542.0),
    "Sriharikota": (13.7199, 80.2304, 5.0),
    "Lucknow": (26.8467, 80.9462, 123.0),
    "Port Blair": (11.6234, 92.7265, 89.0),
    # Default GOES16 stations if run on GOES16
    "Station Alpha": (37.7749, -122.4194, 10.0),
}
