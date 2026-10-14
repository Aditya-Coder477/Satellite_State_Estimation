"""
src/observation_generation/observation_geometry.py
==================================================
Builds the multi-station, long-form geometry DataFrame.

For each ground station, calls :class:`VisibilityCalculator` to produce
topocentric Az/El/Range/Range-Rate geometry, then stacks all station results
into a single DataFrame with one row per (epoch × station).

Classes
-------
ObservationGeometry
    Stateless helper with a single class method ``compute``.
"""

from __future__ import annotations

from typing import List

import pandas as pd

from src.observation_generation.ground_station import GroundStation
from src.observation_generation.visibility import VisibilityCalculator
from src.utils.logger import get_logger

log = get_logger(__name__)


class ObservationGeometry:
    """Compute full topocentric geometry for all stations."""

    @staticmethod
    def compute(
        df: pd.DataFrame,
        stations: List[GroundStation],
        min_elevation_deg: float = 10.0,
    ) -> pd.DataFrame:
        """
        Compute geometry for all stations and return a long-form DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Truth states with ITRS and GCRS columns.
        stations : list[GroundStation]
            All ground stations to process.
        min_elevation_deg : float
            Passed to :class:`VisibilityCalculator`.

        Returns
        -------
        pd.DataFrame
            Long-form DataFrame with one row per (epoch × station).
            Added columns: ``station_id``, ``station_name``,
            ``elevation_deg``, ``azimuth_deg``, ``range_km``,
            ``range_rate_kms``, ``visible``.

        Notes
        -----
        The input DataFrame is *not* modified.  Memory is allocated once
        per station (N rows) and concatenated at the end for efficiency.
        """
        calc = VisibilityCalculator(min_elevation_deg=min_elevation_deg)
        frames: List[pd.DataFrame] = []

        for station in stations:
            log.info("Computing geometry for station: %s", station.name)
            geom = calc.compute(df, station)

            # Tag every row with station identity
            geom.insert(0, "station_id",   station.station_id)
            geom.insert(1, "station_name", station.name)

            frames.append(geom)

        if not frames:
            raise ValueError("No stations provided — cannot compute geometry.")

        combined = pd.concat(frames, ignore_index=True)

        n_total   = len(combined)
        n_visible = int(combined["visible"].sum())
        log.info(
            "Geometry complete — %d rows (%d stations × %d epochs), "
            "%d visible (%.1f%%)",
            n_total,
            len(stations),
            len(df),
            n_visible,
            100.0 * n_visible / max(n_total, 1),
        )

        return combined
