"""
src/observation_generation/station_manager.py
=============================================
Loads and validates the ground station network from ``ground_stations.json``.

The JSON schema expects a top-level object with a ``"stations"`` array.
Each station entry must have: ``id``, ``name``, ``country``,
``latitude_deg``, ``longitude_deg``, ``height_m``.

Example
-------
::

    from src.observation_generation.station_manager import StationManager

    stations = StationManager.load("data/metadata/ground_stations.json")
    for st in stations:
        print(st.summary())
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from src.observation_generation.ground_station import GroundStation
from src.utils.logger import get_logger

log = get_logger(__name__)

# Required keys every station entry must carry
_REQUIRED_KEYS = {"id", "name", "country", "latitude_deg", "longitude_deg", "height_m"}
# Optional keys with their target GroundStation attribute names
_OPTIONAL_KEYS = {
    "station_code": "station_code",
    "timezone":     "timezone",
}


class StationManager:
    """
    Factory and registry for ground stations.

    Usage
    -----
    ::

        stations = StationManager.load(path)      # list[GroundStation]
        by_id    = StationManager.as_dict(stations)  # {id: GroundStation}
    """

    @staticmethod
    def load(path: str | Path) -> List[GroundStation]:
        """
        Parse ``ground_stations.json`` and return a validated list of
        :class:`~src.observation_generation.ground_station.GroundStation`.

        Parameters
        ----------
        path : str or Path
            Absolute or relative path to the JSON file.

        Returns
        -------
        list[GroundStation]
            Ordered list of station objects.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ValueError
            If the JSON schema is invalid or any station entry is missing
            required keys.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Ground stations file not found: {path}")

        log.info("Loading ground stations from: %s", path)

        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)

        # Support both flat array and nested {"stations": [...]} schema
        if isinstance(raw, list):
            entries = raw
            network_name = "unknown"
        elif isinstance(raw, dict) and "stations" in raw:
            entries = raw["stations"]
            network_name = raw.get("network_name", "unknown")
        else:
            raise ValueError(
                "ground_stations.json must be a JSON array or an object "
                "with a 'stations' key."
            )

        log.info("Network: %s — %d station entries found", network_name, len(entries))

        stations: List[GroundStation] = []
        for idx, entry in enumerate(entries):
            StationManager._validate_entry(idx, entry)
            kwargs = {
                "station_id": int(entry["id"]),
                "name":       str(entry["name"]),
                "country":    str(entry["country"]),
                "lat_deg":    float(entry["latitude_deg"]),
                "lon_deg":    float(entry["longitude_deg"]),
                "height_m":   float(entry["height_m"]),
            }
            for json_key, attr in _OPTIONAL_KEYS.items():
                if json_key in entry:
                    kwargs[attr] = str(entry[json_key])

            station = GroundStation(**kwargs)
            stations.append(station)
            log.debug(station.summary())

        log.info("Successfully loaded %d ground stations.", len(stations))
        return stations

    @staticmethod
    def as_dict(stations: List[GroundStation]) -> dict[int, GroundStation]:
        """
        Convert a list of stations to a dict keyed by ``station_id``.

        Parameters
        ----------
        stations : list[GroundStation]
            Output of :meth:`load`.

        Returns
        -------
        dict[int, GroundStation]
        """
        return {st.station_id: st for st in stations}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_entry(idx: int, entry: dict) -> None:
        """Raise ValueError if a station entry is missing required keys."""
        missing = _REQUIRED_KEYS - set(entry.keys())
        if missing:
            raise ValueError(
                f"Station entry at index {idx} is missing required keys: "
                f"{sorted(missing)}.  Entry: {entry}"
            )
        # Sanity-check numeric ranges
        lat = float(entry["latitude_deg"])
        lon = float(entry["longitude_deg"])
        h   = float(entry["height_m"])
        if not (-90.0 <= lat <= 90.0):
            raise ValueError(
                f"Station '{entry.get('name', idx)}' has invalid latitude: {lat}"
            )
        if not (-180.0 <= lon <= 180.0):
            raise ValueError(
                f"Station '{entry.get('name', idx)}' has invalid longitude: {lon}"
            )
        if h < -500.0:
            raise ValueError(
                f"Station '{entry.get('name', idx)}' has unrealistic height: {h} m"
            )
