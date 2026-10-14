"""
src/ground_station/station_manager.py
======================================
GroundStationManager Class.

Loads the ground station configuration JSON file and instantiates GroundStation objects.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.ground_station.ground_station import GroundStation
from src.utils.logger import get_logger

log = get_logger(__name__)


class GroundStationManager:
    """Loads and manages a network of Earth-based ground stations."""

    def __init__(self, station_json_path: Path | str, default_min_elevation: float = 10.0) -> None:
        self.path = Path(station_json_path)
        self.default_min_elevation = default_min_elevation
        self.stations: list[GroundStation] = []

        if not self.path.is_file():
            raise FileNotFoundError(f"Ground stations config file not found: {self.path}")

        self._load_stations()

    def _load_stations(self) -> None:
        """Parse the JSON config and instantiate stations."""
        log.info("Loading ground stations from: %s", self.path)
        with open(self.path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        # Get stations list
        station_list = data.get("stations", [])
        min_elev = data.get("minimum_elevation_deg", self.default_min_elevation)

        for s in station_list:
            gs = GroundStation(
                station_id=s["id"],
                name=s["name"],
                lat_deg=s["latitude_deg"],
                lon_deg=s["longitude_deg"],
                height_m=s["height_m"],
                min_elevation_deg=s.get("minimum_elevation_deg", min_elev)
            )
            self.stations.append(gs)

        log.info("Successfully loaded %d ground stations.", len(self.stations))

    def get_station_by_id(self, station_id: int) -> GroundStation | None:
        """Find a station by its unique ID."""
        for s in self.stations:
            if s.id == station_id:
                return s
        return None

    def __repr__(self) -> str:
        return f"GroundStationManager(stations={len(self.stations)}, source={self.path.name!r})"
