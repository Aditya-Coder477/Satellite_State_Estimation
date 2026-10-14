"""
dashboard/backend/satellite_registry.py
========================================
Registry-driven satellite configuration for the Satellite Orbit Lab dashboard.

No if/elif chains. Each satellite is described by a SatelliteEntry dataclass.
Adding a new satellite is purely a configuration/data operation — no code changes required
in the API or service layers.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SatelliteEntry:
    """Immutable registry entry for one satellite."""
    id: str                        # canonical key, e.g. "GOES16"
    display_name: str              # human-readable, e.g. "GOES-16"
    operator: str
    orbit_type: str                # "GEO", "LEO", etc.
    norad_id: int
    omm_path: str                  # relative to project root
    checkpoint_dir: str            # relative to project root
    metadata_dir: str              # relative to project root
    ground_stations_json: str      # relative to project root
    ai_supported: bool
    description: str = ""
    nominal_longitude_deg: Optional[float] = None


# ---------------------------------------------------------------------------
# Registry — modify here to add satellites, never change API/service code
# ---------------------------------------------------------------------------
_REGISTRY_DATA: List[Dict[str, Any]] = [
    {
        "id": "GOES16",
        "display_name": "GOES-16",
        "operator": "NOAA",
        "orbit_type": "GEO",
        "norad_id": 41866,
        "omm_path": "data/raw/omm/GOES16.txt",
        "checkpoint_dir": "checkpoints/GOES 16",
        "metadata_dir": "data/metadata/GOES 16",
        "ground_stations_json": "data/metadata/GOES 16/ground_stations.json",
        "ai_supported": True,
        "description": "Geostationary Operational Environmental Satellite — East (75.2°W). "
                       "NOAA/NASA weather observation satellite.",
        "nominal_longitude_deg": -75.2,
    },
    {
        "id": "GOES19",
        "display_name": "GOES-19",
        "operator": "NOAA",
        "orbit_type": "GEO",
        "norad_id": 60133,
        "omm_path": "data/raw/omm/GOES19.txt",
        "checkpoint_dir": "checkpoints/GOES19",
        "metadata_dir": "data/metadata/GOES19",
        "ground_stations_json": "data/metadata/GOES 16/ground_stations.json",
        "ai_supported": True,
        "description": "Geostationary Operational Environmental Satellite (2024). "
                       "Next-generation NOAA weather satellite.",
        "nominal_longitude_deg": None,
    },
    {
        "id": "INSAT3D",
        "display_name": "INSAT-3D",
        "operator": "ISRO",
        "orbit_type": "GEO",
        "norad_id": 39216,
        "omm_path": "data/raw/omm/INSAT3D.txt",
        "checkpoint_dir": "checkpoints/INSAT3D",
        "metadata_dir": "data/metadata/INSAT3D",
        "ground_stations_json": "data/metadata/INSAT3D/ground_stations.json",
        "ai_supported": True,
        "description": "Indian National Satellite — 3D (82°E). "
                       "ISRO meteorological and search-and-rescue satellite.",
        "nominal_longitude_deg": 82.0,
    },
]


class SatelliteRegistry:
    """
    Registry of satellites supported by the Satellite Orbit Lab.

    Usage
    -----
    registry = SatelliteRegistry(project_root)
    entry = registry.get("GOES16")
    all_entries = registry.all()
    """

    def __init__(self, project_root: Path) -> None:
        self._root = Path(project_root).resolve()
        self._entries: Dict[str, SatelliteEntry] = {}
        self._load()

    def _load(self) -> None:
        """Instantiate all SatelliteEntry objects from registry data."""
        for data in _REGISTRY_DATA:
            entry = SatelliteEntry(**data)
            self._entries[entry.id] = entry
        log.info("SatelliteRegistry loaded: %d satellites", len(self._entries))

    def all(self) -> List[SatelliteEntry]:
        return list(self._entries.values())

    def get(self, satellite_id: str) -> SatelliteEntry:
        if satellite_id not in self._entries:
            raise KeyError(
                f"Satellite '{satellite_id}' not found in registry. "
                f"Registered: {list(self._entries.keys())}"
            )
        return self._entries[satellite_id]

    def ids(self) -> List[str]:
        return list(self._entries.keys())

    def omm_path(self, satellite_id: str) -> Path:
        return self._root / self.get(satellite_id).omm_path

    def checkpoint_dir(self, satellite_id: str) -> Path:
        return self._root / self.get(satellite_id).checkpoint_dir

    def metadata_dir(self, satellite_id: str) -> Path:
        return self._root / self.get(satellite_id).metadata_dir

    def ground_stations_json(self, satellite_id: str) -> Path:
        return self._root / self.get(satellite_id).ground_stations_json

    def processed_dir(self, satellite_id: str) -> Path:
        """
        Returns the processed data directory for the satellite.
        Looks up from config.yaml satellite registry.
        """
        # Map registry IDs to processed directory names
        _processed_dir_map = {
            "GOES16": "GOES16",
            "GOES19": "GOES19",
            "INSAT3D": "INSAT3D",
        }
        proc_name = _processed_dir_map.get(satellite_id, satellite_id)
        return self._root / "data" / "processed" / proc_name

    def normalization_json(self, satellite_id: str) -> Path:
        """Path to input normalization stats JSON (for PI-STN)."""
        return self.processed_dir(satellite_id) / "dataset" / "normalization.json"

    def target_normalization_json(self, satellite_id: str) -> Path:
        """Path to target normalization stats JSON (for PI-STN de-normalization)."""
        return self.processed_dir(satellite_id) / "training" / "transition_normalization.json"

    def feature_mapping_json(self, satellite_id: str) -> Path:
        """Path to feature mapping JSON (input_cols / target_cols for PI-STN)."""
        return self.processed_dir(satellite_id) / "dataset" / "feature_mapping.json"

    def load_ground_stations(self, satellite_id: str) -> List[Dict[str, Any]]:
        """Load and return ground station list for this satellite."""
        path = self.ground_stations_json(satellite_id)
        if not path.is_file():
            log.warning("Ground station file not found: %s", path)
            return []
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data.get("stations", [])

    def checkpoint_exists(self, satellite_id: str) -> bool:
        """Return True if best_model.pt exists in the checkpoint directory."""
        ckpt = self.checkpoint_dir(satellite_id) / "best_model.pt"
        return ckpt.is_file()

    def ai_available(self, satellite_id: str) -> bool:
        """True only if satellite is marked ai_supported, checkpoint exists, and target normalization exists."""
        entry = self.get(satellite_id)
        return (
            entry.ai_supported
            and self.checkpoint_exists(satellite_id)
            and self.target_normalization_json(satellite_id).is_file()
        )
