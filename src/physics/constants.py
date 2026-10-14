"""
src/physics/constants.py
========================
Centralized physical and astronomical constants for Phase 8 (Physics Engine).

Loads values from ``data/metadata/INSAT3D/physics_constants.json`` at module import time
and exposes them as variables. Handles nested dictionary parsing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)

_ROOT_DIR = Path(__file__).resolve().parents[2]
_DEFAULT_JSON_PATH = _ROOT_DIR / "data" / "metadata" / "INSAT3D" / "physics_constants.json"


class ConstantsRegistry:
    """Registry holding all loaded physical and mathematical constants."""
    
    def __init__(self, json_path: Path) -> None:
        self.json_path = json_path
        self.constants: Dict[str, float] = {}
        self.load()

    def load(self) -> None:
        """Loads physical constants from the JSON file, handling nested structures."""
        log.info("Loading physical constants from: %s", self.json_path)
        try:
            if not self.json_path.is_file():
                log.warning(
                    "Constants JSON file not found at %s. Using standard defaults.",
                    self.json_path,
                )
                self._use_hardcoded_fallbacks()
                return

            with open(self.json_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                
            self.constants = {}
            
            # Parse nested "earth" section
            earth = data.get("earth", {})
            self.constants["EARTH_MU_KM3_S2"] = float(earth.get("gravitational_parameter_km3_s2", 398600.4418))
            self.constants["EARTH_RADIUS_KM"] = float(earth.get("equatorial_radius_km", 6378.137))
            self.constants["EARTH_ROTATION_RATE_RAD_S"] = float(earth.get("rotation_rate_rad_s", 7.292115e-5))
            
            # Parse nested "gravity_model" section
            gravity_model = data.get("gravity_model", {})
            self.constants["EARTH_J2"] = float(gravity_model.get("J2", 0.00108262668))
            self.constants["EARTH_J3"] = float(gravity_model.get("J3", -2.532153e-6))
            self.constants["EARTH_J4"] = float(gravity_model.get("J4", -1.610988e-6))
            
            # Add conversion defaults
            self.constants["KM_TO_M"] = 1000.0
            self.constants["M_TO_KM"] = 0.001
            self.constants["DAYS_TO_SEC"] = 86400.0
            self.constants["SEC_TO_DAYS"] = 1.1574074074074073e-5
            self.constants["STANDARD_GRAVITY_M_S2"] = 9.80665
            
        except Exception as e:
            log.error("Failed to load physical constants: %s. Using hardcoded fallback values.", str(e))
            self._use_hardcoded_fallbacks()

    def _use_hardcoded_fallbacks(self) -> None:
        """Fallbacks to prevent import failure if file IO completely fails."""
        self.constants = {
            "EARTH_MU_KM3_S2": 398600.4418,
            "EARTH_RADIUS_KM": 6378.137,
            "EARTH_J2": 0.00108262668,
            "EARTH_J3": -2.532153e-6,
            "EARTH_J4": -1.610988e-6,
            "EARTH_ROTATION_RATE_RAD_S": 7.292115e-5,
            "STANDARD_GRAVITY_M_S2": 9.80665,
            "KM_TO_M": 1000.0,
            "M_TO_KM": 0.001,
            "DAYS_TO_SEC": 86400.0,
            "SEC_TO_DAYS": 1.1574074074074073e-5
        }


# Initialize standard registry at module import
_registry = ConstantsRegistry(_DEFAULT_JSON_PATH)

# Expose constants as top-level module variables
MU: float = _registry.constants.get("EARTH_MU_KM3_S2", 398600.4418)
R_E: float = _registry.constants.get("EARTH_RADIUS_KM", 6378.137)
J2: float = _registry.constants.get("EARTH_J2", 0.00108262668)
J4: float = _registry.constants.get("EARTH_J4", -1.610988e-6)
OMEGA_E: float = _registry.constants.get("EARTH_ROTATION_RATE_RAD_S", 7.292115e-5)
G0: float = _registry.constants.get("STANDARD_GRAVITY_M_S2", 9.80665)

# Unit conversions
KM_TO_M: float = _registry.constants.get("KM_TO_M", 1000.0)
M_TO_KM: float = _registry.constants.get("M_TO_KM", 0.001)
DAYS_TO_SEC: float = _registry.constants.get("DAYS_TO_SEC", 86400.0)
SEC_TO_DAYS: float = _registry.constants.get("SEC_TO_DAYS", 1.1574074074074073e-5)


def reload_constants(json_path: str | Path) -> None:
    """
    Reloads module constants from an alternative configuration file.
    """
    global MU, R_E, J2, J4, OMEGA_E, G0, KM_TO_M, M_TO_KM, DAYS_TO_SEC, SEC_TO_DAYS
    p = Path(json_path).resolve()
    new_reg = ConstantsRegistry(p)
    
    MU = new_reg.constants.get("EARTH_MU_KM3_S2", MU)
    R_E = new_reg.constants.get("EARTH_RADIUS_KM", R_E)
    J2 = new_reg.constants.get("EARTH_J2", J2)
    J4 = new_reg.constants.get("EARTH_J4", J4)
    OMEGA_E = new_reg.constants.get("EARTH_ROTATION_RATE_RAD_S", OMEGA_E)
    G0 = new_reg.constants.get("STANDARD_GRAVITY_M_S2", G0)
    KM_TO_M = new_reg.constants.get("KM_TO_M", KM_TO_M)
    M_TO_KM = new_reg.constants.get("M_TO_KM", M_TO_KM)
    DAYS_TO_SEC = new_reg.constants.get("DAYS_TO_SEC", DAYS_TO_SEC)
    SEC_TO_DAYS = new_reg.constants.get("SEC_TO_DAYS", SEC_TO_DAYS)
    log.info("Physical constants reloaded successfully from: %s", p)
