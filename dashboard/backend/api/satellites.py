"""
dashboard/backend/api/satellites.py
=====================================
FastAPI router: satellite listing and detail endpoints.
"""

from typing import List
from fastapi import APIRouter, HTTPException
from dashboard.backend.satellite_registry import SatelliteRegistry
from dashboard.backend.models.schemas import SatelliteInfo, SatelliteDetailResponse, GroundStation

router = APIRouter(prefix="/api/satellites", tags=["satellites"])


def _to_info(entry, registry: SatelliteRegistry) -> SatelliteInfo:
    return SatelliteInfo(
        id=entry.id,
        display_name=entry.display_name,
        operator=entry.operator,
        orbit_type=entry.orbit_type,
        norad_id=entry.norad_id,
        ai_supported=entry.ai_supported,
        checkpoint_available=registry.ai_available(entry.id),
        description=entry.description,
        nominal_longitude_deg=entry.nominal_longitude_deg,
    )


def _parse_stations(raw_stations: list) -> List[GroundStation]:
    result = []
    for s in raw_stations:
        result.append(GroundStation(
            id=s.get("id", 0),
            name=s.get("name", ""),
            lat_deg=float(s.get("latitude_deg", s.get("lat_deg", 0.0))),
            lon_deg=float(s.get("longitude_deg", s.get("lon_deg", 0.0))),
            height_m=float(s.get("height_m", 0.0)),
            station_code=s.get("station_code"),
            country=s.get("country"),
        ))
    return result


@router.get("", response_model=List[SatelliteInfo])
def list_satellites():
    """List all registered satellites with availability status."""
    from dashboard.backend.main import get_registry
    registry = get_registry()
    return [_to_info(e, registry) for e in registry.all()]


@router.get("/{satellite_id}", response_model=SatelliteDetailResponse)
def get_satellite(satellite_id: str):
    """Get detailed information about a specific satellite."""
    from dashboard.backend.main import get_registry
    registry = get_registry()
    try:
        entry = registry.get(satellite_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Satellite '{satellite_id}' not found.")

    raw_stations = registry.load_ground_stations(satellite_id)
    stations = _parse_stations(raw_stations)

    info = _to_info(entry, registry)
    return SatelliteDetailResponse(**info.model_dump(), ground_stations=[s.model_dump() for s in stations])
