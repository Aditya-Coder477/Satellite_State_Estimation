"""
dashboard/backend/api/simulation.py
=====================================
FastAPI router: SGP4-only simulation endpoint.
POST /api/simulation/sgp4

Returns a full SimulationResult with reference trajectory only.
"""

import logging
import math
from fastapi import APIRouter, HTTPException
from dashboard.backend.models.schemas import (
    SimulationRequest,
    SimulationResult,
    SatelliteInfo,
    TrajectoryData,
    OrbitalElementsSeries,
    GroundStation,
    SimulationMeta,
)
from dashboard.backend.api.satellites import _parse_stations

router = APIRouter(prefix="/api/simulation", tags=["simulation"])
log = logging.getLogger(__name__)


@router.post("/sgp4", response_model=SimulationResult)
def run_sgp4_simulation(request: SimulationRequest):
    """
    Run SGP4 propagation for the given satellite and return a full SimulationResult.

    No PI-STN inference — reference trajectory only.
    """
    from dashboard.backend.main import get_registry, get_sgp4_service

    registry = get_registry()
    sgp4_svc = get_sgp4_service()

    try:
        entry = registry.get(request.satellite_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Satellite '{request.satellite_id}' not found.")

    omm_path = registry.omm_path(request.satellite_id)

    try:
        result = sgp4_svc.run(
            omm_path=omm_path,
            duration_hours=request.duration_hours,
            step_seconds=request.step_seconds,
        )
    except Exception as exc:
        log.exception("SGP4 simulation failed for %s", request.satellite_id)
        raise HTTPException(status_code=500, detail=f"SGP4 simulation failed: {exc}")

    # Build orbital elements series (may contain NaN → convert to None-safe float)
    oe = result["orbital_elements"]
    def safe_list(arr):
        return [v if math.isfinite(v) else None for v in arr]

    ref = TrajectoryData(
        x_km=result["x_gcrs_km"],
        y_km=result["y_gcrs_km"],
        z_km=result["z_gcrs_km"],
        vx_kms=result["vx_gcrs_kms"],
        vy_kms=result["vy_gcrs_kms"],
        vz_kms=result["vz_gcrs_kms"],
        lat_deg=result["lat_deg"],
        lon_deg=result["lon_deg"],
        alt_km=result["alt_km"],
        orbital_elements=OrbitalElementsSeries(
            a_km=safe_list(oe.get("a_km", [])),
            e=safe_list(oe.get("e", [])),
            i_deg=safe_list(oe.get("i_deg", [])),
            raan_deg=safe_list(oe.get("raan_deg", [])),
            argp_deg=safe_list(oe.get("argp_deg", [])),
            nu_deg=safe_list(oe.get("nu_deg", [])),
        ),
    )

    raw_stations = registry.load_ground_stations(request.satellite_id)
    stations = _parse_stations(raw_stations)

    n_steps = len(result["timestamps"]) - 1
    meta = SimulationMeta(
        satellite_id=request.satellite_id,
        satellite_name=result["satellite_name"],
        initial_timestamp_utc=result["epoch_utc"],
        initial_frame="GCRS",
        initial_state_gcrs=result["initial_state_gcrs"],
        delta_t_seconds=request.step_seconds,
        duration_hours=request.duration_hours,
        n_steps=n_steps,
        mode="sgp4",
        pistn_available=registry.ai_available(request.satellite_id),
    )

    sat_info = SatelliteInfo(
        id=entry.id,
        display_name=entry.display_name,
        operator=entry.operator,
        orbit_type=entry.orbit_type,
        norad_id=entry.norad_id,
        ai_supported=entry.ai_supported,
        checkpoint_available=registry.checkpoint_exists(request.satellite_id),
        description=entry.description,
        nominal_longitude_deg=entry.nominal_longitude_deg,
    )

    return SimulationResult(
        satellite=sat_info,
        timestamps=result["timestamps"],
        elapsed_seconds=result["elapsed_seconds"],
        reference=ref,
        prediction=None,
        comparison=None,
        ground_stations=[s.model_dump() for s in stations],
        metadata=meta,
    )
