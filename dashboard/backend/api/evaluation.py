"""
dashboard/backend/api/evaluation.py
=====================================
FastAPI router: unified compare endpoint.
POST /api/evaluation/compare

Runs both SGP4 and PI-STN and returns a full SimulationResult
with reference, prediction, and comparison (metrics + error arrays).

The PI-STN rollout starts from the SAME initial GCRS state as SGP4 at epoch.
"""

import logging
import math
from typing import Optional
import numpy as np
from fastapi import APIRouter, HTTPException

from dashboard.backend.models.schemas import (
    SimulationRequest,
    SimulationResult,
    SatelliteInfo,
    TrajectoryData,
    OrbitalElementsSeries,
    ComparisonData,
    MetricsSummary,
    GroundStation,
    SimulationMeta,
)
from dashboard.backend.api.satellites import _parse_stations

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])
log = logging.getLogger(__name__)


def _oe_series(oe: dict) -> OrbitalElementsSeries:
    def safe_list(arr):
        return [v if v is not None and math.isfinite(v) else None for v in arr]
    return OrbitalElementsSeries(
        a_km=safe_list(oe.get("a_km", [])),
        e=safe_list(oe.get("e", [])),
        i_deg=safe_list(oe.get("i_deg", [])),
        raan_deg=safe_list(oe.get("raan_deg", [])),
        argp_deg=safe_list(oe.get("argp_deg", [])),
        nu_deg=safe_list(oe.get("nu_deg", [])),
    )


@router.post("/compare", response_model=SimulationResult)
def run_compare(request: SimulationRequest):
    """
    Run SGP4 + PI-STN rollout and return full comparison.

    Both trajectories start from the same GCRS initial state at the OMM epoch.
    If PI-STN is unavailable, returns SGP4 reference only (prediction=null).
    """
    from dashboard.backend.main import (
        get_registry, get_sgp4_service, get_pistn_service, get_eval_service,
    )

    registry = get_registry()
    sgp4_svc = get_sgp4_service()
    pistn_svc = get_pistn_service()
    eval_svc = get_eval_service()

    try:
        entry = registry.get(request.satellite_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Satellite '{request.satellite_id}' not found.")

    omm_path = registry.omm_path(request.satellite_id)
    ai_available = registry.ai_available(request.satellite_id)

    # ── Step 1: SGP4 reference trajectory ────────────────────────────
    try:
        sgp4_result = sgp4_svc.run(
            omm_path=omm_path,
            duration_hours=request.duration_hours,
            step_seconds=request.step_seconds,
        )
    except Exception as exc:
        log.exception("SGP4 failed for %s", request.satellite_id)
        raise HTTPException(status_code=500, detail=f"SGP4 failed: {exc}")

    timestamps = sgp4_result["timestamps"]
    elapsed_seconds = sgp4_result["elapsed_seconds"]
    initial_state_gcrs = sgp4_result["initial_state_gcrs"]
    epoch_utc = sgp4_result["epoch_utc"]
    n_steps = len(timestamps) - 1

    ref = TrajectoryData(
        x_km=sgp4_result["x_gcrs_km"],
        y_km=sgp4_result["y_gcrs_km"],
        z_km=sgp4_result["z_gcrs_km"],
        vx_kms=sgp4_result["vx_gcrs_kms"],
        vy_kms=sgp4_result["vy_gcrs_kms"],
        vz_kms=sgp4_result["vz_gcrs_kms"],
        lat_deg=sgp4_result["lat_deg"],
        lon_deg=sgp4_result["lon_deg"],
        alt_km=sgp4_result["alt_km"],
        orbital_elements=_oe_series(sgp4_result["orbital_elements"]),
    )

    prediction: Optional[TrajectoryData] = None
    comparison: Optional[ComparisonData] = None
    checkpoint_loaded: Optional[str] = None

    # ── Step 2: PI-STN rollout (from same initial state) ────────────
    if ai_available:
        try:
            pistn_result = pistn_svc.run(
                satellite_id=request.satellite_id,
                checkpoint_dir=registry.checkpoint_dir(request.satellite_id),
                norm_json_path=registry.normalization_json(request.satellite_id),
                target_norm_json_path=registry.target_normalization_json(request.satellite_id),
                feature_mapping_json=registry.feature_mapping_json(request.satellite_id),
                initial_state_gcrs=initial_state_gcrs,   # ← EXACT same as SGP4
                initial_elapsed_s=0.0,                    # ← at epoch
                initial_timestamp_utc=epoch_utc,          # ← same epoch
                dt_s=request.step_seconds,
                n_steps=n_steps,
            )

            checkpoint_loaded = str(registry.checkpoint_dir(request.satellite_id) / "best_model.pt")

            prediction = TrajectoryData(
                x_km=pistn_result["x_gcrs_km"],
                y_km=pistn_result["y_gcrs_km"],
                z_km=pistn_result["z_gcrs_km"],
                vx_kms=pistn_result["vx_gcrs_kms"],
                vy_kms=pistn_result["vy_gcrs_kms"],
                vz_kms=pistn_result["vz_gcrs_kms"],
                lat_deg=pistn_result["lat_deg"],
                lon_deg=pistn_result["lon_deg"],
                alt_km=pistn_result["alt_km"],
                orbital_elements=_oe_series(pistn_result["orbital_elements"]),
            )

            # ── Step 3: Evaluation metrics ───────────────────────────
            ref_arr = np.column_stack([
                sgp4_result["x_gcrs_km"], sgp4_result["y_gcrs_km"], sgp4_result["z_gcrs_km"],
                sgp4_result["vx_gcrs_kms"], sgp4_result["vy_gcrs_kms"], sgp4_result["vz_gcrs_kms"],
            ])
            pred_arr = np.column_stack([
                pistn_result["x_gcrs_km"], pistn_result["y_gcrs_km"], pistn_result["z_gcrs_km"],
                pistn_result["vx_gcrs_kms"], pistn_result["vy_gcrs_kms"], pistn_result["vz_gcrs_kms"],
            ])
            elapsed_arr = np.array(elapsed_seconds)

            eval_result = eval_svc.compare(ref_arr, pred_arr, elapsed_arr)
            m = eval_result["metrics"]

            comparison = ComparisonData(
                position_error_km=eval_result["position_error_km"],
                velocity_error_kms=eval_result["velocity_error_kms"],
                err_x_km=eval_result["err_x_km"],
                err_y_km=eval_result["err_y_km"],
                err_z_km=eval_result["err_z_km"],
                err_vx_kms=eval_result["err_vx_kms"],
                err_vy_kms=eval_result["err_vy_kms"],
                err_vz_kms=eval_result["err_vz_kms"],
                metrics=MetricsSummary(
                    position_rmse_km=m.get("position_rmse_km"),
                    position_mae_km=m.get("position_mae_km"),
                    position_max_error_km=m.get("position_max_error_km"),
                    velocity_rmse_kms=m.get("velocity_rmse_kms"),
                    velocity_mae_kms=m.get("velocity_mae_kms"),
                    cumulative_drift_km=m.get("cumulative_drift_km"),
                    error_growth_rate_kmps=m.get("error_growth_rate_kmps"),
                    physics_residual_mean=m.get("physics_residual_mean"),
                    physics_residual_max=m.get("physics_residual_max"),
                    physics_available=m.get("physics_available", False),
                    a_km_mae=m.get("a_km_mae"),
                    a_km_rmse=m.get("a_km_rmse"),
                    e_mae=m.get("e_mae"),
                    e_rmse=m.get("e_rmse"),
                    i_deg_mae=m.get("i_deg_mae"),
                    i_deg_rmse=m.get("i_deg_rmse"),
                ),
            )

        except Exception as exc:
            log.exception("PI-STN rollout failed for %s", request.satellite_id)
            # Graceful degradation — return SGP4 reference only
            log.warning("Returning SGP4-only result due to PI-STN failure: %s", exc)
            ai_available = False

    raw_stations = registry.load_ground_stations(request.satellite_id)
    stations = _parse_stations(raw_stations)

    meta = SimulationMeta(
        satellite_id=request.satellite_id,
        satellite_name=sgp4_result["satellite_name"],
        initial_timestamp_utc=epoch_utc,
        initial_frame="GCRS",
        initial_state_gcrs=initial_state_gcrs,
        delta_t_seconds=request.step_seconds,
        duration_hours=request.duration_hours,
        n_steps=n_steps,
        mode="compare" if ai_available else "sgp4",
        pistn_available=ai_available,
        pistn_rollout_start_utc=epoch_utc if ai_available else None,
        checkpoint_loaded=checkpoint_loaded,
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
        timestamps=timestamps,
        elapsed_seconds=elapsed_seconds,
        reference=ref,
        prediction=prediction,
        comparison=comparison,
        ground_stations=[s.model_dump() for s in stations],
        metadata=meta,
    )
