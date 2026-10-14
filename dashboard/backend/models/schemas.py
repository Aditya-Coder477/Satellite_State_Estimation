"""
dashboard/backend/models/schemas.py
=====================================
Pydantic request/response schemas for the Satellite Orbit Lab API.

One unified SimulationResult feeds all four dashboard views.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class SimulationRequest(BaseModel):
    """Request body for SGP4 simulation, PI-STN prediction, or compare."""
    satellite_id: str = Field(..., description="Satellite registry ID, e.g. 'GOES16'")
    duration_hours: float = Field(24.0, ge=1.0, le=168.0, description="Simulation duration [hours]")
    step_seconds: float = Field(60.0, ge=10.0, le=3600.0, description="Time step [seconds]")
    # Future: start_time_utc: Optional[str] = None  (defaults to OMM epoch)


# ---------------------------------------------------------------------------
# Satellite Info
# ---------------------------------------------------------------------------

class SatelliteInfo(BaseModel):
    id: str
    display_name: str
    operator: str
    orbit_type: str
    norad_id: int
    ai_supported: bool
    checkpoint_available: bool
    description: str
    nominal_longitude_deg: Optional[float] = None


class SatelliteDetailResponse(SatelliteInfo):
    ground_stations: List[Dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Trajectory Data (one frame, used for both reference & prediction)
# ---------------------------------------------------------------------------

class TrajectoryData(BaseModel):
    """Trajectory in GCRS frame (km, km/s)."""
    # Position — GCRS Cartesian [km]
    x_km: List[float]
    y_km: List[float]
    z_km: List[float]
    # Velocity — GCRS Cartesian [km/s]
    vx_kms: List[float]
    vy_kms: List[float]
    vz_kms: List[float]
    # Geographic (from ITRS)
    lat_deg: List[float]
    lon_deg: List[float]
    alt_km: List[float]
    # Orbital elements per step (from GCRS state)
    orbital_elements: Optional[OrbitalElementsSeries] = None


class OrbitalElementsSeries(BaseModel):
    """Per-timestep Keplerian elements."""
    a_km: List[float]     # semi-major axis [km]
    e: List[float]         # eccentricity
    i_deg: List[float]     # inclination [deg]
    raan_deg: List[float]  # RAAN [deg]
    argp_deg: List[float]  # argument of perigee [deg]
    nu_deg: List[float]    # true anomaly [deg]


# ---------------------------------------------------------------------------
# Comparison / Metrics
# ---------------------------------------------------------------------------

class MetricsSummary(BaseModel):
    position_rmse_km: Optional[float] = None
    position_mae_km: Optional[float] = None
    position_max_error_km: Optional[float] = None
    velocity_rmse_kms: Optional[float] = None
    velocity_mae_kms: Optional[float] = None
    cumulative_drift_km: Optional[float] = None
    error_growth_rate_kmps: Optional[float] = None
    physics_residual_mean: Optional[float] = None
    physics_residual_max: Optional[float] = None
    physics_available: bool = False
    # Orbital element drift
    a_km_mae: Optional[float] = None
    a_km_rmse: Optional[float] = None
    e_mae: Optional[float] = None
    e_rmse: Optional[float] = None
    i_deg_mae: Optional[float] = None
    i_deg_rmse: Optional[float] = None


class ComparisonData(BaseModel):
    """Per-timestep error arrays and summary metrics."""
    # Per-step error norms
    position_error_km: List[float]
    velocity_error_kms: List[float]
    # Per-axis errors
    err_x_km: List[float]
    err_y_km: List[float]
    err_z_km: List[float]
    err_vx_kms: List[float]
    err_vy_kms: List[float]
    err_vz_kms: List[float]
    # Summary metrics
    metrics: MetricsSummary


# ---------------------------------------------------------------------------
# Ground Stations
# ---------------------------------------------------------------------------

class GroundStation(BaseModel):
    id: int
    name: str
    lat_deg: float
    lon_deg: float
    height_m: float
    station_code: Optional[str] = None
    country: Optional[str] = None


# ---------------------------------------------------------------------------
# Simulation Metadata
# ---------------------------------------------------------------------------

class SimulationMeta(BaseModel):
    """Records the exact parameters used for this simulation run."""
    satellite_id: str
    satellite_name: str
    initial_timestamp_utc: str           # ISO-8601 (OMM epoch)
    initial_frame: str = "GCRS"          # frame of initial state
    initial_state_gcrs: List[float]      # [x,y,z,vx,vy,vz] km/km/s
    delta_t_seconds: float
    duration_hours: float
    n_steps: int
    mode: str                            # "sgp4" | "pistn" | "compare"
    pistn_available: bool
    pistn_rollout_start_utc: Optional[str] = None
    checkpoint_loaded: Optional[str] = None


# ---------------------------------------------------------------------------
# Main Unified Response — all four views derive from this
# ---------------------------------------------------------------------------

class SimulationResult(BaseModel):
    """
    Single unified response object.
    All four dashboard views (3D Orbit, Ground Track, State Information,
    Prediction Comparison) derive their data from this object.
    """
    satellite: SatelliteInfo
    timestamps: List[str]            # UTC ISO-8601 per step
    elapsed_seconds: List[float]     # seconds from epoch per step

    reference: TrajectoryData        # SGP4 trajectory (GCRS)
    prediction: Optional[TrajectoryData] = None    # PI-STN rollout (GCRS), None if unavailable
    comparison: Optional[ComparisonData] = None    # error analysis, None if no prediction

    ground_stations: List[GroundStation] = []
    metadata: SimulationMeta
