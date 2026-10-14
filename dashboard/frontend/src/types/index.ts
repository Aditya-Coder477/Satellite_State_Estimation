// src/types/index.ts
// Complete TypeScript type definitions for the Satellite Orbit Lab.
// Mirrors the Pydantic schemas in dashboard/backend/models/schemas.py.

export interface SatelliteInfo {
  id: string;
  display_name: string;
  operator: string;
  orbit_type: string;
  norad_id: number;
  ai_supported: boolean;
  checkpoint_available: boolean;
  description: string;
  nominal_longitude_deg: number | null;
}

export interface GroundStation {
  id: number;
  name: string;
  lat_deg: number;
  lon_deg: number;
  height_m: number;
  station_code?: string;
  country?: string;
}

export interface OrbitalElementsSeries {
  a_km: (number | null)[];
  e: (number | null)[];
  i_deg: (number | null)[];
  raan_deg: (number | null)[];
  argp_deg: (number | null)[];
  nu_deg: (number | null)[];
}

/** Trajectory in GCRS frame (km, km/s) + geographic lat/lon/alt */
export interface TrajectoryData {
  x_km: number[];
  y_km: number[];
  z_km: number[];
  vx_kms: number[];
  vy_kms: number[];
  vz_kms: number[];
  lat_deg: number[];
  lon_deg: number[];
  alt_km: number[];
  orbital_elements?: OrbitalElementsSeries;
}

export interface MetricsSummary {
  position_rmse_km: number | null;
  position_mae_km: number | null;
  position_max_error_km: number | null;
  velocity_rmse_kms: number | null;
  velocity_mae_kms: number | null;
  cumulative_drift_km: number | null;
  error_growth_rate_kmps: number | null;
  physics_residual_mean: number | null;
  physics_residual_max: number | null;
  physics_available: boolean;
  a_km_mae: number | null;
  a_km_rmse: number | null;
  e_mae: number | null;
  e_rmse: number | null;
  i_deg_mae: number | null;
  i_deg_rmse: number | null;
}

export interface ComparisonData {
  position_error_km: number[];
  velocity_error_kms: number[];
  err_x_km: number[];
  err_y_km: number[];
  err_z_km: number[];
  err_vx_kms: number[];
  err_vy_kms: number[];
  err_vz_kms: number[];
  metrics: MetricsSummary;
}

export interface SimulationMeta {
  satellite_id: string;
  satellite_name: string;
  initial_timestamp_utc: string;
  initial_frame: string;
  initial_state_gcrs: number[];
  delta_t_seconds: number;
  duration_hours: number;
  n_steps: number;
  mode: 'sgp4' | 'pistn' | 'compare';
  pistn_available: boolean;
  pistn_rollout_start_utc: string | null;
  checkpoint_loaded: string | null;
}

/** One unified simulation result — all 4 views derive from this */
export interface SimulationResult {
  satellite: SatelliteInfo;
  timestamps: string[];
  elapsed_seconds: number[];
  reference: TrajectoryData;
  prediction: TrajectoryData | null;
  comparison: ComparisonData | null;
  ground_stations: GroundStation[];
  metadata: SimulationMeta;
}

export interface SimulationRequest {
  satellite_id: string;
  duration_hours: number;
  step_seconds: number;
}

export type DashboardMode = 'explorer' | 'research';
export type SimulationMode = 'sgp4' | 'pistn' | 'compare';
