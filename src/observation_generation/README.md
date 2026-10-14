# Observation Generation — `src/observation_generation/`

## Overview

This package implements **Phases 4 & 5** of the Satellite PINN State Estimation project:
**Ground Station Simulation** and **Observation Generation**.

It ingests the ground-truth transformed states from `data/processed/<SAT>/truth_states_transformed.csv`
and produces realistic synthetic optical tracking observations as seen from a configurable network
of ground stations.

---

## Pipeline Architecture

```
truth_states_transformed.csv
          │
          ▼
  StationManager.load()      ← reads data/metadata/ground_stations.json
          │
          ▼
  ObservationGeometry        ← Az, El, Range, Range-Rate (ITRS, vectorized NumPy)
          │
          ▼
  RADecGenerator             ← Topocentric RA/Dec via Astropy GCRS
          │
          ▼
  Scenario mask              ← missing observations, weather outages, sampling
          │
          ▼
  NoiseModelFactory          ← GaussianNoise / UniformNoise / LaplaceNoise
          │
          ▼
  ObservationValidator       ← 8-check validation suite
          │
          ▼
  observations.csv           noisy_observations.csv   observation_summary.json
          │
          ▼
  ObservationVisualizer      ← 9 publication-quality figures
```

---

## Modules

| Module | Purpose |
|---|---|
| `ground_station.py` | `GroundStation` dataclass — geodetic position, ITRS XYZ, ENU rotation matrix |
| `station_manager.py` | Parses `ground_stations.json` → `list[GroundStation]` |
| `line_of_sight.py` | Pure-geometry LOS: range vector, scalar range, unit vector (vectorized) |
| `visibility.py` | `VisibilityCalculator` — Az/El/Range/Range-Rate from ITRS (NumPy matrix multiply) |
| `observation_geometry.py` | Multi-station long-form geometry builder |
| `ra_dec_generator.py` | Topocentric RA/Dec via Astropy GCRS + `EarthLocation.get_gcrs_posvel()` |
| `noise_models.py` | Strategy-pattern noise: `GaussianNoise`, `UniformNoise`, `LaplaceNoise`, `NoiseModelFactory` |
| `observation_generator.py` | Full pipeline orchestrator — calls all stages, writes CSVs + JSON |
| `validator.py` | `ObservationValidator` — 8 checks, collects all failures |
| `visualizer.py` | `ObservationVisualizer` — 9 dark-theme publication figures |
| `run_observation_generation.py` | CLI entry point |

---

## Quick Start

```bash
# From the project root
python -m src.observation_generation.run_observation_generation --sat GOES16

# Specific scenario
python -m src.observation_generation.run_observation_generation --sat GOES16 --scenario missing_5pct

# Run all 5 scenarios
python -m src.observation_generation.run_observation_generation --sat GOES16 --scenario all

# Skip plots (faster, for debugging)
python -m src.observation_generation.run_observation_generation --sat GOES16 --no-plots
```

---

## Observation Scenarios

Scenarios are defined in `config.yaml` under `observation_generation.scenarios`.
Each scenario can independently configure:

| Scenario | Description |
|---|---|
| `perfect` | No noise, no missing observations |
| `gaussian_noise` | Gaussian RA/Dec noise at σ = 0.005° (≈18 arcseconds) |
| `missing_5pct` | Gaussian noise + 5% randomly missing |
| `weather_outage` | Gaussian noise + random 30-minute cloud outage windows |
| `sparse_sampling` | Gaussian noise, 5-minute sampling cadence |

New scenarios can be added to `config.yaml` without any code modifications.

---

## Output Schema

### `observations.csv` / `noisy_observations.csv`

| Column | Type | Description |
|---|---|---|
| `observation_id` | str | Unique ID: `<sim_id>_<SAT>_<index>` |
| `simulation_id` | str | Unique 8-char run identifier |
| `utc_time` | str | ISO-8601 UTC timestamp |
| `elapsed_seconds` | float | Seconds from propagation epoch |
| `satellite_name` | str | e.g. "GOES 16" |
| `norad_id` | int | NORAD catalog number |
| `station_id` | int | Ground station ID |
| `station_name` | str | e.g. "Cape Canaveral" |
| `x_gcrs_km` | float | Satellite GCRS X (km) |
| `y_gcrs_km` | float | Satellite GCRS Y (km) |
| `z_gcrs_km` | float | Satellite GCRS Z (km) |
| `x_itrs_km` | float | Satellite ITRS X (km) |
| `y_itrs_km` | float | Satellite ITRS Y (km) |
| `z_itrs_km` | float | Satellite ITRS Z (km) |
| `elevation_deg` | float | Topocentric elevation [−90, 90] ° |
| `azimuth_deg` | float | Topocentric azimuth [0, 360) ° |
| `range_km` | float | Slant range (km) |
| `range_rate_kms` | float | Radial range rate (km/s) |
| `right_ascension_deg` | float | Topocentric RA [0, 360) ° |
| `declination_deg` | float | Topocentric Dec [−90, 90] ° |
| `visible` | bool | Elevation ≥ min_elevation_deg |
| `station_visible` | bool | Same as visible (pre-mask) |
| `observation_quality` | str | `good` / `below_horizon` / `missing` / `weather_outage` |
| `noise_model` | str | Noise model name |
| `noise_seed` | int | RNG seed |
| `ra_noisy_deg` | float | Noisy RA (noisy CSV only) |
| `dec_noisy_deg` | float | Noisy Dec (noisy CSV only) |

### `observation_summary.json`

Compact summary suitable for experiment tracking and paper tables:

```json
{
  "satellite": "GOES16",
  "simulation_id": "a3f9b2c1",
  "scenario": "gaussian_noise",
  "total_observations": 50405,
  "visible_observations": 42105,
  "mean_elevation_deg": 55.2,
  "noise_model": "gaussian",
  "noise_std_ra_deg": 0.005,
  "validation_passed": true
}
```

---

## Mathematics

### Topocentric Geometry (ITRS)

Satellite ITRS position $\vec{r}_{sat}$ and station ITRS position $\vec{r}_{st}$:

$$\vec{\rho} = \vec{r}_{sat} - \vec{r}_{st}, \quad \text{range} = |\vec{\rho}|, \quad \hat{\rho} = \vec{\rho}/|\vec{\rho}|$$

ENU projection via station rotation matrix $R_{ENU}$:

$$[E,\, N,\, U]^T = R_{ENU}\,\vec{\rho}$$

$$\text{el} = \arcsin(U / |\vec{\rho}|), \quad \text{az} = \operatorname{atan2}(E, N) \bmod 360°$$

Range rate (station-fixed in ITRS, so $\vec{v}_{st} = 0$):

$$\dot{\rho} = \vec{v}_{sat,ITRS} \cdot \hat{\rho}$$

### Topocentric RA/Dec (Astropy GCRS)

```python
obsgeoloc, _ = observer.get_gcrs_posvel(obstime)
topo_gcrs = sat_gcrs.transform_to(GCRS(obstime=t, obsgeoloc=obsgeoloc))
ra_deg, dec_deg = topo_gcrs.ra.deg, topo_gcrs.dec.deg
```

---

## Design Decisions

1. **Separate from Phase 3** — `GroundStation` here is the full domain object (JSON-backed, with `station_id`, `station_code`). Phase 3's `station_frame.py` handles only ENU rotation math and is not imported here.

2. **Vectorized operations** — no per-row Python loops. Geometry is computed via `rho_vec @ R.T` (NumPy broadcasting over N epochs).

3. **Strategy pattern for noise** — the `apply(observations, station, timestamps, config)` interface supports future extensions (station-specific accuracy, time-varying calibration, telescope models).

4. **Scenario-driven configuration** — all five scenarios are defined in `config.yaml`; switching scenarios requires only a CLI flag change.

5. **Observation metadata** — every row carries `observation_id`, `simulation_id`, `noise_model`, `noise_seed`, `observation_quality`, and `station_visible` to support experiment reproducibility and ablation studies.

---

## References

- Vallado, D.A. (2013). *Fundamentals of Astrodynamics and Applications*. Microcosm Press.
- Astropy Collaboration (2022). *The Astropy Project*. ApJ, 935, 167.
- NOAA/NESDIS GOES-16 Mission Description. https://www.goes-r.gov/
