"""
src/observation_generation/observation_generator.py
===================================================
Orchestrates the complete observation generation pipeline.

Pipeline stages
---------------
1. Load transformed truth states from ``truth_states_transformed.csv``.
2. Load ground stations from ``ground_stations.json``.
3. Apply scenario settings (noise flag, missing fraction, outage model,
   sampling interval).
4. Compute visibility geometry (Az, El, Range, Range-Rate) for all stations.
5. Compute topocentric RA / Dec via Astropy GCRS.
6. Enrich DataFrame with Cartesian coordinate columns, observation metadata.
7. Apply scenario-specific missing-observation masks.
8. Apply noise model (strategy pattern) to produce noisy observations.
9. Validate both DataFrames.
10. Persist observations.csv, noisy_observations.csv, observation_summary.json.

Classes
-------
ObservationGenerator
    Main orchestrator class.  Instantiated with project config and a
    satellite key, then driven by calling :meth:`generate`.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.observation_generation.ground_station import GroundStation
from src.observation_generation.noise_models import NoiseModelFactory
from src.observation_generation.observation_geometry import ObservationGeometry
from src.observation_generation.ra_dec_generator import RADecGenerator
from src.observation_generation.station_manager import StationManager
from src.observation_generation.validator import ObservationValidator
from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger

log = get_logger(__name__)

# Columns from truth_states_transformed.csv to carry into observations.csv
_TRUTH_COLS = [
    "utc_time", "elapsed_seconds", "satellite_name", "norad_id",
    "x_gcrs_km", "y_gcrs_km", "z_gcrs_km",
    "x_itrs_km", "y_itrs_km", "z_itrs_km",
]

# Final output column order for observations.csv
_OBS_COLUMNS = [
    # --- Metadata ---
    "observation_id",
    "simulation_id",
    "utc_time",
    "elapsed_seconds",
    "satellite_name",
    "norad_id",
    "station_id",
    "station_name",
    # --- Cartesian coordinates ---
    "x_gcrs_km", "y_gcrs_km", "z_gcrs_km",
    "x_itrs_km", "y_itrs_km", "z_itrs_km",
    # --- Observation geometry ---
    "elevation_deg",
    "azimuth_deg",
    "range_km",
    "range_rate_kms",
    # --- Astronomical ---
    "right_ascension_deg",
    "declination_deg",
    # --- Flags & quality ---
    "visible",
    "station_visible",
    "observation_quality",
    "noise_model",
    "noise_seed",
]

_NOISY_EXTRA_COLS = ["ra_noisy_deg", "dec_noisy_deg"]


class ObservationGenerator:
    """
    Full observation generation pipeline for one satellite.

    Parameters
    ----------
    cfg : dict
        Project configuration loaded by :class:`~src.utils.config_loader.ConfigLoader`.
    satellite_key : str
        Satellite key (e.g. ``"GOES16"``).
    scenario : str, optional
        Scenario name from ``cfg["observation_generation"]["scenarios"]``.
        Defaults to ``"gaussian_noise"``.
    simulation_id : str, optional
        Unique identifier for this simulation run.  Auto-generated if None.
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        satellite_key: str,
        scenario: str = "gaussian_noise",
        simulation_id: Optional[str] = None,
    ) -> None:
        self._cfg           = cfg
        self._satellite_key = satellite_key
        self._scenario_name = scenario
        self._simulation_id = simulation_id or str(uuid.uuid4())[:8]

        self._obs_cfg = cfg.get("observation_generation", {})
        self._root    = Path(cfg["_root"]).resolve()

        # Processed data directory for this satellite
        sat_cfg = cfg["satellites"][satellite_key]
        self._processed_dir = (
            self._root / cfg["paths"]["data"]["processed"]["base"]
            / sat_cfg["processed_dir"]
        )

        # Output directories
        self._results_dir = self._root / cfg["paths"]["results"]["observations"]
        self._log_dir     = self._root / cfg["paths"]["logs"]["observations"]
        self._results_dir.mkdir(parents=True, exist_ok=True)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        # Resolve scenario
        scenarios = self._obs_cfg.get("scenarios", {})
        if scenario not in scenarios:
            raise ValueError(
                f"Scenario '{scenario}' not found in config. "
                f"Available: {sorted(scenarios)}"
            )
        self._scenario = scenarios[scenario]
        log.info(
            "ObservationGenerator init — sat=%s, scenario=%s, sim_id=%s",
            satellite_key, scenario, self._simulation_id,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the complete observation generation pipeline.

        Returns
        -------
        observations : pd.DataFrame
            Clean (noiseless) observations DataFrame.
        noisy_observations : pd.DataFrame
            Observations with RA/Dec noise applied.

        Side Effects
        ------------
        Writes the following files:

        * ``data/processed/<SAT>/observations.csv``
        * ``data/processed/<SAT>/noisy_observations.csv``
        * ``results/observations/observation_summary.json``
        """
        # ---- Stage 1: Load truth states ----
        truth_df = self._load_truth_states()

        # ---- Stage 2: Load stations ----
        stations_json = self._root / self._obs_cfg["ground_stations_json"]
        stations      = StationManager.load(stations_json)
        stations_dict = StationManager.as_dict(stations)

        # ---- Stage 3: Apply sampling interval ----
        truth_df = self._apply_sampling(truth_df)

        # ---- Stage 4: Compute visibility geometry ----
        min_el = self._obs_cfg.get("visibility", {}).get("min_elevation_deg", 10.0)
        geometry_df = ObservationGeometry.compute(truth_df, stations, min_el)

        # ---- Stage 5: Compute RA / Dec ----
        obs_df = RADecGenerator.compute(geometry_df, stations_dict)

        # ---- Stage 6: Enrich with metadata columns ----
        obs_df = self._add_metadata(obs_df)

        # ---- Stage 7: Apply missing-observations mask ----
        obs_df = self._apply_missing_mask(obs_df)

        # ---- Stage 8: Reorder columns ----
        obs_df = self._reorder_columns(obs_df, _OBS_COLUMNS)

        # ---- Stage 9: Build noisy observations ----
        noisy_df = self._apply_noise(obs_df, stations_dict)
        noisy_df = self._reorder_columns(
            noisy_df, _OBS_COLUMNS + _NOISY_EXTRA_COLS
        )

        # ---- Stage 10: Validate ----
        validator = ObservationValidator(self._cfg)
        res_clean = validator.validate(obs_df)
        res_noisy = validator.validate(noisy_df)

        if not res_clean.passed:
            log.warning("Clean observations validation FAILED — check logs.")
        if not res_noisy.passed:
            log.warning("Noisy observations validation FAILED — check logs.")

        # ---- Stage 11: Save ----
        self._save_csv(
            obs_df,
            self._processed_dir / self._obs_cfg["output"]["observations_csv"],
        )
        self._save_csv(
            noisy_df,
            self._processed_dir / self._obs_cfg["output"]["noisy_observations_csv"],
        )
        self._save_summary(obs_df, noisy_df, stations, res_clean)

        return obs_df, noisy_df

    # ------------------------------------------------------------------
    # Private pipeline stages
    # ------------------------------------------------------------------

    def _load_truth_states(self) -> pd.DataFrame:
        """Load truth_states_transformed.csv and validate columns."""
        csv_path = self._processed_dir / self._obs_cfg["input_csv"]
        if not csv_path.is_file():
            raise FileNotFoundError(f"Truth states not found: {csv_path}")
        log.info("Loading truth states: %s", csv_path)
        df = pd.read_csv(csv_path, parse_dates=["utc_time"])
        df["utc_time"] = df["utc_time"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
        log.info("Loaded %d rows × %d columns.", len(df), len(df.columns))
        return df

    def _apply_sampling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Subsample to the scenario sampling interval (seconds)."""
        step_s = self._scenario.get("sampling_interval_seconds", 60)
        base_step = self._cfg["propagation"]["step_seconds"]
        if step_s <= base_step:
            log.debug("Sampling interval %ds matches propagation step — no subsampling.", step_s)
            return df
        factor = round(step_s / base_step)
        sub = df.iloc[::factor].reset_index(drop=True)
        log.info("Subsampled from %d to %d rows (every %d-th row).", len(df), len(sub), factor)
        return sub

    def _add_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add observation_id, simulation_id, noise_model, noise_seed, quality columns."""
        noise_cfg = self._obs_cfg.get("noise", {})

        df = df.copy()
        n  = len(df)

        df["simulation_id"] = self._simulation_id
        df["observation_id"] = [
            f"{self._simulation_id}_{self._satellite_key}_{i:06d}"
            for i in range(n)
        ]
        df["noise_model"]        = noise_cfg.get("model", "none")
        df["noise_seed"]         = int(noise_cfg.get("seed", 42))
        df["station_visible"]    = df["visible"]
        df["observation_quality"] = np.where(
            df["visible"], "good", "below_horizon"
        )
        return df

    def _apply_missing_mask(self, df: pd.DataFrame) -> pd.DataFrame:
        """Randomly drop observations and mark outage windows per scenario."""
        df = df.copy()
        scenario = self._scenario

        # Random missing fraction
        missing_fraction = float(scenario.get("missing_fraction", 0.0))
        if missing_fraction > 0.0:
            seed = int(scenario.get("missing_seed", 999))
            rng  = np.random.default_rng(seed)
            mask = rng.random(len(df)) < missing_fraction
            df.loc[mask, "visible"]             = False
            df.loc[mask, "observation_quality"] = "missing"
            log.info("Applied %.1f%% random missing mask (%d rows).", missing_fraction * 100, int(mask.sum()))

        # Weather outage windows
        if scenario.get("outage_enabled", False):
            df = self._apply_outages(df, scenario)

        return df

    def _apply_outages(self, df: pd.DataFrame, scenario: dict) -> pd.DataFrame:
        """Simulate random weather outage windows per station."""
        outage_prob = float(scenario.get("outage_probability", 0.10))
        outage_dur  = int(scenario.get("outage_duration_min", 30))
        seed        = int(scenario.get("outage_seed", 456))
        rng         = np.random.default_rng(seed)

        step_s   = self._scenario.get("sampling_interval_seconds", 60)
        dur_rows = max(1, (outage_dur * 60) // step_s)

        total_outage = 0
        for sid in df["station_id"].unique():
            mask = df["station_id"] == sid
            n_st = int(mask.sum())
            idx  = df.index[mask].tolist()

            i = 0
            while i < n_st:
                if rng.random() < outage_prob:
                    end_i = min(i + dur_rows, n_st)
                    bad_idx = idx[i:end_i]
                    df.loc[bad_idx, "visible"]             = False
                    df.loc[bad_idx, "observation_quality"] = "weather_outage"
                    total_outage += len(bad_idx)
                    i = end_i
                else:
                    i += 1

        log.info("Weather outage mask applied: %d rows marked as outage.", total_outage)
        return df

    def _apply_noise(
        self,
        obs_df: pd.DataFrame,
        stations_dict: Dict[int, GroundStation],
    ) -> pd.DataFrame:
        """Apply noise model to each station's observations independently."""
        noise_cfg    = self._obs_cfg.get("noise", {})
        scenario     = self._scenario
        noise_enabled = scenario.get("noise_enabled", True)

        if not noise_enabled:
            log.info("Noise disabled for scenario '%s' — copying observations.", self._scenario_name)
            noisy = obs_df.copy()
            noisy["ra_noisy_deg"]  = noisy["right_ascension_deg"]
            noisy["dec_noisy_deg"] = noisy["declination_deg"]
            return noisy

        model_name   = scenario.get("noise_model", noise_cfg.get("model", "gaussian"))
        ra_std_deg   = float(noise_cfg.get("ra_std_deg",  0.005))
        dec_std_deg  = float(noise_cfg.get("dec_std_deg", 0.005))
        seed         = int(noise_cfg.get("seed", 42))

        noise_model  = NoiseModelFactory.create(
            model_name, ra_std_deg=ra_std_deg, dec_std_deg=dec_std_deg, seed=seed
        )

        frames: List[pd.DataFrame] = []
        for sid in obs_df["station_id"].unique():
            station_mask = obs_df["station_id"] == sid
            sub = obs_df.loc[station_mask].copy()
            station = stations_dict.get(int(sid))
            ts = sub["utc_time"].to_numpy()
            sub = noise_model.apply(sub, station, ts, noise_cfg)
            frames.append(sub)

        result = pd.concat(frames).sort_index()
        log.info("Noise applied — model=%s, σ_RA=%.5f°, σ_Dec=%.5f°",
                 model_name, ra_std_deg, dec_std_deg)
        return result

    def _reorder_columns(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Keep only known columns in the specified order, plus any extras."""
        ordered    = [c for c in columns if c in df.columns]
        remaining  = [c for c in df.columns if c not in ordered]
        return df[ordered + remaining]

    def _save_csv(self, df: pd.DataFrame, path: Path) -> None:
        """Persist a DataFrame to CSV with logging."""
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        size_mb = path.stat().st_size / 1_048_576
        log.info("Saved %d rows → %s (%.2f MB)", len(df), path, size_mb)

    def _save_summary(
        self,
        obs_df: pd.DataFrame,
        noisy_df: pd.DataFrame,
        stations: List[GroundStation],
        validation: Any,
    ) -> None:
        """Write observation_summary.json to results/observations/."""
        noise_cfg = self._obs_cfg.get("noise", {})
        scenario  = self._scenario

        summary = {
            "satellite":            self._satellite_key,
            "simulation_id":        self._simulation_id,
            "scenario":             self._scenario_name,
            "scenario_description": scenario.get("description", ""),
            "generated_at_utc":     datetime.now(timezone.utc).isoformat(),
            "stations":             len(stations),
            "station_names":        [st.name for st in stations],
            "total_observations":   len(obs_df),
            "visible_observations": int(obs_df.get("visible", pd.Series(dtype=bool)).sum()),
            "hidden_observations":  int((~obs_df.get("visible", pd.Series(dtype=bool))).sum()),
            "mean_elevation_deg":   float(obs_df["elevation_deg"].mean()) if "elevation_deg" in obs_df else None,
            "max_elevation_deg":    float(obs_df["elevation_deg"].max()) if "elevation_deg" in obs_df else None,
            "min_elevation_deg":    float(obs_df["elevation_deg"].min()) if "elevation_deg" in obs_df else None,
            "mean_range_km":        float(obs_df["range_km"].mean()) if "range_km" in obs_df else None,
            "noise_model":          noise_cfg.get("model", "none"),
            "noise_enabled":        scenario.get("noise_enabled", True),
            "noise_std_ra_deg":     noise_cfg.get("ra_std_deg", 0.005),
            "noise_std_dec_deg":    noise_cfg.get("dec_std_deg", 0.005),
            "noise_seed":           noise_cfg.get("seed", 42),
            "missing_fraction":     scenario.get("missing_fraction", 0.0),
            "outage_enabled":       scenario.get("outage_enabled", False),
            "validation_passed":    validation.passed,
            "validation_errors":    validation.errors,
            "validation_warnings":  validation.warnings,
        }

        out_path = self._results_dir / self._obs_cfg["output"]["summary_json"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=str)

        log.info("Summary written → %s", out_path)

        # Also write a satellite-specific copy alongside the CSVs
        local_copy = self._processed_dir / self._obs_cfg["output"]["summary_json"]
        with local_copy.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=str)

        # Print human-readable summary to console
        vis  = summary["visible_observations"]
        tot  = summary["total_observations"]
        log.info(
            "\n"
            "=" * 60 + "\n"
            "  Observation Generation Summary\n"
            "=" * 60 + "\n"
            "  Satellite    : %s\n"
            "  Scenario     : %s\n"
            "  Stations     : %d\n"
            "  Total rows   : %d\n"
            "  Visible      : %d (%.1f%%)\n"
            "  Hidden       : %d (%.1f%%)\n"
            "  Mean El (°)  : %.2f\n"
            "  Max El  (°)  : %.2f\n"
            "  Noise model  : %s\n"
            "  σ_RA (°)     : %.5f\n"
            "  σ_Dec (°)    : %.5f\n"
            "=" * 60,
            self._satellite_key,
            self._scenario_name,
            summary["stations"],
            tot,
            vis,  100.0 * vis / max(tot, 1),
            summary["hidden_observations"], 100.0 * summary["hidden_observations"] / max(tot, 1),
            summary["mean_elevation_deg"] or 0.0,
            summary["max_elevation_deg"] or 0.0,
            summary["noise_model"],
            summary["noise_std_ra_deg"],
            summary["noise_std_dec_deg"],
        )
