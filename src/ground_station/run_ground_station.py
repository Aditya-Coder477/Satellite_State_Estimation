"""
src/ground_station/run_ground_station.py
=========================================
Entry-point script for **Phase 5: Ground Station Simulation**.

Usage
-----
From the project root::

    python -m src.ground_station.run_ground_station              # defaults to INSAT3D
    python -m src.ground_station.run_ground_station --sat INSAT3D

Pipeline Steps
--------------
1. Load configuration and resolve paths.
2. Load ground stations from ground_stations.json.
3. Load transformed satellite trajectory (truth_states_transformed.csv from Phase 4).
4. Perform vectorized satellite-to-station geometry computations (azimuth, elevation, range, range-rate).
5. Run validations on the computed geometry.
6. Persist deliverables (ground_station_geometry.csv, station_passes.csv, visibility_summary.json, station_statistics.json).
7. Generate 9 visualization plots.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── Make src importable when run as a script from project root ─────────
_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from src.ground_station.station_manager import GroundStationManager
from src.ground_station.station_geometry import StationGeometryEngine
from src.ground_station.validator import GroundStationValidator
from src.ground_station.statistics import GroundStationStatistics
from src.ground_station.visualizer import GroundStationVisualizer
from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger, setup_logging
from src.utils.path_manager import PathManager


# ---------------------------------------------------------------------------
# Path resolution helpers
# ---------------------------------------------------------------------------

def _resolve_paths(cfg: dict[str, Any], satellite_key: str) -> dict[str, Path]:
    """Resolve all Phase-5 filesystem paths from config."""
    root = Path(cfg["_root"])
    processed_base = root / cfg["paths"]["data"]["processed"]["base"]
    sat_dir = processed_base / cfg["satellites"][satellite_key]["processed_dir"]
    metadata_dir = root / cfg["paths"]["data"]["metadata"] / satellite_key

    gs_cfg = cfg.get("ground_station", {})

    return {
        "input_csv": sat_dir / gs_cfg.get("input_file", "truth_states_transformed.csv"),
        "station_json": metadata_dir / gs_cfg.get("station_file", "ground_stations.json"),
        "processed_dir": sat_dir,
        "output_csv": sat_dir / gs_cfg.get("output_file", "ground_station_geometry.csv"),
        "geometry_config_used": sat_dir / "geometry_config_used.json",
        "log_dir": root / "logs" / "ground_stations" / satellite_key,
        "results_dir": root / "results" / "ground_stations" / satellite_key,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(satellite_key: str, config_path: str | None = None) -> int:
    """
    Execute the ground station simulation pipeline.

    Returns 0 on success, 1 on failure.
    """
    # ── 1. Load config ─────────────────────────────────────────────────
    cfg: dict[str, Any] = ConfigLoader.load(config_path)

    if satellite_key not in cfg.get("satellites", {}):
        import logging
        logging.basicConfig(level=logging.ERROR)
        logging.getLogger(__name__).error(
            "Satellite '%s' not found in config. Available: %s",
            satellite_key, list(cfg.get("satellites", {}).keys()),
        )
        return 1

    # ── 2. Resolve paths ───────────────────────────────────────────────
    paths = _resolve_paths(cfg, satellite_key)

    # ── 3. Setup logging ───────────────────────────────────────────────
    log_file = paths["log_dir"] / "ground_station.log"
    setup_logging(cfg, log_file=log_file)
    log = get_logger(__name__)

    log.info("=" * 80)
    log.info("Satellite PINN State Estimation — Phase 5: Ground Station Simulation")
    log.info("=" * 80)
    log.info("Satellite    : %s", satellite_key)
    log.info("Input CSV    : %s", paths["input_csv"])
    log.info("Stations JSON: %s", paths["station_json"])
    log.info("Output CSV   : %s", paths["output_csv"])
    log.info("Log file     : %s", log_file)
    log.info("Results dir  : %s", paths["results_dir"])

    wall_start = time.perf_counter()

    try:
        # ── 4. Load Ground Stations ────────────────────────────────────
        log.info("-" * 60)
        log.info("STEP 1/6: Load Ground Stations configuration")
        
        min_elev = float(cfg.get("ground_station", {}).get("minimum_elevation_deg", 10.0))
        manager = GroundStationManager(paths["station_json"], default_min_elevation=min_elev)
        stations = manager.stations

        # Validate station metadata
        validator = GroundStationValidator(cfg)
        meta_val = validator.validate_metadata(stations)
        if not meta_val.passed:
            log.error("Ground station metadata validation failed. Aborting.")
            return 1

        # ── 5. Load Satellite Trajectory ───────────────────────────────
        log.info("-" * 60)
        log.info("STEP 2/6: Load transformed trajectory")
        if not paths["input_csv"].is_file():
            log.error("Input CSV not found: %s. Please run Phase 4 first.", paths["input_csv"])
            return 1

        df_transformed = pd.read_csv(paths["input_csv"])
        n_samples = len(df_transformed)
        log.info("  Loaded %d states from %s", n_samples, paths["input_csv"].name)

        # ── 6. Perform Vectorized Geometry Computations ────────────────
        log.info("-" * 60)
        log.info("STEP 3/6: Compute satellite-station geometry (vectorized)")
        
        # Extract ITRS position & velocity arrays
        sat_pos = df_transformed[["x_itrs_km", "y_itrs_km", "z_itrs_km"]].to_numpy(dtype=np.float64)
        sat_vel = df_transformed[["vx_itrs_kms", "vy_itrs_kms", "vz_itrs_kms"]].to_numpy(dtype=np.float64)
        times = df_transformed["elapsed_seconds"].to_numpy(dtype=np.float64)
        utc_times = df_transformed["utc_time"].to_numpy()

        # Compute geometry for all stations and concatenate
        station_dfs = []
        for s in stations:
            df_s = StationGeometryEngine.compute_geometry(s, sat_pos, sat_vel, times, utc_times)
            station_dfs.append(df_s)

        df_geom = pd.concat(station_dfs, ignore_index=True)

        # ── 7. Validate Geometry ───────────────────────────────────────
        log.info("-" * 60)
        log.info("STEP 4/6: Validate computed geometry dataset")
        geom_val = validator.validate_geometry(df_geom, stations)
        if not geom_val.passed:
            log.error("Computed geometry validation failed. Aborting save.")
            return 1

        # ── 8. Persist Outputs ─────────────────────────────────────────
        log.info("-" * 60)
        log.info("STEP 5/6: Save geometry CSV & statistics JSONs")
        
        # Save geometry CSV
        df_geom.to_csv(paths["output_csv"], index=False, float_format="%.10f")
        log.info("  Saved geometry data to: %s", paths["output_csv"])

        # Calculate passes and write stats
        stats_engine = GroundStationStatistics(cfg, output_dir=paths["processed_dir"])
        stats_engine.process(df_geom, stations)

        # Save config copy
        with open(paths["geometry_config_used"], "w", encoding="utf-8") as fh:
            json.dump(cfg.get("ground_station", {}), fh, indent=2)
        log.info("  Saved config snapshot to: %s", paths["geometry_config_used"])

        # ── 9. Generate Visualizations ──────────────────────────────────
        log.info("-" * 60)
        log.info("STEP 6/6: Generate visualisations (9 figures)")
        visualizer = GroundStationVisualizer(cfg, results_dir=paths["results_dir"])
        visualizer.plot_all(df_geom, df_transformed, stations, satellite_name="INSAT-3D")

        wall_elapsed = time.perf_counter() - wall_start
        log.info("=" * 80)
        log.info("Ground Station Simulation COMPLETE in %.3f s", wall_elapsed)
        log.info("=" * 80)
        return 0

    except Exception as exc:
        log.exception("Unexpected error during Ground Station Simulation pipeline: %s", exc)
        return 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_ground_station",
        description="Phase 5: Ground Station Simulation for Satellite PINN State Estimation.",
    )
    parser.add_argument(
        "--sat",
        type=str,
        default="INSAT3D",
        metavar="SATELLITE_KEY",
        help="Satellite key from config.yaml (e.g. INSAT3D). Default: INSAT3D",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to config.yaml. Auto-discovered if not given.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.exit(run(satellite_key=args.sat, config_path=args.config))


if __name__ == "__main__":
    main()
