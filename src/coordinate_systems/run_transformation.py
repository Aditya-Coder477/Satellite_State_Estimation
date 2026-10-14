"""
src/coordinate_systems/run_transformation.py
============================================
Entry-point script for **Phase 3: Coordinate Transformation Module**.

Usage
-----
From the project root:

    python -m src.coordinate_systems.run_transformation              # Defaults to GOES16
    python -m src.coordinate_systems.run_transformation --sat GOES18
    python -m src.coordinate_systems.run_transformation --sat INSAT3D --config config.yaml

This script orchestrates the pipeline:
1. Load config and resolve paths.
2. Setup logging to logs/coordinate_transforms/coordinate_transform.log.
3. Read data/processed/<SAT>/truth_states.csv.
4. Run coordinate transformations (TEME -> GCRS -> ITRS) and add JD/MJD time columns.
5. Validate transformed states (schema, ranges, continuity, norm-preservation).
6. Save truth_states_transformed.csv.
7. Generate visualization plots in results/coordinate_transforms/.
8. Write transformation summary to log.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add source root to path if needed (though running with -m handles this)
_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from src.coordinate_systems.coordinate_transformer import CoordinateTransformer
from src.coordinate_systems.validator import CoordinateTransformationValidator
from src.coordinate_systems.visualizer import CoordinateVisualizer
from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger, setup_logging
from src.utils.path_manager import PathManager


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="run_transformation",
        description="Coordinate transformations (TEME -> GCRS -> ITRS) for Satellite PINN State Estimation.",
    )
    parser.add_argument(
        "--sat",
        type=str,
        default="GOES16",
        metavar="SATELLITE_KEY",
        help="Satellite key from config.yaml (e.g. GOES16, GOES18, INSAT3D). Default: GOES16",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to config.yaml. Auto-discovered if not given.",
    )
    return parser.parse_args()


def run(satellite_key: str, config_path: str | None = None) -> int:
    """
    Execute the coordinate transformation pipeline for a single satellite.

    Parameters
    ----------
    satellite_key : str
        Key into cfg["satellites"] (e.g. "GOES16").
    config_path : str | None
        Optional explicit path to config.yaml.

    Returns
    -------
    int
        Exit code: 0 = success, 1 = failure.
    """
    # 1. Load config
    cfg: dict[str, Any] = ConfigLoader.load(config_path)

    # 2. Resolve paths using PathManager & config overrides
    try:
        pm = PathManager(cfg, satellite_key)
    except (KeyError, FileNotFoundError) as exc:
        import logging
        logging.basicConfig(level=logging.ERROR)
        logging.getLogger(__name__).error("PathManager initialization failed: %s", exc)
        return 1

    # Check for Phase 4 config block
    coord_cfg = cfg.get("coordinate_transformation", {})
    input_filename = coord_cfg.get("input_file", "truth_states.csv")
    output_filename = coord_cfg.get("output_file", "truth_states_transformed.csv")

    input_csv = pm.processed_dir / input_filename
    output_csv = pm.processed_dir / output_filename

    # Resolve logs and results directories, isolated by satellite_key
    paths_cfg = cfg["paths"]
    log_dir = pm.root / paths_cfg["logs"]["coordinate_transforms"] / satellite_key
    results_dir = pm.root / paths_cfg["results"]["coordinate_transforms"] / satellite_key

    log_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "transformation.log"

    # 3. Setup logging (log file + console)
    setup_logging(cfg, log_file=log_file)
    log = get_logger(__name__)

    log.info("=" * 80)
    log.info("Satellite PINN State Estimation — Phase 4: Coordinate Transformations")
    log.info("=" * 80)
    log.info("Satellite key : %s", satellite_key)
    log.info("Input CSV     : %s", input_csv)
    log.info("Output CSV    : %s", output_csv)
    log.info("Log file      : %s", log_file)
    log.info("Results dir   : %s", results_dir)

    # Verify input exists
    if not input_csv.is_file():
        log.error("Input file does not exist: %s. Please run prior phases first.", input_csv)
        return 1

    wall_start = time.perf_counter()
    status = "FAILED"

    try:
        # Step 1: Read input CSV
        log.info("-" * 60)
        log.info("STEP 1/4: Read input states")
        df_in = pd.read_csv(input_csv)
        log.info("  Loaded %d rows from %s", len(df_in), input_csv.name)

        # Step 2: Transform Coordinates
        log.info("-" * 60)
        log.info("STEP 2/4: Transform Coordinate Frames (TEME -> GCRS -> ITRS)")
        
        # Astropy transforms expect x_teme_km, etc. Let's make sure they are present
        if "x_teme_km" not in df_in.columns and "x_km" in df_in.columns:
            # Map input columns if they are named x_km instead of x_teme_km
            df_in["x_teme_km"] = df_in["x_km"]
            df_in["y_teme_km"] = df_in["y_km"]
            df_in["z_teme_km"] = df_in["z_km"]
            df_in["vx_teme_kms"] = df_in["vx_km_s"]
            df_in["vy_teme_kms"] = df_in["vy_km_s"]
            df_in["vz_teme_kms"] = df_in["vz_km_s"]

        df_out = CoordinateTransformer.transform_dataframe(df_in)

        # Step 3: Validate Transformations
        log.info("-" * 60)
        log.info("STEP 3/4: Validate Transformed Dataset")
        validator = CoordinateTransformationValidator(cfg)
        val_result = validator.validate(df_out, df_in=df_in)

        if not val_result.passed:
            log.error("Validation FAILED. Aborting output generation.")
            return 1

        # Save output CSV
        df_out.to_csv(output_csv, index=False, float_format="%.10f")
        log.info("  Saved transformed dataset to: %s", output_csv)

        # Step 4: Generate Visualization Plots
        log.info("-" * 60)
        log.info("STEP 4/4: Generate Visualizations")
        visualizer = CoordinateVisualizer(cfg, results_dir=results_dir)
        visualizer.plot_all(df_out, satellite_name=df_out["satellite_name"].iloc[0])

        wall_elapsed = time.perf_counter() - wall_start
        status = "SUCCESS"

        # Calculate coordinate differences for summary JSON
        dx_ti = df_out["x_teme_km"].to_numpy() - df_out["x_itrs_km"].to_numpy()
        dy_ti = df_out["y_teme_km"].to_numpy() - df_out["y_itrs_km"].to_numpy()
        dz_ti = df_out["z_teme_km"].to_numpy() - df_out["z_itrs_km"].to_numpy()
        diff_mag = np.sqrt(dx_ti**2 + dy_ti**2 + dz_ti**2)

        # Write transformation_summary.json
        summary_data = {
            "Input frame": coord_cfg.get("frames", {}).get("input", "TEME"),
            "Output frames": [
                coord_cfg.get("frames", {}).get("intermediate", "GCRS"),
                coord_cfg.get("frames", {}).get("output", "ITRS")
            ],
            "Number of samples": int(len(df_out)),
            "Transformation success": True,
            "Maximum coordinate difference": float(diff_mag.max()),
            "Average coordinate difference": float(diff_mag.mean()),
            "Execution time": float(wall_elapsed),
        }
        summary_path = pm.processed_dir / "transformation_summary.json"
        with open(summary_path, "w", encoding="utf-8") as fh:
            import json
            json.dump(summary_data, fh, indent=2)
        log.info("  Saved summary to: %s", summary_path)

        # Write frame_statistics.json
        stats_data = {}
        for frame in ["teme", "gcrs", "itrs"]:
            r = np.sqrt(df_out[f"x_{frame}_km"]**2 + df_out[f"y_{frame}_km"]**2 + df_out[f"z_{frame}_km"]**2)
            v = np.sqrt(df_out[f"vx_{frame}_kms"]**2 + df_out[f"vy_{frame}_kms"]**2 + df_out[f"vz_{frame}_kms"]**2)
            stats_data[frame.upper()] = {
                "position_min_km": float(r.min()),
                "position_max_km": float(r.max()),
                "position_mean_km": float(r.mean()),
                "position_std_km": float(r.std()),
                "velocity_min_kms": float(v.min()),
                "velocity_max_kms": float(v.max()),
                "velocity_mean_kms": float(v.mean()),
                "velocity_std_kms": float(v.std()),
            }
        stats_path = pm.processed_dir / "frame_statistics.json"
        with open(stats_path, "w", encoding="utf-8") as fh:
            json.dump(stats_data, fh, indent=2)
        log.info("  Saved frame stats to: %s", stats_path)

        # Write transformation_config_used.json
        config_path_used = pm.processed_dir / "transformation_config_used.json"
        with open(config_path_used, "w", encoding="utf-8") as fh:
            json.dump(coord_cfg, fh, indent=2)
        log.info("  Saved config snapshot to: %s", config_path_used)

        # Log final summary block
        _write_log_summary(
            log,
            satellite_name=df_out["satellite_name"].iloc[0],
            n_samples=len(df_out),
            wall_elapsed=wall_elapsed,
            status=status,
            val_result=val_result,
        )
        
        log.info("=" * 80)
        log.info("Coordinate Transformation COMPLETE in %.3f s", wall_elapsed)
        log.info("=" * 80)
        return 0

    except Exception as exc:
        log.exception("Unexpected error during coordinate transformation pipeline: %s", exc)
        return 1


def _write_log_summary(
    log: Any,
    satellite_name: str,
    n_samples: int,
    wall_elapsed: float,
    status: str,
    val_result: Any,
) -> None:
    """Emit a structured summary block to the log."""
    log.info("-" * 60)
    log.info("COORDINATE TRANSFORMATION PIPELINE SUMMARY")
    log.info("  Satellite name     : %s", satellite_name)
    log.info("  Number of samples  : %d", n_samples)
    log.info("  Frames converted   : TEME -> GCRS -> ITRS")
    log.info("  Execution time     : %.3f s", wall_elapsed)
    log.info("  Validation status  : %s", status)
    if val_result.stats:
        log.info("  Mean GCRS Altitude : %.2f km", val_result.stats.get("mean_r_gcrs", float("nan")))
        log.info("  Mean ITRS Altitude : %.2f km", val_result.stats.get("mean_r_itrs", float("nan")))
        log.info("  Max Position Diff  : %.3f m (preservation diff)", val_result.stats.get("max_norm_diff_gcrs_m", float("nan")))
    log.info("-" * 60)


if __name__ == "__main__":
    args = parse_args()
    sys.exit(run(satellite_key=args.sat, config_path=args.config))
