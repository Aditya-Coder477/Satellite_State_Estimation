"""
src/observation_generation/run_observation_generation.py
=========================================================
Entry-point script for **Phase 6: Observation Generation**.

Usage
-----
From the project root::

    python -m src.observation_generation.run_observation_generation              # defaults to INSAT3D
    python -m src.observation_generation.run_observation_generation --sat INSAT3D
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

from src.observation_generation.scenario_generator import ScenarioGenerator
from src.observation_generation.validator import ObservationValidator
from src.observation_generation.statistics import ObservationStatistics
from src.observation_generation.visualizer import ObservationVisualizer
from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger, setup_logging
from src.utils.path_manager import PathManager


# ---------------------------------------------------------------------------
# Path resolution helper
# ---------------------------------------------------------------------------

def _resolve_paths(cfg: dict[str, Any], satellite_key: str) -> dict[str, Path]:
    """Resolve all Phase-6 filesystem paths from config."""
    root = Path(cfg["_root"])
    processed_base = root / cfg["paths"]["data"]["processed"]["base"]
    sat_dir = processed_base / cfg["satellites"][satellite_key]["processed_dir"]

    ob_cfg = cfg.get("observation_generation", {})

    return {
        "input_geometry": sat_dir / ob_cfg.get("input_geometry", "ground_station_geometry.csv"),
        "input_states": sat_dir / ob_cfg.get("input_states", "truth_states_transformed.csv"),
        "processed_dir": sat_dir,
        "output_clean": sat_dir / ob_cfg.get("output_clean", "observations.csv"),
        "output_noisy": sat_dir / ob_cfg.get("output_noisy", "noisy_observations.csv"),
        "quality_csv": sat_dir / "observation_quality.csv",
        "visibility_summary_csv": sat_dir / "station_visibility_summary.csv",
        "config_used": sat_dir / "observation_config_used.json",
        "log_dir": root / "logs" / "observations" / satellite_key,
        "results_dir": root / "results" / "observations" / satellite_key,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(satellite_key: str, config_path: str | None = None) -> int:
    """
    Execute the Phase 6 observation generation pipeline.

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
    log_file = paths["log_dir"] / "observation_generation.log"
    setup_logging(cfg, log_file=log_file)
    log = get_logger(__name__)

    log.info("=" * 80)
    log.info("Satellite PINN State Estimation — Phase 6: Observation Generation")
    log.info("=" * 80)
    log.info("Satellite        : %s", satellite_key)
    log.info("Geometry Input   : %s", paths["input_geometry"])
    log.info("States Input     : %s", paths["input_states"])
    log.info("Clean Output     : %s", paths["output_clean"])
    log.info("Noisy Output     : %s", paths["output_noisy"])
    log.info("Log file         : %s", log_file)
    log.info("Results dir      : %s", paths["results_dir"])

    wall_start = time.perf_counter()

    try:
        # ── 4. Verify inputs exist ─────────────────────────────────────
        log.info("-" * 60)
        log.info("STEP 1/5: Load inputs")
        if not paths["input_geometry"].is_file():
            log.error("Geometry input file not found: %s. Please run Phase 5 first.", paths["input_geometry"])
            return 1
        if not paths["input_states"].is_file():
            log.error("Transformed states input file not found: %s. Please run Phase 4 first.", paths["input_states"])
            return 1

        df_geom = pd.read_csv(paths["input_geometry"])
        df_states = pd.read_csv(paths["input_states"])
        log.info(
            "  Loaded %d geometry rows and %d state rows.",
            len(df_geom), len(df_states)
        )

        # ── 5. Generate Scenarios ──────────────────────────────────────
        log.info("-" * 60)
        log.info("STEP 2/5: Generate Scenarios (Scenarios A to F)")
        generator = ScenarioGenerator(cfg)
        df_clean, df_noisy, df_quality, df_vis_summary = generator.generate_all(df_geom, df_states)

        if len(df_clean) == 0:
            log.error("Observation generation failed — no data generated.")
            return 1

        # ── 6. Validate Datasets ───────────────────────────────────────
        log.info("-" * 60)
        log.info("STEP 3/5: Validate generated observations")
        validator = ObservationValidator(cfg)
        val_res = validator.validate(df_clean, df_noisy, df_quality)
        if not val_res.passed:
            log.error("Generated observations validation failed. Aborting save.")
            return 1

        # ── 7. Save Deliverables ───────────────────────────────────────
        log.info("-" * 60)
        log.info("STEP 4/5: Save CSVs & JSONs")
        
        # Save CSVs
        df_clean.to_csv(paths["output_clean"], index=False, float_format="%.10f")
        df_noisy.to_csv(paths["output_noisy"], index=False, float_format="%.10f")
        df_quality.to_csv(paths["quality_csv"], index=False, float_format="%.10f")
        df_vis_summary.to_csv(paths["visibility_summary_csv"], index=False, float_format="%.4f")
        
        log.info("  Saved clean observations: %d rows", len(df_clean))
        log.info("  Saved noisy observations: %d rows", len(df_noisy))
        log.info("  Saved observation quality: %d rows", len(df_quality))
        log.info("  Saved visibility summary: %d rows", len(df_vis_summary))

        # Save stats and summary JSONs
        stats_engine = ObservationStatistics(cfg, output_dir=paths["processed_dir"])
        stats_engine.compute_and_save(df_clean, df_noisy, df_quality, val_stats=val_res.stats)

        # Save config copy
        with open(paths["config_used"], "w", encoding="utf-8") as fh:
            json.dump(cfg.get("observation_generation", {}), fh, indent=2)
        log.info("  Saved config snapshot to: %s", paths["config_used"])

        # ── 8. Generate Visualizations ──────────────────────────────────
        log.info("-" * 60)
        log.info("STEP 5/5: Generate visualisations (10 figures)")
        visualizer = ObservationVisualizer(cfg, results_dir=paths["results_dir"])
        visualizer.plot_all(df_clean, df_noisy, df_quality, satellite_name="INSAT-3D")

        wall_elapsed = time.perf_counter() - wall_start
        log.info("=" * 80)
        log.info("Observation Generation COMPLETE in %.3f s", wall_elapsed)
        log.info("=" * 80)
        return 0

    except Exception as exc:
        log.exception("Unexpected error during Observation Generation pipeline: %s", exc)
        return 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_observation_generation",
        description="Phase 6: Observation Generation for Satellite PINN State Estimation.",
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
