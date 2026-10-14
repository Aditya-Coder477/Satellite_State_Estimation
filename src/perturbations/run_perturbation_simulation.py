"""
src/perturbations/run_perturbation_simulation.py
==================================================
Entry-point script for **Phase 3: Synthetic Perturbation & Residual Dynamics**.

Usage
-----
From the project root::

    python -m src.perturbations.run_perturbation_simulation              # defaults to INSAT3D
    python -m src.perturbations.run_perturbation_simulation --sat INSAT3D

Pipeline Steps
--------------
1.  Load config & resolve paths.
2.  Load nominal trajectory (truth_states.csv from Phase 2).
3.  Build PerturbationOrchestrator from config.
4.  Compute residual accelerations for all time steps.
5.  Integrate residual dynamics → disturbed orbit.
6.  Validate outputs.
7.  Save disturbed CSV + residual CSV.
8.  Compute & save statistics (JSON files + disturbance_log.csv).
9.  Generate visualisation figures.
10. Write perturbation.log summary.
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

from src.perturbations.disturbance_generator import PerturbationOrchestrator
from src.perturbations.residual_dynamics import ResidualDynamicsIntegrator
from src.perturbations.validator import PerturbationValidator
from src.perturbations.statistics import PerturbationStatistics
from src.perturbations.visualizer import PerturbationVisualizer
from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger, setup_logging


# ---------------------------------------------------------------------------
# Path resolution helpers (Phase-3-specific, does not touch PathManager)
# ---------------------------------------------------------------------------

def _resolve_paths(cfg: dict[str, Any], satellite_key: str) -> dict[str, Path]:
    """Resolve all Phase-3 filesystem paths from config."""
    root = Path(cfg["_root"])
    processed_base = root / cfg["paths"]["data"]["processed"]["base"]
    sat_dir = processed_base / cfg["satellites"][satellite_key]["processed_dir"]

    pert_out_cfg = cfg.get("perturbation", {}).get("output", {})

    return {
        "nominal_csv": sat_dir / cfg["propagation"]["output_csv"],
        "processed_dir": sat_dir,
        "disturbed_csv": sat_dir / pert_out_cfg.get("disturbed_csv", "truth_states_disturbed.csv"),
        "residual_csv": sat_dir / pert_out_cfg.get("residual_csv", "residual_acceleration.csv"),
        "disturbance_log_csv": sat_dir / pert_out_cfg.get("disturbance_log_csv", "disturbance_log.csv"),
        "summary_json": sat_dir / pert_out_cfg.get("summary_json", "perturbation_summary.json"),
        "statistics_json": sat_dir / pert_out_cfg.get("statistics_json", "disturbance_statistics.json"),
        "config_used_json": sat_dir / pert_out_cfg.get("config_used_json", "perturbation_config_used.json"),
        "log_dir": root / "logs" / "perturbations" / satellite_key,
        "results_dir": root / "results" / "perturbations" / satellite_key,
    }


# ---------------------------------------------------------------------------
# Output builders
# ---------------------------------------------------------------------------

def _build_residual_csv(df_nominal: pd.DataFrame, result) -> pd.DataFrame:
    """Build residual_acceleration.csv with per-model breakdown."""
    return pd.DataFrame({
        "utc_time": df_nominal["utc_time"].values,
        "elapsed_seconds": df_nominal["elapsed_seconds"].values,
        # SRP components
        "ax_srp": result.acc_srp[:, 0],
        "ay_srp": result.acc_srp[:, 1],
        "az_srp": result.acc_srp[:, 2],
        # Maneuver components
        "ax_maneuver": result.acc_maneuver[:, 0],
        "ay_maneuver": result.acc_maneuver[:, 1],
        "az_maneuver": result.acc_maneuver[:, 2],
        # Random disturbance components
        "ax_random": result.acc_random[:, 0],
        "ay_random": result.acc_random[:, 1],
        "az_random": result.acc_random[:, 2],
        # Bias components
        "ax_bias": result.acc_bias[:, 0],
        "ay_bias": result.acc_bias[:, 1],
        "az_bias": result.acc_bias[:, 2],
        # Total residual
        "ax_res": result.acc_total[:, 0],
        "ay_res": result.acc_total[:, 1],
        "az_res": result.acc_total[:, 2],
    })


def _extract_epoch_jd(df: pd.DataFrame) -> float | None:
    """Extract Julian date of epoch from the nominal trajectory's first timestamp."""
    try:
        import astropy.time
        t0_str = str(df["utc_time"].iloc[0])
        t_astropy = astropy.time.Time(t0_str, format="isot", scale="utc")
        return float(t_astropy.jd)
    except Exception:
        # Fallback: approximate JD for 2026-06-25
        return 2461217.5


def _write_csv(df: pd.DataFrame, path: Path, float_fmt: str = "%.12e") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format=float_fmt)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(satellite_key: str, config_path: str | None = None) -> int:
    """
    Execute the full Phase-3 perturbation pipeline.

    Returns 0 on success, 1 on failure.
    """
    # ── 1. Load config ─────────────────────────────────────────────────
    cfg: dict[str, Any] = ConfigLoader.load(config_path)

    # ── 2. Resolve paths ───────────────────────────────────────────────
    if satellite_key not in cfg.get("satellites", {}):
        import logging
        logging.basicConfig(level=logging.ERROR)
        logging.getLogger(__name__).error(
            "Satellite '%s' not found in config. Available: %s",
            satellite_key, list(cfg.get("satellites", {}).keys()),
        )
        return 1

    paths = _resolve_paths(cfg, satellite_key)

    # ── 3. Setup logging ───────────────────────────────────────────────
    log_file = paths["log_dir"] / "perturbation.log"
    setup_logging(cfg, log_file=log_file)
    log = get_logger(__name__)

    log.info("=" * 70)
    log.info("Satellite PINN State Estimation — Phase 3: Perturbation Simulation")
    log.info("=" * 70)
    log.info("Satellite    : %s", satellite_key)
    log.info("Nominal CSV  : %s", paths["nominal_csv"])
    log.info("Disturbed CSV: %s", paths["disturbed_csv"])
    log.info("Log file     : %s", log_file)
    log.info("Results dir  : %s", paths["results_dir"])

    wall_start = time.perf_counter()
    status = "FAILED"

    try:
        # ── 4. Load nominal trajectory ─────────────────────────────────
        log.info("─" * 50)
        log.info("STEP 1/7  Load nominal trajectory")

        if not paths["nominal_csv"].is_file():
            log.error("Nominal CSV not found: %s", paths["nominal_csv"])
            log.error("Please run Phase 2 (propagation) first.")
            return 1

        df_nominal = pd.read_csv(paths["nominal_csv"])
        n_rows = len(df_nominal)
        log.info("  Loaded %d rows from: %s", n_rows, paths["nominal_csv"])

        # ── 5. Build orchestrator ──────────────────────────────────────
        log.info("─" * 50)
        log.info("STEP 2/7  Build perturbation models from config")

        epoch_jd = _extract_epoch_jd(df_nominal)
        log.info("  Epoch JD = %.4f", epoch_jd)

        pert_cfg = cfg.get("perturbation", {})
        if not pert_cfg.get("enabled", True):
            log.warning("Perturbation simulation disabled in config (perturbation.enabled=false).")
            return 0

        orchestrator = PerturbationOrchestrator.from_config(cfg, epoch_jd=epoch_jd)

        # ── 6. Compute perturbations ───────────────────────────────────
        log.info("─" * 50)
        log.info("STEP 3/7  Compute residual accelerations")

        times = df_nominal["elapsed_seconds"].to_numpy(dtype=np.float64)
        positions = df_nominal[["x_teme_km", "y_teme_km", "z_teme_km"]].to_numpy(dtype=np.float64)

        pert_result = orchestrator.compute(times, positions)

        # ── 7. Integrate residual dynamics ─────────────────────────────
        log.info("─" * 50)
        log.info("STEP 4/7  Integrate residual dynamics → disturbed orbit")

        step_s = float(cfg.get("propagation", {}).get("step_seconds", 60.0))
        integrator = ResidualDynamicsIntegrator(step_seconds=step_s)
        df_disturbed = integrator.apply(df_nominal, pert_result)
        log.info("  Disturbed orbit: %d rows", len(df_disturbed))

        # ── 8. Build residual CSV ──────────────────────────────────────
        df_residual = _build_residual_csv(df_nominal, pert_result)

        # ── 9. Validate ────────────────────────────────────────────────
        log.info("─" * 50)
        log.info("STEP 5/7  Validate outputs")

        expected_rows = int(cfg.get("propagation", {}).get("expected_rows", n_rows))
        validator = PerturbationValidator(cfg, expected_rows=expected_rows, step_seconds=step_s)
        val_result = validator.validate(df_nominal, df_disturbed, df_residual)

        if not val_result.passed:
            log.error("Validation FAILED — aborting output save.")
            return 1
        log.info("  Validation: PASSED ✓")

        # ── 10. Save CSVs ─────────────────────────────────────────────
        log.info("─" * 50)
        log.info("STEP 6/7  Save output CSVs")

        _write_csv(df_disturbed, paths["disturbed_csv"])
        log.info("  Saved disturbed CSV: %d rows → %s", len(df_disturbed), paths["disturbed_csv"])

        _write_csv(df_residual, paths["residual_csv"])
        log.info("  Saved residual CSV:  %d rows → %s", len(df_residual), paths["residual_csv"])

        # ── 11. Save statistics ────────────────────────────────────────
        log.info("─" * 50)
        log.info("STEP 7/7  Compute statistics & generate figures")

        stats = PerturbationStatistics(cfg, output_dir=paths["processed_dir"])
        stats.compute_and_save(
            df_nominal, df_disturbed, pert_result, val_stats=val_result.stats
        )

        # ── 12. Visualise ──────────────────────────────────────────────
        viz = PerturbationVisualizer(cfg, results_dir=paths["results_dir"])
        viz.plot_all(df_nominal, df_disturbed, pert_result, satellite_name="INSAT-3D")

        # ── 13. Final summary ──────────────────────────────────────────
        wall_elapsed = time.perf_counter() - wall_start
        status = "SUCCESS"

        mag = pert_result.perturbation_magnitude
        log.info("=" * 70)
        log.info("PERTURBATION SIMULATION COMPLETE")
        log.info("=" * 70)
        log.info("  Status               : %s", status)
        log.info("  Execution time       : %.2f s", wall_elapsed)
        log.info("  Samples processed    : %d", n_rows)
        log.info("  Max |a_res|          : %.4e km/s²", float(mag.max()))
        log.info("  Mean |a_res|         : %.4e km/s²", float(mag.mean()))
        log.info("  Max pos deviation    : %.4f km", float(val_result.stats.get("max_position_deviation_km", 0.0)))
        log.info("  Max vel deviation    : %.4e km/s", float(val_result.stats.get("max_velocity_deviation_kms", 0.0)))
        log.info("  Disturbed CSV        : %s", paths["disturbed_csv"])
        log.info("  Residual CSV         : %s", paths["residual_csv"])
        log.info("  Results dir          : %s", paths["results_dir"])
        log.info("=" * 70)
        return 0

    except FileNotFoundError as exc:
        log.exception("File not found: %s", exc)
        return 1
    except Exception as exc:  # pylint: disable=broad-except
        log.exception("Unexpected error during perturbation simulation: %s", exc)
        return 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_perturbation_simulation",
        description="Phase 3: Synthetic Perturbation Simulation for Satellite PINN State Estimation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
