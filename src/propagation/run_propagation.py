"""
src/propagation/run_propagation.py
===================================
Entry-point script for **Step 2: SGP4 Orbit Propagation**.

Usage
-----
From the project root::

    python -m src.propagation.run_propagation              # defaults to GOES16
    python -m src.propagation.run_propagation --sat GOES18
    python -m src.propagation.run_propagation --sat INSAT3D --config my.yaml

The script orchestrates the full pipeline:

1. Load configuration.
2. Resolve paths (PathManager).
3. Parse OMM file (OMMParser).
4. Propagate orbit (SGP4Propagator).
5. Validate trajectory (TrajectoryValidator).
6. Save truth_states.csv.
7. Generate result plots (PropagationVisualizer).
8. Write propagation.log summary.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

# ── Make src importable when run as a script from project root ─────────────
# When executed as ``python -m src.propagation.run_propagation`` this is
# handled automatically.  The sys.path insert guards against direct execution.
_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from src.propagation.omm_parser import OMMParser
from src.propagation.sgp4_propagator import SGP4Propagator, SGP4PropagationError
from src.propagation.validator import TrajectoryValidator
from src.propagation.visualizer import PropagationVisualizer
from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger, setup_logging
from src.utils.path_manager import PathManager


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="run_propagation",
        description="SGP4 orbit propagation for Satellite PINN State Estimation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sat",
        type=str,
        default="GOES16",
        metavar="SATELLITE_KEY",
        help=(
            "Satellite key from config.yaml (e.g. GOES16, GOES18, INSAT3D). "
            "Default: GOES16"
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to config.yaml.  Auto-discovered if not given.",
    )
    return parser.parse_args()


def run(satellite_key: str, config_path: str | None = None) -> int:
    """
    Execute the full propagation pipeline for a single satellite.

    Parameters
    ----------
    satellite_key : str
        Key into ``cfg["satellites"]`` (e.g. ``"GOES16"``).
    config_path : str | None
        Optional explicit path to ``config.yaml``.

    Returns
    -------
    int
        Exit code: 0 = success, 1 = failure.
    """
    # ── 1. Load config ─────────────────────────────────────────────────
    cfg: dict[str, Any] = ConfigLoader.load(config_path)

    # ── 2. Resolve paths ───────────────────────────────────────────────
    try:
        pm = PathManager(cfg, satellite_key)
    except (KeyError, FileNotFoundError) as exc:
        # Setup minimal logging to stderr before we have a log file
        import logging
        logging.basicConfig(level=logging.ERROR)
        logging.getLogger(__name__).error("Path resolution failed: %s", exc)
        return 1

    # ── 3. Setup logging (file + console) ──────────────────────────────
    setup_logging(cfg, log_file=pm.propagation_log)
    log = get_logger(__name__)

    log.info("=" * 70)
    log.info("Satellite PINN State Estimation — Step 2: SGP4 Propagation")
    log.info("=" * 70)
    log.info("Satellite key : %s", satellite_key)
    log.info("Config        : %s", cfg.get("_root"))
    log.info("OMM file      : %s", pm.omm_file)
    log.info("Output CSV    : %s", pm.truth_csv)
    log.info("Log file      : %s", pm.propagation_log)
    log.info("Results dir   : %s", pm.results_dir)

    wall_start = time.perf_counter()
    propagation_status = "FAILED"

    try:
        # ── 4. Parse OMM ───────────────────────────────────────────────
        log.info("─" * 50)
        log.info("STEP 1/5  Parse OMM")
        parser = OMMParser(pm.omm_file)
        meta = parser.metadata
        log.info(
            "  Satellite : %s  (NORAD %s)",
            meta["satellite_name"],
            meta["norad_id"],
        )
        log.info("  Epoch     : %s", meta["epoch"].isoformat())

        # ── 5. Propagate ───────────────────────────────────────────────
        log.info("─" * 50)
        log.info("STEP 2/5  Propagate with SGP4")
        prop_cfg = cfg["propagation"]
        propagator = SGP4Propagator(
            parser=parser,
            duration_days=float(prop_cfg["duration_days"]),
            step_seconds=float(prop_cfg["step_seconds"]),
        )
        t_sim_start = propagator.epoch
        t_sim_end = propagator.epoch
        from datetime import timedelta
        t_sim_end = propagator.epoch + timedelta(days=propagator.duration_days)

        df: pd.DataFrame = propagator.propagate()
        log.info("  n_steps   : %d", len(df))
        log.info("  t_start   : %s", df["utc_time"].iloc[0])
        log.info("  t_end     : %s", df["utc_time"].iloc[-1])

        # ── 6. Validate ────────────────────────────────────────────────
        log.info("─" * 50)
        log.info("STEP 3/5  Validate trajectory")
        validator = TrajectoryValidator(
            expected_rows=int(prop_cfg["expected_rows"]),
            step_seconds=float(prop_cfg["step_seconds"]),
            required_columns=list(prop_cfg["csv_columns"]),
        )
        val_result = validator.validate(df)

        if not val_result.passed:
            log.error("Validation FAILED — aborting.")
            _write_log_summary(
                log,
                satellite_name=meta["satellite_name"],
                t_sim_start=t_sim_start,
                t_sim_end=t_sim_end,
                wall_elapsed=time.perf_counter() - wall_start,
                n_samples=len(df),
                status="FAILED (validation errors)",
                val_result=val_result,
            )
            return 1

        # ── 7. Save CSV ────────────────────────────────────────────────
        log.info("─" * 50)
        log.info("STEP 4/5  Save truth_states.csv")
        _save_csv(df, pm.truth_csv, list(prop_cfg["csv_columns"]))
        log.info("  Saved %d rows → %s", len(df), pm.truth_csv)

        # ── 8. Visualise ───────────────────────────────────────────────
        log.info("─" * 50)
        log.info("STEP 5/5  Generate figures")
        viz = PropagationVisualizer(cfg, results_dir=pm.results_dir)
        viz.plot_all(df, satellite_name=meta["satellite_name"])

        wall_elapsed = time.perf_counter() - wall_start
        propagation_status = "SUCCESS"

        # ── 9. Final log summary ───────────────────────────────────────
        _write_log_summary(
            log,
            satellite_name=meta["satellite_name"],
            t_sim_start=t_sim_start,
            t_sim_end=t_sim_end,
            wall_elapsed=wall_elapsed,
            n_samples=len(df),
            status=propagation_status,
            val_result=val_result,
        )
        log.info("=" * 70)
        log.info("Propagation COMPLETE in %.2f s", wall_elapsed)
        log.info("=" * 70)
        return 0

    except SGP4PropagationError as exc:
        log.exception("SGP4 propagation error: %s", exc)
        return 1
    except FileNotFoundError as exc:
        log.exception("File not found: %s", exc)
        return 1
    except Exception as exc:  # pylint: disable=broad-except
        log.exception("Unexpected error during propagation: %s", exc)
        return 1


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _save_csv(df: pd.DataFrame, path: Path, columns: list[str]) -> None:
    """
    Write *df* to a CSV at *path* with column order enforced by *columns*.

    Parameters
    ----------
    df : pd.DataFrame
    path : Path
        Destination file (parent directory must exist).
    columns : list[str]
        Ordered list of column names to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df[columns].to_csv(path, index=False, float_format="%.10f")


def _write_log_summary(
    log,
    satellite_name: str,
    t_sim_start,
    t_sim_end,
    wall_elapsed: float,
    n_samples: int,
    status: str,
    val_result,
) -> None:
    """Emit a structured summary block to the log."""
    log.info("─" * 50)
    log.info("PROPAGATION SUMMARY")
    log.info("  Satellite name   : %s", satellite_name)
    log.info("  Simulation start : %s", t_sim_start.isoformat())
    log.info("  Simulation end   : %s", t_sim_end.isoformat())
    log.info("  Execution time   : %.3f s", wall_elapsed)
    log.info("  Number of samples: %d", n_samples)
    log.info("  Propagation status: %s", status)
    if val_result.stats:
        log.info(
            "  Mean |r|         : %.2f km", val_result.stats.get("r_mean_km", float("nan"))
        )
        log.info(
            "  Std  |r|         : %.4f km", val_result.stats.get("r_std_km", float("nan"))
        )
        log.info(
            "  Mean |v|         : %.5f km/s", val_result.stats.get("v_mean_kms", float("nan"))
        )
    log.info("─" * 50)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Command-line entry point."""
    args = parse_args()
    exit_code = run(satellite_key=args.sat, config_path=args.config)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
