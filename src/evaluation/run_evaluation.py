"""
src/evaluation/run_evaluation.py
=================================
Command-line entry point for Phase 9 — Future 2-Day Prediction Evaluation.

Usage
-----
    # Full evaluation using best_model.pt checkpoint
    python -m src.evaluation.run_evaluation --satellite GOES16

    # Specify a custom checkpoint
    python -m src.evaluation.run_evaluation --satellite GOES16 \\
        --checkpoint checkpoints/last_model.pt

    # Skip physics residual computation
    python -m src.evaluation.run_evaluation --satellite GOES16 --no-physics

    # Use a specific config file
    python -m src.evaluation.run_evaluation --satellite GOES16 \\
        --config path/to/config.yaml

Outputs
-------
  results/evaluation/figures/      — Diagnostic plots
  results/evaluation/reports/      — evaluation_report.md + .json
  results/evaluation/metrics/      — Per-step metric CSVs (archived)
  CSV Outputs/                     — Deliverable CSVs (prediction comparison,
                                     metrics summary, orbital drift)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logging, get_logger

log: logging.Logger = get_logger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_evaluation",
        description="Phase 9 — Future 2-Day Prediction Evaluation Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--satellite",
        type=str,
        default="GOES16",
        help="Satellite key (must match an entry in config.yaml satellites registry).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Path to trained model checkpoint (.pt). "
            "Defaults to value in config.yaml (evaluation.checkpoint)."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml. Auto-detected if not provided.",
    )
    parser.add_argument(
        "--no-physics",
        action="store_true",
        help="Skip physics consistency residual computation.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip figure generation (metrics and report still produced).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point. Returns 0 on success, 1 on error."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    # ── Load configuration ────────────────────────────────────────────
    cfg = ConfigLoader.load(args.config)
    root = Path(cfg["_root"])

    # ── Setup logging ─────────────────────────────────────────────────
    log_dir = root / "logs" / "evaluation"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "evaluation.log"
    setup_logging(cfg, log_file=log_file)

    log.info("=" * 72)
    log.info("Satellite PINN State Estimation — Phase 9: Evaluation")
    log.info("=" * 72)
    log.info("Satellite   : %s", args.satellite)
    log.info("Log file    : %s", log_file)

    # ── Check PyTorch availability ────────────────────────────────────
    try:
        import torch
        TORCH_AVAILABLE = True
        log.info("PyTorch %s detected. Device: %s",
                 torch.__version__,
                 "CUDA" if torch.cuda.is_available() else "CPU")
    except ImportError:
        TORCH_AVAILABLE = False
        log.error("PyTorch not found. Install PyTorch to run PINN evaluation.")
        return 1

    # ── Determine checkpoint path ─────────────────────────────────────
    eval_cfg = cfg.get("evaluation", {})
    checkpoint_rel = args.checkpoint or eval_cfg.get("checkpoint", "checkpoints/best_model.pt")
    checkpoint_path = root / checkpoint_rel

    if not checkpoint_path.is_file():
        log.error(
            "Checkpoint not found: %s\n"
            "Run Phase 8 training first: python -m src.pinn.run_training",
            checkpoint_path,
        )
        return 1

    log.info("Checkpoint  : %s", checkpoint_path)

    # ── Initialise Physics Engine ─────────────────────────────────────
    physics_engine = None
    if not args.no_physics:
        try:
            from src.physics.physics_engine import PhysicsEngine
            physics_engine = PhysicsEngine(cfg)
            log.info("PhysicsEngine initialised for residual computation.")
        except Exception as exc:
            log.warning("PhysicsEngine initialisation failed: %s. Skipping.", exc)

    # ── Initialise PINN Engine and load checkpoint ────────────────────
    log.info("Loading trained PINN model from checkpoint...")
    try:
        from src.physics.physics_engine import PhysicsEngine as _PE
        _pe = _PE(cfg) if physics_engine is None else physics_engine

        from src.pinn.pinn_engine import PINNEngine
        pinn_engine = PINNEngine(cfg=cfg, physics_engine=_pe)
        pinn_engine.load(is_best=(args.checkpoint is None))
        log.info("PINN model loaded successfully.")
    except Exception as exc:
        log.error("Failed to load PINN model: %s", exc, exc_info=True)
        return 1

    # ── Build Predictor Adapter ───────────────────────────────────────
    # Wraps PINNEngine.predict() to satisfy the PredictorProtocol interface.
    class _PINNPredictorAdapter:
        def __init__(self, engine: "PINNEngine") -> None:
            self._engine = engine

        def predict(self, times: "np.ndarray") -> "np.ndarray":
            return self._engine.predict(times)

    import numpy as np
    predictor = _PINNPredictorAdapter(pinn_engine)

    # ── Optionally monkey-patch visualizer to skip figures ────────────
    if args.no_plots:
        import src.evaluation.visualizer as _vis_mod
        class _NoOpVisualizer:
            def generate_all(self, *a, **kw):
                log.info("--no-plots flag set; skipping figure generation.")
        # Patch at class level for this session
        _vis_mod.EvaluationVisualizer.generate_all = _NoOpVisualizer.generate_all

    # ── Run Evaluation ────────────────────────────────────────────────
    log.info("Starting evaluation pipeline...")
    try:
        from src.evaluation.evaluator import EvaluationEngine
        engine = EvaluationEngine(
            cfg=cfg,
            satellite_key=args.satellite,
            physics_engine=physics_engine,
        )
        report = engine.run(predictor)
    except Exception as exc:
        log.error("Evaluation pipeline failed: %s", exc, exc_info=True)
        return 1

    # ── Print summary to console ──────────────────────────────────────
    grades = report.get("grades", {})
    pos_m  = report.get("metrics", {}).get("position", {})
    vel_m  = report.get("metrics", {}).get("velocity", {})

    print("\n" + "=" * 60)
    print(f"  EVALUATION SUMMARY — {args.satellite}")
    print("=" * 60)
    print(f"  Position RMSE  : {pos_m.get('rmse_km', float('nan')):.4f} km")
    print(f"  Velocity RMSE  : {vel_m.get('rmse_kms', float('nan')):.6f} km/s")
    print(f"  Position grade : {grades.get('position', '?')}")
    print(f"  Velocity grade : {grades.get('velocity', '?')}")
    print(f"  Physics grade  : {grades.get('physics', '?')}")
    print(f"  Overall grade  : {grades.get('overall', '?')}")
    print("=" * 60)
    print(f"\n  Reports  → results/evaluation/reports/")
    print(f"  Figures  → results/evaluation/figures/")
    print(f"  CSVs     → CSV Outputs/")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
