"""
src/pinn/run_training.py
========================
Entry-point execution script for Phase 8 (PINN Framework).

Coordinates PINN model training, validation, testing, metric calculations,
checkpointing, and visualization rendering.

Usage
-----
From the project root:

    python -m src.pinn.run_training --sat GOES16
    python -m src.pinn.run_training --sat GOES16 --epochs 10 --no-plots
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Make src importable when run directly
_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

# PyTorch import handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.dataset.dataset_manager import DatasetManager
from src.physics.physics_engine import PhysicsEngine
from src.pinn.metrics import compute_metrics
from src.pinn.pinn_engine import PINNEngine
from src.pinn.validator import PINNValidator
from src.pinn.visualizer import PINNVisualizer
from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger, setup_logging


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="run_training",
        description="Trains a Physics-Informed Neural Network (PINN) for satellite state estimation.",
    )
    parser.add_argument(
        "--sat",
        type=str,
        default="GOES16",
        help="Satellite identifier (e.g. GOES16, GOES18, INSAT3D). Default: GOES16",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to config.yaml.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of training epochs.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable generating training diagnostic PNG plots.",
    )
    return parser.parse_args()


def run(satellite_key: str, config_path: str | None = None, epochs_override: int | None = None, no_plots: bool = False) -> int:
    """
    Executes the training pipeline.

    Parameters
    ----------
    satellite_key : str
        Satellite identifier.
    config_path : str | None
        Path to config.yaml.
    epochs_override : int | None
        Optional epoch count override.
    no_plots : bool
        If True, disables plot rendering.

    Returns
    -------
    int
        Exit code: 0 for success, 1 for failure.
    """
    start_wall = time.perf_counter()

    # 1. Load config
    cfg: Dict[str, Any] = ConfigLoader.load(config_path)
    root = Path(cfg["_root"]).resolve()

    # 2. Setup Logging
    log_dir = root / "logs" / "training"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "training.log"
    setup_logging(cfg, log_file=log_file)
    log = get_logger(__name__)

    log.info("=" * 80)
    log.info("Satellite PINN State Estimation — Phase 8: Neural Network Training")
    log.info("=" * 80)
    log.info("Satellite   : %s", satellite_key)
    log.info("Log file    : %s", log_file)

    if not TORCH_AVAILABLE:
        log.error("PyTorch not found in the environment. PyTorch is a strict requirement for PINN training.")
        return 1

    try:
        # Override configuration epochs if CLI flag is provided
        if epochs_override is not None:
            log.info("Overriding training epochs to: %d", epochs_override)
            cfg["pinn"]["training"]["epochs"] = epochs_override

        # 3. Load Dataset Splits using the DatasetManager (Never read CSVs directly)
        log.info("Loading datasets via Phase 6 DatasetManager...")
        manager = DatasetManager(cfg, satellite_key)
        
        train_df = manager.load_csv(manager.get_dataset_path("train.csv"))
        val_df = manager.load_csv(manager.get_dataset_path("validation.csv"))
        test_df = manager.load_csv(manager.get_dataset_path("test.csv"))
        colloc_df = manager.load_csv(manager.get_dataset_path("collocation_points.csv"))

        log.info(
            "Datasets loaded: train=%d, validation=%d, test=%d, collocation=%d",
            len(train_df),
            len(val_df),
            len(test_df),
            len(colloc_df),
        )

        # 4. Instantiate Physics Engine (Phase 7)
        log.info("Loading Phase 7 Physics Engine...")
        physics_engine = PhysicsEngine(cfg)

        # 5. Instantiate PINN Engine
        pinn_engine = PINNEngine(cfg, physics_engine)

        # 6. Validate Network Shapes before training
        shapes_ok, shape_errors = PINNValidator.validate_network_shapes(
            pinn_engine.network,
            input_dim=int(cfg["pinn"]["network"]["input_dim"]),
            output_dim=int(cfg["pinn"]["network"]["output_dim"]),
        )
        if not shapes_ok:
            log.error("Structural validation checks failed: %s", shape_errors)
            return 1

        # 7. Run Training
        loss_history = pinn_engine.train(train_df, val_df, colloc_df)

        # 8. Load Best Checkpoint for Evaluation
        log.info("Loading best model weights from checkpoints for final evaluation...")
        pinn_engine.load(is_best=True)

        # 9. Evaluate on Test Dataset
        log.info("Running evaluation on test split...")
        test_metrics = pinn_engine.evaluate(test_df)

        # 10. Compute Complete Trajectory Predictions (for plotting comparison)
        log.info("Generating predictions across full test trajectory...")
        t_test = test_df["elapsed_seconds"].values.astype(np.float32)
        pos_cols = ["x_gcrs_km", "y_gcrs_km", "z_gcrs_km"]
        vel_cols = ["vx_gcrs_kms", "vy_gcrs_kms", "vz_gcrs_kms"]
        state_true = test_df[pos_cols + vel_cols].values.astype(np.float32)
        
        state_pred = pinn_engine.predict(t_test)

        # Compute full statistical metrics
        full_metrics = compute_metrics(
            state_pred=state_pred,
            state_true=state_true,
            residuals=None,  # residuals evaluated dynamically in validation
            constraint_error=test_metrics.get("constraint_violations", 0.0),
        )

        # 11. Run Post-Training Checkpoint Validator
        log.info("Validating checkpoint folder integrity...")
        ckpt_dir = root / cfg["pinn"]["checkpoint"]["dir"]
        ckpt_ok, ckpt_errors = PINNValidator.validate_checkpoint_integrity(ckpt_dir)
        if not ckpt_ok:
            log.warning("Checkpoint integrity checks failed: %s", ckpt_errors)

        # 12. Visualization
        if not no_plots:
            log.info("Generating training diagnostic plots...")
            results_dir = root / "results" / "training"
            visualizer = PINNVisualizer(results_dir)
            
            # Loss history
            visualizer.plot_loss_history(loss_history)
            # Trajectory prediction comparison
            visualizer.plot_trajectory_comparison(
                times=t_test,
                states_true=state_true,
                states_pred=state_pred,
            )
            # Error norms
            visualizer.plot_prediction_errors(
                times=t_test,
                states_true=state_true,
                states_pred=state_pred,
            )

        duration = time.perf_counter() - start_wall
        log.info("=" * 80)
        log.info("PINN TRAINING AND EVALUATION COMPLETED SUCCESSFUL")
        log.info("=" * 80)
        log.info("  Execution time            : %.2f seconds", duration)
        log.info("  Final Test Position RMSE  : %.6e km", full_metrics["position_rmse_km"])
        log.info("  Final Test Velocity RMSE  : %.6e km/s", full_metrics["velocity_rmse_kms"])
        log.info("  Final Test Position MAE   : %.6e km", full_metrics["position_mae_km"])
        log.info("  Final Test Velocity MAE   : %.6e km/s", full_metrics["velocity_mae_kms"])
        log.info("  Constraint Violations     : %d", int(full_metrics["constraint_error"]))
        log.info("=" * 80)

        return 0

    except Exception as e:
        log.error("PINN training run failed: %s", str(e), exc_info=True)
        return 1


if __name__ == "__main__":
    args = parse_args()
    sys.exit(
        run(
            satellite_key=args.sat,
            config_path=args.config,
            epochs_override=args.epochs,
            no_plots=args.no_plots,
        )
    )
