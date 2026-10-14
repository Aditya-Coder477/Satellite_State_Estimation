"""
src/pistn/run_training.py
=========================
Entry-point script for Phase 9 — PI-STN Training.

Orchestrates loading datasets, target normalization generation, training execution,
performance metrics logs, prediction CSV generation, and convergence plots.

Usage
-----
From the project root:

    python -m src.pistn.run_training --sat INSAT3D
    python -m src.pistn.run_training --sat INSAT3D --no-plots
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Make src importable when run directly
_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

# PyTorch import handling
try:
    import torch
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.dataset.pytorch_dataset import ObservationDataset
from src.pistn.pistn_engine import PISTNEngine
from src.pistn.visualizer import TrainingVisualizer
from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger, setup_logging


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="run_training",
        description="Phase 9: Training the Physics-Informed State Transition Network (PI-STN).",
    )
    parser.add_argument(
        "--sat",
        type=str,
        default="INSAT3D",
        help="Satellite identifier (e.g. INSAT3D). Default: INSAT3D",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to config.yaml.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable generating diagnostic training PNG plots.",
    )
    return parser.parse_args()


def _resolve_paths(cfg: Dict[str, Any], satellite_key: str) -> Dict[str, Path]:
    root = Path(cfg["_root"]).resolve()
    processed_base = root / cfg["paths"]["data"]["processed"]["base"]
    sat_dir = processed_base / cfg["satellites"][satellite_key]["processed_dir"]
    ds_dir = sat_dir / cfg.get("dataset", {}).get("output_dir", "dataset")
    physics_dir = sat_dir / "physics"
    training_out_dir = sat_dir / "training"
    checkpoint_dir = root / "checkpoints" / satellite_key

    return {
        "sat_dir": sat_dir,
        "dataset_dir": ds_dir,
        "train_csv": ds_dir / "train.csv",
        "validation_csv": ds_dir / "validation.csv",
        "normalization_json": ds_dir / "normalization.json",
        "feature_mapping_json": ds_dir / "feature_mapping.json",
        "physics_targets_csv": physics_dir / "physics_targets.csv",
        "physics_constraints_csv": physics_dir / "physics_constraints.csv",
        "checkpoint_dir": checkpoint_dir,
        "training_out_dir": training_out_dir,
        "log_dir": root / "logs" / "training" / satellite_key,
        "results_dir": root / "results" / "training" / satellite_key,
    }


def run(satellite_key: str, config_path: str | None = None, no_plots: bool = False) -> int:
    """
    Executes the training and post-processing steps.
    """
    start_wall = time.perf_counter()

    # 1. Load config
    cfg: Dict[str, Any] = ConfigLoader.load(config_path)
    paths = _resolve_paths(cfg, satellite_key)

    # Create folders
    paths["checkpoint_dir"].mkdir(parents=True, exist_ok=True)
    paths["training_out_dir"].mkdir(parents=True, exist_ok=True)
    paths["log_dir"].mkdir(parents=True, exist_ok=True)

    # 2. Setup Logging
    log_file = paths["log_dir"] / "training.log"
    setup_logging(cfg, log_file=log_file)
    log = get_logger(__name__)

    # Write additional gradient debug log header
    grad_log_file = paths["log_dir"] / "gradient_debug.log"
    with open(grad_log_file, "w") as gf:
        gf.write(f"Gradient Debug Log for {satellite_key} - Initialized at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    log.info("=" * 80)
    log.info("Satellite PINN State Estimation — Phase 9: PI-STN Training")
    log.info("=" * 80)
    log.info("Satellite        : %s", satellite_key)
    log.info("Training Log     : %s", log_file)
    log.info("Gradient Debug   : %s", grad_log_file)
    log.info("Checkpoint Dir   : %s", paths["checkpoint_dir"])

    if not TORCH_AVAILABLE:
        log.error("PyTorch is not available. PyTorch is a strict requirement for training neural networks.")
        return 1

    try:
        # 3. Load feature mapping and config lists
        if not paths["feature_mapping_json"].is_file():
            log.error("Feature mapping file missing at: %s", paths["feature_mapping_json"])
            return 1

        with open(paths["feature_mapping_json"], "r") as f:
            feat_map = json.load(f)

        input_cols: List[str] = feat_map.get("inputs", [])
        target_cols: List[str] = feat_map.get("targets", [])

        log.info("Input Columns (%d): %s", len(input_cols), input_cols)
        log.info("Target Columns (%d): %s", len(target_cols), target_cols)

        # Priority 3: Create transition_normalization.json for targets
        log.info("Generating transition_normalization.json for target scale tracking...")
        raw_path = paths["dataset_dir"] / "transition_dataset_raw.csv"
        
        target_stats = {}
        if raw_path.is_file():
            df_raw = pd.read_csv(raw_path)
            for col in target_cols:
                target_stats[col] = {
                    "mean": float(df_raw[col].mean()),
                    "std": float(df_raw[col].std()),
                    "min": float(df_raw[col].min()),
                    "max": float(df_raw[col].max())
                }
        else:
            log.warning("Raw transition dataset not found at %s. Falling back to normalization.json stats.", raw_path)
            # Fallback: Read stats from normalization.json
            with open(paths["normalization_json"], "r") as f:
                norm_data = json.load(f)
            col_stats = norm_data.get("stats", {})
            for col in target_cols:
                col_stat = col_stats.get(col, {"mean": 0.0, "std": 1.0})
                target_stats[col] = {
                    "mean": float(col_stat.get("mean", 0.0)),
                    "std": float(col_stat.get("std", 1.0)),
                    "min": float(col_stat.get("mean", 0.0) - 3.0 * col_stat.get("std", 1.0)),
                    "max": float(col_stat.get("mean", 0.0) + 3.0 * col_stat.get("std", 1.0))
                }

        target_norm_data = {
            "method": "standard",
            "columns": target_cols,
            "stats": target_stats
        }
        target_norm_json_path = paths["training_out_dir"] / "transition_normalization.json"
        with open(target_norm_json_path, "w", encoding="utf-8") as f:
            json.dump(target_norm_data, f, indent=4)
        log.info("Saved transition_normalization.json to %s", target_norm_json_path)

        # 4. Load datasets
        if not paths["train_csv"].is_file() or not paths["validation_csv"].is_file():
            log.error("Train or Validation dataset CSVs are missing.")
            return 1

        df_train = pd.read_csv(paths["train_csv"])
        df_val = pd.read_csv(paths["validation_csv"])

        log.info("Loaded Train Data shape: %s", df_train.shape)
        log.info("Loaded Val Data shape: %s", df_val.shape)

        # 5. Create PyTorch datasets and Dataloaders
        batch_size = int(cfg.get("pistn", {}).get("training", {}).get("batch_size", 512))
        
        train_dataset = ObservationDataset(df_train, input_cols, target_cols)
        val_dataset = ObservationDataset(df_val, input_cols, target_cols)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        # 6. Initialize PISTNEngine
        log.info("Initializing PISTNEngine...")
        engine = PISTNEngine(
            cfg=cfg,
            checkpoint_dir=paths["checkpoint_dir"],
            norm_json_path=paths["normalization_json"],
            target_norm_json_path=target_norm_json_path
        )

        # 7. Start training loop (STEP 2/4)
        log.info("-" * 60)
        log.info("STEP 2/4: Executing model training...")
        history = engine.train_model(train_loader, val_loader)

        # Save training loss history CSV (includes rollout diagnostics when enabled)
        n_epochs = len(history["train_loss"])
        df_history = pd.DataFrame({
            "epoch": range(1, n_epochs + 1),
            "transition_loss": history["train_transition_loss"],
            "physics_loss": history["train_physics_loss"],
            "residual_loss": history["train_residual_loss"],
            "constraint_loss": history["train_constraint_loss"],
            "total_loss": history["train_loss"],
            "gradient_norm": history["gradient_norm"],
            "weight_norm": history["weight_norm"],
            "transition_percent": history["transition_percent"],
            "physics_percent": history["physics_percent"],
            "residual_percent": history["residual_percent"],
            "constraint_percent": history["constraint_percent"],
            "learning_rate": history["learning_rate"],
            # Multi-step rollout diagnostics (present when rollout training is enabled)
            "rollout_k": history.get("rollout_steps", [1] * n_epochs),
            "rollout_loss": history.get("rollout_loss", history["train_loss"]),
            "one_step_loss": history.get("one_step_loss", history["train_loss"]),
            "final_step_loss": history.get("final_step_loss", history["train_loss"]),
            "mean_rollout_pos_error": history.get("mean_rollout_pos_error", [0.0] * n_epochs),
            "max_rollout_pos_error": history.get("max_rollout_pos_error", [0.0] * n_epochs),
        })
        df_history.to_csv(paths["training_out_dir"] / "loss_history.csv", index=False, float_format="%.10f")

        # Copy/Save training_loss_breakdown.csv for compatibility
        df_history.to_csv(paths["training_out_dir"] / "training_loss_breakdown.csv", index=False, float_format="%.10f")

        # Save epoch metrics CSV (val components + rollout K per epoch)
        df_metrics = pd.DataFrame({
            "epoch": range(1, n_epochs + 1),
            "val_loss": history["val_loss"],
            "val_transition_loss": history["val_transition_loss"],
            "learning_rate": history["learning_rate"],
            # Constraint components
            "radius_penalty": history["radius_penalty"],
            "velocity_penalty": history["velocity_penalty"],
            "energy_penalty": history["energy_penalty"],
            "momentum_penalty": history["momentum_penalty"],
            # Rollout schedule tracking
            "rollout_k": history.get("rollout_steps", [1] * n_epochs),
        })
        df_metrics.to_csv(paths["training_out_dir"] / "epoch_metrics.csv", index=False, float_format="%.10f")

        # 8. Load best model checkpoint and run predictions (STEP 3/4)
        log.info("-" * 60)
        log.info("STEP 3/4: Generating predictions on Train/Val splits...")
        
        # Load best weights into the engine model
        engine.checkpoint_manager.load_checkpoint(engine.model, load_best=True)

        # Generate predictions reporting CSVs
        train_pred_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        engine.predict(train_pred_loader, paths["training_out_dir"] / "train_predictions.csv")
        engine.predict(val_loader, paths["training_out_dir"] / "validation_predictions.csv")

        # 9. Visualization (STEP 4/4)
        if not no_plots:
            log.info("-" * 60)
            log.info("STEP 4/4: Generating convergence plots...")
            predictions_df = pd.read_csv(paths["training_out_dir"] / "validation_predictions.csv")
            
            viz = TrainingVisualizer(paths["results_dir"])
            viz.plot_all(history, predictions_df)

        duration = time.perf_counter() - start_wall
        log.info("=" * 80)
        log.info("PI-STN TRAINING PIPELINE EXECUTION COMPLETED")
        log.info("=" * 80)
        log.info("  Execution time          : %.2f seconds", duration)
        log.info("  Total epochs completed  : %d", len(history["train_loss"]))
        log.info("  Best validation loss    : %.4e", min(history["val_loss"]))
        if history.get("rollout_steps"):
            final_k = int(history["rollout_steps"][-1])
            max_pos_err = max(history.get("max_rollout_pos_error", [0.0]))
            log.info("  Final rollout K         : %d steps", final_k)
            log.info("  Max rollout pos error   : %.4e km", max_pos_err)
        log.info("=" * 80)

        return 0

    except Exception as e:
        log.error("PI-STN training pipeline failed: %s", str(e), exc_info=True)
        return 1


if __name__ == "__main__":
    args = parse_args()
    # Add satellite to pistn config dynamically for compatibility
    cfg: Dict[str, Any] = ConfigLoader.load(args.config)
    if "pistn" not in cfg:
        cfg["pistn"] = {}
    cfg["pistn"]["satellite"] = args.sat
    
    sys.exit(run(satellite_key=args.sat, config_path=args.config, no_plots=args.no_plots))
