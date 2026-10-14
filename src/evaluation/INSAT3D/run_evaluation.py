"""
src/evaluation/INSAT3D/run_evaluation.py
========================================
Main execution script for Phase 10 — Model Evaluation & Performance Analysis.

Orchestrates test dataset prediction, trajectory reconstruction, physics residual analysis,
orbital elements conversion, diagnostic plotting, and report generation.

Usage
-----
From the project root:

    python -m src.evaluation.INSAT3D.run_evaluation
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Make src importable when run directly
_SRC_ROOT = Path(__file__).resolve().parents[3]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

# PyTorch import handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.dataset.pytorch_dataset import ObservationDataset
from src.physics.physics_engine import PhysicsEngine
from src.pistn.forward_model import PISTNForwardModel
from src.pistn.normalization import DeNormalizer
from src.evaluation.INSAT3D.evaluator import PerformanceEvaluator
from src.evaluation.INSAT3D.metrics_calculator import PerformanceMetricsCalculator
from src.evaluation.INSAT3D.state_error import StateErrorCalculator
from src.evaluation.INSAT3D.physics_analysis import PhysicsResidualAnalyzer
from src.evaluation.INSAT3D.trajectory_analysis import TrajectoryAnalyzer
from src.evaluation.INSAT3D.orbit_analysis import KeplerianElementsAnalyzer
from src.evaluation.INSAT3D.visualizer import EvaluationVisualizer
from src.evaluation.INSAT3D.report_generator import PerformanceReportGenerator
from src.evaluation.INSAT3D.validator import EvaluationValidator
from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger, setup_logging


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="run_evaluation",
        description="Phase 10: PI-STN Model Evaluation & Performance Analysis.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to config.yaml.",
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
    
    # Phase 10 directories
    eval_out_dir = sat_dir / "evaluation"
    results_dir = root / "results" / "evaluation" / satellite_key
    log_dir = root / "logs" / "evaluation" / satellite_key

    return {
        "sat_dir": sat_dir,
        "dataset_dir": ds_dir,
        "test_csv": ds_dir / "test.csv",
        "normalization_json": ds_dir / "normalization.json",
        "feature_mapping_json": ds_dir / "feature_mapping.json",
        "checkpoint_model": checkpoint_dir / "best_model.pt",
        "target_norm_json": training_out_dir / "transition_normalization.json",
        
        # Output paths
        "eval_out_dir": eval_out_dir,
        "results_dir": results_dir,
        "log_dir": log_dir,
        "figures_dir": results_dir / "figures",
        "reports_dir": results_dir / "reports",
        "metrics_dir": results_dir / "metrics",
    }


def run(satellite_key: str = "INSAT3D", config_path: str | None = None) -> int:
    """
    Executes the Phase 10 evaluation pipeline.
    """
    start_wall = time.perf_counter()

    # Load configuration
    cfg: Dict[str, Any] = ConfigLoader.load(config_path)
    
    paths = _resolve_paths(cfg, satellite_key)

    # Create directories
    paths["eval_out_dir"].mkdir(parents=True, exist_ok=True)
    paths["log_dir"].mkdir(parents=True, exist_ok=True)
    paths["figures_dir"].mkdir(parents=True, exist_ok=True)
    paths["reports_dir"].mkdir(parents=True, exist_ok=True)
    paths["metrics_dir"].mkdir(parents=True, exist_ok=True)

    # Setup Logging
    log_file = paths["log_dir"] / "evaluation.log"
    setup_logging(cfg, log_file=log_file)
    log = get_logger(__name__)

    log.info("=" * 80)
    log.info("Satellite PI-STN State Estimation — Phase 10: Model Evaluation")
    log.info("=" * 80)
    log.info("Satellite          : %s", satellite_key)
    log.info("Evaluation Log     : %s", log_file)
    log.info("Checkpoint Path    : %s", paths["checkpoint_model"])
    log.info("Test Dataset CSV   : %s", paths["test_csv"])
    log.info("Output Folder      : %s", paths["eval_out_dir"])
    log.info("Results Folder     : %s", paths["results_dir"])

    if not TORCH_AVAILABLE:
        log.error("PyTorch is required for model evaluation.")
        return 1

    try:
        # Step 1 — Load Test Dataset
        if not paths["test_csv"].is_file():
            log.error("Test dataset CSV missing at %s", paths["test_csv"])
            return 1
            
        df_test = pd.read_csv(paths["test_csv"])
        
        # Load feature mapping
        with open(paths["feature_mapping_json"], "r") as f:
            feat_map = json.load(f)
        input_cols: List[str] = feat_map.get("inputs", [])
        target_cols: List[str] = feat_map.get("targets", [])

        # Validate inputs and targets
        inputs_arr = df_test[input_cols].to_numpy()
        targets_arr = df_test[target_cols].to_numpy()
        if not EvaluationValidator.validate_inputs_outputs(inputs_arr, targets_arr):
            log.error("Test dataset contains NaNs/Infs or failed checks.")
            return 1

        # Step 2 — Load Model
        log.info("Instantiating forward model network...")
        model = PISTNForwardModel(cfg)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("Loading best model weights from %s", paths["checkpoint_model"])
        checkpoint = torch.load(paths["checkpoint_model"], map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.eval()

        # Step 3 — Prediction & De-normalization
        denormalizer = DeNormalizer(paths["normalization_json"], paths["target_norm_json"])
        evaluator = PerformanceEvaluator(model, denormalizer, input_cols, target_cols)
        
        batch_size = int(cfg.get("evaluation", {}).get("prediction", {}).get("batch_size", 1024))
        true_deltas, pred_deltas, true_res, pred_res = evaluator.run_inference(
            df_test, batch_size=batch_size, device=device
        )

        # Validate prediction outputs
        if not EvaluationValidator.validate_predictions(pred_deltas, true_deltas, pred_res, true_res):
            log.error("Prediction outputs validation failed.")
            return 1

        # Extract current raw Cartesian states from test dataset
        state_cols = ["x", "y", "z", "vx", "vy", "vz"]
        
        # Read raw test dataset to de-normalize state if needed
        # Since test.csv is normalized, we de-normalize the input columns to get raw coordinates
        inputs_tensor = torch.tensor(inputs_arr, dtype=torch.float32)
        state_raw_t, dt_raw_t = denormalizer.denormalize_state(inputs_tensor)
        states_raw = state_raw_t.numpy()
        dt_raw = dt_raw_t.numpy()

        # Step 4 — Compute Metrics
        log.info("-" * 60)
        log.info("Computing evaluation metrics...")
        
        # 1. Statistical performance metrics
        metrics = PerformanceMetricsCalculator.compute_all_metrics(pred_deltas, true_deltas)
        EvaluationValidator.validate_metrics(metrics)

        # 2. Position & Velocity error metrics
        errors = StateErrorCalculator.compute_errors(pred_deltas, true_deltas)

        # 3. Physics engine residual evaluation
        physics_engine = PhysicsEngine(cfg)
        physics_analyzer = PhysicsResidualAnalyzer(physics_engine)
        physics_summary = physics_analyzer.evaluate_residuals(
            states_raw, true_deltas, pred_res, dt_raw
        )

        # 4. Trajectory Reconstruction & Orbit Rollout
        trajectory_analyzer = TrajectoryAnalyzer(model, denormalizer, paths["normalization_json"])
        
        # Trajectory 1-step error
        trajectory_summary = trajectory_analyzer.evaluate_1step_trajectory(true_deltas, pred_deltas)
        
        # Continuous Rollout (Orbit Drift)
        # We need raw test dataset for starting state S0 and exact timeline
        # Since test.csv is normalized, let's load raw transition dataset and slice test indices
        # Let's align test indices. Since train, val, test are sliced chronologically,
        # the test set is the last segment of length len(df_test).
        raw_ds_path = paths["dataset_dir"] / "transition_dataset_raw.csv"
        df_raw_all = pd.read_csv(raw_ds_path)
        df_test_raw = df_raw_all.iloc[-len(df_test):].copy().reset_index(drop=True)

        rollout_states, orbit_summary = trajectory_analyzer.evaluate_orbit_drift(df_test_raw, device=device)
        EvaluationValidator.validate_orbit_consistency(df_test_raw[state_cols].to_numpy(), rollout_states)

        # 5. Orbital Keplerian Element deviations
        log.info("Computing classical Keplerian orbital elements deviations...")
        orbit_analyzer = KeplerianElementsAnalyzer()
        true_states_raw = df_test_raw[state_cols].to_numpy()
        
        deviations, orbit_elem_summary = orbit_analyzer.analyze_orbital_elements(
            true_states_raw, rollout_states
        )

        # Step 5 — Save Deliverables (CSV and JSON)
        log.info("-" * 60)
        log.info("Saving evaluation summaries and results...")

        # 1. Save prediction_results.csv (identical layout to validation_predictions.csv)
        df_pred_out = pd.DataFrame({
            "x": states_raw[:, 0], "y": states_raw[:, 1], "z": states_raw[:, 2],
            "vx": states_raw[:, 3], "vy": states_raw[:, 4], "vz": states_raw[:, 5],
            "pred_delta_x": pred_deltas[:, 0], "pred_delta_y": pred_deltas[:, 1], "pred_delta_z": pred_deltas[:, 2],
            "pred_delta_vx": pred_deltas[:, 3], "pred_delta_vy": pred_deltas[:, 4], "pred_delta_vz": pred_deltas[:, 5],
            "true_delta_x": true_deltas[:, 0], "true_delta_y": true_deltas[:, 1], "true_delta_z": true_deltas[:, 2],
            "true_delta_vx": true_deltas[:, 3], "true_delta_vy": true_deltas[:, 4], "true_delta_vz": true_deltas[:, 5],
            "pred_ax_res": pred_res[:, 0], "pred_ay_res": pred_res[:, 1], "pred_az_res": pred_res[:, 2],
            "true_ax_res": true_res[:, 0], "true_ax_res_y": true_res[:, 1], "true_ax_res_z": true_res[:, 2], # wait, labels
        })
        # Fix column labels
        df_pred_out.columns = [
            "x", "y", "z", "vx", "vy", "vz",
            "pred_delta_x", "pred_delta_y", "pred_delta_z", "pred_delta_vx", "pred_delta_vy", "pred_delta_vz",
            "true_delta_x", "true_delta_y", "true_delta_z", "true_delta_vx", "true_delta_vy", "true_delta_vz",
            "pred_ax_res", "pred_ay_res", "pred_az_res", "true_ax_res", "true_ay_res", "true_az_res"
        ]
        df_pred_out.to_csv(paths["eval_out_dir"] / "prediction_results.csv", index=False, float_format="%.10f")

        # 2. Save trajectory_comparison.csv
        drift_arr = np.linalg.norm(true_states_raw[:, :3] - rollout_states[:, :3], axis=1)
        df_traj_out = pd.DataFrame({
            "time": df_test_raw["elapsed_seconds"],
            "true_x": true_states_raw[:, 0], "true_y": true_states_raw[:, 1], "true_z": true_states_raw[:, 2],
            "pred_x": rollout_states[:, 0], "pred_y": rollout_states[:, 1], "pred_z": rollout_states[:, 2],
            "drift": drift_arr
        })
        df_traj_out.to_csv(paths["eval_out_dir"] / "trajectory_comparison.csv", index=False, float_format="%.10f")

        # 3. Create consolidated metrics summary
        # Maps keys to requested display labels
        metrics_summary_map = {
            "RMSE Position": metrics["rmse_position"],
            "RMSE Velocity": metrics["rmse_velocity"],
            "MAE Position": metrics["mae_position"],
            "MAE Velocity": metrics["mae_velocity"],
            "R² Position": metrics["r2_position"],
            "R² Velocity": metrics["r2_velocity"],
            "Mean Position Error": errors["mean_position_error"],
            "Mean Velocity Error": errors["mean_velocity_error"],
            "Mean Physics Residual": physics_summary["mean_physics_residual"],
            "Mean Trajectory Error": trajectory_summary["mean_trajectory_error"],
            "Mean Orbit Drift": orbit_summary["mean_orbit_drift"],
            "Mean Δa": orbit_elem_summary["mean_delta_a"],
            "Mean Δe": orbit_elem_summary["mean_delta_e"],
            "Mean Δi": orbit_elem_summary["mean_delta_i"],
            "Mean ΔRAAN": orbit_elem_summary["mean_delta_raan"],
            "Mean Δω": orbit_elem_summary["mean_delta_omega"],
            "Mean Δν": orbit_elem_summary["mean_delta_nu"],
        }
        
        # Save metrics_summary.csv
        df_metrics_summary = pd.DataFrame({
            "Metric": list(metrics_summary_map.keys()),
            "Value": list(metrics_summary_map.values())
        })
        df_metrics_summary.to_csv(paths["eval_out_dir"] / "metrics_summary.csv", index=False, float_format="%.10f")
        df_metrics_summary.to_csv(paths["metrics_dir"] / "metrics_summary.csv", index=False, float_format="%.10f")

        # Save metrics_summary.json
        with open(paths["eval_out_dir"] / "metrics_summary.json", "w", encoding="utf-8") as f:
            json.dump(metrics_summary_map, f, indent=4)
        with open(paths["metrics_dir"] / "metrics_summary.json", "w", encoding="utf-8") as f:
            json.dump(metrics_summary_map, f, indent=4)

        # Save orbit_analysis.json
        with open(paths["eval_out_dir"] / "orbit_analysis.json", "w", encoding="utf-8") as f:
            json.dump(orbit_elem_summary, f, indent=4)
        with open(paths["metrics_dir"] / "orbit_analysis.json", "w", encoding="utf-8") as f:
            json.dump(orbit_elem_summary, f, indent=4)

        # Save physics_summary.json
        with open(paths["eval_out_dir"] / "physics_summary.json", "w", encoding="utf-8") as f:
            json.dump(physics_summary, f, indent=4)
        with open(paths["metrics_dir"] / "physics_summary.json", "w", encoding="utf-8") as f:
            json.dump(physics_summary, f, indent=4)

        # Save evaluation_config.json
        eval_config = {
            "satellite": satellite_key,
            "random_seed": int(cfg.get("random_seed", 42)),
            "evaluation_settings": cfg.get("evaluation", {}),
            "model_architecture": cfg.get("pistn", {}).get("network", {}),
        }
        with open(paths["eval_out_dir"] / "evaluation_config.json", "w", encoding="utf-8") as f:
            json.dump(eval_config, f, indent=4)
        with open(paths["metrics_dir"] / "evaluation_config.json", "w", encoding="utf-8") as f:
            json.dump(eval_config, f, indent=4)

        # Step 6 — Visualization (Figures 1-10)
        log.info("-" * 60)
        log.info("Generating evaluation figures...")
        
        # Merge physical target predictions with raw true transitions
        physics_residuals = true_res - pred_res
        
        viz = EvaluationVisualizer(paths["figures_dir"], cfg)
        viz.plot_all(
            true_deltas=true_deltas,
            pred_deltas=pred_deltas,
            true_states=true_states_raw,
            pred_states=rollout_states,
            physics_residuals=physics_residuals
        )

        # Step 7 — Reports Generation (evaluation_report.md & HTML)
        log.info("-" * 60)
        log.info("Generating evaluation reports...")
        reporter = PerformanceReportGenerator(paths["reports_dir"])
        reporter.generate(
            metrics=metrics,
            errors=errors,
            physics=physics_summary,
            trajectory=trajectory_summary,
            orbit=orbit_summary,
            num_samples=len(df_test),
            config={"sat": satellite_key}
        )
        
        # Also copy report to the processed evaluation dir for easy user access
        reporter_eval = PerformanceReportGenerator(paths["eval_out_dir"])
        reporter_eval.generate(
            metrics=metrics,
            errors=errors,
            physics=physics_summary,
            trajectory=trajectory_summary,
            orbit=orbit_summary,
            num_samples=len(df_test),
            config={"sat": satellite_key}
        )

        duration = time.perf_counter() - start_wall
        log.info("=" * 80)
        log.info("PI-STN EVALUATION PIPELINE COMPLETED SUCCESSFULLY")
        log.info("=" * 80)
        log.info("  Total time taken       : %.2f seconds", duration)
        log.info("  RMSE Position          : %.4f km", metrics["rmse_position"])
        log.info("  Mean Orbit Drift       : %.4f km", orbit_summary["mean_orbit_drift"])
        log.info("=" * 80)

        return 0

    except Exception as e:
        log.error("Evaluation pipeline failed: %s", str(e), exc_info=True)
        return 1


if __name__ == "__main__":
    args = parse_args()
    sys.exit(run(satellite_key="INSAT3D", config_path=args.config))
