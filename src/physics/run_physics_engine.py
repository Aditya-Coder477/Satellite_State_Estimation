"""
src/physics/run_physics_engine.py
=================================
Entry-point script for Phase 8 — Physics Engine.

Orchestrates loading transition datasets, running physics integrations, evaluating
governing residuals and constraints, computing statistics, rendering dark-theme plots,
and writing output targets.

Usage
-----
From the project root:

    python -m src.physics.run_physics_engine --sat INSAT3D
    python -m src.physics.run_physics_engine --sat INSAT3D --no-plots
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

# Make src importable when run directly
_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

# PyTorch import handling (optional but recommended for verifying autograd)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.dataset.normalization import Normalizer
from src.physics.coordinate_dynamics import (
    angular_momentum as compute_h,
    specific_mechanical_energy as compute_e,
    norm as compute_vec_norm,
)
from src.physics.physics_engine import PhysicsEngine
from src.physics.validator import PhysicsValidator
from src.physics.statistics import DatasetStatistics
from src.physics.visualizer import PhysicsVisualizer
from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger, setup_logging


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="run_physics_engine",
        description="Phase 8: State Transition Physics Engine for PI-STN.",
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
        help="Disable generating diagnostic PNG plots.",
    )
    return parser.parse_args()


def _resolve_paths(cfg: Dict[str, Any], satellite_key: str) -> Dict[str, Path]:
    root = Path(cfg["_root"]).resolve()
    processed_base = root / cfg["paths"]["data"]["processed"]["base"]
    sat_dir = processed_base / cfg["satellites"][satellite_key]["processed_dir"]
    ds_dir = sat_dir / cfg.get("dataset", {}).get("output_dir", "dataset")
    out_dir = sat_dir / "physics"
    metadata_dir = root / "data" / "metadata" / satellite_key

    return {
        "sat_dir": sat_dir,
        "dataset_dir": ds_dir,
        "transition_dataset": ds_dir / "transition_dataset.csv",
        "train_dataset": ds_dir / "train.csv",
        "validation_dataset": ds_dir / "validation.csv",
        "test_dataset": ds_dir / "test.csv",
        "norm_json": ds_dir / "normalization.json",
        "physics_constants": metadata_dir / "physics_constants.json",
        "output_dir": out_dir,
        "log_dir": root / "logs" / "physics" / satellite_key,
        "results_dir": root / "results" / "physics" / satellite_key,
    }


def run(satellite_key: str, config_path: str | None = None, no_plots: bool = False) -> int:
    """
    Executes the physics engine pipeline.
    """
    start_wall = time.perf_counter()

    # 1. Load Config
    cfg: Dict[str, Any] = ConfigLoader.load(config_path)
    paths = _resolve_paths(cfg, satellite_key)
    
    # Create directories
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    paths["log_dir"].mkdir(parents=True, exist_ok=True)

    # 2. Setup Logging
    log_file = paths["log_dir"] / "physics_engine.log"
    setup_logging(cfg, log_file=log_file)
    log = get_logger(__name__)

    log.info("=" * 80)
    log.info("Satellite PINN State Estimation — Phase 8: Physics Engine")
    log.info("=" * 80)
    log.info("Satellite   : %s", satellite_key)
    log.info("Log file    : %s", log_file)

    try:
        # 3. Instantiate Physics Engine
        engine = PhysicsEngine(cfg)

        # 4. Load Dataset Files
        log.info("-" * 60)
        log.info("STEP 1/5: Loading and un-normalizing datasets...")
        if not paths["transition_dataset"].is_file():
            log.error("Transition dataset missing at: %s", paths["transition_dataset"])
            return 1

        df_norm = pd.read_csv(paths["transition_dataset"])
        log.info("Loaded transition dataset of shape: %s", df_norm.shape)

        # Load normalizer to reconstruct raw physical states
        if not paths["norm_json"].is_file():
            log.error("Normalization parameters JSON missing at: %s", paths["norm_json"])
            return 1

        normalizer = Normalizer()
        normalizer.load(paths["norm_json"])
        
        # Reconstruct raw physical states for ODE evaluation
        log.info("Un-normalizing dataset to retrieve raw Cartesian parameters (km, km/s)...")
        df_raw = normalizer.inverse_transform(df_norm)

        # 5. Extract state arrays as NumPy arrays for the core Physics Engine
        state_np = df_raw[["x", "y", "z", "vx", "vy", "vz"]].to_numpy(dtype=np.float64)
        delta_v_np = df_raw[["delta_vx", "delta_vy", "delta_vz"]].to_numpy(dtype=np.float64)
        delta_t_np = df_raw["delta_time"].to_numpy(dtype=np.float64)
        delta_s_true_np = df_raw[["delta_x", "delta_y", "delta_z", "delta_vx", "delta_vy", "delta_vz"]].to_numpy(dtype=np.float64)

        # 6. Execute Core Physics Math (STEP 2/5)
        log.info("-" * 60)
        log.info("STEP 2/5: Computing orbital dynamics and state transitions...")
        
        # Gravity magnitude at each position
        gravity_acc_np = engine.compute_gravity(state_np[:, :3])

        # Integrate Expected Transition: ΔS_phys = Integrator(St, Δt)
        expected_trans_np = engine.compute_transition(state_np, delta_t_np)

        # Compute Residuals: a_trans = Δv/Δt, a_phys = gravity, a_res = a_trans - a_phys
        a_trans_np, a_phys_np, a_res_np = engine.compute_residuals(state_np, delta_v_np, delta_t_np)

        # Evaluate physical constraints
        constr_sat, penalties_np = engine.validate_prediction(state_np, a_res_np)

        # Compute conservation quantities
        energy_np = compute_e(state_np)
        momentum_np = compute_h(state_np)

        # Compute mock losses to test PhysicsLossEvaluator (evaluated on mean values)
        # In actual training, this is computed dynamically in PyTorch backpropagation.
        losses = engine.compute_loss(
            pred_delta=expected_trans_np,
            true_delta=delta_s_true_np,
            phys_delta=expected_trans_np,
            pred_res=a_res_np,
            true_res=a_res_np,  # mock exact prediction for residual testing
            constraint_penalty=penalties_np["total_constraint_penalty"],
        )

        # 7. Validate Outputs (STEP 3/5)
        log.info("-" * 60)
        log.info("STEP 3/5: Running validator check suite...")
        is_valid, validation_errors = PhysicsValidator.validate_physics_outputs(
            state=state_np,
            gravity_acc=gravity_acc_np,
            expected_transition=expected_trans_np,
            residual_acc=a_res_np,
        )
        if not is_valid:
            log.error("Physics Engine Validation FAILED.")
            return 1
        log.info("Physics Engine Validation PASSED successfully.")

        # 8. Write Output CSV Files (STEP 4/5)
        log.info("-" * 60)
        log.info("STEP 4/5: Generating and saving deliverables...")

        # File 1: physics_targets.csv
        df_targets = pd.DataFrame({
            "utc_time": df_raw["utc_time"],
            "elapsed_seconds": df_raw["elapsed_seconds"],
            "delta_time": df_raw["delta_time"],
            "x": state_np[:, 0],
            "y": state_np[:, 1],
            "z": state_np[:, 2],
            "vx": state_np[:, 3],
            "vy": state_np[:, 4],
            "vz": state_np[:, 5],
            "delta_x_phys": expected_trans_np[:, 0],
            "delta_y_phys": expected_trans_np[:, 1],
            "delta_z_phys": expected_trans_np[:, 2],
            "delta_vx_phys": expected_trans_np[:, 3],
            "delta_vy_phys": expected_trans_np[:, 4],
            "delta_vz_phys": expected_trans_np[:, 5],
            "ax_res": a_res_np[:, 0],
            "ay_res": a_res_np[:, 1],
            "az_res": a_res_np[:, 2],
            "orbital_energy": energy_np,
            "angular_momentum": compute_vec_norm(momentum_np, keepdims=False),
        })
        df_targets.to_csv(paths["output_dir"] / "physics_targets.csv", index=False, float_format="%.10f")

        # File 2: residual_dynamics.csv
        df_res_dyn = pd.DataFrame({
            "ax_expected": a_phys_np[:, 0],
            "ay_expected": a_phys_np[:, 1],
            "az_expected": a_phys_np[:, 2],
            "ax_observed": a_trans_np[:, 0],
            "ay_observed": a_trans_np[:, 1],
            "az_observed": a_trans_np[:, 2],
            "ax_res": a_res_np[:, 0],
            "ay_res": a_res_np[:, 1],
            "az_res": a_res_np[:, 2],
        })
        df_res_dyn.to_csv(paths["output_dir"] / "residual_dynamics.csv", index=False, float_format="%.10f")

        # File 3: physics_constraints.csv
        # Computes elementwise constraints for plotting and validation
        pos_norm = compute_vec_norm(state_np[:, :3], keepdims=False)
        vel_norm = compute_vec_norm(state_np[:, 3:6], keepdims=False)
        df_constr = pd.DataFrame({
            "radius_constraint": pos_norm,
            "velocity_constraint": vel_norm,
            "energy_constraint": energy_np,
            "momentum_constraint": compute_vec_norm(momentum_np, keepdims=False),
            "constraint_penalty": penalties_np["radius_penalty"] + penalties_np["velocity_penalty"],
        })
        df_constr.to_csv(paths["output_dir"] / "physics_constraints.csv", index=False, float_format="%.10f")

        # File 4: state_transition_targets.csv
        df_st_targets = pd.DataFrame({
            "delta_x_phys": expected_trans_np[:, 0],
            "delta_y_phys": expected_trans_np[:, 1],
            "delta_z_phys": expected_trans_np[:, 2],
            "delta_vx_phys": expected_trans_np[:, 3],
            "delta_vy_phys": expected_trans_np[:, 4],
            "delta_vz_phys": expected_trans_np[:, 5],
        })
        df_st_targets.to_csv(paths["output_dir"] / "state_transition_targets.csv", index=False, float_format="%.10f")

        # File 5: physics_statistics.json
        stats = DatasetStatistics.compute_physics_statistics(
            gravity_acc=gravity_acc_np,
            residual_acc=a_res_np,
            constraints=penalties_np,
            orbital_energy=energy_np,
            angular_momentum=momentum_np,
        )
        with open(paths["output_dir"] / "physics_statistics.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4)

        # File 6: physics_summary.json
        summary = {
            "satellite": satellite_key,
            "integrator": engine.transition_model.integrator,
            "gravity_model": engine.gravity_model.name,
            "total_samples": len(df_raw),
            "mean_orbital_radius_km": float(np.mean(pos_norm)),
            "mean_velocity_kms": float(np.mean(vel_norm)),
            "mean_residual_acc_magnitude_kms2": float(np.mean(np.linalg.norm(a_res_np, axis=-1))),
            "status": "VALIDATED",
        }
        with open(paths["output_dir"] / "physics_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

        # File 7: physics_config_used.json
        with open(paths["output_dir"] / "physics_config_used.json", "w", encoding="utf-8") as f:
            json.dump(cfg.get("physics", {}), f, indent=4)

        log.info("Saved all 7 output deliverables to %s", paths["output_dir"])

        # 9. Visualization (STEP 5/5)
        if not no_plots:
            log.info("-" * 60)
            log.info("STEP 5/5: Rendering diagnostic plots...")
            viz = PhysicsVisualizer(paths["results_dir"])
            
            # Reconstruct constraint dictionary of arrays for plotting
            plot_constraints = {
                "total_constraint_penalty": np.full(len(df_raw), penalties_np["total_constraint_penalty"]),
            }
            
            viz.plot_all(
                times=df_raw["elapsed_seconds"].to_numpy(dtype=np.float64),
                states=state_np,
                gravity_acc=gravity_acc_np,
                residual_acc=a_res_np,
                expected_transition=expected_trans_np,
                actual_transition=delta_s_true_np,
                constraints=plot_constraints,
                orbital_energy=energy_np,
                angular_momentum=momentum_np,
                losses=losses,
            )

        # 10. Run Integrator Comparison
        log.info("-" * 60)
        log.info("Running numerical integrator comparisons (Euler, RK4, Leapfrog, Velocity Verlet)...")
        try:
            from src.physics.compare_integrators import run_integrator_comparison
            run_integrator_comparison(
                satellite_key=satellite_key,
                config_path=config_path,
                output_dir=paths["output_dir"],
                results_dir=paths["results_dir"],
            )
        except Exception as e_comp:
            log.error("Failed to run integrator comparison: %s", str(e_comp), exc_info=True)

        duration = time.perf_counter() - start_wall
        log.info("=" * 80)
        log.info("PHYSICS ENGINE EXECUTION COMPLETE")
        log.info("=" * 80)
        log.info("  Execution time          : %.2f seconds", duration)
        log.info("  Total samples processed : %d", len(df_raw))
        log.info("  Integrator selected     : %s", engine.transition_model.integrator)
        log.info("  Gravity Model           : %s", engine.gravity_model.name)
        log.info("  Validation Status       : %s", "PASSED" if is_valid else "FAILED")
        log.info("=" * 80)

        return 0 if is_valid else 1

    except Exception as e:
        log.error("Physics pipeline execution failed: %s", str(e), exc_info=True)
        return 1


if __name__ == "__main__":
    args = parse_args()
    sys.exit(run(satellite_key=args.sat, config_path=args.config, no_plots=args.no_plots))
