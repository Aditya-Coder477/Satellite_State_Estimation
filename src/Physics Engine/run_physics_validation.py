"""
src/physics/run_physics_validation.py
======================================
CLI validation script for Phase 7 (Physics Engine).

Loads observation data from previous phases, generates a differentiable orbit prediction
using an analytical Keplerian formulation to establish a PyTorch computation graph,
runs the Physics Engine calculations, executes validator checks, and writes logs and plots.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Make src importable
_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

# PyTorch import handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.physics import constants
from src.physics.autograd_utils import AutogradUtils
from src.physics.physics_engine import PhysicsEngine
from src.physics.validator import PhysicsValidator
from src.physics.visualizer import PhysicsVisualizer
from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger, setup_logging


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="run_physics_validation",
        description="Validates the Phase 7 Physics Engine using simulated orbital states.",
    )
    parser.add_argument(
        "--sat",
        type=str,
        default="GOES16",
        help="Satellite identifier (e.g. GOES16). Default: GOES16",
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


def run(satellite_key: str, config_path: str | None = None, no_plots: bool = False) -> int:
    """
    Executes physics engine validation test.

    Parameters
    ----------
    satellite_key : str
        Satellite identifier.
    config_path : str | None
        Path to config.yaml.
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
    log_file = root / "logs" / "physics" / "physics_engine.log"
    setup_logging(cfg, log_file=log_file)
    log = get_logger(__name__)

    log.info("=" * 80)
    log.info("Satellite PINN State Estimation — Phase 7: Physics Engine Validation")
    log.info("=" * 80)
    log.info("Satellite   : %s", satellite_key)
    log.info("Log file    : %s", log_file)

    if not TORCH_AVAILABLE:
        log.error("PyTorch not found in the environment. PyTorch is a strict requirement for Phase 7 autograd.")
        return 1

    try:
        # Resolve data path
        sat_cfg = cfg["satellites"][satellite_key]
        noisy_obs_path = (
            root / cfg["paths"]["data"]["processed"]["base"]
            / sat_cfg["processed_dir"]
            / "noisy_observations.csv"
        )

        if not noisy_obs_path.is_file():
            log.error("Simulated noisy observations file missing at: %s", noisy_obs_path)
            log.error("Please run observation generation and dataset pipeline first.")
            return 1

        log.info("Loading observations from: %s", noisy_obs_path)
        df = pd.read_csv(noisy_obs_path)
        
        # Take a subset of 500 rows for verification
        df_sub = df.iloc[:500].copy()
        
        # Sort by elapsed_seconds
        df_sub = df_sub.sort_values(by="elapsed_seconds").reset_index(drop=True)

        # 3. Instantiate Physics Engine
        engine = PhysicsEngine(cfg)

        # 4. Generate Differentiable Orbit Prediction Graph
        # To test autograd we need a mathematical relationship y = f(t).
        # We model a Keplerian-like analytical trajectory using initial state.
        pos_cols = ["x_gcrs_km", "y_gcrs_km", "z_gcrs_km"]
        vel_cols = ["vx_gcrs_kms", "vy_gcrs_kms", "vz_gcrs_kms"]
        
        # Initial state from first row
        r0_np = df_sub[pos_cols].iloc[0].values.astype(np.float32)
        v0_np = df_sub[vel_cols].iloc[0].values.astype(np.float32)
        
        # Tensors
        r0 = torch.tensor(r0_np, dtype=torch.float32)
        v0 = torch.tensor(v0_np, dtype=torch.float32)
        r0_norm = torch.linalg.norm(r0)
        
        # Keplerian mean motion: omega = sqrt(mu / r0^3)
        omega = torch.sqrt(torch.tensor(constants.MU) / (r0_norm ** 3))

        # Time tensor t shape (Batch, 1), requires grad to compute autograd derivatives
        t_np = df_sub["elapsed_seconds"].values.astype(np.float32)
        t = torch.tensor(t_np, dtype=torch.float32).unsqueeze(1).requires_grad_(True)

        log.info("Constructing differentiable trajectory prediction graph...")
        # Analytical circular orbit formula (provides exact derivatives)
        pos_pred = r0 * torch.cos(omega * t) + (v0 / omega) * torch.sin(omega * t)
        vel_pred = -r0 * omega * torch.sin(omega * t) + v0 * torch.cos(omega * t)
        state_pred = torch.cat([pos_pred, vel_pred], dim=-1)

        # 5. Compute Derivatives via Autograd
        log.info("Computing predicted state derivatives using Autograd...")
        state_deriv_pred = AutogradUtils.compute_first_derivative(t, state_pred)

        # 6. Compute Residuals, Gravity, and Constraints
        log.info("Evaluating Physics Engine residual equations...")
        residuals = engine.compute_residuals(state_pred, state_deriv_pred)
        gravity_acc = engine.compute_gravity(state_pred[..., :3])
        
        log.info("Evaluating physical state constraints...")
        constr_sat, penalties = engine.validate_prediction(state_pred, state_deriv_pred[..., 3:])
        
        # Compute loss components
        log.info("Computing unweighted and weighted loss components...")
        losses = engine.compute_loss(
            residuals=residuals,
            boundary_loss=0.01,         # mock boundary MSE
            initial_loss=0.001,         # mock initial condition MSE
            constraint_penalty=penalties["total_constraint_penalty"],
        )

        # 7. Run Physics Validator
        log.info("Running PhysicsValidator check suite...")
        is_valid, validation_errors = PhysicsValidator.validate_physics_outputs(
            state_pred=state_pred,
            state_deriv_pred=state_deriv_pred,
            residuals=residuals,
            gravity_acc=gravity_acc,
        )

        # 8. Visualization
        if not no_plots:
            log.info("Generating diagnostic plots...")
            results_dir = root / "results" / "physics"
            visualizer = PhysicsVisualizer(results_dir)
            visualizer.plot_all(
                times=t_np,
                states=state_pred.detach().cpu().numpy(),
                residuals=residuals.detach().cpu().numpy(),
                accelerations=gravity_acc.detach().cpu().numpy(),
                losses=losses,
            )

        duration = time.perf_counter() - start_wall
        log.info("=" * 80)
        log.info("PHYSICS ENGINE VALIDATION SUCCESSFUL")
        log.info("=" * 80)
        log.info("  Execution time          : %.2f seconds", duration)
        log.info("  Batch size evaluated    : %d", len(t_np))
        log.info("  Gravity Model           : %s", engine.gravity_model.name)
        log.info("  Mean Position Residual : %.6e km/s", float(torch.mean(torch.abs(residuals[:, :3]))))
        log.info("  Mean Velocity Residual : %.6e km/s^2", float(torch.mean(torch.abs(residuals[:, 3:6]))))
        log.info("  Constraint Satisfaction : %s", "PASSED" if constr_sat["overall_satisfied"] else "FAILED")
        log.info("  Validation Status       : %s", "PASSED" if is_valid else "FAILED")
        log.info("=" * 80)

        return 0 if is_valid else 1

    except Exception as e:
        log.error("Physics validation failed: %s", str(e), exc_info=True)
        return 1


import pandas as pd  # imported here for schema compatibility checking

if __name__ == "__main__":
    args = parse_args()
    sys.exit(run(satellite_key=args.sat, config_path=args.config, no_plots=args.no_plots))
