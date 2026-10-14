"""
src/physics/compare_integrators.py
==================================
Compares numerical integration methods (Euler, RK4, Leapfrog, Velocity Verlet)
on long-term satellite orbit propagation. Calculates and plots energy drift,
angular momentum drift, and orbit drift (position error) over time.
Writes comparison results to JSON and generates publication-quality plots.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

from src.physics.constants import MU
from src.physics.coordinate_dynamics import angular_momentum, specific_mechanical_energy, norm
from src.physics.gravity_models import GravityModelFactory
from src.physics.state_transition import IntegratorFactory
from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


def run_integrator_comparison(
    satellite_key: str,
    config_path: str | None = None,
    output_dir: Path | None = None,
    results_dir: Path | None = None,
) -> Dict[str, Any]:
    """
    Runs comparison across Euler, RK4, Leapfrog, and Velocity Verlet.
    Generates comparison plots and saves results as JSON.
    """
    log.info("Starting numerical integrator comparison for %s...", satellite_key)
    
    # 1. Load config and resolve paths
    cfg: Dict[str, Any] = ConfigLoader.load(config_path)
    root = Path(cfg.get("_root", ".")).resolve()
    processed_base = root / cfg["paths"]["data"]["processed"]["base"]
    sat_dir = processed_base / cfg["satellites"][satellite_key]["processed_dir"]
    
    if output_dir is None:
        output_dir = sat_dir / "physics"
    if results_dir is None:
        results_dir = root / "results" / "physics" / satellite_key

    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load dataset
    ds_path = sat_dir / cfg.get("dataset", {}).get("output_dir", "dataset") / "transition_dataset_raw.csv"
    if not ds_path.is_file():
        raise FileNotFoundError(f"Raw transition dataset not found at {ds_path}")

    import pandas as pd
    df_raw = pd.read_csv(ds_path)
    log.info("Loaded raw dataset with %d samples for integrator comparison.", len(df_raw))

    # Extract state vector and delta times
    state_np = df_raw[["x", "y", "z", "vx", "vy", "vz"]].to_numpy(dtype=np.float64)
    delta_t_np = df_raw["delta_time"].to_numpy(dtype=np.float64)
    elapsed_seconds = df_raw["elapsed_seconds"].to_numpy(dtype=np.float64)

    # Resolve gravity model
    gravity_model_name = cfg.get("physics", {}).get("gravity_model", "two_body")
    gravity_model = GravityModelFactory.create(gravity_model_name)
    acc_fn = lambda q: gravity_model.compute_acceleration(q)

    integrators = ["euler", "rk4", "leapfrog", "velocity_verlet"]
    colors = {
        "euler": "#FF6B6B",           # Coral Red
        "rk4": "#FFE66D",             # Yellow
        "leapfrog": "#00D4FF",        # Cyan
        "velocity_verlet": "#4ECDC4", # Teal
    }

    comparison_results: Dict[str, Any] = {}
    trajectories: Dict[str, np.ndarray] = {}

    # Run rollout for each integrator
    for name in integrators:
        start_time = time.perf_counter()
        integrator = IntegratorFactory.create(name)
        
        n_steps = len(df_raw)
        rollout_states = np.zeros_like(state_np)
        rollout_states[0] = state_np[0]

        # Recurrent propagation rollout
        for t in range(n_steps - 1):
            dt = delta_t_np[t]
            rollout_states[t + 1] = integrator.integrate(
                rollout_states[t:t+1], dt, acc_fn
            )[0]

        exec_time = time.perf_counter() - start_time
        trajectories[name] = rollout_states

        # Calculate metrics
        pos_errors = np.linalg.norm(rollout_states[:, :3] - state_np[:, :3], axis=1)
        vel_errors = np.linalg.norm(rollout_states[:, 3:6] - state_np[:, 3:6], axis=1)

        energies = specific_mechanical_energy(rollout_states, MU)
        energy_drifts = np.abs(energies - energies[0])

        momentums = norm(angular_momentum(rollout_states), keepdims=False)
        momentum_drifts = np.abs(momentums - momentums[0])

        comparison_results[name] = {
            "execution_time_seconds": float(exec_time),
            "final_position_error_km": float(pos_errors[-1]),
            "max_position_error_km": float(np.max(pos_errors)),
            "mean_position_error_km": float(np.mean(pos_errors)),
            "final_velocity_error_kms": float(vel_errors[-1]),
            "max_velocity_error_kms": float(np.max(vel_errors)),
            "mean_velocity_error_kms": float(np.mean(vel_errors)),
            "final_energy_drift_km2_s2": float(energy_drifts[-1]),
            "max_energy_drift_km2_s2": float(np.max(energy_drifts)),
            "mean_energy_drift_km2_s2": float(np.mean(energy_drifts)),
            "final_momentum_drift_km2_s": float(momentum_drifts[-1]),
            "max_momentum_drift_km2_s": float(np.max(momentum_drifts)),
            "mean_momentum_drift_km2_s": float(np.mean(momentum_drifts)),
        }
        log.info(
            "Integrator %s completed in %.4fs | Mean Pos Err: %.3f km | Max Energy Drift: %.4e",
            name, exec_time, np.mean(pos_errors), np.max(energy_drifts)
        )

    # 3. Save JSON results
    json_data = {
        "satellite": satellite_key,
        "gravity_model": gravity_model_name,
        "time_span_hours": float((elapsed_seconds[-1] - elapsed_seconds[0]) / 3600.0),
        "steps": int(len(df_raw)),
        "metrics": comparison_results,
    }

    # Write to both output_dir and results_dir
    for path in [output_dir, results_dir]:
        with open(path / "integrator_comparison.json", "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=4)
    log.info("Saved integrator_comparison.json successfully.")

    # 4. Generate comparison plots
    plt.switch_backend("Agg")
    plt.style.use("dark_background")

    # Plot 1: Specific Mechanical Energy Drift
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.set_facecolor("#121212")
    for name in integrators:
        energies = specific_mechanical_energy(trajectories[name], MU)
        drifts = np.abs(energies - energies[0])
        ax.plot(elapsed_seconds / 3600.0, drifts, color=colors[name], label=name, lw=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("Time [hours]", fontsize=10, fontweight="bold")
    ax.set_ylabel("Energy Drift |E(t) - E(0)| [km²/s²]", fontsize=10, fontweight="bold")
    ax.set_title(f"Specific Mechanical Energy conservation comparison ({gravity_model_name})", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.15)
    ax.legend()
    fig.savefig(results_dir / "energy_drift_comparison.png", facecolor="#0d0d0d", bbox_inches="tight")
    plt.close(fig)

    # Plot 2: Angular Momentum Drift
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.set_facecolor("#121212")
    for name in integrators:
        momentums = norm(angular_momentum(trajectories[name]), keepdims=False)
        drifts = np.abs(momentums - momentums[0])
        ax.plot(elapsed_seconds / 3600.0, drifts, color=colors[name], label=name, lw=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("Time [hours]", fontsize=10, fontweight="bold")
    ax.set_ylabel("Angular Momentum Drift |h(t) - h(0)| [km²/s]", fontsize=10, fontweight="bold")
    ax.set_title(f"Specific Angular Momentum conservation comparison ({gravity_model_name})", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.15)
    ax.legend()
    fig.savefig(results_dir / "angular_momentum_comparison.png", facecolor="#0d0d0d", bbox_inches="tight")
    plt.close(fig)

    # Plot 3: Orbit Drift (Position Error compared to truth dataset)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.set_facecolor("#121212")
    for name in integrators:
        pos_errors = np.linalg.norm(trajectories[name][:, :3] - state_np[:, :3], axis=1)
        ax.plot(elapsed_seconds / 3600.0, pos_errors, color=colors[name], label=name, lw=1.2)
    ax.set_xlabel("Time [hours]", fontsize=10, fontweight="bold")
    ax.set_ylabel("Accumulated Position Error [km]", fontsize=10, fontweight="bold")
    ax.set_title(f"Accumulated Orbit Drift / Position Error vs. Truth", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.15)
    ax.legend()
    fig.savefig(results_dir / "orbit_drift_comparison.png", facecolor="#0d0d0d", bbox_inches="tight")
    plt.close(fig)

    log.info("Comparison plots generated and saved successfully to %s", results_dir)
    return comparison_results
