"""
src/physics/visualizer.py
==========================
Renders diagnostic plots for Phase 7 (Physics Engine).

Generates dark-theme visualization plots under ``results/physics/``:
1. Position and velocity residual histograms
2. Acceleration magnitude vs radius (verifying gravitational point-mass curve)
3. Residual magnitudes vs time and orbital radius
4. Weighted loss components and constraint boundary proximity
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np

# Headless matplotlib setup
plt.switch_backend("Agg")
plt.style.use("dark_background")

from src.physics import constants
from src.physics.coordinate_dynamics import compute_norm, get_position, get_velocity, is_tensor
from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class PhysicsVisualizer:
    """
    Generates diagnostic and validation plots for physics engine runs.

    Parameters
    ----------
    output_dir : Path
        Target directory to save PNG figures.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.colors = {
            "position": "#00D4FF",      # cyan
            "velocity": "#FFE66D",      # yellow
            "acceleration": "#FF6B6B",  # red
            "theory": "#4ECDC4",        # teal
            "background": "#121212",
            "accent": "#9B5DE5",
        }
        log.info("PhysicsVisualizer initialised. Output directory: %s", output_dir)

    def plot_all(
        self,
        times: np.ndarray,
        states: np.ndarray,
        residuals: np.ndarray,
        accelerations: np.ndarray,
        losses: Dict[str, float] = None,
    ) -> None:
        """
        Generates and saves all physics engine diagnostic figures.

        Parameters
        ----------
        times : np.ndarray
            1D array of times (seconds from epoch) of shape (Batch,).
        states : np.ndarray
            State vector array [x,y,z,vx,vy,vz] of shape (Batch, 6).
        residuals : np.ndarray
            Physics residual array [rx, ry, rz, rvx, rvy, rvz] of shape (Batch, 6).
        accelerations : np.ndarray
            Gravity acceleration vector [ax, ay, az] of shape (Batch, 3).
        losses : Dict[str, float], optional
            Dictionary of computed loss values, by default None.
        """
        try:
            self.plot_residuals_histograms(residuals)
            self.plot_acceleration_profile(states, accelerations)
            self.plot_residuals_trends(times, states, residuals)
            if losses:
                self.plot_loss_components(losses)
            log.info("All physics diagnostic plots saved successfully.")
        except Exception as e:
            log.error("Failed to generate physics visualizations: %s", str(e), exc_info=True)

    def plot_residuals_histograms(self, residuals: np.ndarray) -> None:
        """Plots histograms of position and velocity residuals."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

        pos_res = residuals[:, :3]
        vel_res = residuals[:, 3:6]

        # Position residuals histogram
        ax1.hist(pos_res[:, 0], bins=50, color="#FF6B6B", alpha=0.6, label="X Residual")
        ax1.hist(pos_res[:, 1], bins=50, color="#4ECDC4", alpha=0.6, label="Y Residual")
        ax1.hist(pos_res[:, 2], bins=50, color="#FFE66D", alpha=0.6, label="Z Residual")
        ax1.set_xlabel("Residual Magnitude [km/s]", fontsize=10)
        ax1.set_ylabel("Frequency", fontsize=10)
        ax1.set_title("Position Dynamics Residuals (dr_pred/dt - v_pred)", fontsize=11)
        ax1.grid(True, linestyle="--", alpha=0.15)
        ax1.legend()

        # Velocity residuals histogram
        ax2.hist(vel_res[:, 0], bins=50, color="#FF6B6B", alpha=0.6, label="Vx Residual")
        ax2.hist(vel_res[:, 1], bins=50, color="#4ECDC4", alpha=0.6, label="Vy Residual")
        ax2.hist(vel_res[:, 2], bins=50, color="#FFE66D", alpha=0.6, label="Vz Residual")
        ax2.set_xlabel("Residual Magnitude [km/s^2]", fontsize=10)
        ax2.set_ylabel("Frequency", fontsize=10)
        ax2.set_title("Velocity Dynamics Residuals (dv_pred/dt - a_grav)", fontsize=11)
        ax2.grid(True, linestyle="--", alpha=0.15)
        ax2.legend()

        fig.suptitle("Physics Residuals Distribution Analysis", fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout()
        fig.savefig(self.output_dir / "physics_residuals_histograms.png", facecolor=self.colors["background"])
        plt.close(fig)

    def plot_acceleration_profile(self, states: np.ndarray, accelerations: np.ndarray) -> None:
        """Verifies computed acceleration magnitude matches point-mass point gravity curve."""
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

        pos = states[:, :3]
        r_norm = np.linalg.norm(pos, axis=-1)
        a_norm = np.linalg.norm(accelerations, axis=-1)

        # Plot computed accelerations
        ax.scatter(r_norm, a_norm, color=self.colors["acceleration"], alpha=0.6, s=15, label="Computed Acceleration")

        # Plot theoretical point-mass gravity curve: a = mu / r^2
        r_range = np.linspace(np.min(r_norm), np.max(r_norm), 500)
        a_theory = constants.MU / (r_range ** 2)
        ax.plot(r_range, a_theory, color=self.colors["theory"], linestyle="--", linewidth=2.0, label=r"Theoretical Two-Body ($\mu / r^2$)")

        ax.set_xlabel("Orbital Radius [km]", fontsize=10)
        ax.set_ylabel("Acceleration Magnitude [km/s^2]", fontsize=10)
        ax.set_title("Gravitational Acceleration Profile vs. Orbital Radius", fontsize=12, fontweight="bold", pad=15)
        ax.grid(True, linestyle="--", alpha=0.2)
        ax.legend()

        plt.tight_layout()
        fig.savefig(self.output_dir / "gravitational_acceleration_profile.png", facecolor=self.colors["background"])
        plt.close(fig)

    def plot_residuals_trends(self, times: np.ndarray, states: np.ndarray, residuals: np.ndarray) -> None:
        """Plots residual norms against time and orbital radius."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

        t_hours = times / 3600.0
        pos = states[:, :3]
        r_norm = np.linalg.norm(pos, axis=-1)

        # Compute L2 norms of position and velocity residuals
        pos_res_norm = np.linalg.norm(residuals[:, :3], axis=-1)
        vel_res_norm = np.linalg.norm(residuals[:, 3:6], axis=-1)

        # Residuals vs Time
        ax1.scatter(t_hours, pos_res_norm, color=self.colors["position"], alpha=0.5, s=10, label="Pos Residual Norm")
        ax1.scatter(t_hours, vel_res_norm, color=self.colors["velocity"], alpha=0.5, s=10, label="Vel Residual Norm")
        ax1.set_xlabel("Elapsed Time [Hours]", fontsize=10)
        ax1.set_ylabel("L2 Residual Norm", fontsize=10)
        ax1.set_title("Physics Residual Norms vs. Time", fontsize=11)
        ax1.grid(True, linestyle="--", alpha=0.15)
        ax1.legend()

        # Residuals vs Radius
        ax2.scatter(r_norm, pos_res_norm, color=self.colors["position"], alpha=0.5, s=10, label="Pos Residual Norm")
        ax2.scatter(r_norm, vel_res_norm, color=self.colors["velocity"], alpha=0.5, s=10, label="Vel Residual Norm")
        ax2.set_xlabel("Orbital Radius [km]", fontsize=10)
        ax2.set_ylabel("L2 Residual Norm", fontsize=10)
        ax2.set_title("Physics Residual Norms vs. Orbital Radius", fontsize=11)
        ax2.grid(True, linestyle="--", alpha=0.15)
        ax2.legend()

        plt.tight_layout()
        fig.savefig(self.output_dir / "physics_residuals_trends.png", facecolor=self.colors["background"])
        plt.close(fig)

    def plot_loss_components(self, losses: Dict[str, float]) -> None:
        """Plots a comparison bar chart of the unweighted physics-informed loss components."""
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

        # Unweighted loss component values
        labels = ["Physics Residual", "Boundary", "Initial Condition", "Constraint Penalty"]
        values = [
            losses.get("physics_loss", 0.0),
            losses.get("boundary_loss", 0.0),
            losses.get("initial_loss", 0.0),
            losses.get("constraint_loss", 0.0),
        ]

        x = np.arange(len(labels))
        # Use log scale if there's a huge difference in magnitudes
        ax.bar(x, values, color=[self.colors["position"], self.colors["theory"], self.colors["velocity"], self.colors["acceleration"]], alpha=0.8, width=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("Loss Component Value (Log Scale)", fontsize=10)
        ax.set_yscale("log")
        ax.set_title("Physics-Informed Loss Components (Unweighted)", fontsize=12, fontweight="bold", pad=15)
        ax.grid(True, linestyle="--", alpha=0.15, axis="y")

        # Annotate actual values on top of bars
        for idx, val in enumerate(values):
            ax.text(idx, val * 1.1, f"{val:.2e}", ha="center", va="bottom", color="white", fontsize=9)

        plt.tight_layout()
        fig.savefig(self.output_dir / "physics_loss_components.png", facecolor=self.colors["background"])
        plt.close(fig)
