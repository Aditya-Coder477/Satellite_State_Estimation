"""
src/pinn/visualizer.py
======================
Renders training diagnostics and prediction plots for Phase 8 (PINN Framework).

Generates dark-theme diagnostic PNG plots under ``results/training/``:
1. Training and validation loss histories (total and components)
2. Trajectory coordinate comparisons (GCRS position and velocity components)
3. Orbit prediction error norms over time
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

# Headless matplotlib setup
plt.switch_backend("Agg")
plt.style.use("dark_background")

from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class PINNVisualizer:
    """
    Renders diagnostic and trajectory plots for PINN training runs.

    Parameters
    ----------
    output_dir : Path
        Target directory to save PNG figures.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.colors = {
            "train": "#00D4FF",         # cyan
            "validation": "#FFE66D",    # yellow
            "physics": "#4ECDC4",       # teal
            "initial": "#9B5DE5",       # purple
            "constraint": "#FF6B6B",    # red
            "true": "#FFFFFF",          # white
            "pred": "#F15BB5",          # magenta
            "background": "#121212",
        }
        log.info("PINNVisualizer initialised. Output directory: %s", output_dir)

    def plot_loss_history(self, history: Dict[str, List[float]]) -> None:
        """
        Plots training and validation total loss and loss components over epochs.

        Parameters
        ----------
        history : Dict[str, List[float]]
            Loss history dictionary recorded by the LossHistory callback.
        """
        if not history:
            log.warning("Empty loss history passed to visualizer. Skipping plot.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
        
        epochs = np.arange(1, len(next(iter(history.values()))) + 1)

        # 1. Total Loss (Train vs Val)
        if "total_loss" in history:
            ax1.plot(epochs, history["total_loss"], color=self.colors["train"], label="Train Total Loss", alpha=0.9, linewidth=1.5)
        if "val_total_loss" in history:
            ax1.plot(epochs, history["val_total_loss"], color=self.colors["validation"], label="Val Total Loss", alpha=0.9, linestyle="--", linewidth=1.5)
        
        ax1.set_xlabel("Epochs", fontsize=10)
        ax1.set_ylabel("Loss (Log Scale)", fontsize=10)
        ax1.set_yscale("log")
        ax1.set_title("Total Training vs. Validation Loss", fontsize=11, fontweight="bold")
        ax1.grid(True, linestyle="--", alpha=0.15)
        ax1.legend()

        # 2. Loss Components Breakdown (Train)
        components = {
            "observation_loss": ("Observation Loss", self.colors["train"]),
            "physics_loss": ("Physics Loss", self.colors["physics"]),
            "boundary_loss": ("Boundary Loss", self.colors["validation"]),
            "initial_loss": ("Initial Loss", self.colors["initial"]),
            "constraint_loss": ("Constraint Loss", self.colors["constraint"]),
        }

        for key, (label, color) in components.items():
            if key in history:
                ax2.plot(epochs, history[key], color=color, label=label, alpha=0.7, linewidth=1.2)

        ax2.set_xlabel("Epochs", fontsize=10)
        ax2.set_ylabel("Loss (Log Scale)", fontsize=10)
        ax2.set_yscale("log")
        ax2.set_title("Loss Components Breakdown", fontsize=11, fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.15)
        ax2.legend()

        fig.suptitle("PINN Loss Convergence Diagnostics", fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout()
        fig.savefig(self.output_dir / "pinn_loss_history.png", facecolor=self.colors["background"])
        plt.close(fig)

    def plot_trajectory_comparison(
        self,
        times: np.ndarray,
        states_true: np.ndarray,
        states_pred: np.ndarray,
    ) -> None:
        """
        Compares predicted vs true satellite Cartesian coordinates over time.

        Parameters
        ----------
        times : np.ndarray
            Time stamps array of shape (N,).
        states_true : np.ndarray
            True state trajectories array of shape (N, 6).
        states_pred : np.ndarray
            Predicted state trajectories array of shape (N, 6).
        """
        fig, axes = plt.subplots(3, 2, figsize=(14, 10), dpi=150, sharex=True)
        t_hours = times / 3600.0

        labels_pos = ["X (GCRS) [km]", "Y (GCRS) [km]", "Z (GCRS) [km]"]
        labels_vel = ["Vx (GCRS) [km/s]", "Vy (GCRS) [km/s]", "Vz (GCRS) [km/s]"]

        # Plot Positions
        for idx in range(3):
            ax = axes[idx, 0]
            ax.plot(t_hours, states_true[:, idx], color=self.colors["true"], label="Ground Truth", alpha=0.8, linewidth=1.5)
            ax.plot(t_hours, states_pred[:, idx], color=self.colors["pred"], label="PINN Prediction", alpha=0.8, linestyle="--", linewidth=1.5)
            ax.set_ylabel(labels_pos[idx], fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.15)
            if idx == 0:
                ax.legend(loc="upper right", framealpha=0.8)

        # Plot Velocities
        for idx in range(3):
            ax = axes[idx, 1]
            ax.plot(t_hours, states_true[:, idx + 3], color=self.colors["true"], label="Ground Truth", alpha=0.8, linewidth=1.5)
            ax.plot(t_hours, states_pred[:, idx + 3], color=self.colors["pred"], label="PINN Prediction", alpha=0.8, linestyle="--", linewidth=1.5)
            ax.set_ylabel(labels_vel[idx], fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.15)
            if idx == 0:
                ax.legend(loc="upper right", framealpha=0.8)

        axes[2, 0].set_xlabel("Elapsed Time [Hours]", fontsize=10)
        axes[2, 1].set_xlabel("Elapsed Time [Hours]", fontsize=10)
        
        fig.suptitle("Predicted vs. Ground-Truth Trajectory Coordinates Comparison", fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout()
        fig.savefig(self.output_dir / "pinn_trajectory_comparison.png", facecolor=self.colors["background"])
        plt.close(fig)

    def plot_prediction_errors(
        self,
        times: np.ndarray,
        states_true: np.ndarray,
        states_pred: np.ndarray,
    ) -> None:
        """
        Plots L2 prediction error norms for positions and velocities over time.

        Parameters
        ----------
        times : np.ndarray
            Time stamps array of shape (N,).
        states_true : np.ndarray
            True state trajectory of shape (N, 6).
        states_pred : np.ndarray
            Predicted state trajectory of shape (N, 6).
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
        t_hours = times / 3600.0

        # Position error norm
        pos_err_norm = np.linalg.norm(states_pred[:, :3] - states_true[:, :3], axis=-1)
        # Velocity error norm
        vel_err_norm = np.linalg.norm(states_pred[:, 3:6] - states_true[:, 3:6], axis=-1)

        ax1.plot(t_hours, pos_err_norm, color=self.colors["constraint"], alpha=0.9, linewidth=1.5)
        ax1.set_xlabel("Elapsed Time [Hours]", fontsize=10)
        ax1.set_ylabel("Position L2 Deviation [km]", fontsize=10)
        ax1.set_title("Position Prediction Error Over Time", fontsize=11, fontweight="bold")
        ax1.grid(True, linestyle="--", alpha=0.15)

        ax2.plot(t_hours, vel_err_norm, color=self.colors["physics"], alpha=0.9, linewidth=1.5)
        ax2.set_xlabel("Elapsed Time [Hours]", fontsize=10)
        ax2.set_ylabel("Velocity L2 Deviation [km/s]", fontsize=10)
        ax2.set_title("Velocity Prediction Error Over Time", fontsize=11, fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.15)

        fig.suptitle("PINN Orbit Trajectory Error Norms", fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout()
        fig.savefig(self.output_dir / "pinn_prediction_errors.png", facecolor=self.colors["background"])
        plt.close(fig)
