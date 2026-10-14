"""
src/physics/visualizer.py
==========================
Renders diagnostic plots for Phase 8 (Physics Engine).

Generates 9 dark-theme visualization plots under ``results/physics/``:
1. gravity_acceleration.png
2. residual_acceleration.png
3. state_transition_error.png
4. constraint_violations.png
5. orbital_energy.png
6. angular_momentum.png
7. physics_loss_breakdown.png
8. residual_histogram.png
9. acceleration_components.png
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

from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class PhysicsVisualizer:
    """
    Generates diagnostic and validation plots for physics engine execution.

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
        log.info("PhysicsVisualizer initialized. Output directory: %s", output_dir)

    def plot_all(
        self,
        times: np.ndarray,
        states: np.ndarray,
        gravity_acc: np.ndarray,
        residual_acc: np.ndarray,
        expected_transition: np.ndarray,
        actual_transition: np.ndarray,
        constraints: Dict[str, np.ndarray],
        orbital_energy: np.ndarray,
        angular_momentum: np.ndarray,
        losses: Dict[str, float] = None,
    ) -> None:
        """
        Generates and saves all 9 Phase 8 figures.
        """
        t_hours = times / 3600.0

        try:
            # 1. Gravity Acceleration Magnitude vs Time
            self._plot_gravity(t_hours, gravity_acc)

            # 2. Residual Acceleration Magnitude vs Time
            self._plot_residuals(t_hours, residual_acc)

            # 3. State Transition Error (Observed vs Physics) vs Time
            self._plot_transition_error(t_hours, expected_transition, actual_transition)

            # 4. Constraint Violations vs Time
            self._plot_constraints(t_hours, constraints)

            # 5. Orbital Energy vs Time
            self._plot_energy(t_hours, orbital_energy)

            # 6. Angular Momentum vs Time
            self._plot_momentum(t_hours, angular_momentum)

            # 7. Physics Loss Components
            if losses:
                self._plot_loss_breakdown(losses)

            # 8. Residual Histogram
            self._plot_residual_histogram(residual_acc)

            # 9. Acceleration Components (Gravity vs Transition vs Residual)
            self._plot_acceleration_components(t_hours, gravity_acc, residual_acc)

            log.info("All 9 physics diagnostic plots saved successfully.")
        except Exception as e:
            log.error("Failed to generate physics visualizations: %s", str(e), exc_info=True)

    def _plot_gravity(self, t_hours: np.ndarray, gravity_acc: np.ndarray) -> None:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
        g_norm = np.linalg.norm(gravity_acc, axis=-1)
        ax.plot(t_hours, g_norm, color=self.colors["theory"], lw=1.5, label="Gravity Acceleration")
        ax.set_xlabel("Elapsed Time [Hours]")
        ax.set_ylabel("Gravity Magnitude [km/s^2]")
        ax.set_title("Gravitational Acceleration vs. Time", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.15)
        fig.savefig(self.output_dir / "gravity_acceleration.png", facecolor=self.colors["background"])
        plt.close(fig)

    def _plot_residuals(self, t_hours: np.ndarray, residual_acc: np.ndarray) -> None:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
        res_norm = np.linalg.norm(residual_acc, axis=-1)
        ax.plot(t_hours, res_norm, color=self.colors["acceleration"], lw=1.5, label="Residual Magnitude")
        ax.set_xlabel("Elapsed Time [Hours]")
        ax.set_ylabel("Residual Acceleration [km/s^2]")
        ax.set_title("Residual Dynamics vs. Time", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.15)
        fig.savefig(self.output_dir / "residual_acceleration.png", facecolor=self.colors["background"])
        plt.close(fig)

    def _plot_transition_error(self, t_hours: np.ndarray, expected: np.ndarray, actual: np.ndarray) -> None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, dpi=150)
        
        pos_err = np.linalg.norm(actual[:, :3] - expected[:, :3], axis=-1)
        vel_err = np.linalg.norm(actual[:, 3:6] - expected[:, 3:6], axis=-1)

        ax1.plot(t_hours, pos_err, color=self.colors["position"], lw=1.2)
        ax1.set_ylabel("Position Delta Error [km]")
        ax1.set_title("Physics State Transition Prediction Error (Observed vs. Two-Body Physics)")
        ax1.grid(True, linestyle="--", alpha=0.15)

        ax2.plot(t_hours, vel_err, color=self.colors["velocity"], lw=1.2)
        ax2.set_xlabel("Elapsed Time [Hours]")
        ax2.set_ylabel("Velocity Delta Error [km/s]")
        ax2.grid(True, linestyle="--", alpha=0.15)

        plt.tight_layout()
        fig.savefig(self.output_dir / "state_transition_error.png", facecolor=self.colors["background"])
        plt.close(fig)

    def _plot_constraints(self, t_hours: np.ndarray, constraints: Dict[str, np.ndarray]) -> None:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
        # Plot total penalty over time
        penalty = constraints.get("total_constraint_penalty", np.zeros_like(t_hours))
        ax.plot(t_hours, penalty, color=self.colors["acceleration"], lw=1.5, label="Constraint Penalty")
        ax.set_xlabel("Elapsed Time [Hours]")
        ax.set_ylabel("Penalty Value")
        ax.set_title("Constraint Boundary Violations vs. Time", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.15)
        fig.savefig(self.output_dir / "constraint_violations.png", facecolor=self.colors["background"])
        plt.close(fig)

    def _plot_energy(self, t_hours: np.ndarray, energy: np.ndarray) -> None:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
        # Show normalized variation in energy to show conservation level
        e_init = energy[0] if len(energy) > 0 else 1.0
        e_rel_diff = (energy - e_init) / abs(e_init) if abs(e_init) > 0 else energy
        ax.plot(t_hours, e_rel_diff, color=self.colors["position"], lw=1.5)
        ax.set_xlabel("Elapsed Time [Hours]")
        ax.set_ylabel("Relative Specific Energy Change (E - E0) / |E0|")
        ax.set_title("Orbital Specific Mechanical Energy Conservation Check", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.15)
        fig.savefig(self.output_dir / "orbital_energy.png", facecolor=self.colors["background"])
        plt.close(fig)

    def _plot_momentum(self, t_hours: np.ndarray, momentum: np.ndarray) -> None:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
        h_norm = np.linalg.norm(momentum, axis=-1)
        h_init = h_norm[0] if len(h_norm) > 0 else 1.0
        h_rel_diff = (h_norm - h_init) / h_init if h_init > 0 else h_norm
        ax.plot(t_hours, h_rel_diff, color=self.colors["velocity"], lw=1.5)
        ax.set_xlabel("Elapsed Time [Hours]")
        ax.set_ylabel("Relative Angular Momentum Change (h - h0) / h0")
        ax.set_title("Orbital Specific Angular Momentum Conservation Check", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.15)
        fig.savefig(self.output_dir / "angular_momentum.png", facecolor=self.colors["background"])
        plt.close(fig)

    def _plot_loss_breakdown(self, losses: Dict[str, float]) -> None:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        labels = ["Transition", "Physics", "Residual", "Constraint"]
        values = [
            losses.get("transition_loss", 0.0),
            losses.get("physics_loss", 0.0),
            losses.get("residual_loss", 0.0),
            losses.get("constraint_loss", 0.0),
        ]
        x = np.arange(len(labels))
        ax.bar(x, values, color=[self.colors["position"], self.colors["theory"], self.colors["velocity"], self.colors["acceleration"]], alpha=0.8, width=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Loss Component Value (Log Scale)")
        ax.set_yscale("log")
        ax.set_title("Physics-Informed Loss Components (Unweighted)", fontsize=12, fontweight="bold", pad=15)
        ax.grid(True, linestyle="--", alpha=0.15, axis="y")
        for idx, val in enumerate(values):
            ax.text(idx, val * 1.1, f"{val:.2e}", ha="center", va="bottom", color="white", fontsize=9)
        plt.tight_layout()
        fig.savefig(self.output_dir / "physics_loss_breakdown.png", facecolor=self.colors["background"])
        plt.close(fig)

    def _plot_residual_histogram(self, residual_acc: np.ndarray) -> None:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
        ax.hist(residual_acc[:, 0], bins=50, color=self.colors["position"], alpha=0.5, label="X Residual")
        ax.hist(residual_acc[:, 1], bins=50, color=self.colors["theory"], alpha=0.5, label="Y Residual")
        ax.hist(residual_acc[:, 2], bins=50, color=self.colors["velocity"], alpha=0.5, label="Z Residual")
        ax.set_xlabel("Acceleration Residual [km/s^2]")
        ax.set_ylabel("Frequency")
        ax.set_title("Acceleration Residual Components Distribution", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.15)
        ax.legend()
        fig.savefig(self.output_dir / "residual_histogram.png", facecolor=self.colors["background"])
        plt.close(fig)

    def _plot_acceleration_components(self, t_hours: np.ndarray, gravity_acc: np.ndarray, residual_acc: np.ndarray) -> None:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        g_norm = np.linalg.norm(gravity_acc, axis=-1)
        res_norm = np.linalg.norm(residual_acc, axis=-1)
        
        ax.plot(t_hours, g_norm, color=self.colors["theory"], lw=1.5, label="Theoretical Two-Body Gravity")
        ax.plot(t_hours, res_norm, color=self.colors["acceleration"], lw=1.5, label="Residual Dynamics acceleration")
        
        ax.set_xlabel("Elapsed Time [Hours]")
        ax.set_ylabel("Acceleration Magnitude [km/s^2]")
        ax.set_yscale("log")
        ax.set_title("Acceleration Component Comparison (Log Scale)", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.15)
        ax.legend()
        fig.savefig(self.output_dir / "acceleration_components.png", facecolor=self.colors["background"])
        plt.close(fig)
