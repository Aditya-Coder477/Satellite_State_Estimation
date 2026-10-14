"""
src/evaluation/INSAT3D/visualizer.py
====================================
Generates publication-quality dark-theme plots for Phase 10 Model Evaluation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict
import matplotlib.pyplot as plt
import numpy as np

# Setup headless matplotlib
plt.switch_backend("Agg")

from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class EvaluationVisualizer:
    """
    Renders 10 publication-quality verification figures.
    """

    def __init__(self, output_dir: Path, cfg: Dict[str, Any]) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        plot_cfg = cfg.get("evaluation", {}).get("plots", {})
        self.dpi = int(plot_cfg.get("dpi", 200))
        self.dark_theme = bool(plot_cfg.get("dark_theme", True))

        if self.dark_theme:
            plt.style.use("dark_background")
            self.colors = {
                "truth": "#00D4FF",      # cyan
                "pred": "#FFE66D",       # yellow
                "error": "#FF6B6B",      # red
                "physics": "#4ECDC4",    # teal
                "card": "#121212",
                "text": "#FFFFFF"
            }
        else:
            plt.style.use("default")
            self.colors = {
                "truth": "#0077B6",      # blue
                "pred": "#F77F00",       # orange
                "error": "#D62828",      # red
                "physics": "#2A9D8F",    # dark teal
                "card": "#F8F9FA",
                "text": "#000000"
            }

        log.info("EvaluationVisualizer initialised. DPI=%d, DarkTheme=%s", self.dpi, self.dark_theme)

    def plot_all(
        self,
        true_deltas: np.ndarray,
        pred_deltas: np.ndarray,
        true_states: np.ndarray,
        pred_states: np.ndarray,
        physics_residuals: np.ndarray,
    ) -> None:
        """Generates all 10 evaluation plots."""
        try:
            samples = np.arange(len(true_deltas))
            
            # Figure 1: Prediction vs Truth (Position Transitions)
            self._plot_prediction_vs_truth(samples, true_deltas, pred_deltas)

            # Figure 2: 3D Orbit Comparison
            self._plot_3d_orbit(true_states, pred_states)

            # Figure 3 & 4: Position and Velocity Errors
            self._plot_error_vs_sample(samples, true_states, pred_states)

            # Figure 5: Physics Residual vs Sample
            self._plot_physics_residual_vs_sample(samples, physics_residuals)

            # Figure 6: Trajectory Drift vs Sample
            self._plot_trajectory_drift(samples, true_states, pred_states)

            # Figure 7: Orbital Elements Comparison
            self._plot_orbital_elements_comparison(true_states, pred_states)

            # Figure 8: Residual Histogram
            self._plot_residual_histogram(physics_residuals)

            # Figure 9: RMSE per Coordinate Bar Chart
            self._plot_rmse_bar_chart(true_deltas, pred_deltas)

            # Figure 10: Prediction Scatter Plot
            self._plot_prediction_scatter(true_deltas, pred_deltas)

            log.info("All 10 Phase 10 evaluation figures generated successfully.")
        except Exception as e:
            log.error("Failed to generate evaluation plots: %s", str(e), exc_info=True)

    def _plot_prediction_vs_truth(self, samples: np.ndarray, true_deltas: np.ndarray, pred_deltas: np.ndarray) -> None:
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, dpi=self.dpi)
        coords = ["X", "Y", "Z"]
        limit = min(len(samples), 300)
        
        for idx, ax in enumerate(axes):
            ax.plot(samples[:limit], true_deltas[:limit, idx], label="Ground Truth", color=self.colors["truth"], alpha=0.9, lw=1.2)
            ax.plot(samples[:limit], pred_deltas[:limit, idx], label="Predicted", color=self.colors["pred"], linestyle="--", alpha=0.9, lw=1.2)
            ax.set_ylabel(f"Δ{coords[idx]} [km]")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, linestyle="--", alpha=0.15)
            
        axes[-1].set_xlabel("Sample Index")
        fig.suptitle("Transition Output Predictions vs Ground Truth", fontsize=12, fontweight="bold")
        plt.tight_layout()
        fig.savefig(self.output_dir / "prediction_vs_truth.png")
        plt.close(fig)

    def _plot_3d_orbit(self, true_states: np.ndarray, pred_states: np.ndarray) -> None:
        fig = plt.figure(figsize=(10, 8), dpi=self.dpi)
        ax = fig.add_subplot(111, projection="3d")
        
        # Plot orbits
        ax.plot(true_states[:, 0], true_states[:, 1], true_states[:, 2], label="True Orbit", color=self.colors["truth"], lw=1.5)
        ax.plot(pred_states[:, 0], pred_states[:, 1], pred_states[:, 2], label="Predicted Orbit (Rollout)", color=self.colors["pred"], linestyle="--", lw=1.5)
        
        # Earth sphere represent
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x_e = 6378.137 * np.outer(np.cos(u), np.sin(v))
        y_e = 6378.137 * np.outer(np.sin(u), np.sin(v))
        z_e = 6378.137 * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x_e, y_e, z_e, color="blue", alpha=0.1)

        ax.set_xlabel("X (GCRS) [km]")
        ax.set_ylabel("Y (GCRS) [km]")
        ax.set_zlabel("Z (GCRS) [km]")
        ax.set_title("3D Trajectory Orbit Rollout Comparison", fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.1)
        plt.tight_layout()
        fig.savefig(self.output_dir / "3d_orbit_comparison.png")
        plt.close(fig)

    def _plot_error_vs_sample(self, samples: np.ndarray, true_states: np.ndarray, pred_states: np.ndarray) -> None:
        # Position Error
        fig, ax = plt.subplots(figsize=(10, 4), dpi=self.dpi)
        pos_err = np.linalg.norm(true_states[:, :3] - pred_states[:, :3], axis=1)
        ax.plot(samples, pos_err, color=self.colors["error"], lw=1.2)
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Euclidean Position Error [km]")
        ax.set_title("Euclidean Position Error Over Time", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.15)
        plt.tight_layout()
        fig.savefig(self.output_dir / "position_error.png")
        plt.close(fig)

        # Velocity Error
        fig, ax = plt.subplots(figsize=(10, 4), dpi=self.dpi)
        vel_err = np.linalg.norm(true_states[:, 3:6] - pred_states[:, 3:6], axis=1)
        ax.plot(samples, vel_err, color=self.colors["error"], lw=1.2)
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Euclidean Velocity Error [km/s]")
        ax.set_title("Euclidean Velocity Error Over Time", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.15)
        plt.tight_layout()
        fig.savefig(self.output_dir / "velocity_error.png")
        plt.close(fig)

    def _plot_physics_residual_vs_sample(self, samples: np.ndarray, residuals: np.ndarray) -> None:
        fig, ax = plt.subplots(figsize=(10, 4), dpi=self.dpi)
        res_norms = np.linalg.norm(residuals, axis=1)
        ax.plot(samples, res_norms, color=self.colors["physics"], lw=1.2)
        ax.set_yscale("log")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Residual Acceleration Magnitude [km/s²]")
        ax.set_title("Physics Acceleration Residual Deviation Over Time", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.15)
        plt.tight_layout()
        fig.savefig(self.output_dir / "physics_residual.png")
        plt.close(fig)

    def _plot_trajectory_drift(self, samples: np.ndarray, true_states: np.ndarray, pred_states: np.ndarray) -> None:
        fig, ax = plt.subplots(figsize=(10, 4), dpi=self.dpi)
        drift = np.linalg.norm(true_states[:, :3] - pred_states[:, :3], axis=1)
        ax.plot(samples, drift, color=self.colors["error"], lw=1.5, label="Cumulative Orbit Drift")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Drift [km]")
        ax.set_title("Trajectory Drift / Rollout Deviation Over Time", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.15)
        ax.legend()
        plt.tight_layout()
        fig.savefig(self.output_dir / "orbit_drift.png")
        plt.close(fig)

    def _plot_orbital_elements_comparison(self, true_states: np.ndarray, pred_states: np.ndarray) -> None:
        from src.evaluation.orbital_elements import OrbitalElementsCalculator
        calc = OrbitalElementsCalculator()
        true_elems = calc.batch_to_elements(true_states)
        pred_elems = calc.batch_to_elements(pred_states)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=self.dpi)
        elements = [("a_km", "Semi-major Axis [km]"), ("e", "Eccentricity"), 
                    ("i_deg", "Inclination [deg]"), ("raan_deg", "RAAN [deg]")]
        
        limit = min(len(true_states), 500)
        samples = np.arange(limit)

        for idx, (key, label) in enumerate(elements):
            row, col = idx // 2, idx % 2
            axes[row, col].plot(samples, true_elems[key][:limit], label="True", color=self.colors["truth"], lw=1.2)
            axes[row, col].plot(samples, pred_elems[key][:limit], label="Pred", color=self.colors["pred"], linestyle="--", lw=1.2)
            axes[row, col].set_title(label, fontsize=10, fontweight="bold")
            axes[row, col].legend(fontsize=8)
            axes[row, col].grid(True, linestyle="--", alpha=0.15)

        fig.suptitle("Classical Keplerian Orbital Elements Comparison (Test Rollout)", fontsize=12, fontweight="bold")
        plt.tight_layout()
        fig.savefig(self.output_dir / "orbital_elements_comparison.png")
        plt.close(fig)

    def _plot_residual_histogram(self, residuals: np.ndarray) -> None:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=self.dpi)
        res_norms = np.linalg.norm(residuals, axis=1)
        ax.hist(res_norms, bins=50, color=self.colors["physics"], alpha=0.7, edgecolor="black")
        ax.set_yscale("log")
        ax.set_xlabel("Acceleration Error [km/s²]")
        ax.set_ylabel("Count (Log Scale)")
        ax.set_title("Distribution of Physics Acceleration Residuals", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.15)
        plt.tight_layout()
        fig.savefig(self.output_dir / "residual_histogram.png")
        plt.close(fig)

    def _plot_rmse_bar_chart(self, true_deltas: np.ndarray, pred_deltas: np.ndarray) -> None:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=self.dpi)
        
        # Calculate RMSE per coordinate
        rmse_vals = np.sqrt(np.mean((true_deltas - pred_deltas) ** 2, axis=0))
        labels = ["Δx", "Δy", "Δz", "Δvx", "Δvy", "Δvz"]
        
        # Create dual bars (Position vs Velocity on different axes or just side-by-side)
        ax.bar(labels[:3], rmse_vals[:3], color=self.colors["truth"], alpha=0.8, width=0.4, label="Position [km]")
        ax2 = ax.twinx()
        ax2.bar(labels[3:], rmse_vals[3:6], color=self.colors["pred"], alpha=0.8, width=0.4, label="Velocity [km/s]")
        
        ax.set_ylabel("Position RMSE [km]")
        ax2.set_ylabel("Velocity RMSE [km/s]")
        ax.set_title("Transition Output Root Mean Squared Error (RMSE) per Coordinate", fontsize=11, fontweight="bold")
        
        # Combined legends
        lines, labels_l = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels_l + labels2, loc="upper right")
        
        plt.tight_layout()
        fig.savefig(self.output_dir / "rmse_bar_chart.png")
        plt.close(fig)

    def _plot_prediction_scatter(self, true_deltas: np.ndarray, pred_deltas: np.ndarray) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=self.dpi)
        
        # Position transition scatter
        t_pos = true_deltas[:, :3].flatten()
        p_pos = pred_deltas[:, :3].flatten()
        axes[0].scatter(t_pos, p_pos, color=self.colors["truth"], alpha=0.3, s=2)
        min_v, max_v = min(t_pos.min(), p_pos.min()), max(t_pos.max(), p_pos.max())
        axes[0].plot([min_v, max_v], [min_v, max_v], color=self.colors["error"], linestyle="--", lw=1.5)
        axes[0].set_xlabel("True ΔPosition [km]")
        axes[0].set_ylabel("Predicted ΔPosition [km]")
        axes[0].set_title("Position Transition Scatter", fontsize=10, fontweight="bold")
        axes[0].grid(True, linestyle="--", alpha=0.15)

        # Velocity transition scatter
        t_vel = true_deltas[:, 3:6].flatten()
        p_vel = pred_deltas[:, 3:6].flatten()
        axes[1].scatter(t_vel, p_vel, color=self.colors["pred"], alpha=0.3, s=2)
        min_v, max_v = min(t_vel.min(), p_vel.min()), max(t_vel.max(), p_vel.max())
        axes[1].plot([min_v, max_v], [min_v, max_v], color=self.colors["error"], linestyle="--", lw=1.5)
        axes[1].set_xlabel("True ΔVelocity [km/s]")
        axes[1].set_ylabel("Predicted ΔVelocity [km/s]")
        axes[1].set_title("Velocity Transition Scatter", fontsize=10, fontweight="bold")
        axes[1].grid(True, linestyle="--", alpha=0.15)

        fig.suptitle("Prediction vs Ground Truth Correlation Plots", fontsize=12, fontweight="bold")
        plt.tight_layout()
        fig.savefig(self.output_dir / "prediction_scatter.png")
        plt.close(fig)
