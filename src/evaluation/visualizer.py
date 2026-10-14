"""
src/evaluation/visualizer.py
==============================
High-resolution diagnostic visualization for Phase 9 evaluation.

Generates the following figures:
  1. trajectory_comparison_3d.png   — 3D orbit trajectory (pred vs truth)
  2. position_error_timeseries.png  — Position error per axis + norm over time
  3. velocity_error_timeseries.png  — Velocity error per axis + norm over time
  4. physics_residuals.png          — ODE physics residual norm over time
  5. orbital_drift.png              — Orbital element drift over time
  6. error_distribution.png        — Error histogram (position + velocity)

All figures are saved to the configured figures directory with configurable DPI.
Matplotlib is used exclusively; no PyTorch dependency.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend — safe on all platforms
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    log.warning("Matplotlib not found. Visualization will be skipped.")


class EvaluationVisualizer:
    """
    Produces all Phase 9 diagnostic plots.

    Parameters
    ----------
    cfg : Dict[str, Any]
        Master project configuration.
    output_dir : Path
        Directory where figures are saved.
    """

    def __init__(self, cfg: Dict[str, Any], output_dir: Path) -> None:
        self._cfg = cfg
        self._eval_cfg = cfg.get("evaluation", {})
        self._vis_cfg = self._eval_cfg.get("visualization", {})
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._dpi = self._vis_cfg.get("dpi", 150)
        self._style = self._vis_cfg.get("style", "dark_background")
        self._colors = self._vis_cfg.get("colors", {})

        log.info("EvaluationVisualizer initialised. Output dir: %s", self.output_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_all(
        self,
        times: np.ndarray,
        predicted: np.ndarray,
        truth: np.ndarray,
        metrics: Dict[str, Any],
        orbital_drift_data: Optional[Dict[str, np.ndarray]] = None,
        physics_residuals_ts: Optional[np.ndarray] = None,
    ) -> None:
        """
        Generate and save all diagnostic figures.

        Parameters
        ----------
        times : np.ndarray, shape (N,)
        predicted : np.ndarray, shape (N, 6)
        truth : np.ndarray, shape (N, 6)
        metrics : Dict[str, Any]
        orbital_drift_data : Dict[str, np.ndarray], optional
            Batch orbital elements from OrbitalElementsCalculator.
        physics_residuals_ts : np.ndarray, optional
            Per-step physics residual norms, shape (N-1,).
        """
        if not MATPLOTLIB_AVAILABLE:
            log.warning("Matplotlib not available. Skipping visualizations.")
            return

        days = times / 86400.0  # convert seconds → days for x-axis

        self._plot_3d_trajectory(predicted, truth)
        self._plot_position_errors(days, predicted, truth)
        self._plot_velocity_errors(days, predicted, truth)
        self._plot_error_distribution(predicted, truth)

        if physics_residuals_ts is not None:
            mid_days = 0.5 * (days[:-1] + days[1:])
            self._plot_physics_residuals(mid_days, physics_residuals_ts)

        if orbital_drift_data is not None:
            self._plot_orbital_drift(days, orbital_drift_data)

        log.info("All evaluation figures saved to: %s", self.output_dir)

    # ------------------------------------------------------------------
    # Individual figure generators
    # ------------------------------------------------------------------

    def _plot_3d_trajectory(
        self, predicted: np.ndarray, truth: np.ndarray
    ) -> None:
        figsize = self._vis_cfg.get("figsize", {}).get("trajectory_3d", [12, 10])
        fig = plt.figure(figsize=figsize, facecolor="#0d0d1a")

        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("#0d0d1a")

        c_pred  = self._colors.get("predicted", "#00D4FF")
        c_truth = self._colors.get("truth", "#FFE66D")

        ax.plot(truth[:, 0], truth[:, 1], truth[:, 2],
                color=c_truth, linewidth=1.5, label="Ground Truth", alpha=0.9)
        ax.plot(predicted[:, 0], predicted[:, 1], predicted[:, 2],
                color=c_pred, linewidth=1.5, linestyle="--", label="PINN Predicted", alpha=0.9)

        # Mark start / end
        ax.scatter(*truth[0, :3], color=c_truth, s=60, zorder=5, label="Start")
        ax.scatter(*truth[-1, :3], color="#FF6B6B", s=60, zorder=5, label="End")

        ax.set_xlabel("X (km)", color="white", labelpad=8)
        ax.set_ylabel("Y (km)", color="white", labelpad=8)
        ax.set_zlabel("Z (km)", color="white", labelpad=8)
        ax.tick_params(colors="white")
        ax.set_title("2-Day Future Trajectory: PINN vs Ground Truth\n(GCRS Frame)",
                     color="white", fontsize=13, pad=15)
        ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=10)

        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("#333355")

        plt.tight_layout()
        self._save(fig, "trajectory_comparison_3d.png")

    def _plot_position_errors(
        self, days: np.ndarray, predicted: np.ndarray, truth: np.ndarray
    ) -> None:
        figsize = self._vis_cfg.get("figsize", {}).get("position_error", [16, 10])
        c_err  = self._colors.get("error", "#FF6B6B")

        pos_err = predicted[:, :3] - truth[:, :3]
        err_norm = np.linalg.norm(pos_err, axis=1)
        labels = ["X (km)", "Y (km)", "Z (km)"]
        comp_colors = ["#FF6B6B", "#4ECDC4", "#FFE66D"]

        with plt.style.context(self._style):
            fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
            fig.suptitle("Position Error — 2-Day Prediction Horizon", fontsize=14, y=1.01)

            for i, (ax, lbl, col) in enumerate(zip(axes[:3], labels, comp_colors)):
                ax.plot(days, pos_err[:, i], color=col, linewidth=0.8)
                ax.axhline(0, color="white", linewidth=0.4, linestyle="--", alpha=0.4)
                ax.set_ylabel(f"Δ{lbl}", fontsize=9)
                ax.grid(True, alpha=0.2)

            axes[3].plot(days, err_norm, color=c_err, linewidth=1.2)
            axes[3].set_ylabel("‖Δpos‖ (km)", fontsize=9)
            axes[3].set_xlabel("Elapsed Time (days)", fontsize=10)
            axes[3].grid(True, alpha=0.2)

            plt.tight_layout()
            self._save(fig, "position_error_timeseries.png")

    def _plot_velocity_errors(
        self, days: np.ndarray, predicted: np.ndarray, truth: np.ndarray
    ) -> None:
        figsize = self._vis_cfg.get("figsize", {}).get("velocity_error", [16, 10])
        c_err  = self._colors.get("error", "#FF6B6B")

        vel_err = predicted[:, 3:] - truth[:, 3:]
        err_norm = np.linalg.norm(vel_err, axis=1)
        labels = ["Vx (km/s)", "Vy (km/s)", "Vz (km/s)"]
        comp_colors = ["#FF6B6B", "#4ECDC4", "#FFE66D"]

        with plt.style.context(self._style):
            fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
            fig.suptitle("Velocity Error — 2-Day Prediction Horizon", fontsize=14, y=1.01)

            for i, (ax, lbl, col) in enumerate(zip(axes[:3], labels, comp_colors)):
                ax.plot(days, vel_err[:, i], color=col, linewidth=0.8)
                ax.axhline(0, color="white", linewidth=0.4, linestyle="--", alpha=0.4)
                ax.set_ylabel(f"Δ{lbl}", fontsize=9)
                ax.grid(True, alpha=0.2)

            axes[3].plot(days, err_norm, color=c_err, linewidth=1.2)
            axes[3].set_ylabel("‖Δvel‖ (km/s)", fontsize=9)
            axes[3].set_xlabel("Elapsed Time (days)", fontsize=10)
            axes[3].grid(True, alpha=0.2)

            plt.tight_layout()
            self._save(fig, "velocity_error_timeseries.png")

    def _plot_physics_residuals(
        self, mid_days: np.ndarray, residuals: np.ndarray
    ) -> None:
        figsize = self._vis_cfg.get("figsize", {}).get("residuals", [14, 6])
        c_res = self._colors.get("residual", "#4ECDC4")

        with plt.style.context(self._style):
            fig, ax = plt.subplots(figsize=figsize)
            ax.semilogy(mid_days, residuals + 1e-16, color=c_res, linewidth=1.0)
            ax.set_xlabel("Elapsed Time (days)", fontsize=11)
            ax.set_ylabel("Physics Residual ‖·‖ (log scale)", fontsize=11)
            ax.set_title("ODE Physics Consistency Residuals — 2-Day Horizon", fontsize=13)
            ax.grid(True, which="both", alpha=0.2)
            plt.tight_layout()
            self._save(fig, "physics_residuals.png")

    def _plot_orbital_drift(
        self, days: np.ndarray, orbital_data: Dict[str, np.ndarray]
    ) -> None:
        figsize = self._vis_cfg.get("figsize", {}).get("orbital_drift", [16, 8])
        plot_elements = [
            ("a_km_pred",    "a_km_true",    "Semi-major axis (km)", "a"),
            ("e_pred",       "e_true",       "Eccentricity",          "e"),
            ("i_deg_pred",   "i_deg_true",   "Inclination (deg)",     "i"),
            ("raan_deg_pred","raan_deg_true","RAAN (deg)",            "Ω"),
        ]

        n_plots = sum(1 for p, t, *_ in plot_elements if p in orbital_data and t in orbital_data)
        if n_plots == 0:
            return

        with plt.style.context(self._style):
            fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
            if n_plots == 1:
                axes = [axes]
            fig.suptitle("Orbital Element Drift — 2-Day Horizon", fontsize=14)

            c_pred  = self._colors.get("predicted", "#00D4FF")
            c_truth = self._colors.get("truth", "#FFE66D")

            ax_idx = 0
            for pred_key, true_key, ylabel, symbol in plot_elements:
                if pred_key not in orbital_data or true_key not in orbital_data:
                    continue
                ax = axes[ax_idx]
                ax.plot(days, orbital_data[true_key], color=c_truth, linewidth=1.2,
                        label="Truth", alpha=0.9)
                ax.plot(days, orbital_data[pred_key], color=c_pred, linewidth=1.2,
                        linestyle="--", label="Predicted", alpha=0.9)
                ax.set_ylabel(ylabel, fontsize=9)
                ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
                ax.grid(True, alpha=0.2)
                ax_idx += 1

            axes[-1].set_xlabel("Elapsed Time (days)", fontsize=10)
            plt.tight_layout()
            self._save(fig, "orbital_drift.png")

    def _plot_error_distribution(
        self, predicted: np.ndarray, truth: np.ndarray
    ) -> None:
        pos_err_norm = np.linalg.norm(predicted[:, :3] - truth[:, :3], axis=1)
        vel_err_norm = np.linalg.norm(predicted[:, 3:] - truth[:, 3:], axis=1)

        c_pos = self._colors.get("error", "#FF6B6B")
        c_vel = self._colors.get("residual", "#4ECDC4")

        with plt.style.context(self._style):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle("Prediction Error Distribution — 2-Day Horizon", fontsize=13)

            ax1.hist(pos_err_norm, bins=80, color=c_pos, alpha=0.85, edgecolor="none")
            ax1.set_xlabel("Position Error Magnitude (km)", fontsize=10)
            ax1.set_ylabel("Count", fontsize=10)
            ax1.set_title("Position Error Distribution", fontsize=11)
            ax1.axvline(np.mean(pos_err_norm), color="#FFE66D", linewidth=1.5,
                        linestyle="--", label=f"Mean = {np.mean(pos_err_norm):.3f} km")
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.2)

            ax2.hist(vel_err_norm, bins=80, color=c_vel, alpha=0.85, edgecolor="none")
            ax2.set_xlabel("Velocity Error Magnitude (km/s)", fontsize=10)
            ax2.set_ylabel("Count", fontsize=10)
            ax2.set_title("Velocity Error Distribution", fontsize=11)
            ax2.axvline(np.mean(vel_err_norm), color="#FFE66D", linewidth=1.5,
                        linestyle="--", label=f"Mean = {np.mean(vel_err_norm):.5f} km/s")
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.2)

            plt.tight_layout()
            self._save(fig, "error_distribution.png")

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _save(self, fig: Any, filename: str) -> None:
        path = self.output_dir / filename
        fig.savefig(path, dpi=self._dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        log.info("Saved figure: %s", path)
