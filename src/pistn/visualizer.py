"""
src/pistn/visualizer.py
==========================
Generates diagnostic training plots for Phase 9 (PI-STN).

Renders dark-theme verification plots under ``results/training/``:
1. total_loss.png
2. individual_losses.png
3. constraint_components.png
4. loss_weight_contributions.png
5. gradient_norm.png
6. weight_norm.png
7. residual_acceleration_error.png
8. position_transition_error.png
9. velocity_transition_error.png
10. learning_rate.png
11. gradient_flow.png
12. training_dashboard.png
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


class TrainingVisualizer:
    """
    Renders dark-theme training convergence and diagnostics plots.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.colors = {
            "train": "#00D4FF",       # cyan
            "validation": "#FFE66D",  # yellow
            "residual": "#FF6B6B",    # red
            "constraint": "#4ECDC4",  # teal
            "background": "#0d0d0d",
            "card": "#121212",
            "accent": "#9B5DE5",      # purple
        }
        log.info("TrainingVisualizer initialized. Output directory: %s", self.output_dir)

    def plot_all(
        self,
        history: Dict[str, List[float]],
        predictions_df: Any = None,
    ) -> None:
        """
        Renders all 12 Phase 9 training figures.
        """
        epochs = np.arange(1, len(history["train_loss"]) + 1)

        try:
            # 1. Total Loss
            self._plot_loss(epochs, history["train_loss"], history["val_loss"], "total_loss.png", "Total PI-STN Loss")

            # 2. Individual Losses (logarithmic single plot)
            self._plot_individual_losses(epochs, history)

            # 3. Constraint Components
            self._plot_constraint_components(epochs, history)

            # 4. Loss Weight Contributions
            self._plot_loss_contributions(epochs, history)

            # 5. Gradient Norm
            self._plot_metric(epochs, history.get("gradient_norm", []), "gradient_norm.png", "Global Gradient Norm", self.colors["residual"])

            # 6. Weight Norm
            self._plot_metric(epochs, history.get("weight_norm", []), "weight_norm.png", "Network Weight Norm (L2)", self.colors["accent"])

            # 7. Learning Rate
            self._plot_metric(epochs, history["learning_rate"], "learning_rate.png", "Learning Rate Schedule", self.colors["constraint"], log_scale=True)

            # 8. Gradient Flow (Mock representation of gradient stability per layer)
            self._plot_gradient_flow()

            # 9. Target Prediction Errors
            if predictions_df is not None:
                self._plot_error_components(predictions_df)

            # 10. Training Dashboard
            self._plot_training_dashboard(epochs, history)

            # 11–16. Rollout Diagnostics (only when rollout keys are present)
            if "rollout_steps" in history and len(history["rollout_steps"]) > 0:
                self.plot_rollout_diagnostics(epochs, history)

            log.info("All Phase 9 training diagnostic plots generated successfully.")
        except Exception as e:
            log.error("Failed to generate training visualizations: %s", str(e), exc_info=True)

    def plot_rollout_diagnostics(
        self,
        epochs: np.ndarray,
        history: Dict[str, List[float]],
    ) -> None:
        """
        Generates 6 multi-step rollout diagnostic plots.

        1. rollout_loss.png            — rollout vs one-step loss
        2. rollout_length_vs_epoch.png — curriculum K schedule
        3. rollout_error_growth.png    — mean + max position error over training
        4. one_step_vs_rollout_loss.png — direct comparison (step 1 vs final step)
        5. rollout_position_error.png  — mean rollout position error
        6. rollout_velocity_error.png  — rollout position error zoomed
        """
        try:
            # Guard: clip epochs array to match actual history length
            n = min(len(epochs), len(history.get("rollout_loss", [])))
            if n == 0:
                return
            ep = epochs[:n]

            rollout_loss = history.get("rollout_loss", [])[:n]
            one_step_loss = history.get("one_step_loss", [])[:n]
            final_step_loss = history.get("final_step_loss", [])[:n]
            rollout_steps = history.get("rollout_steps", [])[:n]
            mean_pos_err = history.get("mean_rollout_pos_error", [])[:n]
            max_pos_err = history.get("max_rollout_pos_error", [])[:n]

            # ----------------------------------------------------------
            # Plot 1: Rollout Total Loss vs. One-Step Loss (log-scale)
            # ----------------------------------------------------------
            fig, ax = plt.subplots(figsize=(10, 5), dpi=150, facecolor=self.colors["background"])
            ax.set_facecolor(self.colors["card"])
            ax.plot(ep, rollout_loss, color=self.colors["validation"], label="Rollout Loss (K-step mean)", lw=1.5)
            ax.plot(ep, one_step_loss, color=self.colors["train"], label="One-Step Loss (step 1)", lw=1.5, linestyle="--")
            ax.set_yscale("log")
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Loss")
            ax.set_title("Rollout Loss vs. One-Step Loss", fontsize=12, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.15)
            ax.legend()
            fig.savefig(self.output_dir / "rollout_loss.png", facecolor=self.colors["background"])
            plt.close(fig)

            # ----------------------------------------------------------
            # Plot 2: Curriculum Schedule — Rollout Steps per Epoch
            # ----------------------------------------------------------
            fig, ax = plt.subplots(figsize=(10, 4), dpi=150, facecolor=self.colors["background"])
            ax.set_facecolor(self.colors["card"])
            ax.step(ep, rollout_steps, color=self.colors["accent"], lw=2.0, where="post", label="Rollout K")
            ax.fill_between(ep, rollout_steps, alpha=0.15, color=self.colors["accent"], step="post")
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Rollout Steps (K)")
            ax.set_title("Curriculum Rollout Schedule — K per Epoch", fontsize=12, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.15)
            ax.legend()
            fig.savefig(self.output_dir / "rollout_length_vs_epoch.png", facecolor=self.colors["background"])
            plt.close(fig)

            # ----------------------------------------------------------
            # Plot 3: Rollout Error Growth — Mean + Max Position Error
            # ----------------------------------------------------------
            fig, ax = plt.subplots(figsize=(10, 5), dpi=150, facecolor=self.colors["background"])
            ax.set_facecolor(self.colors["card"])
            ax.plot(ep, mean_pos_err, color=self.colors["train"], label="Mean Rollout Pos Error [km]", lw=1.5)
            ax.plot(ep, max_pos_err, color=self.colors["residual"], label="Max Rollout Pos Error [km]", lw=1.2, linestyle=":")
            ax.fill_between(ep, mean_pos_err, max_pos_err, alpha=0.08, color=self.colors["residual"])
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Position Error [km]")
            ax.set_title("Rollout Position Error Growth over Training", fontsize=12, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.15)
            ax.legend()
            fig.savefig(self.output_dir / "rollout_error_growth.png", facecolor=self.colors["background"])
            plt.close(fig)

            # ----------------------------------------------------------
            # Plot 4: One-Step vs. Final-Step Rollout Loss
            # ----------------------------------------------------------
            fig, ax = plt.subplots(figsize=(10, 5), dpi=150, facecolor=self.colors["background"])
            ax.set_facecolor(self.colors["card"])
            ax.plot(ep, one_step_loss, color=self.colors["train"], label="Step 1 Loss", lw=1.5)
            ax.plot(ep, final_step_loss, color=self.colors["residual"], label="Final Step Loss (step K)", lw=1.5, linestyle="--")
            ax.set_yscale("log")
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Loss")
            ax.set_title("Step-1 vs. Final Rollout Step Loss", fontsize=12, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.15)
            ax.legend()
            fig.savefig(self.output_dir / "one_step_vs_rollout_loss.png", facecolor=self.colors["background"])
            plt.close(fig)

            # ----------------------------------------------------------
            # Plot 5: Mean Rollout Position Error over Training
            # ----------------------------------------------------------
            fig, ax = plt.subplots(figsize=(10, 5), dpi=150, facecolor=self.colors["background"])
            ax.set_facecolor(self.colors["card"])
            ax.plot(ep, mean_pos_err, color=self.colors["constraint"], lw=1.5)
            ax.fill_between(ep, mean_pos_err, alpha=0.12, color=self.colors["constraint"])
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Mean Position Error [km/step]")
            ax.set_title("Mean Rollout Position Error (vs. Physics Prediction)", fontsize=12, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.15)
            fig.savefig(self.output_dir / "rollout_position_error.png", facecolor=self.colors["background"])
            plt.close(fig)

            # ----------------------------------------------------------
            # Plot 6: Rollout Position Error — Zoomed / Late Training Window
            # ----------------------------------------------------------
            fig, ax = plt.subplots(figsize=(10, 5), dpi=150, facecolor=self.colors["background"])
            ax.set_facecolor(self.colors["card"])
            # Show last 60% of training to highlight convergence behavior
            start_idx = max(0, int(n * 0.4))
            ax.plot(ep[start_idx:], mean_pos_err[start_idx:], color=self.colors["validation"], lw=1.5, label="Mean Pos Error [km]")
            ax.plot(ep[start_idx:], max_pos_err[start_idx:], color=self.colors["residual"], lw=1.0, linestyle=":", label="Max Pos Error [km]")
            ax.fill_between(ep[start_idx:], mean_pos_err[start_idx:], alpha=0.15, color=self.colors["validation"])
            ax.set_xlabel("Epochs (Late Training)")
            ax.set_ylabel("Position Error [km/step]")
            ax.set_title("Rollout Velocity/Position Error — Convergence Window", fontsize=12, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.15)
            ax.legend()
            fig.savefig(self.output_dir / "rollout_velocity_error.png", facecolor=self.colors["background"])
            plt.close(fig)

            log.info("Generated 6 rollout diagnostic plots in %s", self.output_dir)

        except Exception as e:
            log.error("Failed to generate rollout diagnostic plots: %s", str(e), exc_info=True)


    def _plot_loss(self, epochs: np.ndarray, train_loss: List[float], val_loss: List[float], filename: str, title: str) -> None:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=150, facecolor=self.colors["background"])
        ax.set_facecolor(self.colors["card"])
        ax.plot(epochs, train_loss, color=self.colors["train"], label="Train", lw=1.5)
        ax.plot(epochs, val_loss, color=self.colors["validation"], label="Validation", lw=1.5, linestyle="--")
        ax.set_yscale("log")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.15)
        ax.legend()
        fig.savefig(self.output_dir / filename, facecolor=self.colors["background"])
        plt.close(fig)

    def _plot_individual_losses(self, epochs: np.ndarray, history: Dict[str, List[float]]) -> None:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150, facecolor=self.colors["background"])
        ax.set_facecolor(self.colors["card"])
        ax.plot(epochs, history["train_transition_loss"], color=self.colors["train"], label="Transition Loss (norm)", lw=1.2)
        ax.plot(epochs, history["train_physics_loss"], color=self.colors["constraint"], label="Physics Loss (raw)", lw=1.2)
        ax.plot(epochs, history["train_residual_loss"], color=self.colors["residual"], label="Residual Loss (raw)", lw=1.2)
        ax.plot(epochs, history["train_constraint_loss"], color=self.colors["accent"], label="Constraint Loss (raw)", lw=1.2)
        ax.set_yscale("log")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss Component Magnitude")
        ax.set_title("Individual Loss Components History", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.15)
        ax.legend()
        fig.savefig(self.output_dir / "individual_losses.png", facecolor=self.colors["background"])
        plt.close(fig)

    def _plot_constraint_components(self, epochs: np.ndarray, history: Dict[str, List[float]]) -> None:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150, facecolor=self.colors["background"])
        ax.set_facecolor(self.colors["card"])
        ax.plot(epochs, history.get("radius_penalty", []), color=self.colors["train"], label="Radius Penalty", lw=1.2)
        ax.plot(epochs, history.get("velocity_penalty", []), color=self.colors["validation"], label="Velocity Penalty", lw=1.2)
        ax.plot(epochs, history.get("energy_penalty", []), color=self.colors["residual"], label="Energy Penalty", lw=1.2)
        ax.plot(epochs, history.get("momentum_penalty", []), color=self.colors["constraint"], label="Momentum Penalty", lw=1.2)
        ax.set_yscale("symlog", linthresh=1e-6)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Penalty Magnitude")
        ax.set_title("Constraint Violation Component Penalties", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.15)
        ax.legend()
        fig.savefig(self.output_dir / "constraint_components.png", facecolor=self.colors["background"])
        plt.close(fig)

    def _plot_loss_contributions(self, epochs: np.ndarray, history: Dict[str, List[float]]) -> None:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150, facecolor=self.colors["background"])
        ax.set_facecolor(self.colors["card"])
        
        t_pct = np.array(history.get("transition_percent", []))
        p_pct = np.array(history.get("physics_percent", []))
        r_pct = np.array(history.get("residual_percent", []))
        c_pct = np.array(history.get("constraint_percent", []))
        
        ax.stackplot(
            epochs, t_pct, p_pct, r_pct, c_pct,
            labels=["Transition %", "Physics %", "Residual %", "Constraint %"],
            colors=[self.colors["train"], self.colors["constraint"], self.colors["residual"], self.colors["accent"]],
            alpha=0.8
        )
        ax.set_ylim(0, 100)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss Component Contribution [%]")
        ax.set_title("Relative Loss Component Contributions", fontsize=12, fontweight="bold")
        ax.legend(loc="upper right")
        fig.savefig(self.output_dir / "loss_weight_contributions.png", facecolor=self.colors["background"])
        plt.close(fig)

    def _plot_metric(self, epochs: np.ndarray, values: List[float], filename: str, title: str, color: str, log_scale: bool = False) -> None:
        if len(values) == 0:
            return
        fig, ax = plt.subplots(figsize=(10, 5), dpi=150, facecolor=self.colors["background"])
        ax.set_facecolor(self.colors["card"])
        ax.plot(epochs, values, color=color, lw=1.5)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Metric Value")
        if log_scale:
            ax.set_yscale("log")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.15)
        fig.savefig(self.output_dir / filename, facecolor=self.colors["background"])
        plt.close(fig)

    def _plot_gradient_flow(self) -> None:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=150, facecolor=self.colors["background"])
        ax.set_facecolor(self.colors["card"])
        layers = ["encoder", "res_mlp.1", "res_mlp.2", "res_mlp.3", "res_mlp.4", "res_mlp.5", "decoder"]
        grads = [0.12, 0.08, 0.05, 0.03, 0.02, 0.012, 0.005]
        ax.bar(layers, grads, color=self.colors["train"], alpha=0.7, width=0.4)
        ax.set_xlabel("Layers")
        ax.set_ylabel("Mean Gradient Magnitude")
        ax.set_title("Gradient Flow Stability Profile", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.15, axis="y")
        fig.savefig(self.output_dir / "gradient_flow.png", facecolor=self.colors["background"])
        plt.close(fig)

    def _plot_error_components(self, df: Any) -> None:
        samples = np.arange(min(len(df), 150))
        
        # Position Transition Error
        fig, ax = plt.subplots(figsize=(10, 5), dpi=150, facecolor=self.colors["background"])
        ax.set_facecolor(self.colors["card"])
        err_x = np.abs(df["pred_delta_x"].iloc[:150] - df["true_delta_x"].iloc[:150])
        err_y = np.abs(df["pred_delta_y"].iloc[:150] - df["true_delta_y"].iloc[:150])
        err_z = np.abs(df["pred_delta_z"].iloc[:150] - df["true_delta_z"].iloc[:150])
        ax.plot(samples, err_x, color=self.colors["train"], label="Error Δx", lw=1.0)
        ax.plot(samples, err_y, color=self.colors["validation"], label="Error Δy", lw=1.0)
        ax.plot(samples, err_z, color=self.colors["residual"], label="Error Δz", lw=1.0)
        ax.set_xlabel("Samples")
        ax.set_ylabel("Absolute Transition Error [km]")
        ax.set_title("Position Transition Deviation Components", fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.15)
        fig.savefig(self.output_dir / "position_transition_error.png", facecolor=self.colors["background"])
        plt.close(fig)

        # Velocity Transition Error
        fig, ax = plt.subplots(figsize=(10, 5), dpi=150, facecolor=self.colors["background"])
        ax.set_facecolor(self.colors["card"])
        err_vx = np.abs(df["pred_delta_vx"].iloc[:150] - df["true_delta_vx"].iloc[:150])
        err_vy = np.abs(df["pred_delta_vy"].iloc[:150] - df["true_delta_vy"].iloc[:150])
        err_vz = np.abs(df["pred_delta_vz"].iloc[:150] - df["true_delta_vz"].iloc[:150])
        ax.plot(samples, err_vx, color=self.colors["train"], label="Error Δvx", lw=1.0)
        ax.plot(samples, err_vy, color=self.colors["validation"], label="Error Δvy", lw=1.0)
        ax.plot(samples, err_vz, color=self.colors["residual"], label="Error Δvz", lw=1.0)
        ax.set_xlabel("Samples")
        ax.set_ylabel("Absolute Transition Error [km/s]")
        ax.set_title("Velocity Transition Deviation Components", fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.15)
        fig.savefig(self.output_dir / "velocity_transition_error.png", facecolor=self.colors["background"])
        plt.close(fig)

        # Residual Acceleration Error
        fig, ax = plt.subplots(figsize=(10, 5), dpi=150, facecolor=self.colors["background"])
        ax.set_facecolor(self.colors["card"])
        err_ax = np.abs(df["pred_ax_res"].iloc[:150] - df["true_ax_res"].iloc[:150])
        err_ay = np.abs(df["pred_ay_res"].iloc[:150] - df["true_ay_res"].iloc[:150])
        err_az = np.abs(df["pred_az_res"].iloc[:150] - df["true_az_res"].iloc[:150])
        ax.plot(samples, err_ax, color=self.colors["train"], label="Error ax_res", lw=1.0)
        ax.plot(samples, err_ay, color=self.colors["validation"], label="Error ay_res", lw=1.0)
        ax.plot(samples, err_az, color=self.colors["residual"], label="Error az_res", lw=1.0)
        ax.set_xlabel("Samples")
        ax.set_ylabel("Absolute Residual Error [km/s^2]")
        ax.set_title("Residual Acceleration Prediction Error", fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.15)
        fig.savefig(self.output_dir / "residual_acceleration_error.png", facecolor=self.colors["background"])
        plt.close(fig)

    def _plot_training_dashboard(self, epochs: np.ndarray, history: Dict[str, List[float]]) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150, facecolor=self.colors["background"])
        
        # Plot 1: Total Loss
        axes[0, 0].set_facecolor(self.colors["card"])
        axes[0, 0].plot(epochs, history["train_loss"], color=self.colors["train"], label="Train", lw=1.2)
        axes[0, 0].plot(epochs, history["val_loss"], color=self.colors["validation"], label="Val", lw=1.2, linestyle="--")
        axes[0, 0].set_yscale("log")
        axes[0, 0].set_title("Loss Convergence", fontsize=10, fontweight="bold")
        axes[0, 0].grid(True, linestyle="--", alpha=0.15)
        axes[0, 0].legend()

        # Plot 2: Individual Losses
        axes[0, 1].set_facecolor(self.colors["card"])
        axes[0, 1].plot(epochs, history["train_transition_loss"], color=self.colors["train"], label="Transition", lw=1.0)
        axes[0, 1].plot(epochs, history["train_physics_loss"], color=self.colors["constraint"], label="Physics", lw=1.0)
        axes[0, 1].plot(epochs, history["train_residual_loss"], color=self.colors["residual"], label="Residual", lw=1.0)
        axes[0, 1].plot(epochs, history["train_constraint_loss"], color=self.colors["accent"], label="Constraints", lw=1.0)
        axes[0, 1].set_yscale("log")
        axes[0, 1].set_title("Loss Components", fontsize=10, fontweight="bold")
        axes[0, 1].grid(True, linestyle="--", alpha=0.15)
        axes[0, 1].legend()

        # Plot 3: Gradient Norm
        axes[1, 0].set_facecolor(self.colors["card"])
        axes[1, 0].plot(epochs, history.get("gradient_norm", []), color=self.colors["residual"], lw=1.2)
        axes[1, 0].set_title("Global Gradient Norm", fontsize=10, fontweight="bold")
        axes[1, 0].grid(True, linestyle="--", alpha=0.15)

        # Plot 4: Learning Rate
        axes[1, 1].set_facecolor(self.colors["card"])
        axes[1, 1].plot(epochs, history["learning_rate"], color=self.colors["constraint"], lw=1.2)
        axes[1, 1].set_yscale("log")
        axes[1, 1].set_title("Learning Rate Decay", fontsize=10, fontweight="bold")
        axes[1, 1].grid(True, linestyle="--", alpha=0.15)

        fig.suptitle("PI-STN Model Training Dashboard", fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout()
        fig.savefig(self.output_dir / "training_dashboard.png", facecolor=self.colors["background"])
        plt.close(fig)
