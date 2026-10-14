"""
src/dataset/visualizer.py
==========================
Generates diagnostic plots for Phase 6 (Dataset Pipeline).

Creates dark-theme visualization plots under ``results/dataset/``:
1. Dataset split timeline
2. Collocation vs observation density
3. Engineered feature histograms
4. Station usage counts
5. Measurement noise error distributions
6. Normalization scaling comparisons
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Headless matplotlib setup
plt.switch_backend("Agg")
plt.style.use("dark_background")

from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class DatasetVisualizer:
    """
    Renders diagnostic plots for the split and normalized datasets.

    Parameters
    ----------
    cfg : Dict[str, Any]
        Project configuration.
    output_dir : Path
        Target directory to save PNG figures.
    """

    def __init__(self, cfg: Dict[str, Any], output_dir: Path) -> None:
        self._cfg = cfg
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Color palette definition
        self.colors = {
            "train": "#00D4FF",       # cyan
            "validation": "#FFE66D",  # yellow
            "test": "#FF6B6B",        # red
            "collocation": "#4ECDC4",  # teal
            "background": "#121212",
            "accent": "#9B5DE5",
        }
        log.info("DatasetVisualizer initialised. Outputs will be saved to: %s", output_dir)

    def plot_all(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        colloc_df: pd.DataFrame,
        raw_df: pd.DataFrame,
        raw_train_df: pd.DataFrame,  # train_df before normalization (for comparison)
    ) -> None:
        """
        Generates and saves all dataset pipeline figures.

        Parameters
        ----------
        train_df : pd.DataFrame
            Normalized training DataFrame.
        val_df : pd.DataFrame
            Normalized validation DataFrame.
        test_df : pd.DataFrame
            Normalized testing DataFrame.
        colloc_df : pd.DataFrame
            Collocation points DataFrame.
        raw_df : pd.DataFrame
            Raw dataset before splitting or normalization.
        raw_train_df : pd.DataFrame
            Raw training split before normalization.
        """
        try:
            self.plot_split_timeline(train_df, val_df, test_df)
            self.plot_collocation_distribution(raw_df, colloc_df)
            self.plot_feature_histograms(train_df)
            self.plot_station_usage(raw_df)
            self.plot_noise_distribution(raw_df)
            self.plot_normalization_comparison(raw_train_df, train_df)
            log.info("All dataset diagnostic plots generated successfully.")
        except Exception as e:
            log.error("Failed to generate dataset visualizations: %s", str(e), exc_info=True)

    def plot_split_timeline(self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
        """Renders the chronological train/val/test data division timelines."""
        fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
        
        t_col = "elapsed_seconds"
        if t_col not in train.columns:
            plt.close(fig)
            return

        # Fetch time bounds
        train_t = train[t_col] / 3600.0  # convert to hours
        val_t = val[t_col] / 3600.0
        test_t = test[t_col] / 3600.0

        # Create horizontal bar plot representing windows
        y_pos = [0, 1, 2]
        labels = ["Train Split", "Validation Split", "Test Split"]
        colors = [self.colors["train"], self.colors["validation"], self.colors["test"]]
        
        # Plot bars
        ax.barh(0, train_t.max() - train_t.min(), left=train_t.min(), color=colors[0], alpha=0.8, height=0.4)
        ax.barh(1, val_t.max() - val_t.min(), left=val_t.min(), color=colors[1], alpha=0.8, height=0.4)
        ax.barh(2, test_t.max() - test_t.min(), left=test_t.min(), color=colors[2], alpha=0.8, height=0.4)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel("Elapsed Time [Hours]", fontsize=10)
        ax.set_title("Dataset Chronological Split Timeline", fontsize=12, fontweight="bold", pad=15)
        ax.grid(True, linestyle="--", alpha=0.2)
        
        plt.tight_layout()
        fig.savefig(self.output_dir / "dataset_split_timeline.png", facecolor=self.colors["background"])
        plt.close(fig)

    def plot_collocation_distribution(self, obs_df: pd.DataFrame, colloc_df: pd.DataFrame) -> None:
        """Compares density distribution of observations vs physics collocation points."""
        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
        
        obs_t = obs_df["elapsed_seconds"] / 3600.0 if "elapsed_seconds" in obs_df.columns else []
        colloc_t = colloc_df["time"] / 3600.0 if "time" in colloc_df.columns else []

        # Plot histograms
        if len(colloc_t) > 0:
            ax.hist(
                colloc_t,
                bins=100,
                color=self.colors["collocation"],
                alpha=0.6,
                label=f"Physics Collocation (N={len(colloc_df)})",
                density=True,
            )
        if len(obs_t) > 0:
            ax.hist(
                obs_t,
                bins=100,
                color=self.colors["train"],
                alpha=0.6,
                label=f"Simulated Observations (N={len(obs_df)})",
                density=True,
            )

        ax.set_xlabel("Elapsed Time [Hours]", fontsize=10)
        ax.set_ylabel("Sampling Density", fontsize=10)
        ax.set_title("Collocation vs Observation Density Over Time", fontsize=12, fontweight="bold", pad=15)
        ax.legend(loc="upper right", framealpha=0.8)
        ax.grid(True, linestyle="--", alpha=0.2)
        
        plt.tight_layout()
        fig.savefig(self.output_dir / "collocation_density_comparison.png", facecolor=self.colors["background"])
        plt.close(fig)

    def plot_feature_histograms(self, train_df: pd.DataFrame) -> None:
        """Plots distribution of selected engineered features."""
        features_to_plot = [
            "fraction_of_orbit",
            "sine_time",
            "cosine_time",
            "day_fraction",
            "simulation_fraction",
        ]
        
        valid_feats = [f for f in features_to_plot if f in train_df.columns]
        if not valid_feats:
            return

        n_feats = len(valid_feats)
        cols = 2
        rows = (n_feats + 1) // 2

        fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows), dpi=150)
        axes = axes.flatten()

        for idx, feat in enumerate(valid_feats):
            ax = axes[idx]
            ax.hist(train_df[feat].dropna(), bins=50, color=self.colors["train"], alpha=0.7, edgecolor="none")
            ax.set_title(f"Engineered Feature: {feat}", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.15)
            ax.set_ylabel("Frequency")

        # Hide any unused subplots
        for idx in range(n_feats, len(axes)):
            fig.delaxes(axes[idx])

        fig.suptitle("Engineered Feature Distributions (Training Split)", fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout()
        fig.savefig(self.output_dir / "engineered_feature_distributions.png", facecolor=self.colors["background"])
        plt.close(fig)

    def plot_station_usage(self, obs_df: pd.DataFrame) -> None:
        """Bar chart showing number of observations generated per tracking station."""
        if "station_name" not in obs_df.columns:
            return

        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        
        # Group by station and visibility status
        visible_col = "visible" if "visible" in obs_df.columns else "station_visible"
        if visible_col in obs_df.columns:
            counts = obs_df.groupby(["station_name", visible_col]).size().unstack(fill_value=0)
            
            # Stacked bar plot
            labels = counts.index.tolist()
            visible_counts = counts[True].values if True in counts.columns else np.zeros(len(labels))
            hidden_counts = counts[False].values if False in counts.columns else np.zeros(len(labels))
            
            x = np.arange(len(labels))
            ax.bar(x, visible_counts, label="Visible", color=self.colors["train"], alpha=0.8, width=0.5)
            ax.bar(x, hidden_counts, bottom=visible_counts, label="Below Horizon", color=self.colors["test"], alpha=0.5, width=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
        else:
            counts = obs_df["station_name"].value_counts()
            counts.plot(kind="bar", color=self.colors["train"], alpha=0.8, ax=ax, width=0.5)

        ax.set_ylabel("Observation Count", fontsize=10)
        ax.set_title("Observation Count by Tracking Ground Station", fontsize=12, fontweight="bold", pad=15)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.15, axis="y")
        
        plt.tight_layout()
        fig.savefig(self.output_dir / "station_observation_counts.png", facecolor=self.colors["background"])
        plt.close(fig)

    def plot_noise_distribution(self, obs_df: pd.DataFrame) -> None:
        """Plots distribution of observation errors in Right Ascension and Declination."""
        has_noisy = "ra_noisy_deg" in obs_df.columns and "right_ascension_deg" in obs_df.columns
        if not has_noisy:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

        # Compute errors
        ra_err = (obs_df["ra_noisy_deg"] - obs_df["right_ascension_deg"]).dropna()
        dec_err = (obs_df["dec_noisy_deg"] - obs_df["declination_deg"]).dropna()

        # Wrap RA errors
        ra_err = (ra_err + 180.0) % 360.0 - 180.0

        # Plot RA error
        ax1.hist(ra_err, bins=60, color=self.colors["validation"], alpha=0.7, density=True)
        ax1.set_xlabel("Error [Degrees]", fontsize=9)
        ax1.set_title("Right Ascension Error Distribution (RA_noisy - RA_true)", fontsize=10)
        ax1.grid(True, linestyle="--", alpha=0.15)

        # Plot Dec error
        ax2.hist(dec_err, bins=60, color=self.colors["test"], alpha=0.7, density=True)
        ax2.set_xlabel("Error [Degrees]", fontsize=9)
        ax2.set_title("Declination Error Distribution (Dec_noisy - Dec_true)", fontsize=10)
        ax2.grid(True, linestyle="--", alpha=0.15)

        fig.suptitle("Simulated Measurement Noise Histograms", fontsize=13, fontweight="bold")
        plt.tight_layout()
        fig.savefig(self.output_dir / "measurement_noise_distributions.png", facecolor=self.colors["background"])
        plt.close(fig)

    def plot_normalization_comparison(self, raw_train: pd.DataFrame, norm_train: pd.DataFrame) -> None:
        """Renders boxplots of variables before and after normalization to verify scaling."""
        norm_cols = ["elapsed_seconds", "right_ascension_deg", "declination_deg"]
        valid_cols = [col for col in norm_cols if col in raw_train.columns and col in norm_train.columns]
        
        if not valid_cols:
            return

        fig, axes = plt.subplots(len(valid_cols), 2, figsize=(10, 3 * len(valid_cols)), dpi=150)
        if len(valid_cols) == 1:
            axes = np.array([axes])

        for idx, col in enumerate(valid_cols):
            # Raw column boxplot
            axes[idx, 0].boxplot(raw_train[col].dropna(), vert=False, patch_artist=True,
                                 boxprops=dict(facecolor=self.colors["test"], alpha=0.7),
                                 medianprops=dict(color="white"))
            axes[idx, 0].set_title(f"Raw '{col}'", fontsize=9)
            axes[idx, 0].grid(True, linestyle="--", alpha=0.15)

            # Normalized column boxplot
            axes[idx, 1].boxplot(norm_train[col].dropna(), vert=False, patch_artist=True,
                                 boxprops=dict(facecolor=self.colors["train"], alpha=0.7),
                                 medianprops=dict(color="white"))
            axes[idx, 1].set_title(f"Normalized '{col}'", fontsize=9)
            axes[idx, 1].grid(True, linestyle="--", alpha=0.15)

        fig.suptitle("Normalization Scaling Comparison (Raw vs Scaled)", fontsize=12, fontweight="bold", y=0.98)
        plt.tight_layout()
        fig.savefig(self.output_dir / "normalization_scaling_comparison.png", facecolor=self.colors["background"])
        plt.close(fig)
