"""
src/observation_generation/visualizer.py
=========================================
ObservationVisualizer.

Generates 10 figures illustrating observation data qualities, noise distributions,
availability, and scenario comparisons.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

matplotlib.use("Agg")
log = get_logger(__name__)

_BG = "#0D1117"
_FG = "#FFFFFF"
_GRID = "#1E1E3A"
_SPINE = "#333355"
_LABEL = "#AAAAAA"


class ObservationVisualizer:
    """Generates all Phase 6 observation visualization figures."""

    def __init__(self, cfg: dict[str, Any], results_dir: Path) -> None:
        self.cfg = cfg
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = int(cfg.get("visualization", {}).get("dpi", 150))
        plt.style.use(cfg.get("visualization", {}).get("style", "dark_background"))

        # Colors
        self.c_clean = "#00D4FF"
        self.c_noisy = "#FF6B6B"
        self.c_list = ["#FF6B6B", "#4ECDC4", "#FFE66D", "#A78BFA", "#FF8C42", "#38BDF8"]

    def plot_all(
        self,
        df_clean: pd.DataFrame,
        df_noisy: pd.DataFrame,
        df_quality: pd.DataFrame,
        satellite_name: str = "INSAT-3D",
    ) -> None:
        """Generate and save all 10 required figures."""
        log.info("Generating Observation Generation simulation plots...")
        hours_clean = df_clean["elapsed_seconds"].to_numpy() / 3600.0

        self._plot_ra_vs_time(df_clean, hours_clean, satellite_name)
        self._plot_dec_vs_time(df_clean, hours_clean, satellite_name)
        self._plot_sky_plot(df_clean, satellite_name)
        self._plot_noisy_vs_clean_ra(df_noisy, satellite_name)
        self._plot_noisy_vs_clean_dec(df_noisy, satellite_name)
        self._plot_observation_density(df_noisy, satellite_name)
        self._plot_observation_availability(df_quality, satellite_name)
        self._plot_noise_distribution(df_noisy, satellite_name)
        self._plot_station_visibility(df_noisy, satellite_name)
        self._plot_scenario_comparison(df_noisy, satellite_name)

        log.info("All observation plots saved in: %s", self.results_dir)

    # ── Figure 1 ── RA vs Time (clean)
    def _plot_ra_vs_time(self, df: pd.DataFrame, hours: np.ndarray, sat_name: str) -> None:
        fig, ax = plt.subplots(figsize=(12, 6), facecolor=_BG)
        ax.set_facecolor(_BG)

        for i, s_name in enumerate(df["station_name"].unique()):
            df_s = df[df["station_name"] == s_name]
            h = df_s["elapsed_seconds"].to_numpy() / 3600.0
            ax.scatter(h, df_s["ra_deg"], color=self.c_list[i % len(self.c_list)], s=5, label=s_name)

        ax.set_title(f"{sat_name} — Right Ascension vs Time (Ideal)", color=_FG, fontsize=14, pad=15)
        ax.set_xlabel("Elapsed time [hours]", color=_LABEL)
        ax.set_ylabel("Right Ascension [deg]", color=_LABEL)
        ax.set_ylim(0, 360)
        ax.grid(color=_GRID, linestyle="--", linewidth=0.5)
        ax.tick_params(colors=_LABEL)
        ax.spines[:].set_color(_SPINE)
        ax.legend(facecolor="#1A1A2E", labelcolor=_FG)
        self._save(fig, "ra_vs_time.png")

    # ── Figure 2 ── Dec vs Time (clean)
    def _plot_dec_vs_time(self, df: pd.DataFrame, hours: np.ndarray, sat_name: str) -> None:
        fig, ax = plt.subplots(figsize=(12, 6), facecolor=_BG)
        ax.set_facecolor(_BG)

        for i, s_name in enumerate(df["station_name"].unique()):
            df_s = df[df["station_name"] == s_name]
            h = df_s["elapsed_seconds"].to_numpy() / 3600.0
            ax.scatter(h, df_s["dec_deg"], color=self.c_list[i % len(self.c_list)], s=5, label=s_name)

        ax.set_title(f"{sat_name} — Declination vs Time (Ideal)", color=_FG, fontsize=14, pad=15)
        ax.set_xlabel("Elapsed time [hours]", color=_LABEL)
        ax.set_ylabel("Declination [deg]", color=_LABEL)
        ax.set_ylim(-15, 15)  # GEO satellites hover around equator
        ax.grid(color=_GRID, linestyle="--", linewidth=0.5)
        ax.tick_params(colors=_LABEL)
        ax.spines[:].set_color(_SPINE)
        ax.legend(facecolor="#1A1A2E", labelcolor=_FG)
        self._save(fig, "dec_vs_time.png")

    # ── Figure 3 ── Sky Plot (Dec vs RA)
    def _plot_sky_plot(self, df: pd.DataFrame, sat_name: str) -> None:
        fig, ax = plt.subplots(figsize=(10, 8), facecolor=_BG)
        ax.set_facecolor(_BG)

        for i, s_name in enumerate(df["station_name"].unique()):
            df_s = df[df["station_name"] == s_name]
            ax.scatter(df_s["ra_deg"], df_s["dec_deg"], color=self.c_list[i % len(self.c_list)], s=10, label=s_name)

        ax.set_title(f"{sat_name} — Observation Sky Plot (Ideal)", color=_FG, fontsize=14, pad=15)
        ax.set_xlabel("Right Ascension [deg]", color=_LABEL)
        ax.set_ylabel("Declination [deg]", color=_LABEL)
        ax.set_xlim(0, 360)
        ax.set_ylim(-15, 15)
        ax.grid(color=_GRID, linestyle="--", linewidth=0.5)
        ax.tick_params(colors=_LABEL)
        ax.spines[:].set_color(_SPINE)
        ax.legend(facecolor="#1A1A2E", labelcolor=_FG)
        self._save(fig, "sky_plot.png")

    # ── Figure 4 ── Noise Distribution
    def _plot_noise_distribution(self, df: pd.DataFrame, sat_name: str) -> None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=_BG)
        
        # Scenario B (Low Noise)
        df_b = df[df["scenario"] == "Scenario B (Low Noise)"]
        if len(df_b) > 0:
            ax1.set_facecolor(_BG)
            ax1.hist(df_b["noise_ra_deg"], bins=50, color=self.c_noisy, alpha=0.7, density=True)
            ax1.set_title("Low Noise Scenario (RA Error)", color=_FG)
            ax1.set_xlabel("Error [deg]", color=_LABEL)
            ax1.grid(color=_GRID, linestyle="--", linewidth=0.5)
            ax1.tick_params(colors=_LABEL)
            ax1.spines[:].set_color(_SPINE)

        # Scenario C (High Noise)
        df_c = df[df["scenario"] == "Scenario C (High Noise)"]
        if len(df_c) > 0:
            ax2.set_facecolor(_BG)
            ax2.hist(df_c["noise_ra_deg"], bins=50, color="#FF8C42", alpha=0.7, density=True)
            ax2.set_title("High Noise Scenario (RA Error)", color=_FG)
            ax2.set_xlabel("Error [deg]", color=_LABEL)
            ax2.grid(color=_GRID, linestyle="--", linewidth=0.5)
            ax2.tick_params(colors=_LABEL)
            ax2.spines[:].set_color(_SPINE)

        fig.suptitle(f"{sat_name} — Measurement Noise Distributions", color=_FG, fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        self._save(fig, "noise_distribution.png")

    # ── Figure 5 ── Observation Density
    def _plot_observation_density(self, df: pd.DataFrame, sat_name: str) -> None:
        fig, ax = plt.subplots(figsize=(12, 6), facecolor=_BG)
        ax.set_facecolor(_BG)

        # Plot observations histogram over time (by scenario)
        hours = df["elapsed_seconds"].to_numpy() / 3600.0
        ax.hist(hours, bins=100, color=self.c_clean, alpha=0.75, edgecolor=_BG, label="Observations density")

        ax.set_title(f"{sat_name} — Observation Density over Time", color=_FG, fontsize=14, pad=15)
        ax.set_xlabel("Elapsed time [hours]", color=_LABEL)
        ax.set_ylabel("Observation Count", color=_LABEL)
        ax.grid(color=_GRID, linestyle="--", linewidth=0.5)
        ax.tick_params(colors=_LABEL)
        ax.spines[:].set_color(_SPINE)
        self._save(fig, "observation_density.png")

    # ── Figure 6 ── Clean vs Noisy RA (Zoomed segment)
    def _plot_noisy_vs_clean_ra(self, df: pd.DataFrame, sat_name: str) -> None:
        fig, ax = plt.subplots(figsize=(12, 6), facecolor=_BG)
        ax.set_facecolor(_BG)

        # Plot Scenario A (Ideal) vs Scenario C (High Noise) for a small window
        df_a = df[df["scenario"] == "Scenario A (Ideal)"].reset_index(drop=True)
        df_c = df[df["scenario"] == "Scenario C (High Noise)"].reset_index(drop=True)

        if len(df_a) > 0 and len(df_c) > 0:
            # Zoom into a 12-hour window
            mask_a = (df_a["elapsed_seconds"] >= 86400) & (df_a["elapsed_seconds"] <= 129600)
            mask_c = (df_c["elapsed_seconds"] >= 86400) & (df_c["elapsed_seconds"] <= 129600)

            ax.plot(df_a[mask_a]["elapsed_seconds"] / 3600.0, df_a[mask_a]["ra_observed_deg"], color=self.c_clean, label="True RA", linewidth=1.5)
            ax.scatter(df_c[mask_c]["elapsed_seconds"] / 3600.0, df_c[mask_c]["ra_observed_deg"], color=self.c_noisy, s=15, label="Observed RA (High Noise)")

        ax.set_title(f"{sat_name} — Right Ascension Calibration Comparison (Zoomed)", color=_FG, fontsize=14, pad=15)
        ax.set_xlabel("Elapsed time [hours]", color=_LABEL)
        ax.set_ylabel("Right Ascension [deg]", color=_LABEL)
        ax.grid(color=_GRID, linestyle="--", linewidth=0.5)
        ax.tick_params(colors=_LABEL)
        ax.spines[:].set_color(_SPINE)
        ax.legend(facecolor="#1A1A2E", labelcolor=_FG)
        self._save(fig, "noisy_vs_clean_ra.png")

    # ── Figure 7 ── Clean vs Noisy Dec (Zoomed segment)
    def _plot_noisy_vs_clean_dec(self, df: pd.DataFrame, sat_name: str) -> None:
        fig, ax = plt.subplots(figsize=(12, 6), facecolor=_BG)
        ax.set_facecolor(_BG)

        # Plot Scenario A (Ideal) vs Scenario C (High Noise) for a small window
        df_a = df[df["scenario"] == "Scenario A (Ideal)"].reset_index(drop=True)
        df_c = df[df["scenario"] == "Scenario C (High Noise)"].reset_index(drop=True)

        if len(df_a) > 0 and len(df_c) > 0:
            # Zoom into a 12-hour window
            mask_a = (df_a["elapsed_seconds"] >= 86400) & (df_a["elapsed_seconds"] <= 129600)
            mask_c = (df_c["elapsed_seconds"] >= 86400) & (df_c["elapsed_seconds"] <= 129600)

            ax.plot(df_a[mask_a]["elapsed_seconds"] / 3600.0, df_a[mask_a]["dec_observed_deg"], color=self.c_clean, label="True Dec", linewidth=1.5)
            ax.scatter(df_c[mask_c]["elapsed_seconds"] / 3600.0, df_c[mask_c]["dec_observed_deg"], color=self.c_noisy, s=15, label="Observed Dec (High Noise)")

        ax.set_title(f"{sat_name} — Declination Calibration Comparison (Zoomed)", color=_FG, fontsize=14, pad=15)
        ax.set_xlabel("Elapsed time [hours]", color=_LABEL)
        ax.set_ylabel("Declination [deg]", color=_LABEL)
        ax.grid(color=_GRID, linestyle="--", linewidth=0.5)
        ax.tick_params(colors=_LABEL)
        ax.spines[:].set_color(_SPINE)
        ax.legend(facecolor="#1A1A2E", labelcolor=_FG)
        self._save(fig, "noisy_vs_clean_dec.png")

    # ── Figure 8 ── Observation Availability Timeline
    def _plot_observation_availability(self, df_q: pd.DataFrame, sat_name: str) -> None:
        fig, ax = plt.subplots(figsize=(12, 6), facecolor=_BG)
        ax.set_facecolor(_BG)

        scenarios = df_q["scenario"].unique()
        
        for i, sc in enumerate(scenarios):
            df_sc = df_q[df_q["scenario"] == sc]
            hours = df_sc["utc_time"].factorize()[0] / 60.0  # approximate time index
            # Extract missing/gap indices
            missing = df_sc["missing"].to_numpy()
            
            y_val = i + 1
            ax.plot([0, 168], [y_val, y_val], color="#1E1E3A", linewidth=10, zorder=1)
            
            # Scatter active observation intervals (where missing is False)
            active_h = np.linspace(0, 168, len(missing))[~missing]
            color = self.c_list[i % len(self.c_list)]
            ax.scatter(active_h, np.full_like(active_h, y_val), color=color, s=4, marker="|", zorder=2)

        ax.set_title(f"{sat_name} Observation Availability Timeline", color=_FG, fontsize=14, pad=15)
        ax.set_xlabel("Time index [hours]", color=_LABEL)
        ax.set_yticks(range(1, len(scenarios) + 1))
        ax.set_yticklabels(scenarios, color=_FG)
        ax.grid(color=_GRID, linestyle="--", linewidth=0.5)
        ax.tick_params(colors=_LABEL)
        ax.spines[:].set_color(_SPINE)

        self._save(fig, "observation_availability.png")

    # ── Figure 9 ── Station Comparison
    def _plot_station_visibility(self, df: pd.DataFrame, sat_name: str) -> None:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=_BG)
        ax.set_facecolor(_BG)

        # Bar chart comparing visible observation counts per station (Scenario A)
        df_a = df[df["scenario"] == "Scenario A (Ideal)"]
        station_counts = df_a.groupby("station_name").size()

        bars = ax.bar(station_counts.index, station_counts.values, color=self.c_list[:len(station_counts)], width=0.5)
        
        # Value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, height + 50, f"{int(height)}", ha="center", va="bottom", color=_FG)

        ax.set_title(f"{sat_name} — Total Observations by Station", color=_FG, fontsize=14, pad=15)
        ax.set_ylabel("Count", color=_LABEL)
        ax.grid(color=_GRID, linestyle="--", linewidth=0.5, axis="y")
        ax.tick_params(colors=_LABEL)
        ax.spines[:].set_color(_SPINE)

        self._save(fig, "station_visibility.png")

    # ── Figure 10 ── Scenario Comparison
    def _plot_scenario_comparison(self, df: pd.DataFrame, sat_name: str) -> None:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=_BG)
        ax.set_facecolor(_BG)

        # Bar chart comparing observation count per scenario
        counts = df.groupby("scenario").size()

        bars = ax.barh(counts.index, counts.values, color=self.c_list[:len(counts)], height=0.5)

        for bar in bars:
            width = bar.get_width()
            ax.text(width + 100, bar.get_y() + bar.get_height()/2.0, f"{int(width)}", ha="left", va="center", color=_FG)

        ax.set_title(f"{sat_name} — Observations by Scenario", color=_FG, fontsize=14, pad=15)
        ax.set_xlabel("Count", color=_LABEL)
        ax.grid(color=_GRID, linestyle="--", linewidth=0.5, axis="x")
        ax.tick_params(colors=_LABEL)
        ax.spines[:].set_color(_SPINE)

        self._save(fig, "scenario_comparison.png")

    def _save(self, fig: plt.Figure, filename: str) -> None:
        out = self.results_dir / filename
        fig.savefig(out, dpi=self.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        log.info("Saved: %s", out)
