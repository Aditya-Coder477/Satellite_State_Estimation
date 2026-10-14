"""
src/perturbations/visualizer.py
================================
PerturbationVisualizer — Generates 5 publication-quality figures for
the Phase 3 perturbation simulation.

Figures
-------
1. ``disturbed_vs_nominal.png``       — 3D trajectory comparison
2. ``residual_acceleration.png``      — Total residual acceleration vs time
3. ``acceleration_components.png``    — Per-model acceleration components
4. ``perturbation_statistics.png``    — Perturbation magnitude histogram + CDF
5. ``trajectory_difference_3d.png``  — 3D position difference vector field

Style matches the project's dark-background aesthetic (``#0D1117`` background,
project colour palette from cfg["visualization"]["colors"]).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from src.perturbations.disturbance_generator import PerturbationResult
from src.utils.logger import get_logger

matplotlib.use("Agg")

log: logging.Logger = get_logger(__name__)

_BG = "#0D1117"
_FG = "#FFFFFF"
_GRID = "#1E1E3A"
_SPINE = "#333355"
_LABEL = "#AAAAAA"
_EARTH_R_KM = 6_371.0


class PerturbationVisualizer:
    """
    Generates and saves perturbation visualisation figures.

    Parameters
    ----------
    cfg : dict[str, Any]
        Project configuration.
    results_dir : Path
        Directory where figures are saved.
    """

    def __init__(self, cfg: dict[str, Any], results_dir: Path) -> None:
        viz_cfg = cfg.get("visualization", {})
        self._results_dir = Path(results_dir)
        self._results_dir.mkdir(parents=True, exist_ok=True)
        self._dpi = int(viz_cfg.get("dpi", 150))

        plt.style.use(viz_cfg.get("style", "dark_background"))
        colors = viz_cfg.get("colors", {})
        self._c_nominal = colors.get("trajectory", "#00D4FF")
        self._c_disturbed = "#FF6B6B"
        self._c_srp = "#FFE66D"
        self._c_maneuver = "#FF8C42"
        self._c_random = "#4ECDC4"
        self._c_bias = "#A78BFA"
        self._c_total = "#00D4FF"
        self._c_earth = colors.get("earth", "#1A3A5C")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plot_all(
        self,
        df_nominal: pd.DataFrame,
        df_disturbed: pd.DataFrame,
        result: PerturbationResult,
        satellite_name: str = "INSAT-3D",
    ) -> None:
        """Generate and save all five figures."""
        hours = df_nominal["elapsed_seconds"].to_numpy() / 3600.0

        self._plot_disturbed_vs_nominal_3d(df_nominal, df_disturbed, satellite_name)
        self._plot_residual_acceleration(hours, result, satellite_name)
        self._plot_acceleration_components(hours, result, satellite_name)
        self._plot_perturbation_statistics(result, satellite_name)
        self._plot_trajectory_difference_3d(df_nominal, df_disturbed, satellite_name)

        log.info("All perturbation figures saved to: %s", str(self._results_dir))

    # ------------------------------------------------------------------
    # Figure 1 — Disturbed vs Nominal 3D orbit
    # ------------------------------------------------------------------

    def _plot_disturbed_vs_nominal_3d(
        self,
        df_nominal: pd.DataFrame,
        df_disturbed: pd.DataFrame,
        satellite_name: str,
    ) -> None:
        fig = plt.figure(figsize=(11, 9), facecolor=_BG)
        ax = fig.add_subplot(111, projection="3d", facecolor=_BG)

        # Earth
        u_ang = np.linspace(0, 2 * np.pi, 60)
        v_ang = np.linspace(0, np.pi, 30)
        ex = _EARTH_R_KM * np.outer(np.cos(u_ang), np.sin(v_ang))
        ey = _EARTH_R_KM * np.outer(np.sin(u_ang), np.sin(v_ang))
        ez = _EARTH_R_KM * np.outer(np.ones_like(u_ang), np.cos(v_ang))
        ax.plot_surface(ex, ey, ez, color=self._c_earth, alpha=0.5, linewidth=0)

        # Orbits
        ax.plot(df_nominal["x_teme_km"], df_nominal["y_teme_km"], df_nominal["z_teme_km"],
                color=self._c_nominal, linewidth=0.8, alpha=0.9, label="Nominal")
        ax.plot(df_disturbed["x_teme_km"], df_disturbed["y_teme_km"], df_disturbed["z_teme_km"],
                color=self._c_disturbed, linewidth=0.7, alpha=0.7, linestyle="--", label="Disturbed")

        ax.set_xlabel("X TEME [km]", color=_LABEL, labelpad=8)
        ax.set_ylabel("Y TEME [km]", color=_LABEL, labelpad=8)
        ax.set_zlabel("Z TEME [km]", color=_LABEL, labelpad=8)
        ax.tick_params(colors=_LABEL)
        ax.set_title(f"{satellite_name} — Nominal vs Disturbed Orbit", color=_FG, fontsize=13, pad=12)
        ax.legend(facecolor="#1A1A2E", labelcolor=_FG, fontsize=9)

        self._save(fig, "disturbed_vs_nominal.png")

    # ------------------------------------------------------------------
    # Figure 2 — Total Residual Acceleration vs Time
    # ------------------------------------------------------------------

    def _plot_residual_acceleration(
        self, hours: np.ndarray, result: PerturbationResult, satellite_name: str
    ) -> None:
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), facecolor=_BG, sharex=True)
        fig.suptitle(f"{satellite_name} — Residual Acceleration Components", color=_FG, fontsize=13)

        data = [
            (result.acc_total[:, 0], "ax_res [km/s²]", self._c_total),
            (result.acc_total[:, 1], "ay_res [km/s²]", self._c_total),
            (result.acc_total[:, 2], "az_res [km/s²]", self._c_total),
            (result.perturbation_magnitude, "|a_res| [km/s²]", "#FFFFFF"),
        ]

        for ax, (y, label, color) in zip(axes, data):
            ax.set_facecolor(_BG)
            ax.plot(hours, y, color=color, linewidth=0.7, alpha=0.9)
            ax.set_ylabel(label, color=_LABEL, fontsize=9)
            ax.tick_params(colors=_LABEL)
            ax.spines[:].set_color(_SPINE)
            ax.grid(color=_GRID, linewidth=0.5, linestyle="--")

        axes[-1].set_xlabel("Elapsed time [hours]", color=_LABEL, fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        self._save(fig, "residual_acceleration.png")

    # ------------------------------------------------------------------
    # Figure 3 — Per-model Acceleration Components
    # ------------------------------------------------------------------

    def _plot_acceleration_components(
        self, hours: np.ndarray, result: PerturbationResult, satellite_name: str
    ) -> None:
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), facecolor=_BG, sharex=True)
        fig.suptitle(f"{satellite_name} — Perturbation Model Breakdown", color=_FG, fontsize=13)

        srp_mag = np.linalg.norm(result.acc_srp, axis=-1)
        man_mag = np.linalg.norm(result.acc_maneuver, axis=-1)
        rnd_mag = np.linalg.norm(result.acc_random, axis=-1)
        bias_mag = np.linalg.norm(result.acc_bias, axis=-1)

        data = [
            (srp_mag, "SRP |a| [km/s²]", self._c_srp),
            (man_mag, "Maneuver |a| [km/s²]", self._c_maneuver),
            (rnd_mag, "Random |a| [km/s²]", self._c_random),
            (bias_mag, "Bias |a| [km/s²]", self._c_bias),
        ]

        for ax, (y, label, color) in zip(axes, data):
            ax.set_facecolor(_BG)
            ax.plot(hours, y, color=color, linewidth=0.8, alpha=0.9)
            ax.set_ylabel(label, color=_LABEL, fontsize=9)
            ax.tick_params(colors=_LABEL)
            ax.spines[:].set_color(_SPINE)
            ax.grid(color=_GRID, linewidth=0.5, linestyle="--")

        axes[-1].set_xlabel("Elapsed time [hours]", color=_LABEL, fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        self._save(fig, "acceleration_components.png")

    # ------------------------------------------------------------------
    # Figure 4 — Perturbation Magnitude Histogram + CDF
    # ------------------------------------------------------------------

    def _plot_perturbation_statistics(
        self, result: PerturbationResult, satellite_name: str
    ) -> None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=_BG)
        fig.suptitle(f"{satellite_name} — Perturbation Magnitude Statistics", color=_FG, fontsize=13)

        mag = result.perturbation_magnitude

        # Histogram
        ax1.set_facecolor(_BG)
        ax1.hist(mag, bins=60, color=self._c_total, alpha=0.85, edgecolor="#0D1117")
        ax1.axvline(mag.mean(), color="#FFE66D", linestyle="--", linewidth=1.5, label=f"Mean={mag.mean():.2e}")
        ax1.axvline(np.percentile(mag, 95), color="#FF6B6B", linestyle=":", linewidth=1.5, label=f"P95={np.percentile(mag,95):.2e}")
        ax1.set_xlabel("|a_res| [km/s²]", color=_LABEL)
        ax1.set_ylabel("Count", color=_LABEL)
        ax1.set_title("Magnitude Histogram", color=_FG)
        ax1.tick_params(colors=_LABEL)
        ax1.spines[:].set_color(_SPINE)
        ax1.grid(color=_GRID, linewidth=0.5, linestyle="--")
        ax1.legend(facecolor="#1A1A2E", labelcolor=_FG, fontsize=9)

        # CDF
        ax2.set_facecolor(_BG)
        sorted_mag = np.sort(mag)
        cdf = np.arange(1, len(sorted_mag) + 1) / len(sorted_mag)
        ax2.plot(sorted_mag, cdf, color=self._c_total, linewidth=1.5)
        ax2.axhline(0.95, color="#FF6B6B", linestyle=":", linewidth=1.2, label="95th percentile")
        ax2.set_xlabel("|a_res| [km/s²]", color=_LABEL)
        ax2.set_ylabel("CDF", color=_LABEL)
        ax2.set_title("Cumulative Distribution", color=_FG)
        ax2.tick_params(colors=_LABEL)
        ax2.spines[:].set_color(_SPINE)
        ax2.grid(color=_GRID, linewidth=0.5, linestyle="--")
        ax2.legend(facecolor="#1A1A2E", labelcolor=_FG, fontsize=9)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        self._save(fig, "perturbation_statistics.png")

    # ------------------------------------------------------------------
    # Figure 5 — Trajectory Difference 3D
    # ------------------------------------------------------------------

    def _plot_trajectory_difference_3d(
        self,
        df_nominal: pd.DataFrame,
        df_disturbed: pd.DataFrame,
        satellite_name: str,
    ) -> None:
        # Downsample to avoid overplotting (plot every Nth vector)
        step = max(1, len(df_nominal) // 200)
        idx = np.arange(0, len(df_nominal), step)

        x_nom = df_nominal["x_teme_km"].to_numpy()[idx]
        y_nom = df_nominal["y_teme_km"].to_numpy()[idx]
        z_nom = df_nominal["z_teme_km"].to_numpy()[idx]

        dx = (df_disturbed["x_teme_km"].to_numpy() - df_nominal["x_teme_km"].to_numpy())[idx]
        dy = (df_disturbed["y_teme_km"].to_numpy() - df_nominal["y_teme_km"].to_numpy())[idx]
        dz = (df_disturbed["z_teme_km"].to_numpy() - df_nominal["z_teme_km"].to_numpy())[idx]

        dr_mag = np.sqrt(dx**2 + dy**2 + dz**2)
        scale_factor = 500.0 / max(float(dr_mag.max()), 1e-12)

        fig = plt.figure(figsize=(11, 9), facecolor=_BG)
        ax = fig.add_subplot(111, projection="3d", facecolor=_BG)

        # Nominal orbit (dim)
        ax.plot(df_nominal["x_teme_km"], df_nominal["y_teme_km"], df_nominal["z_teme_km"],
                color=self._c_nominal, linewidth=0.5, alpha=0.3, label="Nominal")

        # Perturbation difference vectors (colour-coded by magnitude)
        norm = plt.Normalize(dr_mag.min(), dr_mag.max())
        cmap = plt.cm.plasma
        colors = cmap(norm(dr_mag))
        for i in range(len(idx)):
            ax.quiver(x_nom[i], y_nom[i], z_nom[i],
                      dx[i] * scale_factor, dy[i] * scale_factor, dz[i] * scale_factor,
                      color=colors[i], linewidth=0.5, alpha=0.8)

        ax.set_xlabel("X TEME [km]", color=_LABEL, labelpad=8)
        ax.set_ylabel("Y TEME [km]", color=_LABEL, labelpad=8)
        ax.set_zlabel("Z TEME [km]", color=_LABEL, labelpad=8)
        ax.tick_params(colors=_LABEL)
        ax.set_title(
            f"{satellite_name} — Position Difference Vectors (×{scale_factor:.0f})",
            color=_FG, fontsize=12, pad=12,
        )

        # Colourbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, pad=0.1, shrink=0.6)
        cb.set_label("|Δr| [km]", color=_LABEL)
        cb.ax.yaxis.set_tick_params(color=_LABEL)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=_LABEL)

        self._save(fig, "trajectory_difference_3d.png")

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _save(self, fig: plt.Figure, filename: str) -> None:
        out = self._results_dir / filename
        fig.savefig(out, dpi=self._dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        log.info("Saved: %s", out)
