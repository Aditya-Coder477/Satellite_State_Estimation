"""
src/coordinate_systems/visualizer.py
====================================
Visualisation utilities for coordinate frame transformations.

Generates 3D orbit plots, comparison overlays, and differences in position
and velocity over time. Save plots to the results/coordinate_transforms directory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from src.utils.logger import get_logger

matplotlib.use("Agg")  # Non-interactive backend safe for headless servers
log = get_logger(__name__)

_EARTH_RADIUS_KM = 6371.0


class CoordinateVisualizer:
    """
    Plots transformed trajectories and coordinate system differences.

    Parameters
    ----------
    cfg : dict[str, Any]
        Full project configuration.
    results_dir : Path
        Directory to save generated plots.
    """

    def __init__(self, cfg: Dict[str, Any], results_dir: Path) -> None:
        self.cfg = cfg
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.viz_cfg = cfg.get("visualization", {})
        style = self.viz_cfg.get("style", "dark_background")
        plt.style.use(style)

        self.dpi = int(self.viz_cfg.get("dpi", 150))
        
        colors = self.viz_cfg.get("colors", {})
        self.c_traj = colors.get("trajectory", "#00D4FF")
        self.c_x = colors.get("x", "#FF6B6B")
        self.c_y = colors.get("y", "#4ECDC4")
        self.c_z = colors.get("z", "#FFE66D")
        self.c_earth = colors.get("earth", "#1A3A5C")

    def plot_all(self, df: pd.DataFrame, satellite_name: str) -> None:
        """
        Generate and save all coordinate transformation plots.

        Parameters
        ----------
        df : pd.DataFrame
            Validated transformed DataFrame.
        satellite_name : str
            Display name of the satellite.
        """
        log.info("Generating coordinate transformation visualization plots...")
        hours = df["elapsed_seconds"].to_numpy() / 3600.0

        self._plot_3d_frame(df, "teme", "TEME", satellite_name)
        self._plot_3d_frame(df, "gcrs", "GCRS", satellite_name)
        self._plot_3d_frame(df, "itrs", "ITRS", satellite_name)

        self._plot_3d_overlay(df, "teme", "gcrs", "TEME vs GCRS", satellite_name)
        self._plot_3d_overlay(df, "gcrs", "itrs", "GCRS vs ITRS", satellite_name)
        self._plot_frame_comparison(df, satellite_name)

        self._plot_differences_xyz(df, hours, satellite_name)
        self._plot_differences_vxyz(df, hours, satellite_name)
        self._plot_transformation_validation(df, hours, satellite_name)

        log.info("All plots saved in: %s", self.results_dir)

    def _draw_earth(self, ax: Any) -> None:
        """Helper to draw Earth sphere."""
        u_ang = np.linspace(0, 2 * np.pi, 50)
        v_ang = np.linspace(0, np.pi, 25)
        ex = _EARTH_RADIUS_KM * np.outer(np.cos(u_ang), np.sin(v_ang))
        ey = _EARTH_RADIUS_KM * np.outer(np.sin(u_ang), np.sin(v_ang))
        ez = _EARTH_RADIUS_KM * np.outer(np.ones_like(u_ang), np.cos(v_ang))
        ax.plot_surface(ex, ey, ez, color=self.c_earth, alpha=0.5, linewidth=0)

    def _plot_3d_frame(self, df: pd.DataFrame, prefix: str, frame_name: str, sat_name: str) -> None:
        """Plot a single 3D trajectory."""
        fig = plt.figure(figsize=(10, 8), facecolor="#0D1117")
        ax = fig.add_subplot(111, projection="3d", facecolor="#0D1117")
        
        self._draw_earth(ax)

        x = df[f"x_{prefix}_km"].to_numpy()
        y = df[f"y_{prefix}_km"].to_numpy()
        z = df[f"z_{prefix}_km"].to_numpy()

        ax.plot(x, y, z, color=self.c_traj, linewidth=1.0, label=frame_name)
        ax.scatter(x[0], y[0], z[0], color="#FFFFFF", s=30, label="Epoch")

        ax.set_xlabel(f"X {frame_name} [km]", color="#AAAAAA")
        ax.set_ylabel(f"Y {frame_name} [km]", color="#AAAAAA")
        ax.set_zlabel(f"Z {frame_name} [km]", color="#AAAAAA")
        ax.tick_params(colors="#AAAAAA")
        ax.set_title(f"{sat_name} — 3D Trajectory in {frame_name}", color="#FFFFFF", fontsize=14)
        ax.legend(facecolor="#1A1A2E", labelcolor="#FFFFFF")

        out_path = self.results_dir / f"{prefix}_frame.png"
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        log.debug("Saved: %s", out_path)

    def _plot_3d_overlay(
        self, df: pd.DataFrame, f1: str, f2: str, label: str, sat_name: str
    ) -> None:
        """Plot an overlay of two frames in 3D."""
        fig = plt.figure(figsize=(10, 8), facecolor="#0D1117")
        ax = fig.add_subplot(111, projection="3d", facecolor="#0D1117")

        self._draw_earth(ax)

        # Frame 1
        x1 = df[f"x_{f1}_km"].to_numpy()
        y1 = df[f"y_{f1}_km"].to_numpy()
        z1 = df[f"z_{f1}_km"].to_numpy()
        ax.plot(x1, y1, z1, color=self.c_x, linewidth=0.9, alpha=0.8, label=f1.upper())

        # Frame 2
        x2 = df[f"x_{f2}_km"].to_numpy()
        y2 = df[f"y_{f2}_km"].to_numpy()
        z2 = df[f"z_{f2}_km"].to_numpy()
        ax.plot(x2, y2, z2, color=self.c_y, linewidth=0.9, alpha=0.8, label=f2.upper())

        ax.set_xlabel("X [km]", color="#AAAAAA")
        ax.set_ylabel("Y [km]", color="#AAAAAA")
        ax.set_zlabel("Z [km]", color="#AAAAAA")
        ax.tick_params(colors="#AAAAAA")
        ax.set_title(f"{sat_name} — Orbit Comparison: {label}", color="#FFFFFF", fontsize=14)
        ax.legend(facecolor="#1A1A2E", labelcolor="#FFFFFF")

        out_path = self.results_dir / f"{f1}_vs_{f2}.png"
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        log.debug("Saved: %s", out_path)

    def _plot_differences_xyz(self, df: pd.DataFrame, hours: np.ndarray, sat_name: str) -> None:
        """Plot component-wise coordinate differences over time."""
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), facecolor="#0D1117", sharex=True)
        fig.suptitle(f"{sat_name} — Coordinate Differences", color="#FFFFFF", fontsize=14)

        # TEME - GCRS
        dx_tg = df["x_teme_km"].to_numpy() - df["x_gcrs_km"].to_numpy()
        dy_tg = df["y_teme_km"].to_numpy() - df["y_gcrs_km"].to_numpy()
        dz_tg = df["z_teme_km"].to_numpy() - df["z_gcrs_km"].to_numpy()

        # GCRS - ITRS
        dx_gi = df["x_gcrs_km"].to_numpy() - df["x_itrs_km"].to_numpy()
        dy_gi = df["y_gcrs_km"].to_numpy() - df["y_itrs_km"].to_numpy()
        dz_gi = df["z_gcrs_km"].to_numpy() - df["z_itrs_km"].to_numpy()

        diffs = [
            (dx_tg, dx_gi, "X Difference [km]"),
            (dy_tg, dy_gi, "Y Difference [km]"),
            (dz_tg, dz_gi, "Z Difference [km]"),
        ]

        for ax, (diff_tg, diff_gi, ylabel) in zip(axes, diffs):
            ax.set_facecolor("#0D1117")
            ax.plot(hours, diff_tg, color=self.c_x, label="TEME - GCRS", linewidth=0.8, alpha=0.9)
            ax.plot(hours, diff_gi, color=self.c_y, label="GCRS - ITRS", linewidth=0.8, alpha=0.9)
            ax.set_ylabel(ylabel, color="#AAAAAA")
            ax.tick_params(colors="#AAAAAA")
            ax.grid(color="#1E1E3A", linewidth=0.5, linestyle="--")
            ax.legend(facecolor="#1A1A2E", labelcolor="#FFFFFF", loc="upper right")

        axes[-1].set_xlabel("Elapsed time [hours]", color="#AAAAAA")
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        out_path = self.results_dir / "coordinate_difference.png"
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        log.debug("Saved: %s", out_path)

    def _plot_differences_vxyz(self, df: pd.DataFrame, hours: np.ndarray, sat_name: str) -> None:
        """Plot component-wise velocity differences over time."""
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), facecolor="#0D1117", sharex=True)
        fig.suptitle(f"{sat_name} — Velocity Differences", color="#FFFFFF", fontsize=14)

        # TEME - GCRS
        dvx_tg = df["vx_teme_kms"].to_numpy() - df["vx_gcrs_kms"].to_numpy()
        dvy_tg = df["vy_teme_kms"].to_numpy() - df["vy_gcrs_kms"].to_numpy()
        dvz_tg = df["vz_teme_kms"].to_numpy() - df["vz_gcrs_kms"].to_numpy()

        # GCRS - ITRS
        dvx_gi = df["vx_gcrs_kms"].to_numpy() - df["vx_itrs_kms"].to_numpy()
        dvy_gi = df["vy_gcrs_kms"].to_numpy() - df["vy_itrs_kms"].to_numpy()
        dvz_gi = df["vz_gcrs_kms"].to_numpy() - df["vz_itrs_kms"].to_numpy()

        diffs = [
            (dvx_tg, dvx_gi, "Vx Difference [km/s]"),
            (dvy_tg, dvy_gi, "Vy Difference [km/s]"),
            (dvz_tg, dvz_gi, "Vz Difference [km/s]"),
        ]

        for ax, (diff_tg, diff_gi, ylabel) in zip(axes, diffs):
            ax.set_facecolor("#0D1117")
            ax.plot(hours, diff_tg, color=self.c_x, label="TEME - GCRS", linewidth=0.8, alpha=0.9)
            ax.plot(hours, diff_gi, color=self.c_y, label="GCRS - ITRS", linewidth=0.8, alpha=0.9)
            ax.set_ylabel(ylabel, color="#AAAAAA")
            ax.tick_params(colors="#AAAAAA")
            ax.grid(color="#1E1E3A", linewidth=0.5, linestyle="--")
            ax.legend(facecolor="#1A1A2E", labelcolor="#FFFFFF", loc="upper right")

        axes[-1].set_xlabel("Elapsed time [hours]", color="#AAAAAA")
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        out_path = self.results_dir / "velocity_difference.png"
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        log.debug("Saved: %s", out_path)

    def _plot_frame_comparison(self, df: pd.DataFrame, sat_name: str) -> None:
        """Plot the comparison between inertial (GCRS) and rotating (ITRS) frames in 3D."""
        fig = plt.figure(figsize=(10, 8), facecolor="#0D1117")
        ax = fig.add_subplot(111, projection="3d", facecolor="#0D1117")

        self._draw_earth(ax)

        # GCRS (Inertial)
        x_g = df["x_gcrs_km"].to_numpy()
        y_g = df["y_gcrs_km"].to_numpy()
        z_g = df["z_gcrs_km"].to_numpy()
        ax.plot(x_g, y_g, z_g, color=self.c_x, linewidth=1.0, label="GCRS (Inertial)")

        # ITRS (Rotating)
        x_i = df["x_itrs_km"].to_numpy()
        y_i = df["y_itrs_km"].to_numpy()
        z_i = df["z_itrs_km"].to_numpy()
        ax.plot(x_i, y_i, z_i, color=self.c_y, linewidth=0.8, alpha=0.8, linestyle="--", label="ITRS (Rotating)")

        ax.set_xlabel("X [km]", color="#AAAAAA")
        ax.set_ylabel("Y [km]", color="#AAAAAA")
        ax.set_zlabel("Z [km]", color="#AAAAAA")
        ax.tick_params(colors="#AAAAAA")
        ax.set_title(f"{sat_name} — Frame Comparison (GCRS vs ITRS)", color="#FFFFFF", fontsize=14)
        ax.legend(facecolor="#1A1A2E", labelcolor="#FFFFFF")

        out_path = self.results_dir / "frame_comparison.png"
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        log.debug("Saved: %s", out_path)

    def _plot_transformation_validation(self, df: pd.DataFrame, hours: np.ndarray, sat_name: str) -> None:
        """Plot validation metrics showing norm preservation errors to verify conversion correctness."""
        fig, ax = plt.subplots(figsize=(12, 6), facecolor="#0D1117")
        ax.set_facecolor("#0D1117")

        r_teme = np.sqrt(df["x_teme_km"]**2 + df["y_teme_km"]**2 + df["z_teme_km"]**2)
        r_gcrs = np.sqrt(df["x_gcrs_km"]**2 + df["y_gcrs_km"]**2 + df["z_gcrs_km"]**2)
        r_itrs = np.sqrt(df["x_itrs_km"]**2 + df["y_itrs_km"]**2 + df["z_itrs_km"]**2)

        # Norm preservation error in meters
        err_gcrs = np.abs(r_teme - r_gcrs) * 1000.0
        err_itrs = np.abs(r_teme - r_itrs) * 1000.0

        ax.plot(hours, err_gcrs, color=self.c_x, label="TEME vs GCRS Position Norm Error", linewidth=1.2)
        ax.plot(hours, err_itrs, color=self.c_y, label="TEME vs ITRS Position Norm Error", linewidth=1.0, linestyle="--")

        ax.set_title(f"{sat_name} — Position Norm Preservation Validation", color="#FFFFFF", fontsize=14)
        ax.set_xlabel("Elapsed time [hours]", color="#AAAAAA")
        ax.set_ylabel("Norm Preservation Error [meters]", color="#AAAAAA")
        ax.tick_params(colors="#AAAAAA")
        ax.grid(color="#1E1E3A", linewidth=0.5, linestyle="--")
        ax.legend(facecolor="#1A1A2E", labelcolor="#FFFFFF")

        out_path = self.results_dir / "transformation_validation.png"
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        log.debug("Saved: %s", out_path)
