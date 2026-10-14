"""
src/propagation/visualizer.py
==============================
Publication-quality visualisation of propagated satellite trajectories.

Generates three figures
-----------------------
1. **trajectory_3d.png** — 3-D TEME orbit with Earth sphere overlay.
2. **position_xyz.png**  — X / Y / Z TEME position vs. time.
3. **velocity_xyz.png**  — Vx / Vy / Vz TEME velocity vs. time.

All figures are written to ``results/propagation/`` as managed by
:class:`~src.utils.path_manager.PathManager`.

Design decisions
----------------
* Uses ``matplotlib``'s ``dark_background`` style and project colour palette
  (configurable via ``cfg["visualization"]``) for a consistent research aesthetic.
* Earth sphere radius = 6 371 km (IAU mean).
* Time axis is expressed in decimal hours from epoch for readability.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

from src.utils.logger import get_logger

matplotlib.use("Agg")  # non-interactive backend — safe in headless environments

log: logging.Logger = get_logger(__name__)

_EARTH_RADIUS_KM = 6_371.0


class PropagationVisualizer:
    """
    Generates and saves all trajectory visualisation figures.

    Parameters
    ----------
    cfg : dict[str, Any]
        Project configuration (``cfg["visualization"]`` block is used).
    results_dir : Path
        Directory in which figures are saved.

    Examples
    --------
    >>> viz = PropagationVisualizer(cfg, results_dir=pm.results_dir)
    >>> viz.plot_all(df, satellite_name="GOES 16")
    """

    def __init__(self, cfg: dict[str, Any], results_dir: Path) -> None:
        self._viz_cfg: dict[str, Any] = cfg.get("visualization", {})
        self._results_dir = Path(results_dir)
        self._results_dir.mkdir(parents=True, exist_ok=True)

        style = self._viz_cfg.get("style", "dark_background")
        plt.style.use(style)

        self._dpi: int = int(self._viz_cfg.get("dpi", 150))
        colors_cfg = self._viz_cfg.get("colors", {})
        self._c_traj: str = colors_cfg.get("trajectory", "#00D4FF")
        self._c_x: str = colors_cfg.get("x", "#FF6B6B")
        self._c_y: str = colors_cfg.get("y", "#4ECDC4")
        self._c_z: str = colors_cfg.get("z", "#FFE66D")
        self._c_earth: str = colors_cfg.get("earth", "#1A3A5C")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plot_all(self, df: pd.DataFrame, satellite_name: str) -> None:
        """
        Generate and save all three standard figures.

        Parameters
        ----------
        df : pd.DataFrame
            Validated propagated trajectory DataFrame.
        satellite_name : str
            Display name (used in plot titles).
        """
        hours = df["elapsed_seconds"].to_numpy() / 3600.0

        self._plot_trajectory_3d(df, satellite_name)
        self._plot_position_xyz(df, hours, satellite_name)
        self._plot_velocity_xyz(df, hours, satellite_name)

        log.info(
            "All figures saved to: %s", str(self._results_dir)
        )

    # ------------------------------------------------------------------
    # Individual figure generators
    # ------------------------------------------------------------------

    def _plot_trajectory_3d(
        self, df: pd.DataFrame, satellite_name: str
    ) -> None:
        """3-D orbit plot with Earth sphere."""
        fs_cfg = self._viz_cfg.get("figsize", {})
        figsize = tuple(fs_cfg.get("trajectory_3d", [10, 8]))

        fig = plt.figure(figsize=figsize, facecolor="#0D1117")
        ax: Axes3D = fig.add_subplot(111, projection="3d", facecolor="#0D1117")

        x = df["x_teme_km"].to_numpy()
        y = df["y_teme_km"].to_numpy()
        z = df["z_teme_km"].to_numpy()

        # ── Earth sphere ───────────────────────────────────────────────
        u_ang = np.linspace(0, 2 * np.pi, 60)
        v_ang = np.linspace(0, np.pi, 30)
        ex = _EARTH_RADIUS_KM * np.outer(np.cos(u_ang), np.sin(v_ang))
        ey = _EARTH_RADIUS_KM * np.outer(np.sin(u_ang), np.sin(v_ang))
        ez = _EARTH_RADIUS_KM * np.outer(np.ones_like(u_ang), np.cos(v_ang))
        ax.plot_surface(ex, ey, ez, color=self._c_earth, alpha=0.6, linewidth=0)

        # ── Orbit trace ────────────────────────────────────────────────
        ax.plot(x, y, z, color=self._c_traj, linewidth=0.8, alpha=0.9)
        # Mark start
        ax.scatter(x[0], y[0], z[0], color="#FFFFFF", s=40, zorder=5, label="Epoch")

        ax.set_xlabel("X TEME [km]", color="#AAAAAA", labelpad=8)
        ax.set_ylabel("Y TEME [km]", color="#AAAAAA", labelpad=8)
        ax.set_zlabel("Z TEME [km]", color="#AAAAAA", labelpad=8)
        ax.tick_params(colors="#AAAAAA")
        ax.set_title(
            f"{satellite_name} — 7-Day TEME Trajectory",
            color="#FFFFFF",
            fontsize=13,
            pad=12,
        )
        ax.legend(facecolor="#1A1A2E", labelcolor="#FFFFFF", fontsize=9)

        out = self._results_dir / "trajectory_3d.png"
        fig.savefig(out, dpi=self._dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        log.info("Saved: %s", out)

    def _plot_position_xyz(
        self, df: pd.DataFrame, hours: np.ndarray, satellite_name: str
    ) -> None:
        """Time-series plot of X, Y, Z TEME positions."""
        fs_cfg = self._viz_cfg.get("figsize", {})
        figsize = tuple(fs_cfg.get("position_xyz", [14, 8]))

        fig, axes = plt.subplots(3, 1, figsize=figsize, facecolor="#0D1117", sharex=True)
        fig.suptitle(
            f"{satellite_name} — TEME Position (7 days)",
            color="#FFFFFF",
            fontsize=13,
            y=0.98,
        )

        components = [
            ("x_teme_km", "X", self._c_x),
            ("y_teme_km", "Y", self._c_y),
            ("z_teme_km", "Z", self._c_z),
        ]

        for ax, (col, label, color) in zip(axes, components):
            ax.set_facecolor("#0D1117")
            ax.plot(hours, df[col].to_numpy(), color=color, linewidth=0.7, alpha=0.9)
            ax.set_ylabel(f"{label} [km]", color="#AAAAAA", fontsize=10)
            ax.tick_params(colors="#AAAAAA")
            ax.spines[:].set_color("#333355")
            ax.grid(color="#1E1E3A", linewidth=0.5, linestyle="--")

        axes[-1].set_xlabel("Elapsed time [hours]", color="#AAAAAA", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        out = self._results_dir / "position_xyz.png"
        fig.savefig(out, dpi=self._dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        log.info("Saved: %s", out)

    def _plot_velocity_xyz(
        self, df: pd.DataFrame, hours: np.ndarray, satellite_name: str
    ) -> None:
        """Time-series plot of Vx, Vy, Vz TEME velocities."""
        fs_cfg = self._viz_cfg.get("figsize", {})
        figsize = tuple(fs_cfg.get("velocity_xyz", [14, 8]))

        fig, axes = plt.subplots(3, 1, figsize=figsize, facecolor="#0D1117", sharex=True)
        fig.suptitle(
            f"{satellite_name} — TEME Velocity (7 days)",
            color="#FFFFFF",
            fontsize=13,
            y=0.98,
        )

        components = [
            ("vx_teme_kms", "Vx", self._c_x),
            ("vy_teme_kms", "Vy", self._c_y),
            ("vz_teme_kms", "Vz", self._c_z),
        ]

        for ax, (col, label, color) in zip(axes, components):
            ax.set_facecolor("#0D1117")
            ax.plot(hours, df[col].to_numpy(), color=color, linewidth=0.7, alpha=0.9)
            ax.set_ylabel(f"{label} [km/s]", color="#AAAAAA", fontsize=10)
            ax.tick_params(colors="#AAAAAA")
            ax.spines[:].set_color("#333355")
            ax.grid(color="#1E1E3A", linewidth=0.5, linestyle="--")

        axes[-1].set_xlabel("Elapsed time [hours]", color="#AAAAAA", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        out = self._results_dir / "velocity_xyz.png"
        fig.savefig(out, dpi=self._dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        log.info("Saved: %s", out)
