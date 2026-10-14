"""
src/ground_station/visualizer.py
=================================
GroundStationVisualizer.

Generates 9 high-quality visualisations for Phase 5:
  1. station_locations.png     - Ground Station Map (Lat/Lon)
  2. ground_track.png           - Satellite ground track overlaying stations
  3. elevation_vs_time.png      - Elevation vs Time for all stations
  4. azimuth_vs_time.png        - Azimuth vs Time for all stations
  5. range_vs_time.png          - Slant Range vs Time
  6. range_rate_vs_time.png     - Slant Range Rate vs Time
  7. station_pass_timeline.png  - Horizontal pass timeline
  8. sky_visibility.png         - Polar Sky Plot (Azimuth vs Elevation)
  9. visibility_map.png         - Coverage zone map
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.ground_station.ground_station import GroundStation
from src.utils.logger import get_logger

matplotlib.use("Agg")
log = get_logger(__name__)

_BG = "#0D1117"
_FG = "#FFFFFF"
_GRID = "#1E1E3A"
_SPINE = "#333355"
_LABEL = "#AAAAAA"


class GroundStationVisualizer:
    """Generates all Phase 5 visualization plots."""

    def __init__(self, cfg: dict[str, Any], results_dir: Path) -> None:
        self.cfg = cfg
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = int(cfg.get("visualization", {}).get("dpi", 150))
        plt.style.use(cfg.get("visualization", {}).get("style", "dark_background"))

        # Colors
        self.c_sat = "#00D4FF"
        self.c_vis = "#FF6B6B"
        self.c_list = ["#FF6B6B", "#4ECDC4", "#FFE66D", "#A78BFA", "#FF8C42", "#38BDF8"]

    def plot_all(
        self,
        df_geom: pd.DataFrame,
        df_transformed: pd.DataFrame,
        stations: list[GroundStation],
        satellite_name: str = "INSAT-3D",
    ) -> None:
        """Generate and save all 9 required figures."""
        log.info("Generating Ground Station simulation plots...")
        hours = df_geom["elapsed_seconds"].unique() / 3600.0

        # Calculate satellite geodetic coordinates from ITRS
        x_itrs = df_transformed["x_itrs_km"].to_numpy()
        y_itrs = df_transformed["y_itrs_km"].to_numpy()
        z_itrs = df_transformed["z_itrs_km"].to_numpy()
        
        sat_lon = np.degrees(np.arctan2(y_itrs, x_itrs))
        sat_lat = np.degrees(np.arctan2(z_itrs, np.sqrt(x_itrs**2 + y_itrs**2)))

        self._plot_station_locations(stations)
        self._plot_ground_track(sat_lon, sat_lat, stations, satellite_name)
        self._plot_time_series(df_geom, hours, "elevation_deg", "Elevation [deg]", "elevation_vs_time.png", satellite_name)
        self._plot_time_series(df_geom, hours, "azimuth_deg", "Azimuth [deg]", "azimuth_vs_time.png", satellite_name)
        self._plot_time_series(df_geom, hours, "range_km", "Slant Range [km]", "range_vs_time.png", satellite_name)
        self._plot_time_series(df_geom, hours, "range_rate_km_s", "Range Rate [km/s]", "range_rate_vs_time.png", satellite_name)
        self._plot_pass_timeline(df_geom, stations, satellite_name)
        self._plot_polar_sky_plot(df_geom, stations, satellite_name)
        self._plot_visibility_map(sat_lon, sat_lat, df_geom, stations, satellite_name)

        log.info("All Ground Station plots saved in: %s", self.results_dir)

    # ── Figure 1 ── Ground Station Map
    def _plot_station_locations(self, stations: list[GroundStation]) -> None:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=_BG)
        ax.set_facecolor(_BG)

        for i, s in enumerate(stations):
            color = self.c_list[i % len(self.c_list)]
            ax.scatter(s.lon_deg, s.lat_deg, color=color, s=100, marker="^", edgecolors=_FG, zorder=5)
            ax.text(s.lon_deg + 0.5, s.lat_deg + 0.5, s.name, color=_FG, fontsize=10, fontweight="bold")

        ax.set_title("Ground Station Network Locations", color=_FG, fontsize=14, pad=15)
        ax.set_xlabel("Longitude [deg]", color=_LABEL)
        ax.set_ylabel("Latitude [deg]", color=_LABEL)
        ax.grid(color=_GRID, linestyle="--", linewidth=0.5)
        ax.tick_params(colors=_LABEL)
        ax.spines[:].set_color(_SPINE)

        # Set bounds to encompass India region (where INSAT3D network is)
        ax.set_xlim(60, 105)
        ax.set_ylim(0, 40)

        self._save(fig, "station_locations.png")

    # ── Figure 2 ── Ground Track
    def _plot_ground_track(
        self, sat_lon: np.ndarray, sat_lat: np.ndarray, stations: list[GroundStation], sat_name: str
    ) -> None:
        fig, ax = plt.subplots(figsize=(12, 6), facecolor=_BG)
        ax.set_facecolor(_BG)

        # Satellite ground track
        ax.plot(sat_lon, sat_lat, color=self.c_sat, linewidth=1.5, label=f"{sat_name} Track", zorder=3)
        ax.scatter(sat_lon[0], sat_lat[0], color="#FFFFFF", s=50, marker="o", label="Epoch", zorder=4)

        # Ground stations
        for i, s in enumerate(stations):
            color = self.c_list[i % len(self.c_list)]
            ax.scatter(s.lon_deg, s.lat_deg, color=color, s=80, marker="^", edgecolors=_FG, zorder=5, label=s.name)

        ax.set_title(f"{sat_name} — 7-Day Ground Track and Ground Station Network", color=_FG, fontsize=14, pad=15)
        ax.set_xlabel("Longitude [deg]", color=_LABEL)
        ax.set_ylabel("Latitude [deg]", color=_LABEL)
        ax.grid(color=_GRID, linestyle="--", linewidth=0.5)
        ax.tick_params(colors=_LABEL)
        ax.spines[:].set_color(_SPINE)
        
        # GEO satellite ground track is localized, set regional/global view
        ax.set_xlim(30, 130)
        ax.set_ylim(-30, 50)
        ax.legend(facecolor="#1A1A2E", labelcolor=_FG, loc="upper right")

        self._save(fig, "ground_track.png")

    # ── Figures 3, 4, 5, 6 ── Time Series Plots
    def _plot_time_series(
        self, df: pd.DataFrame, hours: np.ndarray, column: str, ylabel: str, filename: str, sat_name: str
    ) -> None:
        fig, ax = plt.subplots(figsize=(12, 6), facecolor=_BG)
        ax.set_facecolor(_BG)

        for i, s_name in enumerate(df["station_name"].unique()):
            df_s = df[df["station_name"] == s_name]
            y = df_s[column].to_numpy()
            color = self.c_list[i % len(self.c_list)]
            ax.plot(hours, y, color=color, linewidth=1.0, label=s_name)

        ax.set_title(f"{sat_name} — {ylabel} vs Time", color=_FG, fontsize=14, pad=15)
        ax.set_xlabel("Elapsed time [hours]", color=_LABEL)
        ax.set_ylabel(ylabel, color=_LABEL)
        ax.grid(color=_GRID, linestyle="--", linewidth=0.5)
        ax.tick_params(colors=_LABEL)
        ax.spines[:].set_color(_SPINE)
        ax.legend(facecolor="#1A1A2E", labelcolor=_FG, loc="upper right")

        self._save(fig, filename)

    # ── Figure 7 ── Pass Timeline
    def _plot_pass_timeline(
        self, df: pd.DataFrame, stations: list[GroundStation], sat_name: str
    ) -> None:
        fig, ax = plt.subplots(figsize=(12, 6), facecolor=_BG)
        ax.set_facecolor(_BG)

        hours = df["elapsed_seconds"].unique() / 3600.0

        for i, s in enumerate(stations):
            df_s = df[df["station_id"] == s.id].reset_index(drop=True)
            if len(df_s) == 0:
                continue

            vis = df_s["visible"].to_numpy()
            color = self.c_list[i % len(self.c_list)]
            
            # Plot line for visible periods
            y_val = i + 1
            ax.plot([hours[0], hours[-1]], [y_val, y_val], color="#1E1E3A", linewidth=6, zorder=1)
            
            # Draw visible segments
            visible_hours = hours[vis]
            if len(visible_hours) > 0:
                ax.scatter(visible_hours, np.full_like(visible_hours, y_val), color=color, s=8, marker="|", zorder=2)

        ax.set_title(f"{sat_name} Ground Station Pass Timeline", color=_FG, fontsize=14, pad=15)
        ax.set_xlabel("Elapsed time [hours]", color=_LABEL)
        ax.set_yticks(range(1, len(stations) + 1))
        ax.set_yticklabels([s.name for s in stations], color=_FG)
        ax.grid(color=_GRID, linestyle="--", linewidth=0.5)
        ax.tick_params(colors=_LABEL)
        ax.spines[:].set_color(_SPINE)

        self._save(fig, "station_pass_timeline.png")

    # ── Figure 8 ── Polar Sky Plot (Azimuth vs Elevation)
    def _plot_polar_sky_plot(self, df: pd.DataFrame, stations: list[GroundStation], sat_name: str) -> None:
        fig = plt.figure(figsize=(9, 9), facecolor=_BG)
        ax = fig.add_subplot(111, projection="polar", facecolor=_BG)
        
        # Radii represents 90 - Elevation (so 90 deg elevation is at the center, 0 at the edge)
        # Grid lines for 0 to 90 degrees
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1) # Clockwise
        
        for i, s in enumerate(stations):
            df_s = df[(df["station_id"] == s.id) & (df["visible"])].reset_index(drop=True)
            if len(df_s) == 0:
                continue

            az_rad = np.radians(df_s["azimuth_deg"].to_numpy())
            el = df_s["elevation_deg"].to_numpy()
            radius = 90.0 - el

            color = self.c_list[i % len(self.c_list)]
            ax.scatter(az_rad, radius, color=color, s=5, label=s.name, alpha=0.8)

        ax.set_title(f"{sat_name} — Polar Sky Plot (Visible Gaps Only)", color=_FG, fontsize=14, pad=20)
        
        # Format polar axes
        ax.set_rmax(90)
        ax.set_rticks([0, 30, 60, 90])
        ax.set_yticklabels(["90°", "60°", "30°", "0°"], color=_LABEL)
        ax.tick_params(colors=_LABEL)
        ax.grid(color=_GRID, linestyle="--", linewidth=0.5)
        ax.legend(facecolor="#1A1A2E", labelcolor=_FG, loc="lower right")

        self._save(fig, "sky_visibility.png")

    # ── Figure 9 ── Coverage Zone Map (Visibility Map)
    def _plot_visibility_map(
        self, sat_lon: np.ndarray, sat_lat: np.ndarray, df: pd.DataFrame, stations: list[GroundStation], sat_name: str
    ) -> None:
        fig, ax = plt.subplots(figsize=(12, 6), facecolor=_BG)
        ax.set_facecolor(_BG)

        # Plot full trajectory (dim)
        ax.plot(sat_lon, sat_lat, color="#1E1E3A", linewidth=2.0, zorder=1)

        # Plot visible segments color-coded by station
        for i, s in enumerate(stations):
            df_s = df[df["station_id"] == s.id].reset_index(drop=True)
            if len(df_s) == 0:
                continue

            vis = df_s["visible"].to_numpy()
            color = self.c_list[i % len(self.c_list)]
            
            # Scatter visible satellite coordinates
            ax.scatter(
                sat_lon[vis], sat_lat[vis],
                color=color, s=10, alpha=0.6, label=f"Visible to {s.name}", zorder=2
            )

        # Draw station positions
        for i, s in enumerate(stations):
            color = self.c_list[i % len(self.c_list)]
            ax.scatter(s.lon_deg, s.lat_deg, color=color, s=100, marker="^", edgecolors=_FG, zorder=5)

        ax.set_title(f"{sat_name} — Coverage and Visibility Map", color=_FG, fontsize=14, pad=15)
        ax.set_xlabel("Longitude [deg]", color=_LABEL)
        ax.set_ylabel("Latitude [deg]", color=_LABEL)
        ax.grid(color=_GRID, linestyle="--", linewidth=0.5)
        ax.tick_params(colors=_LABEL)
        ax.spines[:].set_color(_SPINE)
        
        ax.set_xlim(40, 120)
        ax.set_ylim(-20, 40)
        ax.legend(facecolor="#1A1A2E", labelcolor=_FG, loc="upper right")

        self._save(fig, "visibility_map.png")

    def _save(self, fig: plt.Figure, filename: str) -> None:
        out = self.results_dir / filename
        fig.savefig(out, dpi=self.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        log.info("Saved: %s", out)
