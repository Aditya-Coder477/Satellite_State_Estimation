"""
src/ground_station/statistics.py
=================================
GroundStationStatistics.

Computes pass intervals (station_passes.csv) and summarizes visibility/range
metrics per station (station_statistics.json) and for the whole network (visibility_summary.json).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ground_station.ground_station import GroundStation
from src.utils.logger import get_logger

log = get_logger(__name__)


class GroundStationStatistics:
    """Computes passes and summarizes ground station tracking statistics."""

    def __init__(self, cfg: dict[str, Any], output_dir: Path) -> None:
        self.cfg = cfg
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

    def process(self, df_geom: pd.DataFrame, stations: list[GroundStation]) -> None:
        """
        Identify passes and compute statistics, saving JSON and CSV deliverables.

        Parameters
        ----------
        df_geom : pd.DataFrame
            The computed geometry DataFrame.
        stations : list[GroundStation]
            The list of ground stations.
        """
        self._compute_passes(df_geom, stations)
        self._compute_json_stats(df_geom, stations)

    def _compute_passes(self, df: pd.DataFrame, stations: list[GroundStation]) -> None:
        """Identify contiguous visibility blocks (passes) per station and save CSV."""
        log.info("Identifying ground station passes...")
        passes = []

        for s in stations:
            df_s = df[df["station_id"] == s.id].reset_index(drop=True)
            if len(df_s) == 0:
                continue

            vis = df_s["visible"].to_numpy().astype(int)
            t = df_s["elapsed_seconds"].to_numpy()
            utc = df_s["utc_time"].to_numpy()
            el = df_s["elevation_deg"].to_numpy()
            r = df_s["range_km"].to_numpy()

            # Find transitions
            diff = np.diff(vis)
            starts = np.where(diff == 1)[0] + 1
            ends = np.where(diff == -1)[0]

            # Handle boundaries
            if vis[0] == 1:
                starts = np.insert(starts, 0, 0)
            if vis[-1] == 1:
                ends = np.append(ends, len(vis) - 1)

            # Loop through detected passes
            for start_idx, end_idx in zip(starts, ends):
                if end_idx < start_idx:
                    continue

                pass_el = el[start_idx : end_idx + 1]
                pass_r = r[start_idx : end_idx + 1]

                passes.append({
                    "Station": s.name,
                    "Pass Start": str(utc[start_idx]),
                    "Pass End": str(utc[end_idx]),
                    "Maximum Elevation": float(pass_el.max()),
                    "Pass Duration": float(t[end_idx] - t[start_idx]),
                    "Minimum Range": float(pass_r.min()),
                })

        df_passes = pd.DataFrame(passes)
        passes_path = self.out / "station_passes.csv"
        
        # Save output CSV (even if empty, write headers)
        if len(df_passes) == 0:
            df_passes = pd.DataFrame(columns=[
                "Station", "Pass Start", "Pass End", "Maximum Elevation", "Pass Duration", "Minimum Range"
            ])

        df_passes.to_csv(passes_path, index=False, float_format="%.4f")
        log.info("Saved passes (%d intervals) to: %s", len(df_passes), passes_path)

    def _compute_json_stats(self, df: pd.DataFrame, stations: list[GroundStation]) -> None:
        """Compute per-station statistics and visibility summaries, saving JSONs."""
        log.info("Computing ground station summary statistics...")
        stats_dict = {}

        total_visible = 0
        total_invisible = 0

        for s in stations:
            df_s = df[df["station_id"] == s.id]
            if len(df_s) == 0:
                continue

            vis = df_s["visible"].to_numpy()
            el = df_s["elevation_deg"].to_numpy()
            r = df_s["range_km"].to_numpy()

            visible_count = int(np.sum(vis))
            invisible_count = int(np.sum(~vis))

            total_visible += visible_count
            total_invisible += invisible_count

            # Stats only on visible part where appropriate, fallback to full if never visible
            vis_el = el[vis] if visible_count > 0 else el
            vis_r = r[vis] if visible_count > 0 else r

            stats_dict[s.name] = {
                "Station ID": s.id,
                "Coordinates": {
                    "Latitude": s.lat_deg,
                    "Longitude": s.lon_deg,
                    "Height": s.height_m,
                },
                "Visible Samples": visible_count,
                "Invisible Samples": invisible_count,
                "Maximum Elevation": float(el.max()),
                "Average Elevation": float(el.mean()),
                "Maximum Range": float(r.max()),
                "Minimum Range": float(r.min()),
                "Average Range": float(r.mean()),
                "Average Visibility Percentage": float(visible_count / len(df_s) * 100.0),
            }

        # Write station_statistics.json
        stats_path = self.out / "station_statistics.json"
        with open(stats_path, "w", encoding="utf-8") as fh:
            json.dump(stats_dict, fh, indent=2)
        log.info("Saved station statistics to: %s", stats_path)

        # Write visibility_summary.json
        total_samples = total_visible + total_invisible
        summary = {
            "network_name": self.cfg.get("ground_station", {}).get("station_file", "ground_stations.json"),
            "simulation_samples": total_samples,
            "total_visible_samples": total_visible,
            "total_invisible_samples": total_invisible,
            "average_network_visibility_percentage": float(total_visible / max(total_samples, 1) * 100.0),
        }
        summary_path = self.out / "visibility_summary.json"
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        log.info("Saved visibility summary to: %s", summary_path)
