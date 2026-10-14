"""
src/observation_generation/statistics.py
=========================================
ObservationStatistics.

Computes overall metrics, observation densities, and scenario/station breakdowns
for the generated observations and saves JSON reports.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)


class ObservationStatistics:
    """Computes and saves statistics summaries for generated observations."""

    def __init__(self, cfg: dict[str, Any], output_dir: Path) -> None:
        self.cfg = cfg
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

    def compute_and_save(
        self,
        df_clean: pd.DataFrame,
        df_noisy: pd.DataFrame,
        df_quality: pd.DataFrame,
        val_stats: dict[str, Any] | None = None,
    ) -> None:
        """Compute all statistics and save JSON summaries."""
        # ── 1. Compute observation_statistics.json ─────────────────────
        total_obs = len(df_noisy)
        
        # Extract noise arrays (ignoring Scenario A which is zero noise)
        df_noise = df_noisy[df_noisy["scenario"] != "Scenario A (Ideal)"]
        noise_ra = df_noise["noise_ra_deg"].to_numpy()
        noise_dec = df_noise["noise_dec_deg"].to_numpy()
        noise_mag = np.sqrt(noise_ra**2 + noise_dec**2)

        # Count missing samples in df_quality
        missing_count = int(df_quality["missing"].sum())
        visible_count = int(df_quality["visible"].sum())

        # Density = total actual visible observations divided by total possible tracking time (days)
        # Total duration is 7 days
        density_per_day = total_obs / 7.0

        # Per-station counts
        station_counts = df_noisy.groupby("station_name").size().to_dict()

        # Per-scenario counts
        scenario_counts = df_noisy.groupby("scenario").size().to_dict()

        stats_data = {
            "Total observations": total_obs,
            "Visible observations": visible_count,
            "Missing observations": missing_count,
            "Average noise": float(noise_mag.mean()) if len(noise_mag) > 0 else 0.0,
            "Maximum noise": float(noise_mag.max()) if len(noise_mag) > 0 else 0.0,
            "Observation density": {
                "total_observations_per_day": float(density_per_day),
                "total_observation_count": total_obs,
            },
            "Per-station observation counts": {str(k): int(v) for k, v in station_counts.items()},
            "Per-scenario observation counts": {str(k): int(v) for k, v in scenario_counts.items()},
        }

        stats_path = self.out / "observation_statistics.json"
        with open(stats_path, "w", encoding="utf-8") as fh:
            json.dump(stats_data, fh, indent=2)
        log.info("Saved observation statistics to: %s", stats_path)

        # ── 2. Compute observation_summary.json ────────────────────────
        summary_data = {
            "phase": "Phase 6 — Optical Observation Generation",
            "satellite": "INSAT-3D",
            "scenarios_generated": list(df_noisy["scenario"].unique()),
            "clean_observation_dataset": {
                "file": "observations.csv",
                "samples": len(df_clean),
                "columns": list(df_clean.columns),
            },
            "noisy_observation_dataset": {
                "file": "noisy_observations.csv",
                "samples": len(df_noisy),
                "columns": list(df_noisy.columns),
            },
            "overall_status": "SUCCESS",
        }
        if val_stats:
            summary_data["validation"] = val_stats

        summary_path = self.out / "observation_summary.json"
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary_data, fh, indent=2)
        log.info("Saved observation summary to: %s", summary_path)

        # ── 3. Write observation_scenarios.json ────────────────────────
        # Summary description of the generated scenarios
        scenarios_desc = {
            "Scenario A (Ideal)": "Zero-noise observations on a uniform 60s cadence.",
            "Scenario B (Low Noise)": "Gaussian sensor noise with sigma = 0.005 degrees.",
            "Scenario C (High Noise)": "Gaussian sensor noise with sigma = 0.020 degrees.",
            "Scenario D (Missing Obs)": "Gaussian sensor noise (0.005 deg) with 20% random dropouts.",
            "Scenario E (Variable Sampling)": "Gaussian sensor noise (0.005 deg) with non-uniform variable sampling rate (~33% density).",
            "Scenario F (Station Outage)": "Gaussian sensor noise (0.005 deg) with station 1 offline during Day 3-4.",
        }
        scenarios_path = self.out / "observation_scenarios.json"
        with open(scenarios_path, "w", encoding="utf-8") as fh:
            json.dump(scenarios_desc, fh, indent=2)
        log.info("Saved observation scenarios description to: %s", scenarios_path)
