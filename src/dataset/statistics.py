"""
src/dataset/statistics.py
==========================
Computes comprehensive dataset and feature statistics.

Generates `dataset_summary.json` containing high-level metadata (splits, stations,
visibility percentages, and noise errors) and `feature_statistics.json` containing
column-wise parameters (mean, std, min, max).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class DatasetStatistics:
    """
    Computes and formats statistics for simulated observation datasets.
    """

    @staticmethod
    def compute_feature_statistics(df: pd.DataFrame, columns: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Computes mean, std, min, max for continuous numeric features in a dataset.

        Parameters
        ----------
        df : pd.DataFrame
            The dataset (usually raw training split).
        columns : List[str]
            List of column names to analyze.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Nested dictionary of statistics per column.
        """
        log.info("Computing feature statistics for %d columns", len(columns))
        stats_out = {}
        
        for col in columns:
            if col not in df.columns:
                continue
            
            series = df[col].dropna().astype(float)
            if len(series) == 0:
                continue

            stats_out[col] = {
                "mean": float(series.mean()),
                "std": float(series.std()) if len(series) > 1 else 0.0,
                "min": float(series.min()),
                "max": float(series.max()),
                "count": int(len(series)),
            }

        return stats_out

    @classmethod
    def compute_dataset_summary(
        cls,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        raw_obs_df: pd.DataFrame,
        collocation_df: pd.DataFrame | None = None,
    ) -> Dict[str, Any]:
        """
        Computes high-level summaries including counts, splits, ground stations,
        visibility fractions, missing percentages, and noise statistics.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training split.
        val_df : pd.DataFrame
            Validation split.
        test_df : pd.DataFrame
            Testing split.
        raw_obs_df : pd.DataFrame
            Full dataset before splitting.
        collocation_df : pd.DataFrame, optional
            Physics collocation points, by default None.

        Returns
        -------
        Dict[str, Any]
            The compiled dataset summary.
        """
        log.info("Computing comprehensive dataset summary")
        
        # Satellite metadata
        sat_name = str(raw_obs_df["satellite_name"].iloc[0]) if "satellite_name" in raw_obs_df.columns else "Unknown"
        norad_id = int(raw_obs_df["norad_id"].iloc[0]) if "norad_id" in raw_obs_df.columns else -1
        
        # Station counts
        n_stations = int(raw_obs_df["station_id"].nunique()) if "station_id" in raw_obs_df.columns else 0
        stations_list = raw_obs_df["station_name"].unique().tolist() if "station_name" in raw_obs_df.columns else []

        # Visibility metrics (based on 'visible' flag)
        total_rows = len(raw_obs_df)
        visible_rows = 0
        visible_pct = 0.0
        if "visible" in raw_obs_df.columns:
            visible_rows = int(raw_obs_df["visible"].sum())
            visible_pct = float(visible_rows / total_rows * 100) if total_rows > 0 else 0.0
        
        hidden_rows = total_rows - visible_rows
        hidden_pct = 100.0 - visible_pct

        # Outages/Missing observations logic
        # If 'observation_quality' column exists, check for weather outages or sparse drops
        # Real observations have quality='good'. Missing ones have quality='cloud_outage' or 'missing_fraction' etc.
        missing_rows = 0
        if "observation_quality" in raw_obs_df.columns:
            missing_rows = int((raw_obs_df["observation_quality"] != "good").sum())
        
        missing_pct = float(missing_rows / total_rows * 100) if total_rows > 0 else 0.0

        # Time coverage
        t_col = "elapsed_seconds" if "elapsed_seconds" in raw_obs_df.columns else "time"
        t_min = float(raw_obs_df[t_col].min()) if total_rows > 0 else 0.0
        t_max = float(raw_obs_df[t_col].max()) if total_rows > 0 else 0.0
        time_span = t_max - t_min

        utc_col = "utc_time"
        start_utc = str(raw_obs_df[utc_col].min()) if utc_col in raw_obs_df.columns else "N/A"
        end_utc = str(raw_obs_df[utc_col].max()) if utc_col in raw_obs_df.columns else "N/A"

        # Noise statistics
        noise_stats = {}
        noise_model = "none"
        if "noise_model" in raw_obs_df.columns:
            models_present = raw_obs_df["noise_model"].dropna().unique()
            valid_models = [m for m in models_present if m not in ["none", "None", ""]]
            if valid_models:
                noise_model = str(valid_models[0])

        # Evaluate actual error metrics if noisy columns exist
        if "ra_noisy_deg" in raw_obs_df.columns and "right_ascension_deg" in raw_obs_df.columns:
            ra_err = (raw_obs_df["ra_noisy_deg"] - raw_obs_df["right_ascension_deg"]).astype(float)
            dec_err = (raw_obs_df["dec_noisy_deg"] - raw_obs_df["declination_deg"]).astype(float)
            
            # Angle wrap correction for RA errors (since RA is cyclic [0, 360])
            ra_err = (ra_err + 180.0) % 360.0 - 180.0
            
            noise_stats = {
                "ra_error_deg": {
                    "mean": float(ra_err.mean()),
                    "std": float(ra_err.std()),
                    "min": float(ra_err.min()),
                    "max": float(ra_err.max()),
                },
                "dec_error_deg": {
                    "mean": float(dec_err.mean()),
                    "std": float(dec_err.std()),
                    "min": float(dec_err.min()),
                    "max": float(dec_err.max()),
                }
            }

        collocation_count = len(collocation_df) if collocation_df is not None else 0

        summary = {
            "satellite_metadata": {
                "satellite_name": sat_name,
                "norad_id": norad_id,
            },
            "temporal_range": {
                "start_utc": start_utc,
                "end_utc": end_utc,
                "duration_seconds": time_span,
                "min_elapsed_seconds": t_min,
                "max_elapsed_seconds": t_max,
            },
            "ground_station_network": {
                "number_of_stations": n_stations,
                "stations": stations_list,
            },
            "data_fractions": {
                "total_observations": total_rows,
                "visible_observations": visible_rows,
                "visible_pct": visible_pct,
                "hidden_observations": hidden_rows,
                "hidden_pct": hidden_pct,
                "missing_or_corrupted_observations": missing_rows,
                "missing_pct": missing_pct,
            },
            "dataset_splits": {
                "train_samples": len(train_df),
                "val_samples": len(val_df),
                "test_samples": len(test_df),
                "collocation_samples": collocation_count,
            },
            "noise_configuration": {
                "noise_model": noise_model,
                "actual_errors": noise_stats,
            }
        }

        return summary
