"""
src/observation_generation/sampling_models.py
==============================================
Sampling and Dropout Models.

Handles variable sampling cadences and random observation dropouts.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)


class ObservationSampler:
    """Manages sampling rates and missing observation dropouts."""

    @staticmethod
    def apply_sampling_cadence(
        df: pd.DataFrame,
        interval_seconds: int,
    ) -> pd.DataFrame:
        """
        Keep only observations matching a specified sampling cadence.

        Parameters
        ----------
        df : pd.DataFrame
            The observations DataFrame.
        interval_seconds : int
            Desired sampling interval in seconds.

        Returns
        -------
        pd.DataFrame
            Subsampled DataFrame.
        """
        if interval_seconds <= 60:
            return df

        # Vectorized check: elapsed_seconds must align with the requested interval
        # (Allows tolerance for floating-point inaccuracies)
        elapsed = df["elapsed_seconds"].to_numpy()
        mask = np.isclose(elapsed % interval_seconds, 0.0, atol=1e-3) | \
               np.isclose(elapsed % interval_seconds, interval_seconds, atol=1e-3)
        
        df_sub = df[mask].copy()
        log.debug("Subsampled dataset: kept %d/%d rows (interval=%ds)", len(df_sub), len(df), interval_seconds)
        return df_sub

    @staticmethod
    def apply_missing_dropout(
        df: pd.DataFrame,
        missing_fraction: float,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Randomly drop a fraction of observations to simulate data loss.

        Parameters
        ----------
        df : pd.DataFrame
        missing_fraction : float
            Fraction of samples to drop (between 0.0 and 1.0).
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        pd.DataFrame
            DataFrame with dropped rows.
        """
        if missing_fraction <= 0.0:
            return df

        n = len(df)
        rng = np.random.default_rng(seed)
        
        # Vectorized choice: select random rows to keep
        keep_mask = rng.uniform(size=n) >= missing_fraction
        df_dropped = df[keep_mask].copy()
        
        log.debug(
            "Missing dropout applied: dropped %d/%d rows (%.1f%%)",
            n - len(df_dropped), n, missing_fraction * 100.0
        )
        return df_dropped
