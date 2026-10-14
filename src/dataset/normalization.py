"""
src/dataset/normalization.py
============================
Handles normalization of continuous features using standard scaliers.

Supports StandardScaler, MinMaxScaler, and Identity methods. Maintains parameter
state, writes statistics to JSON, and enables exact inverse transforms.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class NormalizationError(ValueError):
    """Raised when errors occur during fit or transform operations."""
    pass


class Normalizer:
    """
    Standardizes or scales numeric columns in a pandas DataFrame.

    Fits scaling parameters on training data and applies them to all sets.
    
    Parameters
    ----------
    method : str
        Scaling strategy to apply: "standard", "minmax", or "identity".
    columns : List[str]
        List of column names to normalize.
    """

    def __init__(self, method: str = "standard", columns: List[str] | None = None) -> None:
        self.method = method.lower()
        self.columns = columns or []
        self.stats: Dict[str, Dict[str, float]] = {}
        
        valid_methods = ["standard", "minmax", "identity"]
        if self.method not in valid_methods:
            raise ValueError(f"Invalid normalization method '{method}'. Valid options: {valid_methods}")

    def fit(self, df: pd.DataFrame) -> None:
        """
        Computes scaling parameters (mean/std or min/max) from a training DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The training dataset to fit statistics on.
        """
        log.info("Fitting normalizer using method='%s' on %d columns", self.method, len(self.columns))
        self.stats = {}

        if self.method == "identity":
            return

        for col in self.columns:
            if col not in df.columns:
                log.warning("Column '%s' not found in training DataFrame during normalizer fit.", col)
                continue

            series = df[col].astype(float)
            
            if self.method == "standard":
                mean_val = float(series.mean())
                std_val = float(series.std())
                # Handle zero variance column
                if np.isnan(std_val) or std_val == 0.0:
                    std_val = 1.0
                self.stats[col] = {"mean": mean_val, "std": std_val}
                
            elif self.method == "minmax":
                min_val = float(series.min())
                max_val = float(series.max())
                # Handle zero range column
                if min_val == max_val:
                    max_val = min_val + 1.0
                self.stats[col] = {"min": min_val, "max": max_val}

        log.debug("Fitted statistics: %s", self.stats)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes the configured columns in a DataFrame using fitted statistics.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to normalize.

        Returns
        -------
        pd.DataFrame
            A copy of the DataFrame with normalized columns.
        """
        df_out = df.copy()
        
        if self.method == "identity":
            return df_out

        for col in self.columns:
            if col not in df_out.columns:
                continue

            if col not in self.stats:
                log.warning("No fitted statistics for column '%s'. Skipping normalization.", col)
                continue

            col_stats = self.stats[col]
            series = df_out[col].astype(float)

            if self.method == "standard":
                df_out[col] = (series - col_stats["mean"]) / col_stats["std"]
            elif self.method == "minmax":
                df_out[col] = (series - col_stats["min"]) / (col_stats["max"] - col_stats["min"])

        return df_out

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Restores normalized columns in a DataFrame to their original scale.

        Parameters
        ----------
        df : pd.DataFrame
            Normalized DataFrame.

        Returns
        -------
        pd.DataFrame
            A copy of the DataFrame with inverse-transformed columns.
        """
        df_out = df.copy()

        if self.method == "identity":
            return df_out

        for col in self.columns:
            if col not in df_out.columns:
                continue

            if col not in self.stats:
                log.warning("No fitted statistics for column '%s'. Skipping inverse transform.", col)
                continue

            col_stats = self.stats[col]
            series = df_out[col].astype(float)

            if self.method == "standard":
                df_out[col] = (series * col_stats["std"]) + col_stats["mean"]
            elif self.method == "minmax":
                df_out[col] = (series * (col_stats["max"] - col_stats["min"])) + col_stats["min"]

        return df_out

    def save(self, path: Path) -> None:
        """Saves fitted normalization parameters to a JSON file."""
        log.info("Saving normalization parameters to: %s", path)
        data = {
            "method": self.method,
            "columns": self.columns,
            "stats": self.stats,
        }
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=4)
        except Exception as e:
            raise NormalizationError(f"Failed to save normalization parameters: {e}")

    def load(self, path: Path) -> None:
        """Loads fitted normalization parameters from a JSON file."""
        log.info("Loading normalization parameters from: %s", path)
        if not path.is_file():
            raise FileNotFoundError(f"Normalization file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.method = data["method"]
            self.columns = data["columns"]
            self.stats = data["stats"]
        except Exception as e:
            raise NormalizationError(f"Failed to load normalization parameters: {e}")

    # ------------------------------------------------------------------
    # Phase 7 convenience API
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on df, then transform and return the normalised DataFrame."""
        self.columns = list(df.select_dtypes(include=[np.number]).columns)
        self.fit(df)
        return self.transform(df)

    def get_params(self) -> dict:
        """Return the fitted normalisation parameters as a JSON-serialisable dict."""
        return {
            "method": self.method,
            "columns": self.columns,
            "stats": self.stats,
        }

    @classmethod
    def from_cfg(cls, cfg: dict) -> "Normalizer":
        """Construct a Normalizer from a project config dict."""
        method = cfg.get("dataset", {}).get("normalization", {}).get("method", "standard")
        return cls(method=method)


