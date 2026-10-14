"""
src/dataset/feature_engineering.py
==================================
Performs feature engineering for the Satellite PINN State Estimation datasets.

Computes physical, periodic, and encoding features such as fraction of orbit,
sine/cosine time, day fraction, simulation fraction, deterministic one-hot station
encodings, visibility encodings, and noise indicators.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class FeatureEngineer:
    """
    Applies feature engineering transforms to satellite state/observation datasets.

    Parameters
    ----------
    cfg : dict[str, Any]
        Project configuration dictionary.
    
    Attributes
    ----------
    orbital_period : float
        Orbital period in seconds, either configured or dynamically estimated.
    known_stations : List[int]
        List of ground station IDs seen during fitting to ensure consistent one-hot columns.
    t_min : float
        Minimum elapsed seconds from training set to resolve simulation fraction.
    t_max : float
        Maximum elapsed seconds from training set to resolve simulation fraction.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self._cfg = cfg
        self._feat_cfg = cfg.get("dataset", {}).get("feature_engineering", {})
        
        self.orbital_period: float | None = self._feat_cfg.get("estimated_period_seconds")
        self.known_stations: List[int] = []
        self.t_min: float = 0.0
        self.t_max: float = 1.0

    def fit(self, train_df: pd.DataFrame, truth_df: pd.DataFrame) -> None:
        """
        Fits feature engineering parameters (orbital period, stations, time bounds)
        on the training and ground-truth datasets.

        Parameters
        ----------
        train_df : pd.DataFrame
            The training dataset split.
        truth_df : pd.DataFrame
            The ground-truth states dataset (used for Keplerian period estimation).
        """
        log.info("Fitting FeatureEngineer parameters on training data")

        # 1. Estimate/Resolve orbital period
        if self.orbital_period is None:
            self.orbital_period = self._estimate_orbital_period(truth_df)
        else:
            log.info("Using configured orbital period: %.2f seconds", self.orbital_period)

        # 2. Fit known station IDs
        if "station_id" in train_df.columns:
            self.known_stations = sorted(train_df["station_id"].dropna().unique().astype(int).tolist())
            log.info("Identified %d ground stations: %s", len(self.known_stations), self.known_stations)
        else:
            self.known_stations = []

        # 3. Fit simulation time bounds
        if "elapsed_seconds" in train_df.columns:
            # Fallback to truth if train is empty or missing elapsed_seconds
            self.t_min = float(train_df["elapsed_seconds"].min())
            self.t_max = float(train_df["elapsed_seconds"].max())
        elif "elapsed_seconds" in truth_df.columns:
            self.t_min = float(truth_df["elapsed_seconds"].min())
            self.t_max = float(truth_df["elapsed_seconds"].max())
        
        # Handle division by zero for single-point simulation
        if self.t_min == self.t_max:
            self.t_max = self.t_min + 1.0

        log.info("Simulation time window fitted: [%.1f, %.1f] seconds", self.t_min, self.t_max)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds engineered features to the input DataFrame.

        Keeps all original columns intact and appends the new engineered columns.

        Parameters
        ----------
        df : pd.DataFrame
            The dataset to engineer features for.

        Returns
        -------
        pd.DataFrame
            DataFrame copy containing original and new features.
        """
        df_out = df.copy()

        if not self._feat_cfg.get("enabled", True):
            log.info("Feature engineering is disabled in config. Returning original DataFrame.")
            return df_out

        # Ensure elapsed_seconds exists
        if "elapsed_seconds" not in df_out.columns:
            log.warning("Column 'elapsed_seconds' not found. Skipping time-based feature engineering.")
            return df_out

        t = df_out["elapsed_seconds"].astype(float)

        # 1. Fraction of Orbit
        T = self.orbital_period if self.orbital_period is not None else 86400.0
        df_out["fraction_of_orbit"] = (t % T) / T

        # 2. Sine and Cosine of Time (periodic representation)
        df_out["sine_time"] = np.sin(2.0 * np.pi * t / T)
        df_out["cosine_time"] = np.cos(2.0 * np.pi * t / T)

        # 3. Day Fraction (fraction of 24h day)
        df_out["day_fraction"] = (t % 86400.0) / 86400.0

        # 4. Simulation Fraction
        df_out["simulation_fraction"] = (t - self.t_min) / (self.t_max - self.t_min)

        # 5. Ground Station Encoding (Deterministic One-Hot Encodings)
        if "station_id" in df_out.columns:
            for sid in self.known_stations:
                df_out[f"station_onehot_{sid}"] = (df_out["station_id"] == sid).astype(float)

        # 6. Visibility Encoding
        if "visible" in df_out.columns:
            df_out["visibility_encoding"] = df_out["visible"].astype(float)
        elif "station_visible" in df_out.columns:
            df_out["visibility_encoding"] = df_out["station_visible"].astype(float)

        # 7. Noise Indicator
        if "noise_model" in df_out.columns:
            # Consider any model other than 'none', empty, or nan as noisy
            is_noisy = df_out["noise_model"].notna() & (~df_out["noise_model"].isin(["none", "None", ""]))
            df_out["noise_indicator"] = is_noisy.astype(float)
        else:
            df_out["noise_indicator"] = 0.0

        return df_out

    def _estimate_orbital_period(self, truth_df: pd.DataFrame) -> float:
        """Estimates the orbital period from Cartesian positions using Kepler's Third Law."""
        pos_cols = ["x_gcrs_km", "y_gcrs_km", "z_gcrs_km"]
        if all(col in truth_df.columns for col in pos_cols):
            # Dynamic estimate based on mean orbit radius
            r = np.sqrt(
                truth_df[pos_cols[0]].astype(float) ** 2
                + truth_df[pos_cols[1]].astype(float) ** 2
                + truth_df[pos_cols[2]].astype(float) ** 2
            )
            a = float(r.mean())
            log.info("Computed average orbital radius: %.2f km from GCRS state norm", a)
        else:
            # Fallback to standard geostationary orbit radius (approx 42164 km)
            a = 42164.14
            log.warning(
                "Position columns %s not found in truth states. "
                "Defaulting to Geostationary semi-major axis: %.2f km",
                pos_cols,
                a,
            )

        # Standard gravitational parameter for Earth (mu = G * M_earth) in km^3 / s^2
        mu = 398600.4418
        T = float(2.0 * np.pi * np.sqrt(a**3 / mu))
        log.info(
            "Dynamically estimated orbital period: %.2f seconds (%.2f hours) via Kepler's 3rd Law",
            T,
            T / 3600.0,
        )
        return T

    # ------------------------------------------------------------------
    # Phase 7: Transition-pair physical feature engineering
    # ------------------------------------------------------------------

    def engineer_transition_features(self, df_pairs: pd.DataFrame) -> pd.DataFrame:
        """
        Compute physical orbital features for transition pairs.

        Adds (based on config flags):
          - radius              : |r| = sqrt(x²+y²+z²)  [km]
          - velocity_magnitude  : |v| = sqrt(vx²+vy²+vz²)  [km/s]
          - orbital_energy      : E = 0.5*v² - mu/r  [km²/s²]
          - angular_momentum    : |L| = |r × v|  [km²/s]

        Parameters
        ----------
        df_pairs : pd.DataFrame
            Transition pair dataset produced by TransitionGenerator.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with optional physical feature columns appended.
        """
        _MU = 398600.4418   # km³/s²
        fe_cfg = self._feat_cfg
        df = df_pairs.copy()

        x  = df["x"].to_numpy(dtype=np.float64)
        y  = df["y"].to_numpy(dtype=np.float64)
        z  = df["z"].to_numpy(dtype=np.float64)
        vx = df["vx"].to_numpy(dtype=np.float64)
        vy = df["vy"].to_numpy(dtype=np.float64)
        vz = df["vz"].to_numpy(dtype=np.float64)

        r = np.sqrt(x**2 + y**2 + z**2)
        v = np.sqrt(vx**2 + vy**2 + vz**2)

        if fe_cfg.get("radius", True):
            df["radius"] = r
            log.debug("Added 'radius' feature.")

        if fe_cfg.get("velocity_magnitude", True):
            df["velocity_magnitude"] = v
            log.debug("Added 'velocity_magnitude' feature.")

        if fe_cfg.get("orbital_energy", True):
            r_safe = np.maximum(r, 1e-12)
            df["orbital_energy"] = 0.5 * v**2 - _MU / r_safe
            log.debug("Added 'orbital_energy' feature.")

        if fe_cfg.get("angular_momentum", True):
            # L = r × v (cross product), then take magnitude
            Lx = y * vz - z * vy
            Ly = z * vx - x * vz
            Lz = x * vy - y * vx
            df["angular_momentum"] = np.sqrt(Lx**2 + Ly**2 + Lz**2)
            log.debug("Added 'angular_momentum' feature.")

        added = [c for c in ["radius", "velocity_magnitude", "orbital_energy", "angular_momentum"]
                 if c in df.columns and c not in df_pairs.columns]
        log.info("  Engineered orbital features: %s", added)
        return df
