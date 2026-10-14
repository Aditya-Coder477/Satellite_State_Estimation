"""
src/observation_generation/sensor_model.py
===========================================
Sensor Model hierarchy.

Defines the BaseSensorModel class to support extensibility for future sensor
types (e.g. Radar, Laser Ranging) and the OpticalSensorModel.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from src.observation_generation.ra_dec_generator import RADecGenerator


class BaseSensorModel(ABC):
    """Abstract base class representing a tracking sensor."""

    def __init__(self, sensor_type: str) -> None:
        self.sensor_type = sensor_type

    @abstractmethod
    def generate_observations(
        self,
        df_geom: pd.DataFrame,
        df_states: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate observations from ground station geometry and satellite states.

        Parameters
        ----------
        df_geom : pd.DataFrame
            Ground station geometry dataset from Phase 5.
        df_states : pd.DataFrame
            Transformed satellite states from Phase 4.

        Returns
        -------
        pd.DataFrame
            Generated observations dataset.
        """


class OpticalSensorModel(BaseSensorModel):
    """Optical telescope sensor model. Generates Right Ascension (RA) and Declination (Dec)."""

    def __init__(self) -> None:
        super().__init__(sensor_type="optical")

    def generate_observations(
        self,
        df_geom: pd.DataFrame,
        df_states: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate topocentric RA and Dec for all visible samples.

        Parameters
        ----------
        df_geom : pd.DataFrame
        df_states : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            DataFrame with 'ra_deg' and 'dec_deg' added.
        """
        # Filter for visible observations only
        df_visible = df_geom[df_geom["visible"]].copy()
        if len(df_visible) == 0:
            return pd.DataFrame(columns=df_geom.columns.tolist() + ["ra_deg", "dec_deg"])

        # Compute RA/Dec
        ra_deg, dec_deg = RADecGenerator.compute_vectorized(df_visible, df_states)
        df_visible["ra_deg"] = ra_deg
        df_visible["dec_deg"] = dec_deg

        return df_visible
