"""
src/dataset/dataset_loader.py
============================
Handles loading and schema validation of satellite observation and state datasets.

Validates that raw and simulated CSV files conform to expected column schemas,
data types, and contain no unexpected missing or null values.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class DatasetSchemaError(ValueError):
    """Raised when a dataset fails to conform to the expected schema or types."""
    pass


class DatasetLoader:
    """
    Loads and validates dataframes from simulated CSV datasets.

    Provides robust verification of expected columns, data types, and null values.
    """

    # Expected columns for truth_states_transformed.csv
    TRUTH_SCHEMA: Dict[str, str] = {
        "utc_time": "object",  # treated as string/object initially
        "elapsed_seconds": "float64",
        "satellite_name": "object",
        "norad_id": "int64",
        "x_gcrs_km": "float64",
        "y_gcrs_km": "float64",
        "z_gcrs_km": "float64",
        "vx_gcrs_kms": "float64",
        "vy_gcrs_kms": "float64",
        "vz_gcrs_kms": "float64",
        "x_itrs_km": "float64",
        "y_itrs_km": "float64",
        "z_itrs_km": "float64",
    }

    # Expected columns for observations.csv
    OBSERVATION_SCHEMA: Dict[str, str] = {
        "observation_id": "object",
        "simulation_id": "object",
        "utc_time": "object",
        "elapsed_seconds": "float64",
        "satellite_name": "object",
        "norad_id": "int64",
        "station_id": "int64",
        "station_name": "object",
        "elevation_deg": "float64",
        "azimuth_deg": "float64",
        "range_km": "float64",
        "range_rate_kms": "float64",
        "right_ascension_deg": "float64",
        "declination_deg": "float64",
        "visible": "bool",
        "station_visible": "bool",
        "observation_quality": "object",
    }

    # Extra columns for noisy_observations.csv
    NOISY_OBSERVATION_SCHEMA: Dict[str, str] = {
        **OBSERVATION_SCHEMA,
        "ra_noisy_deg": "float64",
        "dec_noisy_deg": "float64",
    }

    @classmethod
    def load_and_validate(cls, path: Path, file_type: str) -> pd.DataFrame:
        """
        Loads a CSV file and validates it against the expected schema.

        Parameters
        ----------
        path : Path
            Path to the CSV file to load.
        file_type : str
            Type of dataset to load. Must be one of:
            "truth_states", "observations", "noisy_observations".

        Returns
        -------
        pd.DataFrame
            The loaded and validated DataFrame.

        Raises
        ------
        DatasetSchemaError
            If schema structure, types, or null values fail validation.
        """
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found at: {path}")

        log.info("Loading and validating %s from %s", file_type, path)
        df = pd.read_csv(path)

        if file_type == "truth_states":
            schema = cls.TRUTH_SCHEMA
        elif file_type == "observations":
            schema = cls.OBSERVATION_SCHEMA
        elif file_type == "noisy_observations":
            schema = cls.NOISY_OBSERVATION_SCHEMA
        else:
            raise ValueError(f"Invalid file_type: {file_type}")

        cls._validate_schema(df, schema, file_type)
        cls._validate_nulls(df, file_type)

        log.info(
            "Successfully validated %s (%d rows, %d columns)",
            file_type,
            len(df),
            len(df.columns),
        )
        return df

    @staticmethod
    def _validate_schema(df: pd.DataFrame, expected_schema: Dict[str, str], file_type: str) -> None:
        """Helper to validate column existence and basic type checking."""
        # Check missing columns
        missing_cols = [col for col in expected_schema if col not in df.columns]
        if missing_cols:
            raise DatasetSchemaError(
                f"Validation failed for {file_type}: Missing expected columns: {missing_cols}"
            )

        # Check types (coerced if possible, or warning)
        for col, expected_type in expected_schema.items():
            col_dtype = df[col].dtype
            
            # Allow conversions between int and float if compatible
            if expected_type == "float64" and np.issubdtype(col_dtype, np.integer):
                df[col] = df[col].astype(float)
                col_dtype = df[col].dtype

            if expected_type == "bool" and col_dtype != bool:
                # Try to cast boolean from string or int
                try:
                    df[col] = df[col].astype(bool)
                except Exception:
                    raise DatasetSchemaError(
                        f"Validation failed for {file_type}: Column '{col}' is expected to be "
                        f"boolean, but has type '{col_dtype}' and could not be coerced."
                    )
                col_dtype = df[col].dtype

            # If types are incompatible, log warning or raise error
            if expected_type == "float64" and not np.issubdtype(col_dtype, np.floating):
                raise DatasetSchemaError(
                    f"Validation failed for {file_type}: Column '{col}' expected float64, got '{col_dtype}'"
                )
            elif expected_type == "int64" and not np.issubdtype(col_dtype, np.integer):
                raise DatasetSchemaError(
                    f"Validation failed for {file_type}: Column '{col}' expected int64, got '{col_dtype}'"
                )

    @staticmethod
    def _validate_nulls(df: pd.DataFrame, file_type: str) -> None:
        """Helper to ensure there are no NaN or Inf values in critical numeric columns."""
        # Find numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Check NaNs
        nan_counts = df[numeric_cols].isna().sum()
        cols_with_nans = nan_counts[nan_counts > 0].index.tolist()
        if cols_with_nans:
            nan_details = {col: int(nan_counts[col]) for col in cols_with_nans}
            raise DatasetSchemaError(
                f"Validation failed for {file_type}: NaN values detected in columns: {nan_details}"
            )

        # Check Infs
        inf_counts = np.isinf(df[numeric_cols]).sum()
        cols_with_infs = inf_counts[inf_counts > 0].index.tolist()
        if cols_with_infs:
            inf_details = {col: int(inf_counts[col]) for col in cols_with_infs}
            raise DatasetSchemaError(
                f"Validation failed for {file_type}: Inf values detected in columns: {inf_details}"
            )
