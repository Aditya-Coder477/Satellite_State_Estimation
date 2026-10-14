"""
src/dataset/dataset_manager.py
==============================
Manages filesystem paths and I/O operations for Phase 6 (Dataset Pipeline).

This class serves as the central data interface for the dataset pipeline and future
PINN training phases, ensuring that all data inputs and outputs are resolved
consistently based on the configuration and satellite name.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class DatasetManager:
    """
    Manages paths, directory creation, and I/O for Phase 6 dataset pipeline.

    Parameters
    ----------
    cfg : dict[str, Any]
        Project configuration loaded by ConfigLoader.
    satellite_key : str
        Satellite identifier (e.g. "GOES16").

    Attributes
    ----------
    root : Path
        Absolute project root directory.
    satellite_key : str
        Canonical satellite identifier.
    processed_dir : Path
        Directory where simulated truth/observations are stored.
    dataset_dir : Path
        Directory where split datasets, collocation points, and configs are saved.
    """

    def __init__(self, cfg: Dict[str, Any], satellite_key: str) -> None:
        self._cfg = cfg
        self.satellite_key = satellite_key
        self.root = Path(cfg["_root"]).resolve()

        # Resolve paths using config.yaml structure
        paths_cfg = cfg["paths"]
        processed_base = self.root / paths_cfg["data"]["processed"]["base"]
        
        # Per-satellite directory mapping
        sat_cfg = cfg.get("satellites", {}).get(satellite_key)
        if not sat_cfg:
            raise KeyError(
                f"Satellite '{satellite_key}' not found in configuration satellites registry."
            )
            
        self.processed_dir = processed_base / sat_cfg["processed_dir"]
        self.dataset_dir = self.processed_dir / "dataset"

        # Ensure output directories exist
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        log.info(
            "DatasetManager initialised for %s. Dataset directory: %s",
            satellite_key,
            self.dataset_dir,
        )

    def verify_input_files(self) -> None:
        """
        Verifies that the expected input files from previous phases exist.

        Raises
        ------
        FileNotFoundError
            If any expected input file is missing.
        """
        required_files = [
            self.processed_dir / "truth_states.csv",
            self.processed_dir / "truth_states_transformed.csv",
            self.processed_dir / "observations.csv",
            self.processed_dir / "noisy_observations.csv",
        ]

        for path in required_files:
            if not path.is_file():
                log.error("Required input file missing: %s", path)
                raise FileNotFoundError(
                    f"Required input file '{path.name}' not found in {self.processed_dir}. "
                    "Ensure previous phases (SGP4, coordinate transform, observation generation) "
                    "have been run successfully."
                )
        log.info("All required simulated inputs verified successfully in %s", self.processed_dir)

    def get_input_path(self, filename: str) -> Path:
        """
        Resolves the absolute path for an input file in the processed directory.

        Parameters
        ----------
        filename : str
            Name of the file (e.g. "noisy_observations.csv").

        Returns
        -------
        Path
            Absolute path to the file.
        """
        return self.processed_dir / filename

    def get_dataset_path(self, filename: str) -> Path:
        """
        Resolves the absolute path for a file in the dataset subdirectory.

        Parameters
        ----------
        filename : str
            Name of the file (e.g. "train.csv").

        Returns
        -------
        Path
            Absolute path to the file.
        """
        return self.dataset_dir / filename

    def load_csv(self, path: Path) -> pd.DataFrame:
        """
        Loads a CSV file into a pandas DataFrame.

        Parameters
        ----------
        path : Path
            Path to the CSV file.

        Returns
        -------
        pd.DataFrame
            Loaded DataFrame.
        """
        log.info("Loading CSV dataset from: %s", path)
        try:
            return pd.read_csv(path)
        except Exception as e:
            log.error("Failed to load CSV from %s: %s", path, str(e))
            raise

    def save_csv(self, df: pd.DataFrame, path: Path) -> None:
        """
        Saves a DataFrame to a CSV file.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to save.
        path : Path
            Target destination path.
        """
        log.info("Saving CSV dataset (%d rows) to: %s", len(df), path)
        try:
            df.to_csv(path, index=False)
        except Exception as e:
            log.error("Failed to save CSV to %s: %s", path, str(e))
            raise

    def save_json(self, data: Dict[str, Any] | list[Any], path: Path) -> None:
        """
        Saves metadata dictionary as a formatted JSON file.

        Parameters
        ----------
        data : Dict[str, Any] | list[Any]
            The metadata to save.
        path : Path
            Target destination path.
        """
        log.info("Saving JSON metadata to: %s", path)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=4)
        except Exception as e:
            log.error("Failed to save JSON to %s: %s", path, str(e))
            raise

    def load_json(self, path: Path) -> Dict[str, Any] | list[Any]:
        """
        Loads a JSON file.

        Parameters
        ----------
        path : Path
            Path to the JSON file.

        Returns
        -------
        Dict[str, Any] | list[Any]
            Loaded JSON data.
        """
        log.info("Loading JSON metadata from: %s", path)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as e:
            log.error("Failed to load JSON from %s: %s", path, str(e))
            raise
