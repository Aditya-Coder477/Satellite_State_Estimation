"""
src/evaluation/future_prediction.py
=====================================
Future-horizon trajectory prediction interface for Phase 9.

This module loads a trained PINN model (via PINNEngine) and generates
predicted satellite states across a configurable time window that lies
*entirely outside* the training period — the 2-day future prediction window.

The class is model-agnostic: any future model (HNN, Neural ODE, etc.) can
be plugged in as long as it exposes a ``predict(times: np.ndarray) -> np.ndarray``
interface that returns a (N, 6) state array in GCRS Cartesian coordinates.

Design contract
---------------
* Prediction window: Day 5 → Day 7 (as configured in ``evaluation.prediction_window``).
* The module NEVER loads CSVs directly — it queries the DatasetManager API.
* Time stamps are resolved from the ground truth trajectory (elapsed_seconds column).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Protocol, Tuple

import numpy as np
import pandas as pd

from src.dataset.dataset_manager import DatasetManager
from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class PredictorProtocol(Protocol):
    """
    Duck-typed interface any predictor must satisfy to work with
    :class:`FuturePredictionEngine`.
    """

    def predict(self, times: np.ndarray) -> np.ndarray:
        """
        Given elapsed times (seconds from epoch) of shape (N,),
        return predicted states of shape (N, 6): [x, y, z, vx, vy, vz].
        """
        ...


class FuturePredictionEngine:
    """
    Generates future satellite state predictions using a trained model.

    Parameters
    ----------
    cfg : Dict[str, Any]
        Master project configuration.
    dataset_manager : DatasetManager
        Provides path resolution and CSV loading without direct file access.
    predictor : PredictorProtocol
        Any model exposing a ``predict(times: np.ndarray) -> np.ndarray`` method.
    """

    # GCRS state column names in the truth file
    _POS_COLS = ["x_gcrs_km", "y_gcrs_km", "z_gcrs_km"]
    _VEL_COLS = ["vx_gcrs_kms", "vy_gcrs_kms", "vz_gcrs_kms"]
    _STATE_COLS = _POS_COLS + _VEL_COLS

    def __init__(
        self,
        cfg: Dict[str, Any],
        dataset_manager: DatasetManager,
        predictor: PredictorProtocol,
    ) -> None:
        self._cfg = cfg
        self._dm = dataset_manager
        self._predictor = predictor
        self._eval_cfg = cfg.get("evaluation", {})
        log.info("FuturePredictionEngine initialised.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Execute the 2-day future prediction.

        Returns
        -------
        times : np.ndarray, shape (N,)
            Elapsed seconds from epoch for every prediction step.
        predicted_states : np.ndarray, shape (N, 6)
            PINN-predicted [x, y, z, vx, vy, vz] in GCRS (km, km/s).
        truth_states : np.ndarray, shape (N, 6)
            Ground-truth [x, y, z, vx, vy, vz] in GCRS (km, km/s).
        """
        log.info("Loading ground-truth trajectory from DatasetManager...")
        truth_df = self._load_truth_states()

        log.info("Slicing prediction window (Day %s → Day %s)...",
                 self._eval_cfg.get("prediction_window", {}).get("start_day", 5),
                 self._eval_cfg.get("prediction_window", {}).get("end_day", 7))
        window_df = self._slice_prediction_window(truth_df)

        times = window_df["elapsed_seconds"].values.astype(np.float64)
        truth_states = window_df[self._STATE_COLS].values.astype(np.float64)

        log.info("Running model inference over %d time steps...", len(times))
        predicted_states = self._predictor.predict(times.astype(np.float32))
        predicted_states = predicted_states.astype(np.float64)

        log.info(
            "Prediction complete. Steps: %d | Position range (truth): %.1f – %.1f km",
            len(times),
            np.linalg.norm(truth_states[:, :3], axis=1).min(),
            np.linalg.norm(truth_states[:, :3], axis=1).max(),
        )
        return times, predicted_states, truth_states

    def build_comparison_dataframe(
        self,
        times: np.ndarray,
        predicted_states: np.ndarray,
        truth_states: np.ndarray,
    ) -> pd.DataFrame:
        """
        Builds a side-by-side comparison DataFrame for CSV export and analysis.

        Parameters
        ----------
        times : np.ndarray, shape (N,)
        predicted_states : np.ndarray, shape (N, 6)
        truth_states : np.ndarray, shape (N, 6)

        Returns
        -------
        pd.DataFrame
            Columns: elapsed_seconds, pred_x, pred_y, pred_z,
            pred_vx, pred_vy, pred_vz, truth_x, truth_y, truth_z,
            truth_vx, truth_vy, truth_vz,
            err_x, err_y, err_z, err_vx, err_vy, err_vz,
            position_error_km, velocity_error_kms.
        """
        pos_error = predicted_states[:, :3] - truth_states[:, :3]
        vel_error = predicted_states[:, 3:] - truth_states[:, 3:]

        df = pd.DataFrame({
            "elapsed_seconds": times,
            # Predictions
            "pred_x_km":   predicted_states[:, 0],
            "pred_y_km":   predicted_states[:, 1],
            "pred_z_km":   predicted_states[:, 2],
            "pred_vx_kms": predicted_states[:, 3],
            "pred_vy_kms": predicted_states[:, 4],
            "pred_vz_kms": predicted_states[:, 5],
            # Ground Truth
            "truth_x_km":   truth_states[:, 0],
            "truth_y_km":   truth_states[:, 1],
            "truth_z_km":   truth_states[:, 2],
            "truth_vx_kms": truth_states[:, 3],
            "truth_vy_kms": truth_states[:, 4],
            "truth_vz_kms": truth_states[:, 5],
            # Per-axis errors
            "err_x_km":   pos_error[:, 0],
            "err_y_km":   pos_error[:, 1],
            "err_z_km":   pos_error[:, 2],
            "err_vx_kms": vel_error[:, 0],
            "err_vy_kms": vel_error[:, 1],
            "err_vz_kms": vel_error[:, 2],
            # Norms
            "position_error_km":    np.linalg.norm(pos_error, axis=1),
            "velocity_error_kms":   np.linalg.norm(vel_error, axis=1),
        })
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_truth_states(self) -> pd.DataFrame:
        """Loads the ground-truth trajectory via the DatasetManager API."""
        truth_filename = self._eval_cfg.get("truth_states_file", "truth_states.csv")

        # Route to the correct DatasetManager method based on location:
        #   • Files in processed/<SAT>/dataset/  → get_dataset_path(filename)
        #   • Files in processed/<SAT>/           → get_input_path(filename)
        if truth_filename.startswith("dataset/") or truth_filename.startswith("dataset\\"):
            filename_only = truth_filename.split("/")[-1].split("\\")[-1]
            path = self._dm.get_dataset_path(filename_only)
        else:
            path = self._dm.get_input_path(truth_filename)

        log.info("Loading ground-truth states from: %s", path)
        df = self._dm.load_csv(path)

        # Validate expected columns
        required_cols = {"elapsed_seconds"} | set(self._STATE_COLS)
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"'{truth_filename}' is missing required columns: {missing}. "
                f"Available columns: {list(df.columns)}"
            )
        return df

    def _slice_prediction_window(self, truth_df: pd.DataFrame) -> pd.DataFrame:
        """Slices the prediction window from the full ground-truth DataFrame."""
        pred_window = self._eval_cfg.get("prediction_window", {})
        start_day = pred_window.get("start_day", 5)
        end_day = pred_window.get("end_day", 7)

        start_sec = start_day * 86400.0
        end_sec = end_day * 86400.0

        mask = (truth_df["elapsed_seconds"] >= start_sec) & (
            truth_df["elapsed_seconds"] <= end_sec
        )
        window_df = truth_df[mask].copy().reset_index(drop=True)

        if len(window_df) == 0:
            raise ValueError(
                f"Prediction window Day {start_day}–{end_day} yielded zero rows. "
                f"Check that truth_states.csv covers days 0–7 "
                f"(elapsed_seconds range: {truth_df['elapsed_seconds'].min():.0f} "
                f"to {truth_df['elapsed_seconds'].max():.0f})."
            )
        log.info(
            "Prediction window: %.1f – %.1f seconds (%d rows).",
            start_sec, end_sec, len(window_df),
        )
        return window_df
