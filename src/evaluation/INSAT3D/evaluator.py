"""
src/evaluation/INSAT3D/evaluator.py
===================================
Evaluator module coordinating model inference and de-normalization for Phase 10.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

# PyTorch import handling
try:
    import torch
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.dataset.pytorch_dataset import ObservationDataset
from src.pistn.normalization import DeNormalizer
from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class PerformanceEvaluator:
    """
    Coordinates model inference, de-normalization, and target predictions collection.
    """

    def __init__(
        self,
        model: Any,
        denormalizer: DeNormalizer,
        input_cols: list[str],
        target_cols: list[str],
    ) -> None:
        self.model = model
        self.denormalizer = denormalizer
        self.input_cols = input_cols
        self.target_cols = target_cols

    def run_inference(
        self,
        df_test: pd.DataFrame,
        batch_size: int = 1024,
        device: str = "cpu",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Executes frozen-weight model inference and de-normalizes predictions.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            1. true_deltas_raw (N, 6)
            2. pred_deltas_raw (N, 6)
            3. true_res_raw (N, 3)
            4. pred_res_raw (N, 3)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for model inference.")

        self.model.to(device)
        self.model.eval()

        dataset = ObservationDataset(df_test, self.input_cols, self.target_cols)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        all_true_deltas = []
        all_pred_deltas = []
        all_true_res = []
        all_pred_res = []

        log.info("Starting model inference on test dataset (%d samples)...", len(df_test))

        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch
                else:
                    inputs = batch["inputs"]
                    targets = batch["targets"]

                inputs = inputs.to(device, dtype=torch.float32)
                targets = targets.to(device, dtype=torch.float32)

                # Forward pass (slice to input_dim)
                outputs = self.model(inputs[:, :self.model.input_dim])

                # De-normalize outputs and targets
                outputs_raw = self.denormalizer.denormalize_output(outputs)
                targets_raw = self.denormalizer.denormalize_target(targets)

                # Extract components
                pred_delta_raw = outputs_raw[:, :6].cpu().numpy()
                true_delta_raw = targets_raw[:, :6].cpu().numpy()
                
                pred_res_raw = outputs_raw[:, 6:9].cpu().numpy()
                true_res_raw = targets_raw[:, 6:9].cpu().numpy()

                all_pred_deltas.append(pred_delta_raw)
                all_true_deltas.append(true_delta_raw)
                all_pred_res.append(pred_res_raw)
                all_true_res.append(true_res_raw)

        true_deltas_raw = np.concatenate(all_true_deltas, axis=0)
        pred_deltas_raw = np.concatenate(all_pred_deltas, axis=0)
        true_res_raw = np.concatenate(all_true_res, axis=0)
        pred_res_raw = np.concatenate(all_pred_res, axis=0)

        log.info("Model inference and de-normalization complete.")
        return true_deltas_raw, pred_deltas_raw, true_res_raw, pred_res_raw
