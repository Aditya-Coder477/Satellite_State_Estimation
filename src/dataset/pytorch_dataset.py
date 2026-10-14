"""
src/dataset/pytorch_dataset.py
==============================
Defines the PyTorch Dataset implementation for satellite state/observation data.

Implements `ObservationDataset` subclassing `torch.utils.data.Dataset`.
Supports loading input and target columns as PyTorch tensors for neural networks,
with a fallback to NumPy arrays if PyTorch is not installed in the environment.
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)

# Resilient PyTorch import block
try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
    log.info("PyTorch successfully imported. ObservationDataset is fully functional.")
except ImportError:
    # Stub base class for compilation safety
    class Dataset:  # type: ignore
        pass
    TORCH_AVAILABLE = False
    log.warning("PyTorch not installed in this environment. Dataset class will run in NumPy fallback mode.")


class ObservationDataset(Dataset):
    """
    Custom dataset mapping pandas DataFrames to PyTorch tensors or NumPy arrays.

    Parameters
    ----------
    df : pd.DataFrame
        The engineered and normalized DataFrame split (train, val, or test).
    input_cols : List[str]
        List of column names to serve as model inputs (e.g. elapsed_seconds, sine_time).
    target_cols : List[str]
        List of column names to serve as model targets (e.g. ra_noisy_deg, dec_noisy_deg).
    """

    def __init__(self, df: pd.DataFrame, input_cols: List[str], target_cols: List[str]) -> None:
        self.input_cols = input_cols
        self.target_cols = target_cols
        
        # Verify columns exist
        for col in input_cols + target_cols:
            if col not in df.columns:
                raise KeyError(f"Requested column '{col}' not found in the DataFrame.")

        # Extract features and targets as float32 arrays
        self.raw_inputs = df[input_cols].values.astype(np.float32)
        self.raw_targets = df[target_cols].values.astype(np.float32)

        log.info(
            "ObservationDataset initialised: samples=%d, input_dim=%d, target_dim=%d",
            len(self.raw_inputs),
            len(input_cols),
            len(target_cols),
        )

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.raw_inputs)

    def __getitem__(self, idx: int) -> Tuple[Union[Any, np.ndarray], Union[Any, np.ndarray]]:
        """
        Retrieves the input and target tensors (or arrays) for a single index.

        Parameters
        ----------
        idx : int
            Index of the sample.

        Returns
        -------
        Tuple[Tensor/Array, Tensor/Array]
            Input and target features for training/evaluation.
        """
        inputs = self.raw_inputs[idx]
        targets = self.raw_targets[idx]

        if TORCH_AVAILABLE:
            return torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)
        else:
            return inputs, targets
