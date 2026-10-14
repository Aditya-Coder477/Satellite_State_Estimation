"""
src/pinn/predictor.py
=====================
Prediction interface for Phase 8 (PINN Framework).

Loads trained PINN models and generates orbit trajectory states. Supports
single-timestamp queries, batch predictions, and future state extrapolation.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

# PyTorch import handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.pinn.forward_model import ForwardModel
from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class PINNPredictor:
    """
    Handles inference and trajectory extrapolation using trained networks.

    Parameters
    ----------
    forward_model : ForwardModel
        The active wrapped network model.
    """

    def __init__(self, forward_model: ForwardModel) -> None:
        self.forward_model = forward_model
        log.info("PINNPredictor instantiated.")

    def predict_single(self, elapsed_seconds: float) -> np.ndarray:
        """
        Predicts the satellite state at a single elapsed second.

        Parameters
        ----------
        elapsed_seconds : float
            Elapsed seconds from epoch.

        Returns
        -------
        np.ndarray
            Predicted state vector [x, y, z, vx, vy, vz] of shape (6,).
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for prediction.")

        self.forward_model.network.eval()
        
        t_val = np.array([[elapsed_seconds]], dtype=np.float32)
        t = torch.tensor(t_val, dtype=torch.float32, device=self.forward_model.device)
        
        with torch.no_grad():
            state_pred = self.forward_model.predict_state(t, requires_grad=False)
            
        return state_pred.cpu().squeeze(0).numpy()

    def predict_trajectory(self, times: np.ndarray) -> np.ndarray:
        """
        Predicts state vectors across a series of time stamps.

        Parameters
        ----------
        times : np.ndarray
            Flat array of elapsed seconds of shape (N,).

        Returns
        -------
        np.ndarray
            Predicted state trajectory array of shape (N, 6).
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for trajectory prediction.")

        self.forward_model.network.eval()
        
        t_val = times.astype(np.float32).reshape(-1, 1)
        
        # Batch predictions to avoid OOM for extremely long trajectories
        batch_size = 10000
        n_samples = len(times)
        
        state_preds = []
        
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                t_batch_np = t_val[i : i + batch_size]
                t_batch = torch.tensor(t_batch_np, dtype=torch.float32, device=self.forward_model.device)
                
                state_pred = self.forward_model.predict_state(t_batch, requires_grad=False)
                state_preds.append(state_pred.cpu().numpy())

        return np.concatenate(state_preds, axis=0)

    def predict_batch(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates predictions in PyTorch, preserving gradients if required.

        Parameters
        ----------
        t : torch.Tensor
            Input time tensor of shape (Batch, 1).

        Returns
        -------
        torch.Tensor
            Predicted states of shape (Batch, 6).
        """
        self.forward_model.network.eval()
        return self.forward_model.predict_state(t, requires_grad=t.requires_grad)
