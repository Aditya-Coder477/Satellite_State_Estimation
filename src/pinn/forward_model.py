"""
src/pinn/forward_model.py
=========================
Manages forward propagation passes for Phase 8 (PINN Framework).

Wraps the network instance to manage device placement (CPU/GPU) and ensure
correct tensor configurations (requires_grad) for automatic differentiation.
"""

from __future__ import annotations

import logging
from typing import Any

# PyTorch import handling
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class ForwardModel:
    """
    Manages evaluation passes of the PINN network.

    Parameters
    ----------
    network : nn.Module
        The active neural network.
    device : torch.device or str
        Target device for execution (e.g. "cuda", "cpu").
    """

    def __init__(self, network: Any, device: Any = "cpu") -> None:
        self.network = network
        self.device = torch.device(device) if TORCH_AVAILABLE else "cpu"
        
        # Place network on the target device
        if TORCH_AVAILABLE and isinstance(network, nn.Module):
            self.network.to(self.device)

        log.info("ForwardModel initialised on device: %s", self.device)

    def predict_state(self, t: torch.Tensor, requires_grad: bool = True) -> torch.Tensor:
        """
        Runs a forward pass to predict the satellite state at time t.

        Parameters
        ----------
        t : torch.Tensor
            Time input tensor of shape (Batch, 1).
        requires_grad : bool, optional
            Whether to enable gradients for t, by default True. Needed for autograd.

        Returns
        -------
        torch.Tensor
            Predicted state tensor [x,y,z,vx,vy,vz] of shape (Batch, 6).
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to run predict_state.")

        # Ensure t is on correct device
        t_device = t.to(self.device)
        
        # Enable gradient calculation on the input time tensor if requested
        if requires_grad and not t_device.requires_grad:
            t_device = t_device.clone().detach().requires_grad_(True)

        state_pred = self.network(t_device)
        return state_pred
