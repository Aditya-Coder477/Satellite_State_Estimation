"""
src/physics/autograd_utils.py
============================
PyTorch automatic differentiation utilities for Phase 8 (Physics Engine).

Computes vector-Jacobian products, first/second derivatives, and gradient metrics.
"""

from __future__ import annotations

import logging
from typing import Any, List, Union

# PyTorch import handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class AutogradDisconnectedError(RuntimeError):
    """Raised when autograd graph is disconnected."""
    pass


class AutogradUtils:
    """
    Utility class for standard PyTorch automatic differentiation operations.
    """

    @staticmethod
    def compute_first_derivative(t: "torch.Tensor", y: "torch.Tensor") -> "torch.Tensor":
        """
        Computes the first derivative dy/dt of a tensor y with respect to t.

        Parameters
        ----------
        t : torch.Tensor
            Time coordinate tensor of shape (Batch, 1). Must have requires_grad=True.
        y : torch.Tensor
            Dependent variable tensor of shape (Batch, Dim).

        Returns
        -------
        torch.Tensor
            The derivative dy/dt of shape (Batch, Dim).
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available.")
        if not isinstance(t, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise TypeError("Inputs must be PyTorch Tensors.")
        if not t.requires_grad:
            raise ValueError("Input time tensor 't' must have requires_grad=True to compute derivatives.")

        # Batch-wise Jacobian/gradients calculation
        # Create unit vector for sum reduction to compute dy/dt elementwise
        grad_outputs = torch.ones_like(y)
        dy_dt = torch.autograd.grad(
            outputs=y,
            inputs=t,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        if dy_dt is None:
            raise RuntimeError("Gradient computation returned None. Check if graph connection is broken.")

        return dy_dt

    @staticmethod
    def compute_jacobian(inputs: "torch.Tensor", outputs: "torch.Tensor") -> "torch.Tensor":
        """
        Computes the Jacobian matrix of outputs with respect to inputs.
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available.")
        
        batch_size = inputs.shape[0]
        out_dim = outputs.shape[1]
        in_dim = inputs.shape[1]
        
        jacobian = torch.zeros(batch_size, out_dim, in_dim, dtype=outputs.dtype, device=outputs.device)
        
        for i in range(out_dim):
            grad_outputs = torch.zeros_like(outputs)
            grad_outputs[:, i] = 1.0
            grads = torch.autograd.grad(
                outputs=outputs,
                inputs=inputs,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            jacobian[:, i, :] = grads
            
        return jacobian

    @staticmethod
    def compute_gradient_norm(loss: "torch.Tensor", parameters: List["torch.Tensor"]) -> float:
        """
        Calculates the L2 norm of the gradients of the loss with respect to parameters.
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available.")
        
        grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        total_norm = 0.0
        for grad in grads:
            if grad is not None:
                total_norm += float(torch.sum(grad ** 2).item())
        return float(np.sqrt(total_norm)) if total_norm > 0.0 else 0.0
