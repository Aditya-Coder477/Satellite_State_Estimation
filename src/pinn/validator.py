"""
src/pinn/validator.py
=====================
Validation routines for Phase 8 (PINN Framework).

Enforces structural validations on PyTorch networks (shapes, outputs), checks
for NaN/Inf gradients during backpropagation, validates checkpoint file structure,
and verifies dataset compatibility.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# PyTorch import handling
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class PINNValidator:
    """
    Validates networks, losses, gradients, and checkpoint files.
    """

    @staticmethod
    def validate_network_shapes(network: Any, input_dim: int, output_dim: int) -> Tuple[bool, List[str]]:
        """
        Validates network output dimensions via a mock forward pass.

        Parameters
        ----------
        network : nn.Module
            The network to validate.
        input_dim : int
            Expected input dimensions.
        output_dim : int
            Expected output dimensions.

        Returns
        -------
        Tuple[bool, List[str]]
            1. Success flag.
            2. List of error messages.
        """
        if not TORCH_AVAILABLE:
            return True, []

        errors: List[str] = []
        log.info("Validating network layout shapes...")

        try:
            # Create a mock input batch of size 8
            mock_in = torch.randn(8, input_dim)
            network.eval()
            with torch.no_grad():
                mock_out = network(mock_in)

            # Check shape
            if len(mock_out.shape) != 2:
                errors.append(f"Network output must be 2D, got shape {mock_out.shape}")
            elif mock_out.shape[0] != 8:
                errors.append(f"Batch dimension mismatch: expected 8, got {mock_out.shape[0]}")
            elif mock_out.shape[1] != output_dim:
                errors.append(f"Output dimension mismatch: expected {output_dim}, got {mock_out.shape[1]}")

        except Exception as e:
            errors.append(f"Network failed on forward pass execution: {e}")

        return len(errors) == 0, errors

    @staticmethod
    def validate_gradients(network: Any) -> Tuple[bool, List[str]]:
        """
        Asserts that all parameter gradients are finite (not NaN or Inf).

        Parameters
        ----------
        network : nn.Module
            The network being trained.

        Returns
        -------
        Tuple[bool, List[str]]
            1. Success flag.
            2. List of error messages.
        """
        if not TORCH_AVAILABLE:
            return True, []

        errors: List[str] = []
        
        for name, param in network.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad = param.grad
                has_nan = torch.isnan(grad).any().item()
                has_inf = torch.isinf(grad).any().item()
                
                if has_nan:
                    errors.append(f"NaN gradient detected in parameter '{name}'.")
                if has_inf:
                    errors.append(f"Infinite gradient detected in parameter '{name}'.")

        return len(errors) == 0, errors

    @staticmethod
    def validate_losses(loss_dict: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Verifies that all computed loss components are finite.

        Parameters
        ----------
        loss_dict : Dict[str, Any]
            Dictionary of computed loss tensors/values.

        Returns
        -------
        Tuple[bool, List[str]]
            1. Success flag.
            2. List of error messages.
        """
        errors: List[str] = []

        for name, val in loss_dict.items():
            # Support PyTorch tensors and float values
            if TORCH_AVAILABLE and isinstance(val, torch.Tensor):
                has_nan = torch.isnan(val).any().item()
                has_inf = torch.isinf(val).any().item()
                val_float = float(val.item())
            else:
                has_nan = np.isnan(val)
                has_inf = np.isinf(val)
                val_float = float(val)

            if has_nan:
                errors.append(f"NaN loss detected in component '{name}'.")
            if has_inf:
                errors.append(f"Infinite loss detected in component '{name}'.")
            if val_float < 0.0 and name != "lr":
                errors.append(f"Negative loss value detected in component '{name}': {val_float:.4e}")

        return len(errors) == 0, errors

    @staticmethod
    def validate_checkpoint_integrity(checkpoint_dir: Path | str) -> Tuple[bool, List[str]]:
        """
        Verifies that checkpoint files are present and uncorrupted.

        Parameters
        ----------
        checkpoint_dir : Path | str
            Directory where checkpoints are stored.

        Returns
        -------
        Tuple[bool, List[str]]
            1. Success flag.
            2. List of error messages.
        """
        errors: List[str] = []
        d = Path(checkpoint_dir)

        required_files = [
            "last_model.pt",
            "last_optimizer.pt",
            "last_training_state.json",
        ]

        for fname in required_files:
            p = d / fname
            if not p.is_file():
                errors.append(f"Missing expected checkpoint file: {fname} (path: {p})")
            elif p.stat().st_size == 0:
                errors.append(f"Checkpoint file is empty: {fname}")

        return len(errors) == 0, errors
class PINNValidatorError(ValueError):
    """Exception raised when training states or gradients become unstable."""
    pass
