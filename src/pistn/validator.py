"""
src/pistn/validator.py
======================
Validation checks for Phase 9 PI-STN inputs, outputs, and parameters.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

# PyTorch import handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class ModelValidator:
    """
    Validates model tensors, shapes, configurations, and outputs.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.pistn_cfg = cfg.get("pistn", {})
        self.net_cfg = self.pistn_cfg.get("network", {})
        
        self.expected_in = int(self.net_cfg.get("input_dim", 8))
        self.expected_out = int(self.net_cfg.get("output_dim", 9))

    def validate_inputs_outputs(self, batch_in: Any, batch_out: Any) -> Tuple[bool, List[str]]:
        """
        Validates the input and output batch tensors.
        """
        errors = []

        # 1. Type validation
        if TORCH_AVAILABLE and isinstance(batch_in, torch.Tensor):
            if not torch.isfinite(batch_in).all():
                errors.append("Input batch contains NaN/Inf values.")
            if not torch.isfinite(batch_out).all():
                errors.append("Output predictions contain NaN/Inf values.")
            
            in_shape = batch_in.shape
            out_shape = batch_out.shape
        else:
            import numpy as np
            if not np.isfinite(batch_in).all():
                errors.append("Input batch contains NaN/Inf values.")
            if not np.isfinite(batch_out).all():
                errors.append("Output predictions contain NaN/Inf values.")
            
            in_shape = batch_in.shape
            out_shape = batch_out.shape

        # 2. Shape validation
        if len(in_shape) != 2 or in_shape[1] != self.expected_in:
            errors.append(f"Invalid input shape {in_shape}. Expected (Batch, {self.expected_in}).")
            
        if len(out_shape) != 2 or out_shape[1] != self.expected_out:
            errors.append(f"Invalid output shape {out_shape}. Expected (Batch, {self.expected_out}).")

        is_valid = len(errors) == 0
        return is_valid, errors
