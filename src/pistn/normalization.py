"""
src/pistn/normalization.py
==========================
Handles physical de-normalization of tensors during Phase 9 PI-STN loss evaluation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

# PyTorch import handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class DeNormalizer:
    """
    Loads normalization parameters and de-normalizes network inputs and outputs
    back to physical units (km, km/s, km/s^2) for the Physics Engine.
    """

    def __init__(self, norm_json_path: Path, target_norm_json_path: Path) -> None:
        self.norm_json_path = Path(norm_json_path)
        self.target_norm_json_path = Path(target_norm_json_path)
        
        self.input_stats: Dict[str, Dict[str, float]] = {}
        self.target_stats: Dict[str, Dict[str, float]] = {}

        self._load_input_stats()
        self._load_target_stats()

    def _load_input_stats(self) -> None:
        try:
            with open(self.norm_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.input_stats = data.get("stats", {})
            log.info("Successfully loaded input normalization stats from %s", self.norm_json_path)
        except Exception as e:
            log.error("Failed to load input normalization JSON: %s", str(e))

    def _load_target_stats(self) -> None:
        try:
            with open(self.target_norm_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.target_stats = data.get("stats", {})
            log.info("Successfully loaded target normalization stats from %s", self.target_norm_json_path)
        except Exception as e:
            log.error("Failed to load target normalization JSON: %s", str(e))

    def denormalize_state(self, inputs_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        De-normalizes inputs_tensor (Batch, 8+) back to physical [x,y,z,vx,vy,vz] (km, km/s) and dt (s).
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for de-normalization.")

        device = inputs_tensor.device
        dtype = inputs_tensor.dtype

        # Inputs layout: [elapsed_seconds, delta_time, x, y, z, vx, vy, vz, ...]
        # 1. De-normalize delta_time (dt)
        dt_stats = self.input_stats.get("delta_time", {"mean": 60.0, "std": 1.0})
        dt_mean = dt_stats["mean"]
        dt_std = dt_stats["std"]
        dt_raw = inputs_tensor[:, 1] * dt_std + dt_mean

        # 2. De-normalize x, y, z, vx, vy, vz
        pos_vel_cols = ["x", "y", "z", "vx", "vy", "vz"]
        means = []
        stds = []
        for col in pos_vel_cols:
            col_stats = self.input_stats.get(col, {"mean": 0.0, "std": 1.0})
            means.append(col_stats["mean"])
            stds.append(col_stats["std"])

        means_t = torch.tensor(means, dtype=dtype, device=device)
        stds_t = torch.tensor(stds, dtype=dtype, device=device)

        state_norm = inputs_tensor[:, 2:8]
        state_raw = state_norm * stds_t + means_t

        return state_raw, dt_raw

    def denormalize_output(self, outputs_tensor: torch.Tensor) -> torch.Tensor:
        """
        De-normalizes outputs_tensor (Batch, 9) containing predicted ΔS and residual acceleration.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for de-normalization.")

        device = outputs_tensor.device
        dtype = outputs_tensor.dtype

        target_cols = [
            "delta_x", "delta_y", "delta_z",
            "delta_vx", "delta_vy", "delta_vz",
            "ax_res", "ay_res", "az_res"
        ]

        means = []
        stds = []
        for col in target_cols:
            col_stats = self.target_stats.get(col, {"mean": 0.0, "std": 1.0})
            means.append(col_stats["mean"])
            stds.append(col_stats["std"])

        means_t = torch.tensor(means, dtype=dtype, device=device)
        stds_t = torch.tensor(stds, dtype=dtype, device=device)

        return outputs_tensor * stds_t + means_t

    def denormalize_target(self, targets_tensor: torch.Tensor) -> torch.Tensor:
        """
        De-normalizes targets_tensor (Batch, 9) back to physical targets.
        """
        return self.denormalize_output(targets_tensor)

    def normalize_state(
        self,
        state_raw: torch.Tensor,
        elapsed_raw: torch.Tensor,
        dt_raw: torch.Tensor,
    ) -> torch.Tensor:
        """
        Normalizes a raw physical state back into the 8-feature input format.

        This is the inverse of ``denormalize_state`` and is used during multi-step
        rollout training to re-normalize the updated state at each rollout step
        before feeding it into the next network forward pass.

        Parameters
        ----------
        state_raw : Tensor of shape (N, 6)
            Physical state vector [x, y, z, vx, vy, vz] in km and km/s.
        elapsed_raw : Tensor of shape (N,)
            Raw elapsed seconds since epoch.
        dt_raw : Tensor of shape (N,)
            Raw delta-time (fixed timestep) in seconds.

        Returns
        -------
        Tensor of shape (N, 8)
            Normalized input vector:
            [elapsed_seconds_norm, delta_time_norm, x_norm, y_norm, z_norm,
             vx_norm, vy_norm, vz_norm]
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for normalization.")

        device = state_raw.device
        dtype = state_raw.dtype

        # Normalize elapsed_seconds
        el_stats = self.input_stats.get("elapsed_seconds", {"mean": 0.0, "std": 1.0})
        el_std = max(float(el_stats.get("std", 1.0)), 1e-8)
        elapsed_norm = (elapsed_raw - float(el_stats["mean"])) / el_std

        # Normalize delta_time
        dt_stats = self.input_stats.get("delta_time", {"mean": 60.0, "std": 1.0})
        dt_std = max(float(dt_stats.get("std", 1.0)), 1e-8)
        dt_norm = (dt_raw - float(dt_stats["mean"])) / dt_std

        # Normalize position/velocity: [x, y, z, vx, vy, vz]
        pos_vel_cols = ["x", "y", "z", "vx", "vy", "vz"]
        means = []
        stds = []
        for col in pos_vel_cols:
            col_stats = self.input_stats.get(col, {"mean": 0.0, "std": 1.0})
            means.append(float(col_stats.get("mean", 0.0)))
            stds.append(max(float(col_stats.get("std", 1.0)), 1e-8))

        means_t = torch.tensor(means, dtype=dtype, device=device)
        stds_t = torch.tensor(stds, dtype=dtype, device=device)

        state_norm = (state_raw - means_t) / stds_t

        # Assemble (N, 8) normalized input tensor
        return torch.cat([
            elapsed_norm.unsqueeze(1),
            dt_norm.unsqueeze(1),
            state_norm,
        ], dim=1)

