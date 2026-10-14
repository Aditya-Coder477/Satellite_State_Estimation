"""
src/evaluation/INSAT3D/trajectory_analysis.py
============================================
Reconstructs satellite trajectories and analyzes orbit drift over time for Phase 10.
"""

from __future__ import annotations

import logging
import json
from typing import Any, Dict, Tuple
import numpy as np

# PyTorch import handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class TrajectoryAnalyzer:
    """
    Performs trajectory reconstruction (1-step) and continuous rollout (orbit drift).
    """

    def __init__(self, model: Any, denormalizer: Any, normalization_json_path: str) -> None:
        self.model = model
        self.denormalizer = denormalizer
        
        # Load normalization stats for input normalization during rollout
        with open(normalization_json_path, "r", encoding="utf-8") as f:
            norm_data = json.load(f)
        self.input_stats = norm_data.get("stats", {})

    def evaluate_1step_trajectory(
        self,
        true_deltas_raw: np.ndarray,
        pred_deltas_raw: np.ndarray,
    ) -> Dict[str, float]:
        """
        Computes 1-step trajectory reconstruction error (next state = current + pred_delta).
        """
        # 1-step error is the difference between predicted transition and true transition
        errors = np.linalg.norm(pred_deltas_raw[:, :3] - true_deltas_raw[:, :3], axis=1)
        return {
            "mean_trajectory_error": float(np.mean(errors)),
            "max_trajectory_error": float(np.max(errors)),
        }

    def evaluate_orbit_drift(
        self,
        df_test_raw: Any,
        device: str = "cpu",
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Rolls out predictions continuously starting from S0.
        Computes orbit drift (distance between true orbit and predicted orbit over time).
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for orbit rollout.")

        self.model.eval()

        # Extract raw arrays from test dataset
        # We need: elapsed_seconds, delta_time, x, y, z, vx, vy, vz
        t_raw = df_test_raw["elapsed_seconds"].to_numpy()
        dt_raw = df_test_raw["delta_time"].to_numpy()
        
        pos_cols = ["x", "y", "z"]
        vel_cols = ["vx", "vy", "vz"]
        
        true_pos = df_test_raw[pos_cols].to_numpy()
        true_vel = df_test_raw[vel_cols].to_numpy()

        n_steps = len(df_test_raw)
        
        # Initialize predicted arrays with true initial state
        pred_pos = np.zeros_like(true_pos)
        pred_vel = np.zeros_like(true_vel)
        
        pred_pos[0] = true_pos[0]
        pred_vel[0] = true_vel[0]

        # Prepare column stats for normalising inputs
        input_cols = ["elapsed_seconds", "delta_time", "x", "y", "z", "vx", "vy", "vz"]
        means = np.array([self.input_stats[c]["mean"] for c in input_cols])
        stds = np.array([self.input_stats[c]["std"] for c in input_cols])

        # Step-by-step rollout
        for t in range(n_steps - 1):
            # 1. Prepare raw input vector for step t
            # x_pred, y_pred, z_pred, vx_pred, vy_pred, vz_pred are previously predicted values
            raw_in = np.array([
                t_raw[t],
                dt_raw[t],
                pred_pos[t, 0], pred_pos[t, 1], pred_pos[t, 2],
                pred_vel[t, 0], pred_vel[t, 1], pred_vel[t, 2]
            ])

            # 2. Normalize input vector using stats
            norm_in = (raw_in - means) / stds

            # Convert to PyTorch Tensor
            in_tensor = torch.tensor(norm_in, dtype=torch.float32, device=device).unsqueeze(0)

            # 3. Model forward pass (only pass first 8 columns to network)
            with torch.no_grad():
                out_tensor = self.model(in_tensor[:, :self.model.input_dim])

            # 4. De-normalize output
            out_raw = self.denormalizer.denormalize_output(out_tensor).cpu().numpy()[0]

            # Extract transition components
            pred_delta_pos = out_raw[:3]
            pred_delta_vel = out_raw[3:6]

            # 5. Integrate to get next state (Euler transition is used here)
            pred_pos[t + 1] = pred_pos[t] + pred_delta_pos
            pred_vel[t + 1] = pred_vel[t] + pred_delta_vel

        # Compute drift distance over time
        drift_distances = np.linalg.norm(true_pos - pred_pos, axis=1)

        summary = {
            "mean_orbit_drift": float(np.mean(drift_distances)),
            "max_orbit_drift": float(np.max(drift_distances)),
            "final_orbit_drift": float(drift_distances[-1]),
        }

        log.info(
            "Orbit Rollout Drift — Mean: %.4f km | Max: %.4f km | Final: %.4f km",
            summary["mean_orbit_drift"],
            summary["max_orbit_drift"],
            summary["final_orbit_drift"],
        )

        # Concatenate predicted states for visualization
        pred_states = np.hstack([pred_pos, pred_vel])

        return pred_states, summary
