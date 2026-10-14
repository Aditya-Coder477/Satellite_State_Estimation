"""
src/pistn/predictor.py
======================
Generates model predictions and saves de-normalized physical CSV outputs for Phase 9.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# PyTorch import handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.pistn.normalization import DeNormalizer
from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class PISTNPredictor:
    """
    Generates predictions from a trained model and formats them into reporting CSV files (in physical units).
    """

    def __init__(self, model: Any, physics_engine: Any, denormalizer: DeNormalizer) -> None:
        self.model = model
        self.physics_engine = physics_engine
        self.denormalizer = denormalizer

    def predict_to_csv(self, loader: Any, output_path: Path, device: str = "cpu") -> None:
        """
        Runs model inference over a loader, de-normalizes predictions to physical units, and outputs a CSV.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to run model prediction.")

        self.model.eval()
        
        all_x = []
        all_y = []
        all_z = []
        all_vx = []
        all_vy = []
        all_vz = []

        all_pred_dx = []
        all_pred_dy = []
        all_pred_dz = []
        all_pred_dvx = []
        all_pred_dvy = []
        all_pred_dvz = []

        all_true_dx = []
        all_true_dy = []
        all_true_dz = []
        all_true_dvx = []
        all_true_dvy = []
        all_true_dvz = []

        all_pred_ax_res = []
        all_pred_ay_res = []
        all_pred_az_res = []

        all_true_ax_res = []
        all_true_ay_res = []
        all_true_az_res = []

        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch
                else:
                    inputs = batch["inputs"]
                    targets = batch["targets"]

                inputs = inputs.to(device, dtype=torch.float32)
                targets = targets.to(device, dtype=torch.float32)

                # Forward pass
                outputs = self.model(inputs[:, :self.model.input_dim])

                # 1. De-normalize state (inputs) and outputs/targets to physical space
                state_raw, dt_raw = self.denormalizer.denormalize_state(inputs)
                outputs_raw = self.denormalizer.denormalize_output(outputs)
                targets_raw = self.denormalizer.denormalize_target(targets)

                # 2. Extract components in physical units
                states = state_raw.cpu().numpy()
                pred_deltas = outputs_raw[:, :6].cpu().numpy()
                true_deltas = targets_raw[:, :6].cpu().numpy()
                pred_res = outputs_raw[:, 6:9].cpu().numpy()

                # Calculate true residual target in raw units (a_res = a_observed - a_physics)
                _, _, a_res_true_raw = self.physics_engine.compute_residuals(
                    state_raw, targets_raw[:, 3:6], dt_raw
                )
                true_res = a_res_true_raw.cpu().numpy()

                all_x.extend(states[:, 0])
                all_y.extend(states[:, 1])
                all_z.extend(states[:, 2])
                all_vx.extend(states[:, 3])
                all_vy.extend(states[:, 4])
                all_vz.extend(states[:, 5])

                all_pred_dx.extend(pred_deltas[:, 0])
                all_pred_dy.extend(pred_deltas[:, 1])
                all_pred_dz.extend(pred_deltas[:, 2])
                all_pred_dvx.extend(pred_deltas[:, 3])
                all_pred_dvy.extend(pred_deltas[:, 4])
                all_pred_dvz.extend(pred_deltas[:, 5])

                all_true_dx.extend(true_deltas[:, 0])
                all_true_dy.extend(true_deltas[:, 1])
                all_true_dz.extend(true_deltas[:, 2])
                all_true_dvx.extend(true_deltas[:, 3])
                all_true_dvy.extend(true_deltas[:, 4])
                all_true_dvz.extend(true_deltas[:, 5])

                all_pred_ax_res.extend(pred_res[:, 0])
                all_pred_ay_res.extend(pred_res[:, 1])
                all_pred_az_res.extend(pred_res[:, 2])

                all_true_ax_res.extend(true_res[:, 0])
                all_true_ay_res.extend(true_res[:, 1])
                all_true_az_res.extend(true_res[:, 2])

        # Construct reporting DataFrame in raw physical units
        df_out = pd.DataFrame({
            "x": all_x, "y": all_y, "z": all_z,
            "vx": all_vx, "vy": all_vy, "vz": all_vz,
            "pred_delta_x": all_pred_dx, "pred_delta_y": all_pred_dy, "pred_delta_z": all_pred_dz,
            "pred_delta_vx": all_pred_dvx, "pred_delta_vy": all_pred_dvy, "pred_delta_vz": all_pred_dvz,
            "true_delta_x": all_true_dx, "true_delta_y": all_true_dy, "true_delta_z": all_true_dz,
            "true_delta_vx": all_true_dvx, "true_delta_vy": all_true_dvy, "true_delta_vz": all_true_dvz,
            "pred_ax_res": all_pred_ax_res, "pred_ay_res": all_pred_ay_res, "pred_az_res": all_pred_az_res,
            "true_ax_res": all_true_ax_res, "true_ay_res": all_true_ay_res, "true_az_res": all_true_az_res,
        })

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(output_path, index=False, float_format="%.10f")
        log.info("Saved de-normalised predictions reporting CSV to %s", output_path)
