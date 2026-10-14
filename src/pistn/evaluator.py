"""
src/pistn/evaluator.py
======================
Evaluator module for Phase 9 (PI-STN) validation.

Evaluates validation performance in both normalized and de-normalized physical spaces.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

# PyTorch import handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.pistn.metrics import MetricsCalculator
from src.pistn.normalization import DeNormalizer
from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class PISTNEvaluator:
    """
    Evaluates PI-STN model predictions against target transitions and physics constraints.
    """

    def __init__(self, model: Any, loss_fn: Any, physics_engine: Any, denormalizer: DeNormalizer) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.physics_engine = physics_engine
        self.denormalizer = denormalizer

    def evaluate(self, val_loader: Any, device: str = "cpu") -> Tuple[float, Dict[str, float]]:
        """
        Runs validation evaluation.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for model evaluation.")

        self.model.eval()
        
        total_loss = 0.0
        total_trans_loss = 0.0
        total_phys_loss = 0.0
        total_res_loss = 0.0
        total_const_loss = 0.0

        all_pred_deltas_raw = []
        all_true_deltas_raw = []
        all_pred_res_raw = []
        all_true_res_raw = []
        
        batches_count = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch
                else:
                    inputs = batch["inputs"]
                    targets = batch["targets"]

                inputs = inputs.to(device, dtype=torch.float32)
                targets = targets.to(device, dtype=torch.float32)

                # Forward pass
                outputs = self.model(inputs[:, :self.model.input_dim])

                # 1. De-normalize state and time step to raw physical units
                state_raw, dt_raw = self.denormalizer.denormalize_state(inputs)

                # 2. De-normalize model predictions and target labels
                outputs_raw = self.denormalizer.denormalize_output(outputs)
                targets_raw = self.denormalizer.denormalize_target(targets)

                pred_delta_raw = outputs_raw[:, :6]
                true_delta_raw = targets_raw[:, :6]
                
                pred_res_raw = outputs_raw[:, 6:9]
                true_res_raw = targets_raw[:, 6:9]

                # 3. Compute expected physics transition using integrator
                phys_delta_raw = self.physics_engine.compute_transition(state_raw, dt_raw)

                # 4. Compute true residual acceleration (a_res = a_observed - a_physics)
                _, _, a_res_true_raw = self.physics_engine.compute_residuals(
                    state_raw, true_delta_raw[:, 3:6], dt_raw
                )

                # 5. Evaluate constraint violations on raw positions and residual accelerations
                _, penalties = self.physics_engine.validate_prediction(state_raw, pred_res_raw)

                # 6. Compute loss components
                loss_dict = self.loss_fn(
                    pred_delta_norm=outputs[:, :6],
                    true_delta_norm=targets[:, :6],
                    pred_delta_raw=pred_delta_raw,
                    true_delta_raw=true_delta_raw,
                    phys_delta_raw=phys_delta_raw,
                    pred_res_raw=pred_res_raw,
                    true_res_raw=a_res_true_raw,
                    constraint_penalty_raw=penalties["total_constraint_penalty"],
                )

                loss = loss_dict["total_physics_loss"]

                total_loss += float(loss.item())
                total_trans_loss += float(loss_dict["transition_loss"].item())
                total_phys_loss += float(loss_dict["physics_loss"].item())
                total_res_loss += float(loss_dict["residual_loss"].item())
                total_const_loss += float(loss_dict["constraint_loss"].item())

                all_pred_deltas_raw.append(pred_delta_raw.cpu().numpy())
                all_true_deltas_raw.append(true_delta_raw.cpu().numpy())
                all_pred_res_raw.append(pred_res_raw.cpu().numpy())
                all_true_res_raw.append(a_res_true_raw.cpu().numpy())
                
                batches_count += 1

        if batches_count == 0:
            return 0.0, {}

        mean_loss = total_loss / batches_count
        mean_trans_loss = total_trans_loss / batches_count
        mean_phys_loss = total_phys_loss / batches_count
        mean_res_loss = total_res_loss / batches_count
        mean_const_loss = total_const_loss / batches_count

        # Concatenate all batches for metrics
        pred_deltas_raw = np.concatenate(all_pred_deltas_raw, axis=0)
        true_deltas_raw = np.concatenate(all_true_deltas_raw, axis=0)
        pred_res_raw = np.concatenate(all_pred_res_raw, axis=0)
        true_res_raw = np.concatenate(all_true_res_raw, axis=0)

        # Compute regression metrics (on physical units)
        metrics = MetricsCalculator.compute_all(pred_deltas_raw, true_deltas_raw)
        
        # Add component MAEs (Physical units)
        mae_pos = np.mean(np.abs(pred_deltas_raw[:, :3] - true_deltas_raw[:, :3]), axis=0)
        mae_vel = np.mean(np.abs(pred_deltas_raw[:, 3:6] - true_deltas_raw[:, 3:6]), axis=0)
        mae_acc_res = np.mean(np.abs(pred_res_raw - true_res_raw), axis=0)

        metrics.update({
            "val_loss": mean_loss,
            "val_transition_loss": mean_trans_loss,
            "val_physics_loss": mean_phys_loss,
            "val_residual_loss": mean_res_loss,
            "val_constraint_loss": mean_const_loss,
            
            # Position transition component MAEs
            "mae_delta_x": float(mae_pos[0]),
            "mae_delta_y": float(mae_pos[1]),
            "mae_delta_z": float(mae_pos[2]),
            
            # Velocity transition component MAEs
            "mae_delta_vx": float(mae_vel[0]),
            "mae_delta_vy": float(mae_vel[1]),
            "mae_delta_vz": float(mae_vel[2]),
            
            # Residual acceleration component MAEs
            "mae_ax_res": float(mae_acc_res[0]),
            "mae_ay_res": float(mae_acc_res[1]),
            "mae_az_res": float(mae_acc_res[2]),
        })

        return mean_loss, metrics
