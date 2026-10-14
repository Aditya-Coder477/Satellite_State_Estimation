"""
src/pinn/evaluator.py
=====================
Computes validation and testing metrics for Phase 8 (PINN Framework).

Runs model evaluation on given splits to calculate Root Mean Squared Error (RMSE),
Mean Absolute Error (MAE), mean physics residuals, and constraint violations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

# PyTorch import handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.physics.autograd_utils import AutogradUtils
from src.physics.coordinate_dynamics import compute_norm
from src.physics.physics_engine import PhysicsEngine
from src.pinn.forward_model import ForwardModel
from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class PINNEvaluator:
    """
    Evaluates PINN network models against validation/test datasets.

    Parameters
    ----------
    forward_model : ForwardModel
        The wrapped prediction model.
    physics_engine : PhysicsEngine
        The active physics engine.
    """

    def __init__(self, forward_model: ForwardModel, physics_engine: PhysicsEngine) -> None:
        self.forward_model = forward_model
        self.physics_engine = physics_engine

    def evaluate(self, df: Any) -> Dict[str, float]:
        """
        Runs evaluation on the given pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The dataset split to evaluate (validation or test).

        Returns
        -------
        Dict[str, float]
            A dictionary containing evaluation metrics:
            - position_rmse (km)
            - velocity_rmse (km/s)
            - position_mae (km)
            - velocity_mae (km/s)
            - physics_loss (mean squared residual)
            - constraint_violations_count
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for PINN evaluation.")

        if len(df) == 0:
            log.warning("Empty DataFrame passed to PINNEvaluator. Returning zero metrics.")
            return {
                "position_rmse": 0.0,
                "velocity_rmse": 0.0,
                "position_mae": 0.0,
                "velocity_mae": 0.0,
                "physics_loss": 0.0,
                "constraint_violations": 0.0,
            }

        # 1. Prepare Inputs & Targets
        t_np = df["elapsed_seconds"].values.astype(np.float32)
        
        pos_cols = ["x_gcrs_km", "y_gcrs_km", "z_gcrs_km"]
        vel_cols = ["vx_gcrs_kms", "vy_gcrs_kms", "vz_gcrs_kms"]
        state_true_np = df[pos_cols + vel_cols].values.astype(np.float32)

        # Place on GPU/CPU device
        t = torch.tensor(t_np, dtype=torch.float32, device=self.forward_model.device).unsqueeze(1).requires_grad_(True)
        state_true = torch.tensor(state_true_np, dtype=torch.float32, device=self.forward_model.device)

        # 2. Forward Predict
        # We need model in eval mode
        self.forward_model.network.eval()
        
        with torch.set_grad_enabled(True):  # autograd needs gradients enabled even in eval mode!
            state_pred = self.forward_model.predict_state(t, requires_grad=True)
            
            # Compute first derivatives for residuals
            state_deriv_pred = AutogradUtils.compute_first_derivative(t, state_pred)
            
            # Compute residuals
            residuals = self.physics_engine.compute_residuals(state_pred, state_deriv_pred)
            physics_loss = torch.mean(residuals ** 2).item()

        # Place predictions back on CPU for metric calculation
        state_pred_np = state_pred.detach().cpu().numpy()
        state_deriv_pred_np = state_deriv_pred.detach().cpu().numpy()

        pos_pred = state_pred_np[:, :3]
        pos_true = state_true_np[:, :3]
        vel_pred = state_pred_np[:, 3:6]
        vel_true = state_true_np[:, 3:6]

        # 3. Compute Metrics
        pos_err_sq = (pos_pred - pos_true) ** 2
        vel_err_sq = (vel_pred - vel_true) ** 2
        
        pos_rmse = float(np.sqrt(np.mean(pos_err_sq)))
        vel_rmse = float(np.sqrt(np.mean(vel_err_sq)))

        pos_mae = float(np.mean(np.abs(pos_pred - pos_true)))
        vel_mae = float(np.mean(np.abs(vel_pred - vel_true)))

        # 4. Evaluate Constraint Violations
        # Evaluate using CPU numpy arrays inside ConstraintEvaluator (ConstraintEvaluator supports NumPy!)
        acc_pred = state_deriv_pred_np[:, 3:]
        constr_sat, _ = self.physics_engine.validate_prediction(state_pred_np, acc_pred)
        
        # Calculate percent of samples violating constraints
        # ConstraintEvaluator evaluates overall batch. Let's count row-by-row violations for details
        r_norm = np.linalg.norm(pos_pred, axis=-1)
        v_norm = np.linalg.norm(vel_pred, axis=-1)
        
        min_r = self.physics_engine.constraint_evaluator.min_radius
        max_v = self.physics_engine.constraint_evaluator.max_velocity
        
        r_violations = int(np.sum(r_norm < min_r))
        v_violations = int(np.sum(v_norm > max_v))
        total_violations = r_violations + v_violations

        metrics = {
            "position_rmse": pos_rmse,
            "velocity_rmse": vel_rmse,
            "position_mae": pos_mae,
            "velocity_mae": vel_mae,
            "physics_loss": physics_loss,
            "constraint_violations": float(total_violations),
        }

        log.info(
            "Evaluation completed: Pos RMSE=%.4e km, Vel RMSE=%.4e km/s, Phys Loss=%.4e",
            pos_rmse,
            vel_rmse,
            physics_loss,
        )

        return metrics
