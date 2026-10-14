"""
src/pistn/losses.py
===================
Defines loss components and total loss calculation for Phase 9 (PI-STN).

Supports de-normalized physical evaluation for Physics Loss and Constraint Loss,
and dynamic loss weights.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

# PyTorch import handling
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    class nn:  # type: ignore
        Module = object
    TORCH_AVAILABLE = False

from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


if TORCH_AVAILABLE:
    class PISTNLoss(nn.Module):
        """
        Computes the weighted PI-STN loss with optional dynamic loss balancing.
        """

        def __init__(self, cfg: Dict[str, Any]) -> None:
            super().__init__()
            self._cfg = cfg
            loss_cfg = cfg.get("pistn", {}).get("losses", {})

            # Initial weights from config
            self.w_trans = float(loss_cfg.get("transition_weight", 1.0))
            self.w_phys = float(loss_cfg.get("physics_weight", 5.0))
            self.w_resid = float(loss_cfg.get("residual_weight", 2.0))
            self.w_const = float(loss_cfg.get("constraint_weight", 1.0))
            self.w_energy = float(loss_cfg.get("energy_weight", 1.0))
            self.w_ang_mom = float(loss_cfg.get("angular_momentum_weight", 1.0))
            self.w_orb = float(loss_cfg.get("orbital_weight", 0.5))

            log.info(
                "PISTNLoss initialized with weights: "
                "transition=%.3e, physics=%.3e, residual=%.3e, constraint=%.3e, "
                "energy=%.3e, angular_momentum=%.3e, orbital=%.3e",
                self.w_trans,
                self.w_phys,
                self.w_resid,
                self.w_const,
                self.w_energy,
                self.w_ang_mom,
                self.w_orb,
            )

        def forward(
            self,
            pred_delta_norm: torch.Tensor,
            true_delta_norm: torch.Tensor,
            pred_delta_raw: torch.Tensor,
            true_delta_raw: torch.Tensor,
            phys_delta_raw: torch.Tensor,
            pred_res_raw: torch.Tensor,
            true_res_raw: torch.Tensor,
            constraint_penalty_raw: torch.Tensor | float = 0.0,
            energy_loss: torch.Tensor | float = 0.0,
            angular_momentum_loss: torch.Tensor | float = 0.0,
            orbital_loss: torch.Tensor | float = 0.0,
        ) -> Dict[str, torch.Tensor]:
            """
            Evaluates individual loss terms and their weighted total.
            
            - Transition Loss: computed on normalized values for optimization stability.
            - Physics, Residual, and Constraint Losses: computed on de-normalized physical values.
            """
            # 1. Transition Loss: MSE of predicted vs true transitions (normalized)
            transition_loss = torch.mean((pred_delta_norm - true_delta_norm) ** 2)

            # 2. Physics Loss: MSE of predicted vs expected physics transitions (raw physical units)
            physics_loss = torch.mean((pred_delta_raw - phys_delta_raw) ** 2)

            # 3. Residual Loss: MSE of predicted vs true residual accelerations (raw physical units)
            residual_loss = torch.mean((pred_res_raw - true_res_raw) ** 2)

            # 4. Constraint Loss
            constraint_loss = torch.tensor(constraint_penalty_raw, device=pred_delta_norm.device) if not isinstance(constraint_penalty_raw, torch.Tensor) else constraint_penalty_raw

            # 5. New losses
            el = torch.tensor(energy_loss, device=pred_delta_norm.device) if not isinstance(energy_loss, torch.Tensor) else energy_loss
            al = torch.tensor(angular_momentum_loss, device=pred_delta_norm.device) if not isinstance(angular_momentum_loss, torch.Tensor) else angular_momentum_loss
            ol = torch.tensor(orbital_loss, device=pred_delta_norm.device) if not isinstance(orbital_loss, torch.Tensor) else orbital_loss

            # 6. Weighted Total Loss
            total_loss = (
                self.w_trans * transition_loss
                + self.w_phys * physics_loss
                + self.w_resid * residual_loss
                + self.w_const * constraint_loss
                + self.w_energy * el
                + self.w_ang_mom * al
                + self.w_orb * ol
            )

            return {
                "transition_loss": transition_loss,
                "physics_loss": physics_loss,
                "residual_loss": residual_loss,
                "constraint_loss": constraint_loss,
                "energy_loss": el,
                "angular_momentum_loss": al,
                "orbital_loss": ol,
                "total_physics_loss": total_loss,
            }
else:
    class PISTNLoss:  # type: ignore
        def __init__(self, cfg: Dict[str, Any]) -> None: pass
