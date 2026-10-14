"""
src/pinn/losses.py
==================
Computes loss components for Phase 8 (PINN Framework).

Evaluates observation losses, physics residuals losses (retrieved from the
PhysicsEngine), initial condition losses, boundary condition losses, and
constraint penalties, combining them into a single weighted total loss.

Bug Fixes (Phase 9 — Physics Residual NaN elimination):
  * Added pre-flight assertion: t_colloc.requires_grad must be True.
  * Added diagnostic logging of t_colloc.grad_fn, device, dtype before physics loss.
  * Added torch.isfinite assertion on autograd derivative and residual tensors.
  * NaN location is logged (batch index, dimension, value) before raising.
  * Per-loss component logging every call for Step 7 visibility.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

# PyTorch import handling
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.physics.autograd_utils import AutogradUtils, AutogradDisconnectedError
from src.physics.physics_engine import PhysicsEngine
from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class PINNLossCalculator:
    """
    Computes weighted and unweighted loss terms for PINN training.

    Parameters
    ----------
    cfg : Dict[str, Any]
        Project configuration.
    physics_engine : PhysicsEngine
        PhysicsEngine instance used to compute residuals and constraints.
    """

    def __init__(self, cfg: Dict[str, Any], physics_engine: PhysicsEngine) -> None:
        self._cfg = cfg
        self.physics_engine = physics_engine

        # Loss weights configuration
        losses_cfg = cfg.get("pinn", {}).get("losses", {})
        self.w_obs    = float(losses_cfg.get("observation_weight",  1.0))
        self.w_phys   = float(losses_cfg.get("physics_weight",     10.0))
        self.w_bound  = float(losses_cfg.get("boundary_weight",     5.0))
        self.w_init   = float(losses_cfg.get("initial_weight",      5.0))
        self.w_const  = float(losses_cfg.get("constraint_weight",   1.0))

        log.info(
            "PINNLossCalculator initialised with loss weights: "
            "obs=%.3f  physics=%.3f  boundary=%.3f  initial=%.3f  constraint=%.3f",
            self.w_obs, self.w_phys, self.w_bound, self.w_init, self.w_const,
        )

    def calculate_losses(
        self,
        t_obs: "torch.Tensor",
        state_obs_true: "torch.Tensor",
        state_obs_pred: "torch.Tensor",
        t_colloc: "torch.Tensor",
        state_colloc_pred: "torch.Tensor",
        t_init: Optional["torch.Tensor"] = None,
        state_init_true: Optional["torch.Tensor"] = None,
        state_init_pred: Optional["torch.Tensor"] = None,
        t_bound: Optional["torch.Tensor"] = None,
        state_bound_true: Optional["torch.Tensor"] = None,
        state_bound_pred: Optional["torch.Tensor"] = None,
        epoch: int = 0,
        batch_id: int = 0,
    ) -> Dict[str, "torch.Tensor"]:
        """
        Computes all physics-informed loss terms and the total weighted loss.

        Parameters
        ----------
        t_obs : torch.Tensor
            Time stamps for observations, shape (Batch_obs, 1).
        state_obs_true : torch.Tensor
            True/noisy observation states, shape (Batch_obs, 6).
        state_obs_pred : torch.Tensor
            Predicted states at observation times, shape (Batch_obs, 6).
        t_colloc : torch.Tensor
            Physics collocation point times, shape (Batch_col, 1).
            **Must have requires_grad=True** and be connected to state_colloc_pred.
        state_colloc_pred : torch.Tensor
            Predicted states at collocation points, shape (Batch_col, 6).
        epoch : int
            Current training epoch (for diagnostics).
        batch_id : int
            Current mini-batch index (for diagnostics).

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of unweighted loss components and the total loss.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to compute losses.")

        device = state_obs_pred.device

        # ── Step 1: Pre-flight checks ──────────────────────────────────────
        if not t_colloc.requires_grad:
            raise ValueError(
                f"[Epoch {epoch} | Batch {batch_id}] "
                "t_colloc.requires_grad is False — cannot compute autograd physics residual. "
                "Fix in trainer.py mini-batch loop: "
                "t_colloc_batch = t_colloc[idx].clone().detach().requires_grad_(True)"
            )

        log.debug(
            "[Epoch %d | Batch %d] t_colloc: shape=%s  requires_grad=%s  "
            "grad_fn=%s  device=%s  dtype=%s",
            epoch, batch_id,
            tuple(t_colloc.shape),
            t_colloc.requires_grad,
            t_colloc.grad_fn,
            t_colloc.device,
            t_colloc.dtype,
        )
        log.debug(
            "[Epoch %d | Batch %d] state_colloc_pred: shape=%s  requires_grad=%s  "
            "grad_fn=%s",
            epoch, batch_id,
            tuple(state_colloc_pred.shape),
            state_colloc_pred.requires_grad,
            state_colloc_pred.grad_fn,
        )

        # ── 1. Observation Data Loss ───────────────────────────────────────
        obs_loss = torch.mean((state_obs_pred - state_obs_true) ** 2)

        # ── 2. Physics Residual Loss (at Collocation points) ──────────────
        try:
            state_deriv_colloc = AutogradUtils.compute_first_derivative(
                t_colloc, state_colloc_pred
            )
        except (AutogradDisconnectedError, ValueError, RuntimeError) as exc:
            log.error(
                "[Epoch %d | Batch %d] Autograd derivative failed: %s",
                epoch, batch_id, exc,
            )
            raise

        # ── Step 6: Validate derivative before passing to physics engine ───
        if not torch.isfinite(state_deriv_colloc).all():
            n_nan = torch.isnan(state_deriv_colloc).sum().item()
            n_inf = torch.isinf(state_deriv_colloc).sum().item()
            bad_indices = (~torch.isfinite(state_deriv_colloc)).nonzero(as_tuple=False)
            for idx in bad_indices[:5]:
                log.error(
                    "[Epoch %d | Batch %d] NaN/Inf in state_deriv_colloc: "
                    "batch_idx=%d  dim=%d  value=%s",
                    epoch, batch_id, idx[0].item(), idx[1].item(),
                    state_deriv_colloc[idx[0], idx[1]].item(),
                )
            raise RuntimeError(
                f"[Epoch {epoch} | Batch {batch_id}] "
                f"state_deriv_colloc has {n_nan} NaN(s) and {n_inf} Inf(s). "
                "Physics residual cannot be computed."
            )

        # Compute residuals using the PhysicsEngine
        residuals = self.physics_engine.compute_residuals(state_colloc_pred, state_deriv_colloc)

        # ── Step 6: Validate residuals ────────────────────────────────────
        if not torch.isfinite(residuals).all():
            n_nan = torch.isnan(residuals).sum().item()
            n_inf = torch.isinf(residuals).sum().item()
            log.error(
                "[Epoch %d | Batch %d] Physics residuals non-finite: n_nan=%d  n_inf=%d",
                epoch, batch_id, n_nan, n_inf,
            )
            raise RuntimeError(
                f"[Epoch {epoch} | Batch {batch_id}] "
                f"Physics residuals have {n_nan} NaN(s) and {n_inf} Inf(s)."
            )

        reduction = self._cfg.get("physics", {}).get("residual", {}).get("reduction", "mean").lower()
        if reduction == "sum":
            physics_loss = torch.sum(residuals ** 2)
        else:
            physics_loss = torch.mean(residuals ** 2)

        # ── 3. Initial Condition Loss ──────────────────────────────────────
        if t_init is not None and state_init_true is not None and state_init_pred is not None:
            initial_loss = torch.mean((state_init_pred - state_init_true) ** 2)
        else:
            initial_loss = torch.tensor(0.0, dtype=torch.float32, device=device)

        # ── 4. Boundary Condition Loss ─────────────────────────────────────
        if t_bound is not None and state_bound_true is not None and state_bound_pred is not None:
            boundary_loss = torch.mean((state_bound_pred - state_bound_true) ** 2)
        else:
            boundary_loss = torch.tensor(0.0, dtype=torch.float32, device=device)

        # ── 5. Differentiable Physical Constraints Loss ────────────────────
        _, penalties_colloc = self.physics_engine.validate_prediction(
            state_colloc_pred, state_deriv_colloc[..., 3:]
        )
        constraint_loss = penalties_colloc["total_constraint_penalty"]

        # ── 6. Weighted Total Loss ─────────────────────────────────────────
        total_loss = (
            self.w_obs   * obs_loss
            + self.w_phys  * physics_loss
            + self.w_bound * boundary_loss
            + self.w_init  * initial_loss
            + self.w_const * constraint_loss
        )

        # ── Step 7: Per-component diagnostic logging ───────────────────────
        log.debug(
            "[Epoch %d | Batch %d] Losses — obs=%.4e  physics=%.4e  "
            "boundary=%.4e  initial=%.4e  constraint=%.4e  total=%.4e",
            epoch, batch_id,
            obs_loss.item(),
            physics_loss.item(),
            boundary_loss.item(),
            initial_loss.item(),
            constraint_loss.item() if hasattr(constraint_loss, "item") else float(constraint_loss),
            total_loss.item(),
        )

        return {
            "observation_loss": obs_loss,
            "physics_loss":     physics_loss,
            "boundary_loss":    boundary_loss,
            "initial_loss":     initial_loss,
            "constraint_loss":  constraint_loss,
            "total_loss":       total_loss,
        }
