"""
src/pinn/trainer.py
===================
Coordinates neural network training for Phase 8 (PINN Framework).

Implements the PINNTrainer class running optimization loops (Adam, L-BFGS),
validation passes, gradient clipping, scheduler stepping, and callback dispatching.

Bug Fixes (Phase 9 — Physics Residual NaN elimination):
  * BUG 1 (CRITICAL): Mini-batch collocation slice now uses
      t_colloc_batch = t_colloc[idx].clone().detach().requires_grad_(True)
    Indexing a leaf tensor with randint indices destroys the gradient connection;
    re-attaching restores it for autograd differentiation.
  * BUG 5: t_colloc.grad is zeroed at the start of each mini-batch to prevent
    gradient accumulation across batches.
  * Step 7: All 5 loss components are logged per-epoch and saved to
    ``logs/training/training_loss_breakdown.csv``.
  * Step 5/11: GradientDebugger.diagnose() and .plot_gradient_flow() are called
    periodically during training.
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# PyTorch import handling
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.physics.gradient_debugger import GradientDebugger
from src.pinn.callbacks import CallbackManager, LearningRateMonitor, LossHistory
from src.pinn.checkpoint import CheckpointManager
from src.pinn.evaluator import PINNEvaluator
from src.pinn.forward_model import ForwardModel
from src.pinn.losses import PINNLossCalculator
from src.pinn.validator import PINNValidator
from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)

# Loss component CSV columns (Step 7)
_CSV_FIELDNAMES = [
    "epoch",
    "observation_loss",
    "physics_loss",
    "boundary_loss",
    "initial_loss",
    "constraint_loss",
    "total_loss",
    "val_position_rmse",
    "val_velocity_rmse",
]


class PINNTrainer:
    """
    Orchestrates PINN optimization, evaluation, and checkpointing.

    Parameters
    ----------
    forward_model : ForwardModel
        Wrapped neural network model.
    optimizer : torch.optim.Optimizer
        The active PyTorch optimizer.
    scheduler : Any
        The active learning rate scheduler.
    loss_calculator : PINNLossCalculator
        Loss evaluation class.
    checkpoint_manager : CheckpointManager
        Checkpoint manager instance.
    cfg : Dict[str, Any]
        Project configuration.
    """

    def __init__(
        self,
        forward_model: ForwardModel,
        optimizer: "torch.optim.Optimizer",
        scheduler: Any,
        loss_calculator: PINNLossCalculator,
        checkpoint_manager: CheckpointManager,
        cfg: Dict[str, Any],
    ) -> None:
        self.forward_model      = forward_model
        self.optimizer          = optimizer
        self.scheduler          = scheduler
        self.loss_calculator    = loss_calculator
        self.checkpoint_manager = checkpoint_manager
        self._cfg               = cfg

        train_cfg        = cfg.get("pinn", {}).get("training", {})
        self.epochs      = int(train_cfg.get("epochs",         500))
        self.batch_size  = int(train_cfg.get("batch_size",    1024))
        self.grad_clip   = float(train_cfg.get("gradient_clip", 1.0))

        self.stop_training = False

        # Standard default callbacks
        self.loss_history    = LossHistory()
        self.lr_monitor      = LearningRateMonitor()
        self.callback_manager = CallbackManager([self.loss_history, self.lr_monitor])

        # Gradient debugger (Step 5)
        root = Path(cfg.get("_root", "."))
        grad_log_path = root / "logs" / "training" / "gradient_debug.log"
        self._grad_debugger = GradientDebugger(
            log_file=grad_log_path,
            log_every_n_epochs=max(1, self.epochs // 20),   # ~20 snapshots per run
        )

        # Loss breakdown CSV path (Step 7)
        self._loss_csv_path = root / "logs" / "training" / "training_loss_breakdown.csv"
        self._loss_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_file: Optional[Any]    = None
        self._csv_writer: Optional[Any]  = None

        # Gradient flow figure path (Step 11)
        self._grad_flow_path = root / "results" / "training" / "gradient_flow.png"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_callbacks(self, callbacks: List[Any]) -> None:
        """Configures custom callbacks (e.g. EarlyStopping, ModelCheckpoint)."""
        self.callback_manager = CallbackManager(
            [self.loss_history, self.lr_monitor] + callbacks
        )

    def train(
        self,
        t_obs: "torch.Tensor",
        state_obs_true: "torch.Tensor",
        t_colloc: "torch.Tensor",
        val_df: Any,
    ) -> Dict[str, List[float]]:
        """
        Executes the optimization loop over the configured number of epochs.

        Parameters
        ----------
        t_obs : torch.Tensor
            Observation timestamps tensor of shape (N_obs, 1).
        state_obs_true : torch.Tensor
            True GCRS states tensor of shape (N_obs, 6).
        t_colloc : torch.Tensor
            Collocation point timestamps of shape (N_col, 1).
            Will be detached and re-attached with requires_grad=True internally.
        val_df : pd.DataFrame
            Validation split DataFrame used by the evaluator at epoch end.

        Returns
        -------
        Dict[str, List[float]]
            Loss history dictionary.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for training.")

        evaluator = PINNEvaluator(self.forward_model, self.loss_calculator.physics_engine)

        # ── Resolve initial condition ──────────────────────────────────────
        t_init         = None
        state_init_true = None
        zero_mask = (t_obs == 0.0).squeeze()
        if zero_mask.any():
            t_init          = t_obs[zero_mask][:1]
            state_init_true = state_obs_true[zero_mask][:1]

        # ── Ensure t_colloc has requires_grad (full tensor) ────────────────
        # BUG FIX 1/5: Detach and re-attach to create a clean leaf tensor.
        # Later, each mini-batch slice is ALSO re-attached individually.
        t_colloc = t_colloc.clone().detach().requires_grad_(True)
        log.info(
            "Training: t_colloc shape=%s  requires_grad=%s  device=%s",
            tuple(t_colloc.shape), t_colloc.requires_grad, t_colloc.device,
        )

        # ── Open loss breakdown CSV (Step 7) ───────────────────────────────
        self._csv_file   = open(self._loss_csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=_CSV_FIELDNAMES)
        self._csv_writer.writeheader()

        self.callback_manager.on_train_begin(self)

        is_lbfgs = isinstance(self.optimizer, torch.optim.LBFGS)
        log.info(
            "Starting training loop. Optimizer: %s  Epochs: %d  BatchSize: %s",
            "L-BFGS (Full Batch)" if is_lbfgs else "Stochastic Mini-Batch",
            self.epochs,
            "N/A (full batch)" if is_lbfgs else self.batch_size,
        )

        for epoch in range(1, self.epochs + 1):
            if self.stop_training:
                log.info("Stopping training early at epoch %d.", epoch)
                break

            self.forward_model.network.train()

            if is_lbfgs:
                epoch_losses = self._lbfgs_step(
                    t_obs, state_obs_true, t_colloc, t_init, state_init_true, epoch
                )
            else:
                epoch_losses = self._minibatch_step(
                    t_obs, state_obs_true, t_colloc, t_init, state_init_true, epoch
                )

            # ── Validation pass ────────────────────────────────────────────
            self.forward_model.network.eval()
            val_metrics = evaluator.evaluate(val_df)

            epoch_logs = {**epoch_losses}
            epoch_logs["val_position_rmse"] = val_metrics["position_rmse"]
            epoch_logs["val_velocity_rmse"] = val_metrics["velocity_rmse"]
            epoch_logs["val_total_loss"]    = val_metrics["physics_loss"]

            # ── Scheduler step ─────────────────────────────────────────────
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(epoch_logs["val_total_loss"])
                else:
                    self.scheduler.step()

            # ── Training stability check ───────────────────────────────────
            grads_ok,  _  = PINNValidator.validate_gradients(self.forward_model.network)
            losses_ok, _  = PINNValidator.validate_losses(epoch_losses)
            if not grads_ok or not losses_ok:
                log.error(
                    "Training unstable at epoch %d! grads_ok=%s  losses_ok=%s",
                    epoch, grads_ok, losses_ok,
                )
                self.stop_training = True

            # ── Step 7: Write loss breakdown CSV row ───────────────────────
            if self._csv_writer is not None:
                row = {"epoch": epoch}
                for k in _CSV_FIELDNAMES[1:]:
                    v = epoch_logs.get(k, float("nan"))
                    row[k] = v.item() if hasattr(v, "item") else v
                self._csv_writer.writerow(row)
                self._csv_file.flush()

            # ── Step 5: Gradient diagnostics ───────────────────────────────
            self._grad_debugger.diagnose(
                self.forward_model.network,
                epoch=epoch,
                batch_id=0,
            )

            # ── Callbacks ──────────────────────────────────────────────────
            self.callback_manager.on_epoch_end(self, epoch, epoch_logs)

        # ── Post-training cleanup ──────────────────────────────────────────
        self.callback_manager.on_train_end(self)

        if self._csv_file is not None:
            self._csv_file.close()

        # ── Step 11: Gradient flow visualization ───────────────────────────
        self._grad_debugger.plot_gradient_flow(
            self.forward_model.network,
            save_path=self._grad_flow_path,
        )

        log.info(
            "Training complete. Loss breakdown CSV: %s",
            self._loss_csv_path,
        )
        return self.loss_history.history

    # ------------------------------------------------------------------
    # Private training steps
    # ------------------------------------------------------------------

    def _lbfgs_step(
        self,
        t_obs: "torch.Tensor",
        state_obs_true: "torch.Tensor",
        t_colloc: "torch.Tensor",
        t_init: Optional["torch.Tensor"],
        state_init_true: Optional["torch.Tensor"],
        epoch: int,
    ) -> Dict[str, float]:
        """Full-batch L-BFGS step with closure."""

        def closure() -> float:
            self.optimizer.zero_grad()
            state_obs_pred    = self.forward_model.predict_state(t_obs,     requires_grad=True)
            state_colloc_pred = self.forward_model.predict_state(t_colloc,  requires_grad=True)
            state_init_pred   = (
                self.forward_model.predict_state(t_init, requires_grad=True)
                if t_init is not None else None
            )
            loss_dict = self.loss_calculator.calculate_losses(
                t_obs=t_obs,
                state_obs_true=state_obs_true,
                state_obs_pred=state_obs_pred,
                t_colloc=t_colloc,
                state_colloc_pred=state_colloc_pred,
                t_init=t_init,
                state_init_true=state_init_true,
                state_init_pred=state_init_pred,
                epoch=epoch,
                batch_id=0,
            )
            total_loss = loss_dict["total_loss"]
            total_loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.forward_model.network.parameters(), self.grad_clip
                )
            self._current_loss_dict = {k: v.item() for k, v in loss_dict.items()}
            return float(total_loss.item())

        try:
            self.optimizer.step(closure)
            return self._current_loss_dict
        except Exception as exc:
            log.error("L-BFGS step failed at epoch %d: %s", epoch, exc, exc_info=True)
            raise

    def _minibatch_step(
        self,
        t_obs: "torch.Tensor",
        state_obs_true: "torch.Tensor",
        t_colloc: "torch.Tensor",
        t_init: Optional["torch.Tensor"],
        state_init_true: Optional["torch.Tensor"],
        epoch: int,
    ) -> Dict[str, float]:
        """
        Stochastic mini-batch Adam/SGD step.

        BUG 1 FIX: Each collocation slice is re-attached with requires_grad=True.
        BUG 5 FIX: t_colloc.grad is zeroed before slicing to prevent accumulation.
        """
        epoch_loss_accum: Dict[str, float] = {}
        n_samples = len(t_obs)
        indices   = torch.randperm(n_samples, device=t_obs.device)

        for batch_num, start_idx in enumerate(range(0, n_samples, self.batch_size)):
            self.optimizer.zero_grad()

            # ── BUG 5 FIX: zero accumulated gradient on collocation tensor ─
            if t_colloc.grad is not None:
                t_colloc.grad.zero_()

            batch_idx          = indices[start_idx : start_idx + self.batch_size]
            t_obs_batch        = t_obs[batch_idx]
            state_obs_true_batch = state_obs_true[batch_idx]

            # ── BUG 1 FIX: re-attach gradient after indexing ───────────────
            # t_colloc[colloc_indices] creates a non-leaf tensor; .clone().detach()
            # makes it a new leaf and .requires_grad_(True) re-enables autograd.
            colloc_indices     = torch.randint(0, len(t_colloc), (len(batch_idx),), device=t_colloc.device)
            t_colloc_batch     = t_colloc[colloc_indices].clone().detach().requires_grad_(True)

            # ── Forward predictions ────────────────────────────────────────
            state_obs_pred     = self.forward_model.predict_state(t_obs_batch,     requires_grad=True)
            state_colloc_pred  = self.forward_model.predict_state(t_colloc_batch,  requires_grad=True)
            state_init_pred    = (
                self.forward_model.predict_state(t_init, requires_grad=True)
                if t_init is not None else None
            )

            # ── Compute losses ─────────────────────────────────────────────
            loss_dict = self.loss_calculator.calculate_losses(
                t_obs=t_obs_batch,
                state_obs_true=state_obs_true_batch,
                state_obs_pred=state_obs_pred,
                t_colloc=t_colloc_batch,
                state_colloc_pred=state_colloc_pred,
                t_init=t_init,
                state_init_true=state_init_true,
                state_init_pred=state_init_pred,
                epoch=epoch,
                batch_id=batch_num,
            )

            total_loss = loss_dict["total_loss"]
            total_loss.backward()

            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.forward_model.network.parameters(), self.grad_clip
                )

            self.optimizer.step()

            for k, v in loss_dict.items():
                epoch_loss_accum[k] = epoch_loss_accum.get(k, 0.0) + v.item()

        # Average losses over all batches in this epoch
        n_batches   = int(np.ceil(n_samples / self.batch_size))
        epoch_losses = {k: v / n_batches for k, v in epoch_loss_accum.items()}

        log.info(
            "Epoch %d — obs=%.4e  physics=%.4e  boundary=%.4e  "
            "initial=%.4e  constraint=%.4e  total=%.4e",
            epoch,
            epoch_losses.get("observation_loss", float("nan")),
            epoch_losses.get("physics_loss",      float("nan")),
            epoch_losses.get("boundary_loss",      float("nan")),
            epoch_losses.get("initial_loss",       float("nan")),
            epoch_losses.get("constraint_loss",    float("nan")),
            epoch_losses.get("total_loss",         float("nan")),
        )

        return epoch_losses
