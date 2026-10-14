"""
src/pistn/trainer.py
====================
Core training loop coordinator for Phase 9 (PI-STN).

Supports dynamic loss balancing, physical de-normalization, extensive validation logging,
callbacks (early stopping + checkpoints), and multi-step curriculum rollout training.

Multi-Step Rollout Training
---------------------------
When ``pistn.rollout_training.enabled = true``, each batch is processed through
K rollout steps controlled by the curriculum schedule:

* Step 1 (teacher-forced): Full loss — Transition + Physics + Residual + Constraint.
* Steps 2..K (autonomous rollout): Physics + Residual + Constraint losses only.
  Ground-truth ΔState is unavailable at steps k>1, so transition loss is omitted.

Total rollout loss = mean over K steps, backpropagated through the full K-step graph.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# PyTorch import handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.pistn.callbacks import EarlyStopping
from src.pistn.checkpoint import CheckpointManager
from src.pistn.evaluator import PISTNEvaluator
from src.pistn.normalization import DeNormalizer
from src.pistn.rollout_scheduler import RolloutScheduler
from src.losses.orbital_losses import EnergyConservationLoss, AngularMomentumLoss, OrbitalElementLoss
from src.losses.scheduled_sampler import ScheduledSampler
from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class PISTNTrainer:
    """
    Orchestrates the training lifecycle of the PI-STN model.
    """

    def __init__(
        self,
        model: Any,
        optimizer: Any,
        scheduler: Any,
        loss_fn: Any,
        physics_engine: Any,
        denormalizer: DeNormalizer,
        checkpoint_dir: Path,
        cfg: Dict[str, Any],
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.physics_engine = physics_engine
        self.denormalizer = denormalizer
        self.checkpoint_dir = Path(checkpoint_dir)
        self.cfg = cfg

        train_cfg = cfg.get("pistn", {}).get("training", {})
        self.epochs = int(train_cfg.get("epochs", 500))
        self.grad_clip = float(train_cfg.get("gradient_clip", 1.0))

        # Early stopping
        es_cfg = train_cfg.get("early_stopping", {})
        patience = int(es_cfg.get("patience", 20))
        min_delta = float(es_cfg.get("min_delta", 1e-6))
        restore_best = bool(es_cfg.get("restore_best_weights", True))
        self.early_stopping = EarlyStopping(patience=patience, min_delta=min_delta, restore_best_weights=restore_best)

        self.checkpoint_manager = CheckpointManager(self.checkpoint_dir)
        self.evaluator = PISTNEvaluator(self.model, self.loss_fn, self.physics_engine, self.denormalizer)

        # Multi-step rollout curriculum scheduler
        self.rollout_scheduler = RolloutScheduler(cfg)

        # Scheduled sampler
        self.scheduled_sampler = ScheduledSampler(cfg)

        # Invariant and orbital losses
        self.energy_loss_fn = EnergyConservationLoss()
        self.angular_momentum_loss_fn = AngularMomentumLoss()
        
        # Load orbital loss sub-weights from config
        orb_cfg = cfg.get("pistn", {}).get("orbital_loss", {})
        w_sma = float(orb_cfg.get("semi_major_axis", 1.0))
        w_ecc = float(orb_cfg.get("eccentricity", 1.0))
        w_inc = float(orb_cfg.get("inclination", 1.0))
        self.orbital_loss_fn = OrbitalElementLoss(w_a=w_sma, w_e=w_ecc, w_i=w_inc)

        self.history: Dict[str, List[float]] = {
            # Core losses
            "train_loss": [],
            "train_transition_loss": [],
            "train_physics_loss": [],
            "train_residual_loss": [],
            "train_constraint_loss": [],
            "train_energy_loss": [],
            "train_angular_momentum_loss": [],
            "train_orbital_loss": [],
            "train_sma_loss": [],
            "train_ecc_loss": [],
            "train_inc_loss": [],
            "val_loss": [],
            "val_transition_loss": [],
            "val_physics_loss": [],
            "val_residual_loss": [],
            "val_constraint_loss": [],
            "val_energy_loss": [],
            "val_angular_momentum_loss": [],
            "val_orbital_loss": [],
            "learning_rate": [],

            # Gradient / weight diagnostics
            "gradient_norm": [],
            "weight_norm": [],
            "transition_percent": [],
            "physics_percent": [],
            "residual_percent": [],
            "constraint_percent": [],

            # Constraint component breakdown
            "radius_penalty": [],
            "velocity_penalty": [],
            "energy_penalty": [],
            "momentum_penalty": [],

            # Multi-step rollout diagnostics
            "rollout_steps": [],
            "rollout_loss": [],
            "one_step_loss": [],
            "final_step_loss": [],
            "mean_rollout_pos_error": [],
            "max_rollout_pos_error": [],
            "teacher_forcing_prob": [],
        }

    # ------------------------------------------------------------------
    # Loss calibration
    # ------------------------------------------------------------------

    def calibrate_losses(self, train_loader: Any, device: str = "cpu") -> None:
        """
        Runs a forward pass over the training loader to compute mean losses
        and calibrate loss weights.
        """
        if not TORCH_AVAILABLE:
            return

        log.info("Running initial loss calibration pass...")
        self.model.eval()

        sum_trans = 0.0
        sum_phys = 0.0
        sum_resid = 0.0
        sum_const = 0.0
        batches = 0

        with torch.no_grad():
            for batch in train_loader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch
                else:
                    inputs = batch["inputs"]
                    targets = batch["targets"]

                inputs = inputs.to(device, dtype=torch.float32)
                targets = targets.to(device, dtype=torch.float32)

                outputs = self.model(inputs[:, :self.model.input_dim])

                state_raw, dt_raw = self.denormalizer.denormalize_state(inputs)
                outputs_raw = self.denormalizer.denormalize_output(outputs)
                targets_raw = self.denormalizer.denormalize_target(targets)

                pred_delta_raw = outputs_raw[:, :6]
                true_delta_raw = targets_raw[:, :6]
                pred_res_raw = outputs_raw[:, 6:9]
                true_res_raw = targets_raw[:, 6:9]

                phys_delta_raw = self.physics_engine.compute_transition(state_raw, dt_raw)
                _, _, a_res_true_raw = self.physics_engine.compute_residuals(state_raw, true_delta_raw[:, 3:6], dt_raw)
                _, penalties = self.physics_engine.validate_prediction(state_raw, pred_res_raw)

                trans_l = float(torch.mean((outputs[:, :6] - targets[:, :6]) ** 2).item())
                phys_l = float(torch.mean((pred_delta_raw - phys_delta_raw) ** 2).item())
                resid_l = float(torch.mean((pred_res_raw - true_res_raw) ** 2).item())
                const_l = float(penalties["total_constraint_penalty"])

                sum_trans += trans_l
                sum_phys += phys_l
                sum_resid += resid_l
                sum_const += const_l
                batches += 1

                if batches >= 10:
                    break

        mean_trans = sum_trans / batches
        mean_phys = sum_phys / batches
        mean_resid = sum_resid / batches
        mean_const = sum_const / batches

        log.info(
            "Average unweighted losses — Transition: %.4e | Physics: %.4e | Residual: %.4e | Constraint: %.4e",
            mean_trans, mean_phys, mean_resid, mean_const
        )

        w_trans_rec = 1.0
        w_phys_rec = mean_trans / max(mean_phys, 1e-8)
        w_resid_rec = mean_trans / max(mean_resid, 1e-8)
        w_const_rec = mean_trans / max(mean_const, 1e-8)

        log.info(
            "Recommended weights — Transition: %.4e | Physics: %.4e | Residual: %.4e | Constraint: %.4e",
            w_trans_rec, w_phys_rec, w_resid_rec, w_const_rec
        )

        balancing_cfg = self.cfg.get("pistn", {}).get("loss_balancing", {})
        mode = balancing_cfg.get("mode", "auto").lower()

        if mode == "auto":
            log.info("Loss balancing mode is 'auto'. Applying recommended weights.")
            self.loss_fn.w_trans = w_trans_rec
            self.loss_fn.w_phys = w_phys_rec
            self.loss_fn.w_resid = w_resid_rec
            self.loss_fn.w_const = w_const_rec
        else:
            log.info("Loss balancing mode is 'manual'. Applying manual weights from config.")
            manual_w = balancing_cfg.get("manual_weights", {})
            self.loss_fn.w_trans = float(manual_w.get("transition", 1.0))
            self.loss_fn.w_phys = float(manual_w.get("physics", 5.0))
            self.loss_fn.w_resid = float(manual_w.get("residual", 2.0))
            self.loss_fn.w_const = float(manual_w.get("constraint", 1.0))

    # ------------------------------------------------------------------
    # Multi-step rollout training step
    # ------------------------------------------------------------------

    def _rollout_train_step(
        self,
        inputs: "torch.Tensor",
        targets: "torch.Tensor",
        epoch: int,
        device: str,
    ) -> "Tuple[torch.Tensor, Dict[str, Any]]":
        """
        Computes multi-step autonomous rollout loss for one batch.

        Loss composition
        ----------------
        * Step 1 (teacher-forced / scheduled sampling): Transition + Physics + Residual + Constraint + Energy + Angular Momentum + Orbital
        * Steps 2..K (autonomous): Physics + Residual + Constraint + Energy + Angular Momentum + Orbital  (no GT delta)

        Total rollout loss = mean over all K step losses.
        """
        K = self.rollout_scheduler.get_rollout_steps(epoch)
        p_teacher = self.scheduled_sampler.get_probability(epoch)

        # ----------------------------------------------------------------
        # Step 1: Teacher-forced (ground-truth current state S_0 from batch)
        # ----------------------------------------------------------------
        state_raw_k, dt_raw = self.denormalizer.denormalize_state(inputs)
        state_raw_initial = state_raw_k.clone()

        # Extract raw elapsed_seconds for rollout state advancement
        el_stats = self.denormalizer.input_stats.get("elapsed_seconds", {"mean": 0.0, "std": 1.0})
        elapsed_raw_k = inputs[:, 0] * float(el_stats.get("std", 1.0)) + float(el_stats["mean"])

        # Forward pass — step 1
        outputs_1 = self.model(inputs[:, :self.model.input_dim])
        outputs_raw_1 = self.denormalizer.denormalize_output(outputs_1)
        targets_raw = self.denormalizer.denormalize_target(targets)

        pred_delta_raw_1 = outputs_raw_1[:, :6]
        pred_res_raw_1 = outputs_raw_1[:, 6:9]
        true_delta_raw = targets_raw[:, :6]

        # Physics transition and true residual for step 1
        phys_delta_raw_1 = self.physics_engine.compute_transition(state_raw_k, dt_raw)
        _, _, a_res_true_1 = self.physics_engine.compute_residuals(
            state_raw_k, true_delta_raw[:, 3:6], dt_raw
        )
        _, penalties_1 = self.physics_engine.validate_prediction(state_raw_k, pred_res_raw_1)

        # Ground truth next state is the reference target for conservation
        state_true_ref = state_raw_initial + true_delta_raw

        # Initialize lists to collect step-by-step losses for averaging
        physics_losses = []
        residual_losses = []
        constraint_losses = []
        energy_losses = []
        angular_momentum_losses = []
        orbital_losses = []
        sma_losses = []
        ecc_losses = []
        inc_losses = []
        step_pos_errors = []

        # Step 1 computations
        physics_loss_1 = torch.mean((pred_delta_raw_1 - phys_delta_raw_1) ** 2)
        residual_loss_1 = torch.mean((pred_res_raw_1 - a_res_true_1) ** 2)
        constraint_loss_1 = penalties_1["total_constraint_penalty"]
        if not isinstance(constraint_loss_1, torch.Tensor):
            constraint_loss_1 = torch.tensor(float(constraint_loss_1), dtype=torch.float32, device=device)

        physics_losses.append(physics_loss_1)
        residual_losses.append(residual_loss_1)
        constraint_losses.append(constraint_loss_1)

        # Next predicted state for step 1
        state_pred_raw_next = state_raw_k + pred_delta_raw_1

        # Invariant & element losses for step 1
        energy_l = self.energy_loss_fn(state_pred_raw_next, state_true_ref)
        ang_mom_l = self.angular_momentum_loss_fn(state_pred_raw_next, state_true_ref)
        orb_l, sma_l, ecc_l, inc_l = self.orbital_loss_fn(state_pred_raw_next, state_true_ref)

        energy_losses.append(energy_l)
        angular_momentum_losses.append(ang_mom_l)
        orbital_losses.append(orb_l)
        sma_losses.append(sma_l)
        ecc_losses.append(ecc_l)
        inc_losses.append(inc_l)

        # Step 1 full total loss (using loss_fn)
        loss_dict_1 = self.loss_fn(
            pred_delta_norm=outputs_1[:, :6],
            true_delta_norm=targets[:, :6],
            pred_delta_raw=pred_delta_raw_1,
            true_delta_raw=true_delta_raw,
            phys_delta_raw=phys_delta_raw_1,
            pred_res_raw=pred_res_raw_1,
            true_res_raw=a_res_true_1,
            constraint_penalty_raw=constraint_loss_1,
            energy_loss=energy_l,
            angular_momentum_loss=ang_mom_l,
            orbital_loss=orb_l,
        )
        step_losses: List[torch.Tensor] = [loss_dict_1["total_physics_loss"]]

        # Scheduled sampling decision for the next input state
        if p_teacher > 0.0:
            mask = (torch.rand(state_raw_k.shape[0], 1, device=device) < p_teacher).float()
            state_raw_k = mask * state_true_ref + (1.0 - mask) * state_pred_raw_next
        else:
            state_raw_k = state_pred_raw_next

        elapsed_raw_k = elapsed_raw_k + dt_raw

        # ----------------------------------------------------------------
        # Steps 2..K: Autonomous rollout (no ground-truth ΔState)
        # ----------------------------------------------------------------
        for _k in range(2, K + 1):
            # Re-normalize rolled-out state → (N, 8) network input
            inputs_k = self.denormalizer.normalize_state(state_raw_k, elapsed_raw_k, dt_raw)

            # Forward pass — step k
            outputs_k = self.model(inputs_k[:, :self.model.input_dim])
            outputs_raw_k = self.denormalizer.denormalize_output(outputs_k)

            pred_delta_raw_k = outputs_raw_k[:, :6]
            pred_res_raw_k = outputs_raw_k[:, 6:9]

            # Physics transition at predicted state
            phys_delta_raw_k = self.physics_engine.compute_transition(state_raw_k, dt_raw)

            # Self-consistency residual: residual implied by the network's own velocity prediction
            _, _, a_res_implied_k = self.physics_engine.compute_residuals(
                state_raw_k, pred_delta_raw_k[:, 3:6], dt_raw
            )

            # Constraint penalties at predicted state
            _, penalties_k = self.physics_engine.validate_prediction(state_raw_k, pred_res_raw_k)
            const_penalty_k = penalties_k["total_constraint_penalty"]
            if not isinstance(const_penalty_k, torch.Tensor):
                const_penalty_k = torch.tensor(
                    float(const_penalty_k), dtype=torch.float32, device=device
                )

            physics_loss_k = torch.mean((pred_delta_raw_k - phys_delta_raw_k) ** 2)
            residual_loss_k = torch.mean((pred_res_raw_k - a_res_implied_k) ** 2)

            physics_losses.append(physics_loss_k)
            residual_losses.append(residual_loss_k)
            constraint_losses.append(const_penalty_k)

            # Predicted next state
            state_pred_raw_next = state_raw_k + pred_delta_raw_k

            # Invariant & element losses for step k
            energy_l = self.energy_loss_fn(state_pred_raw_next, state_true_ref)
            ang_mom_l = self.angular_momentum_loss_fn(state_pred_raw_next, state_true_ref)
            orb_l, sma_l, ecc_l, inc_l = self.orbital_loss_fn(state_pred_raw_next, state_true_ref)

            energy_losses.append(energy_l)
            angular_momentum_losses.append(ang_mom_l)
            orbital_losses.append(orb_l)
            sma_losses.append(sma_l)
            ecc_losses.append(ecc_l)
            inc_losses.append(inc_l)

            # Rollout step loss: Physics + Residual + Constraint + Energy + Angular Momentum + Orbital (no Transition Loss)
            step_loss_k = (
                self.loss_fn.w_phys * physics_loss_k
                + self.loss_fn.w_resid * residual_loss_k
                + self.loss_fn.w_const * const_penalty_k
                + self.loss_fn.w_energy * energy_l
                + self.loss_fn.w_ang_mom * ang_mom_l
                + self.loss_fn.w_orb * orb_l
            )
            step_losses.append(step_loss_k)

            # Position error vs. physics prediction (diagnostic only)
            with torch.no_grad():
                pos_err = float(
                    torch.mean(
                        torch.norm(pred_delta_raw_k[:, :3] - phys_delta_raw_k[:, :3], dim=1)
                    ).item()
                )
            step_pos_errors.append(pos_err)

            # Integrate: S_k = S_{k-1} + ΔS_pred_{k-1}
            state_raw_k = state_pred_raw_next
            elapsed_raw_k = elapsed_raw_k + dt_raw

        # ----------------------------------------------------------------
        # Aggregate: total rollout loss = mean over all K step losses
        # ----------------------------------------------------------------
        total_rollout_loss = torch.stack(step_losses).mean()

        diagnostics: Dict[str, Any] = {
            "K": K,
            "rollout_loss": float(total_rollout_loss.item()),
            "one_step_loss": float(step_losses[0].item()),
            "final_step_loss": float(step_losses[-1].item()),
            "mean_rollout_pos_error": float(np.mean(step_pos_errors)) if step_pos_errors else 0.0,
            "max_rollout_pos_error": float(np.max(step_pos_errors)) if step_pos_errors else 0.0,
            # Step-1 loss component breakdown for regular history tracking
            "trans_loss": float(loss_dict_1["transition_loss"].item()),
            "phys_loss": float(torch.stack(physics_losses).mean().item()),
            "res_loss": float(torch.stack(residual_losses).mean().item()),
            "const_loss": float(torch.stack(constraint_losses).mean().item()),
            "rad_penalty": float(penalties_1["radius_penalty"]),
            "vel_penalty": float(penalties_1["velocity_penalty"]),
            "eng_penalty": float(penalties_1["energy_penalty"]),
            "mom_penalty": float(penalties_1["momentum_penalty"]),
            # New custom loss metrics
            "energy_loss": float(torch.stack(energy_losses).mean().item()),
            "angular_momentum_loss": float(torch.stack(angular_momentum_losses).mean().item()),
            "orbital_loss": float(torch.stack(orbital_losses).mean().item()),
            "sma_loss": float(torch.stack(sma_losses).mean().item()),
            "ecc_loss": float(torch.stack(ecc_losses).mean().item()),
            "inc_loss": float(torch.stack(inc_losses).mean().item()),
            "p_teacher": p_teacher,
        }

        return total_rollout_loss, diagnostics

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, train_loader: Any, val_loader: Any, device: str = "cpu") -> Dict[str, Any]:
        """
        Runs the full training loop with optional multi-step rollout.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to train the model.")

        self.calibrate_losses(train_loader, device=device)

        rollout_enabled = self.rollout_scheduler.is_enabled()
        log.info(
            "Starting training loop on device: %s for %d epochs | Rollout enabled: %s",
            device, self.epochs, rollout_enabled
        )
        self.model.to(device)

        best_val_loss = float("inf")

        # Output paths for per-epoch CSV logging
        sat_dir = (
            Path(self.cfg["_root"])
            / self.cfg["paths"]["data"]["processed"]["base"]
            / self.cfg["satellites"][self.cfg.get("pistn", {}).get("satellite", "INSAT3D")]["processed_dir"]
        )
        training_out_dir = sat_dir / "training"
        training_out_dir.mkdir(parents=True, exist_ok=True)

        loss_breakdown_csv = training_out_dir / "training_loss_breakdown.csv"
        epoch_metrics_csv = training_out_dir / "epoch_metrics.csv"

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.perf_counter()
            self.model.train()

            # Accumulators for epoch-level means
            epoch_train_loss = 0.0
            epoch_trans_loss = 0.0
            epoch_phys_loss = 0.0
            epoch_res_loss = 0.0
            epoch_const_loss = 0.0
            epoch_rad_penalty = 0.0
            epoch_vel_penalty = 0.0
            epoch_eng_penalty = 0.0
            epoch_mom_penalty = 0.0
            epoch_energy_loss = 0.0
            epoch_angular_momentum_loss = 0.0
            epoch_orbital_loss = 0.0
            epoch_sma_loss = 0.0
            epoch_ecc_loss = 0.0
            epoch_inc_loss = 0.0

            # Rollout diagnostics accumulators
            epoch_rollout_loss = 0.0
            epoch_one_step_loss = 0.0
            epoch_final_step_loss = 0.0
            epoch_mean_pos_error = 0.0
            epoch_max_pos_error = 0.0
            current_K = 1
            current_tf_prob = 1.0

            batches_count = 0

            for batch in train_loader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch
                else:
                    inputs = batch["inputs"]
                    targets = batch["targets"]

                inputs = inputs.to(device, dtype=torch.float32)
                targets = targets.to(device, dtype=torch.float32)

                self.optimizer.zero_grad()

                # Multi-step rollout training step (K=1 is equivalent to original one-step)
                loss, diag = self._rollout_train_step(inputs, targets, epoch, device)

                loss.backward()

                if self.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.optimizer.step()

                # Accumulate
                epoch_train_loss += float(loss.item())
                epoch_trans_loss += diag["trans_loss"]
                epoch_phys_loss += diag["phys_loss"]
                epoch_res_loss += diag["res_loss"]
                epoch_const_loss += diag["const_loss"]
                epoch_rad_penalty += diag["rad_penalty"]
                epoch_vel_penalty += diag["vel_penalty"]
                epoch_eng_penalty += diag["eng_penalty"]
                epoch_mom_penalty += diag["mom_penalty"]
                epoch_energy_loss += diag["energy_loss"]
                epoch_angular_momentum_loss += diag["angular_momentum_loss"]
                epoch_orbital_loss += diag["orbital_loss"]
                epoch_sma_loss += diag["sma_loss"]
                epoch_ecc_loss += diag["ecc_loss"]
                epoch_inc_loss += diag["inc_loss"]

                epoch_rollout_loss += diag["rollout_loss"]
                epoch_one_step_loss += diag["one_step_loss"]
                epoch_final_step_loss += diag["final_step_loss"]
                epoch_mean_pos_error += diag["mean_rollout_pos_error"]
                epoch_max_pos_error = max(epoch_max_pos_error, diag["max_rollout_pos_error"])
                current_K = diag["K"]
                current_tf_prob = diag["p_teacher"]

                batches_count += 1

            # Compute epoch-level means
            n = batches_count
            mean_train_loss = epoch_train_loss / n
            mean_trans_loss = epoch_trans_loss / n
            mean_phys_loss = epoch_phys_loss / n
            mean_res_loss = epoch_res_loss / n
            mean_const_loss = epoch_const_loss / n
            mean_rad = epoch_rad_penalty / n
            mean_vel = epoch_vel_penalty / n
            mean_eng = epoch_eng_penalty / n
            mean_mom = epoch_mom_penalty / n
            mean_energy = epoch_energy_loss / n
            mean_ang_mom = epoch_angular_momentum_loss / n
            mean_orbital = epoch_orbital_loss / n
            mean_sma = epoch_sma_loss / n
            mean_ecc = epoch_ecc_loss / n
            mean_inc = epoch_inc_loss / n

            # Loss percentage contributions (from step-1 breakdown)
            total_unweighted = mean_trans_loss + mean_phys_loss + mean_res_loss + mean_const_loss + mean_energy + mean_ang_mom + mean_orbital
            if total_unweighted > 0.0:
                p_trans = (mean_trans_loss / total_unweighted) * 100.0
                p_phys = (mean_phys_loss / total_unweighted) * 100.0
                p_res = (mean_res_loss / total_unweighted) * 100.0
                p_const = (mean_const_loss / total_unweighted) * 100.0
            else:
                p_trans = p_phys = p_res = p_const = 25.0

            # Gradient norm
            grad_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    grad_norm += float(p.grad.data.norm(2).item() ** 2)
            grad_norm = float(np.sqrt(grad_norm))

            # Weight norm
            weight_norm = 0.0
            for p in self.model.parameters():
                weight_norm += float(p.data.norm(2).item() ** 2)
            weight_norm = float(np.sqrt(weight_norm))

            # Validation
            val_loss, val_metrics = self.evaluator.evaluate(val_loader, device=device)

            # Scheduler step
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Record history
            self.history["train_loss"].append(mean_train_loss)
            self.history["train_transition_loss"].append(mean_trans_loss)
            self.history["train_physics_loss"].append(mean_phys_loss)
            self.history["train_residual_loss"].append(mean_res_loss)
            self.history["train_constraint_loss"].append(mean_const_loss)
            self.history["train_energy_loss"].append(mean_energy)
            self.history["train_angular_momentum_loss"].append(mean_ang_mom)
            self.history["train_orbital_loss"].append(mean_orbital)
            self.history["train_sma_loss"].append(mean_sma)
            self.history["train_ecc_loss"].append(mean_ecc)
            self.history["train_inc_loss"].append(mean_inc)

            self.history["val_loss"].append(val_loss)
            self.history["val_transition_loss"].append(val_metrics.get("val_transition_loss", 0.0))
            self.history["val_physics_loss"].append(val_metrics.get("val_physics_loss", 0.0))
            self.history["val_residual_loss"].append(val_metrics.get("val_residual_loss", 0.0))
            self.history["val_constraint_loss"].append(val_metrics.get("val_constraint_loss", 0.0))
            self.history["val_energy_loss"].append(val_metrics.get("val_energy_loss", 0.0))
            self.history["val_angular_momentum_loss"].append(val_metrics.get("val_angular_momentum_loss", 0.0))
            self.history["val_orbital_loss"].append(val_metrics.get("val_orbital_loss", 0.0))
            self.history["learning_rate"].append(current_lr)

            self.history["gradient_norm"].append(grad_norm)
            self.history["weight_norm"].append(weight_norm)
            self.history["transition_percent"].append(p_trans)
            self.history["physics_percent"].append(p_phys)
            self.history["residual_percent"].append(p_res)
            self.history["constraint_percent"].append(p_const)

            self.history["radius_penalty"].append(mean_rad)
            self.history["velocity_penalty"].append(mean_vel)
            self.history["energy_penalty"].append(mean_eng)
            self.history["momentum_penalty"].append(mean_mom)

            # Rollout diagnostics
            self.history["rollout_steps"].append(float(current_K))
            self.history["rollout_loss"].append(mean_rollout_loss)
            self.history["one_step_loss"].append(mean_one_step_loss)
            self.history["final_step_loss"].append(mean_final_step_loss)
            self.history["mean_rollout_pos_error"].append(mean_pos_error)
            self.history["max_rollout_pos_error"].append(epoch_max_pos_error)
            self.history["teacher_forcing_prob"].append(current_tf_prob)

            # ---- Per-epoch CSV logging ----

            breakdown_dict = {
                "epoch": [epoch],
                "rollout_k": [current_K],
                "transition_loss": [mean_trans_loss],
                "physics_loss": [mean_phys_loss],
                "residual_loss": [mean_res_loss],
                "constraint_loss": [mean_const_loss],
                "energy_loss": [mean_energy],
                "angular_momentum_loss": [mean_ang_mom],
                "orbital_loss": [mean_orbital],
                "sma_loss": [mean_sma],
                "ecc_loss": [mean_ecc],
                "inc_loss": [mean_inc],
                "total_loss": [mean_train_loss],
                "rollout_loss": [mean_rollout_loss],
                "one_step_loss": [mean_one_step_loss],
                "final_step_loss": [mean_final_step_loss],
                "mean_rollout_pos_error": [mean_pos_error],
                "max_rollout_pos_error": [epoch_max_pos_error],
                "gradient_norm": [grad_norm],
                "weight_norm": [weight_norm],
                "transition_percent": [p_trans],
                "physics_percent": [p_phys],
                "residual_percent": [p_res],
                "constraint_percent": [p_const],
                "learning_rate": [current_lr],
                "teacher_forcing_prob": [current_tf_prob],
            }
            df_breakdown = pd.DataFrame(breakdown_dict)
            if not loss_breakdown_csv.is_file():
                df_breakdown.to_csv(loss_breakdown_csv, index=False)
            else:
                df_breakdown.to_csv(loss_breakdown_csv, mode="a", header=False, index=False)

            metrics_dict = {
                "epoch": [epoch],
                "val_loss": [val_loss],
                "val_transition_loss": [val_metrics.get("val_transition_loss", 0.0)],
                "val_physics_loss": [val_metrics.get("val_physics_loss", 0.0)],
                "val_residual_loss": [val_metrics.get("val_residual_loss", 0.0)],
                "val_constraint_loss": [val_metrics.get("val_constraint_loss", 0.0)],
                "val_energy_loss": [val_metrics.get("val_energy_loss", 0.0)],
                "val_angular_momentum_loss": [val_metrics.get("val_angular_momentum_loss", 0.0)],
                "val_orbital_loss": [val_metrics.get("val_orbital_loss", 0.0)],
                "learning_rate": [current_lr],
                "mae_delta_x": [val_metrics.get("mae_delta_x", 0.0)],
                "mae_delta_y": [val_metrics.get("mae_delta_y", 0.0)],
                "mae_delta_z": [val_metrics.get("mae_delta_z", 0.0)],
                "mae_delta_vx": [val_metrics.get("mae_delta_vx", 0.0)],
                "mae_delta_vy": [val_metrics.get("mae_delta_vy", 0.0)],
                "mae_delta_vz": [val_metrics.get("mae_delta_vz", 0.0)],
                "mae_ax_res": [val_metrics.get("mae_ax_res", 0.0)],
                "mae_ay_res": [val_metrics.get("mae_ay_res", 0.0)],
                "mae_az_res": [val_metrics.get("mae_az_res", 0.0)],
                # Rollout columns
                "rollout_k": [current_K],
                "radius_penalty": [mean_rad],
                "velocity_penalty": [mean_vel],
                "energy_penalty": [mean_eng],
                "momentum_penalty": [mean_mom],
                "teacher_forcing_prob": [current_tf_prob],
            }
            df_metrics = pd.DataFrame(metrics_dict)
            if not epoch_metrics_csv.is_file():
                df_metrics.to_csv(epoch_metrics_csv, index=False)
            else:
                df_metrics.to_csv(epoch_metrics_csv, mode="a", header=False, index=False)

            # ---- Progress log ----
            epoch_time = time.perf_counter() - epoch_start
            if epoch == 1 or epoch % 10 == 0 or epoch == self.epochs:
                log.info(
                    "Epoch %3d/%3d | K=%2d | TF Prob: %.2f | Time: %.2fs | RolloutLoss: %.4e "
                    "| 1-step: %.4e | Val: %.4e | Energy: %.2e | AngMom: %.2e | Orb: %.2e",
                    epoch, self.epochs, current_K, current_tf_prob, epoch_time,
                    mean_rollout_loss, mean_one_step_loss, val_loss, mean_energy, mean_ang_mom, mean_orbital
                )

            # Checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                history=self.history,
                config=self.cfg.get("pistn", {}),
                is_best=is_best,
            )

            # Early stopping (monitors val_loss, restores best weights)
            if self.early_stopping(val_loss, self.model):
                log.info(
                    "Early stopping triggered at epoch %d. Best val loss: %.4e",
                    epoch, best_val_loss
                )
                break

        log.info("Training complete. Best validation loss: %.4e", best_val_loss)
        return self.history
