"""
src/physics/physics_engine.py
=============================
Unified public API for Phase 8 (Physics Engine).

Exposes the ``PhysicsEngine`` orchestrator class. All external modules (such as
the future PI-STN network/trainer or running script) interact exclusively through
this class to evaluate physics equations, residuals, and constraints.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

from src.physics.constraints import ConstraintEvaluator
from src.physics.gravity_models import GravityModelFactory
from src.physics.loss_components import PhysicsLossEvaluator
from src.physics.state_transition import StateTransitionModel
from src.physics.residual_dynamics import PhysicsResidualCalculator
from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class PhysicsEngine:
    """
    Orchestrates Earth orbit physics calculations, constraints, and PI-STN training loss metrics.
    Operates purely on NumPy arrays and PyTorch tensors.

    Parameters
    ----------
    cfg : Dict[str, Any]
        Project configuration dictionary.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self._cfg = cfg
        self._physics_cfg = cfg.get("physics", {})

        # 1. Resolve Gravity Model Strategy
        model_name = self._physics_cfg.get("gravity_model", "two_body")
        self.gravity_model = GravityModelFactory.create(model_name)
        log.info("PhysicsEngine instantiated with gravity model: '%s'", self.gravity_model.name)

        # 2. Resolve Integrator
        integrator_name = self._physics_cfg.get("integrator", "rk4")
        self.transition_model = StateTransitionModel(self.gravity_model, integrator_name)

        # 3. Instantiate Helpers
        self.residual_calculator = PhysicsResidualCalculator(self.gravity_model)
        self.constraint_evaluator = ConstraintEvaluator(cfg)
        self.loss_evaluator = PhysicsLossEvaluator(cfg)

    def compute_gravity(self, position: Any) -> Any:
        """
        Computes gravitational acceleration at a given position.
        """
        return self.gravity_model.compute_acceleration(position)

    def compute_transition(self, state: Any, dt: Any) -> Any:
        """
        Computes numerical state transitions (expected ΔState) over dt.
        """
        return self.transition_model.compute_transition(state, dt)

    def compute_residuals(self, state: Any, delta_v: Any, delta_t: Any) -> Tuple[Any, Any, Any]:
        """
        Calculates governing transition, physics, and residual dynamics accelerations.
        """
        return self.residual_calculator.compute_residuals(state, delta_v, delta_t)

    def validate_prediction(self, state: Any, acc: Any = None) -> Tuple[Dict[str, bool], Dict[str, Any]]:
        """
        Evaluates physical constraints and returns status and penalties.
        """
        return self.constraint_evaluator.evaluate(state, acc)

    def compute_loss(
        self,
        pred_delta: Any,
        true_delta: Any,
        phys_delta: Any,
        pred_res: Any,
        true_res: Any,
        constraint_penalty: Any = 0.0,
    ) -> Dict[str, Any]:
        """
        Computes transition, physics, residual, and constraint loss components.
        """
        return self.loss_evaluator.compute_loss(
            pred_delta=pred_delta,
            true_delta=true_delta,
            phys_delta=phys_delta,
            pred_res=pred_res,
            true_res=true_res,
            constraint_penalty=constraint_penalty,
        )
