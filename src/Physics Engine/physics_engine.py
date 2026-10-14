"""
src/physics/physics_engine.py
=============================
Unified public API for Phase 7 (Physics Engine).

Exposes the ``PhysicsEngine`` orchestrator class. All external modules (such as
the future PINN trainer) interact exclusively through this class to evaluate
derivatives, residuals, physical constraints, and loss components.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

from src.physics.constraints import ConstraintEvaluator
from src.physics.gravity_models import GravityModelFactory
from src.physics.loss_components import PhysicsLossEvaluator
from src.physics.orbital_equations import compute_state_derivative
from src.physics.residuals import PhysicsResidualCalculator
from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class PhysicsEngine:
    """
    Orchestrates Earth orbit physics calculations and PINN training components.

    Serves as the main gateway interface to the physics sub-system.

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

        # 2. Instantiate Helpers
        self.residual_calculator = PhysicsResidualCalculator(self.gravity_model)
        self.constraint_evaluator = ConstraintEvaluator(cfg)
        self.loss_evaluator = PhysicsLossEvaluator(cfg)

    def compute_gravity(self, position: Any) -> Any:
        """
        Computes gravitational acceleration at a given position.

        Parameters
        ----------
        position : Any
            Position vector [x, y, z] of shape (..., 3) (km).

        Returns
        -------
        Any
            Gravitational acceleration vector [ax, ay, az] (km/s^2).
        """
        return self.gravity_model.compute_acceleration(position)

    def compute_derivatives(self, state: Any) -> Any:
        """
        Computes analytical state derivatives based on orbital dynamics.

        Parameters
        ----------
        state : Any
            Cartesian state vector [x, y, z, vx, vy, vz] of shape (..., 6).

        Returns
        -------
        Any
            Derivative vector [vx, vy, vz, ax, ay, az] of shape (..., 6).
        """
        return compute_state_derivative(state, self.gravity_model)

    def compute_residuals(self, state_pred: Any, state_deriv_pred: Any) -> Any:
        """
        Calculates governing physics residuals (position and velocity discrepancies).

        Parameters
        ----------
        state_pred : Any
            Predicted state vector [x, y, z, vx, vy, vz] of shape (..., 6).
        state_deriv_pred : Any
            Predicted time derivative of state, computed via autograd (..., 6).

        Returns
        -------
        Any
            Physics residual tensor of shape (..., 6).
        """
        return self.residual_calculator.compute_residuals(state_pred, state_deriv_pred)

    def validate_prediction(self, state_pred: Any, acc_pred: Any = None) -> Tuple[Dict[str, bool], Dict[str, Any]]:
        """
        Evaluates predictions against physical constraints.

        Parameters
        ----------
        state_pred : Any
            Predicted state vector [x, y, z, vx, vy, vz] of shape (..., 6).
        acc_pred : Any, optional
            Predicted acceleration vector of shape (..., 3), by default None.

        Returns
        -------
        Tuple[Dict[str, bool], Dict[str, Any]]
            1. Dictionary of constraint satisfaction boolean flags.
            2. Dictionary of differentiable penalty loss values.
        """
        return self.constraint_evaluator.evaluate(state_pred, acc_pred)

    def compute_loss(
        self,
        residuals: Any,
        boundary_loss: Any = 0.0,
        initial_loss: Any = 0.0,
        constraint_penalty: Any = 0.0,
    ) -> Dict[str, Any]:
        """
        Computes weighted physics-informed loss components.

        Parameters
        ----------
        residuals : Any
            Residual tensor of shape (..., 6).
        boundary_loss : Any, optional
            Precomputed boundary loss value, by default 0.0.
        initial_loss : Any, optional
            Precomputed initial condition loss value, by default 0.0.
        constraint_penalty : Any, optional
            Precomputed constraint penalty sum, by default 0.0.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing physics_loss, boundary_loss, initial_loss,
            constraint_loss, and total_physics_loss.
        """
        return self.loss_evaluator.compute_loss(
            residuals=residuals,
            boundary_loss=boundary_loss,
            initial_loss=initial_loss,
            constraint_penalty=constraint_penalty,
        )
