"""
src/physics/__init__.py
=======================
Phase 7: Physics Engine.

Exposes physical constants, governing orbital dynamics equations, autograd helpers,
physical constraints, and loss evaluators.

Public API
----------
- MU, R_E, J2, J4, OMEGA_E, G0 : Centralized physical constants
- reload_constants             : Dynamic constants loader
- compute_state_derivative     : Calculates state derivatives
- GravityModelFactory          : Instantiates gravity strategies
- AutogradUtils                : PyTorch batched automatic differentiation
- PhysicsEngine                : Public API orchestrator class
- PhysicsValidator             : Quality control checker
- PhysicsVisualizer            : Heads-up plotting utilities
"""

from __future__ import annotations

from src.physics.autograd_utils import AutogradUtils
from src.physics.constants import G0, J2, J4, MU, OMEGA_E, R_E, reload_constants
from src.physics.constraints import ConstraintEvaluator
from src.physics.coordinate_dynamics import (
    compute_dot_product,
    compute_norm,
    compute_relative_position,
    compute_relative_velocity,
    compute_unit_vector,
    get_position,
    get_velocity,
    is_tensor,
)
from src.physics.gravity_models import (
    GravityModel,
    GravityModelFactory,
    HighOrderGravity,
    J2Gravity,
    J4Gravity,
    TwoBodyGravity,
)
from src.physics.loss_components import PhysicsLossEvaluator
from src.physics.orbital_equations import compute_state_derivative
from src.physics.physics_engine import PhysicsEngine
from src.physics.residuals import PhysicsResidualCalculator
from src.physics.validator import PhysicsValidator
from src.physics.visualizer import PhysicsVisualizer

__all__ = [
    "MU",
    "R_E",
    "J2",
    "J4",
    "OMEGA_E",
    "G0",
    "reload_constants",
    "is_tensor",
    "get_position",
    "get_velocity",
    "compute_norm",
    "compute_unit_vector",
    "compute_relative_position",
    "compute_relative_velocity",
    "compute_dot_product",
    "compute_state_derivative",
    "GravityModel",
    "TwoBodyGravity",
    "J2Gravity",
    "J4Gravity",
    "HighOrderGravity",
    "GravityModelFactory",
    "PhysicsResidualCalculator",
    "ConstraintEvaluator",
    "AutogradUtils",
    "PhysicsLossEvaluator",
    "PhysicsEngine",
    "PhysicsValidator",
    "PhysicsVisualizer",
]
