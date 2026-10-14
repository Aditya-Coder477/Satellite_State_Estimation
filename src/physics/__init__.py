"""
src/physics/__init__.py
=======================
Phase 8: Physics Engine.

Exposes physical constants, governing orbital dynamics equations, state transition model,
residual dynamics calculator, physical constraints evaluator, loss evaluators, and validation utilities.
"""

from __future__ import annotations

from src.physics.autograd_utils import AutogradUtils
from src.physics.constants import G0, J2, J4, MU, OMEGA_E, R_E, reload_constants
from src.physics.constraints import ConstraintEvaluator
from src.physics.coordinate_dynamics import (
    angular_momentum,
    compute_cross_product,
    compute_dot_product,
    compute_norm,
    compute_relative_position,
    compute_relative_velocity,
    compute_unit_vector,
    cross,
    dot,
    get_position,
    get_velocity,
    is_tensor,
    norm,
    orbital_energy,
    specific_mechanical_energy,
    unit_vector,
)
from src.physics.gravity_models import (
    GravityModel,
    GravityModelFactory,
    J2Gravity,
    J4Gravity,
    ManeuverGravity,
    SRPGravity,
    TwoBodyGravity,
)
from src.physics.loss_components import PhysicsLossEvaluator
from src.physics.orbital_equations import compute_state_derivative
from src.physics.physics_engine import PhysicsEngine
from src.physics.residual_dynamics import PhysicsResidualCalculator
from src.physics.symplectic_integrator import LeapfrogIntegrator, VelocityVerletIntegrator
from src.physics.state_transition import StateTransitionModel, IntegratorFactory
from src.physics.validator import PhysicsValidator
from src.physics.statistics import DatasetStatistics
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
    "compute_cross_product",
    "angular_momentum",
    "specific_mechanical_energy",
    "norm",
    "unit_vector",
    "dot",
    "cross",
    "orbital_energy",
    "GravityModel",
    "TwoBodyGravity",
    "J2Gravity",
    "J4Gravity",
    "SRPGravity",
    "ManeuverGravity",
    "GravityModelFactory",
    "compute_state_derivative",
    "StateTransitionModel",
    "LeapfrogIntegrator",
    "VelocityVerletIntegrator",
    "IntegratorFactory",
    "PhysicsResidualCalculator",
    "ConstraintEvaluator",
    "AutogradUtils",
    "PhysicsLossEvaluator",
    "PhysicsEngine",
    "PhysicsValidator",
    "DatasetStatistics",
    "PhysicsVisualizer",
]
