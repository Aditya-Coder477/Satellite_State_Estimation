"""
src/physics/state_transition.py
===============================
Implements physics-based state transitions using Euler, RK4, Leapfrog, and Velocity Verlet integrators.
Exposes APIs to integrate states over a time interval delta_t, returning the predicted
physics-based state change (expected ΔState). Supports both PyTorch and NumPy.
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

from src.physics.coordinate_dynamics import is_tensor
from src.physics.gravity_models import GravityModel
from src.physics.symplectic_integrator import LeapfrogIntegrator, VelocityVerletIntegrator
from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class EulerIntegrator:
    """Euler numerical integrator."""

    def integrate(self, state: Any, dt: Any, acceleration_function: Any) -> Any:
        q = state[..., :3]
        v = state[..., 3:6]

        if isinstance(dt, (np.ndarray, torch.Tensor if TORCH_AVAILABLE else Any)):
            if len(dt.shape) < len(state.shape):
                if is_tensor(dt):
                    dt_aligned = dt.unsqueeze(-1)
                else:
                    dt_aligned = np.expand_dims(dt, axis=-1)
            else:
                dt_aligned = dt
        else:
            dt_aligned = dt

        acc = acceleration_function(q)
        q_next = q + dt_aligned * v
        v_next = v + dt_aligned * acc

        if is_tensor(state):
            return torch.cat([q_next, v_next], dim=-1)
        else:
            return np.concatenate([q_next, v_next], axis=-1)


class RK4Integrator:
    """Classical 4th-order Runge-Kutta numerical integrator."""

    def integrate(self, state: Any, dt: Any, acceleration_function: Any) -> Any:
        q = state[..., :3]
        v = state[..., 3:6]

        if isinstance(dt, (np.ndarray, torch.Tensor if TORCH_AVAILABLE else Any)):
            if len(dt.shape) < len(state.shape):
                if is_tensor(dt):
                    dt_aligned = dt.unsqueeze(-1)
                else:
                    dt_aligned = np.expand_dims(dt, axis=-1)
            else:
                dt_aligned = dt
        else:
            dt_aligned = dt

        # k1
        k1_q = v
        k1_v = acceleration_function(q)

        # k2
        q_k2 = q + 0.5 * dt_aligned * k1_q
        k2_q = v + 0.5 * dt_aligned * k1_v
        k2_v = acceleration_function(q_k2)

        # k3
        q_k3 = q + 0.5 * dt_aligned * k2_q
        k3_q = v + 0.5 * dt_aligned * k2_v
        k3_v = acceleration_function(q_k3)

        # k4
        q_k4 = q + dt_aligned * k3_q
        k4_q = v + dt_aligned * k3_v
        k4_v = acceleration_function(q_k4)

        q_next = q + (dt_aligned / 6.0) * (k1_q + 2.0 * k2_q + 2.0 * k3_q + k4_q)
        v_next = v + (dt_aligned / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

        if is_tensor(state):
            return torch.cat([q_next, v_next], dim=-1)
        else:
            return np.concatenate([q_next, v_next], axis=-1)


class IntegratorFactory:
    """
    Factory to construct numerical integrators.
    """
    _INTEGRATORS = {
        "euler": EulerIntegrator,
        "rk4": RK4Integrator,
        "leapfrog": LeapfrogIntegrator,
        "velocity_verlet": VelocityVerletIntegrator,
        "velocity-verlet": VelocityVerletIntegrator,
    }

    @classmethod
    def create(cls, name: str) -> Any:
        name_lower = name.lower().replace("_", "-")
        integrator_class = cls._INTEGRATORS.get(name_lower)
        if integrator_class is None:
            # Try with underscore too
            name_under = name.lower().replace("-", "_")
            integrator_class = cls._INTEGRATORS.get(name_under)
        if integrator_class is None:
            raise ValueError(
                f"Unknown integrator '{name}'. Supported: {list(cls._INTEGRATORS.keys())}"
            )
        return integrator_class()


class StateTransitionModel:
    """
    Computes numerical state transitions for Cartesian orbits using a selected integrator.
    """

    def __init__(self, gravity_model: GravityModel, integrator: Any = "rk4") -> None:
        self.gravity_model = gravity_model
        if isinstance(integrator, dict):
            self.integrator_name = integrator.get("type", "rk4")
        else:
            self.integrator_name = integrator
        # Keep self.integrator as a string for 100% backward compatibility
        self.integrator = self.integrator_name.lower()
        self.integrator_obj = IntegratorFactory.create(self.integrator_name)
        log.info("StateTransitionModel initialized with integrator: '%s'", self.integrator_name)

    def compute_transition(self, state: Any, dt: Any) -> Any:
        """
        Integrates the state vector forward by dt.

        Parameters
        ----------
        state : Any
            Cartesian state vector [x, y, z, vx, vy, vz] of shape (..., 6).
            Supports PyTorch tensors and NumPy arrays.
        dt : Any
            Step time [seconds] of shape (...) or float.

        Returns
        -------
        Any
            Expected ΔState [Δx, Δy, Δz, Δvx, Δvy, Δvz] of shape (..., 6) over dt.
        """
        acc_fn = lambda q: self.gravity_model.compute_acceleration(q)
        next_state = self.integrator_obj.integrate(state, dt, acc_fn)
        return next_state - state
