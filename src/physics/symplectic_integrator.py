"""
src/physics/symplectic_integrator.py
====================================
Implements symplectic numerical integrators (Leapfrog and Velocity Verlet)
for Hamiltonian systems, specifically satellite orbital dynamics under gravity.
Supports both PyTorch and NumPy.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# PyTorch import handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.physics.coordinate_dynamics import is_tensor


class LeapfrogIntegrator:
    """
    Leapfrog numerical integrator (symplectic).
    Uses position-Verlet (drift-kick-drift) form:
        q_{1/2} = q_0 + 0.5 * dt * v_0
        a_{1/2} = acceleration_function(q_{1/2})
        v_1     = v_0 + dt * a_{1/2}
        q_1     = q_{1/2} + 0.5 * dt * v_1
    """

    def integrate(self, state: Any, dt: Any, acceleration_function: Any) -> Any:
        """
        Integrates the Cartesian state vector forward by dt using Leapfrog.

        Parameters
        ----------
        state : Any
            Cartesian state vector [x, y, z, vx, vy, vz] of shape (..., 6).
            Supports PyTorch tensors and NumPy arrays.
        dt : Any
            Step time [seconds] of shape (...) or float.
        acceleration_function : Callable
            A function mapping position (..., 3) to acceleration (..., 3).

        Returns
        -------
        Any
            Next state vector [x, y, z, vx, vy, vz] of shape (..., 6) after dt.
        """
        q = state[..., :3]
        v = state[..., 3:6]

        # Align dt shape
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

        q_half = q + 0.5 * dt_aligned * v
        acc_half = acceleration_function(q_half)
        v_next = v + dt_aligned * acc_half
        q_next = q_half + 0.5 * dt_aligned * v_next

        if is_tensor(state):
            return torch.cat([q_next, v_next], dim=-1)
        else:
            return np.concatenate([q_next, v_next], axis=-1)


class VelocityVerletIntegrator:
    """
    Velocity Verlet numerical integrator (symplectic).
    Uses kick-drift-kick form:
        v_{1/2} = v_0 + 0.5 * dt * acc(q_0)
        q_1     = q_0 + dt * v_{1/2}
        v_1     = v_{1/2} + 0.5 * dt * acc(q_1)
    """

    def integrate(self, state: Any, dt: Any, acceleration_function: Any) -> Any:
        """
        Integrates the Cartesian state vector forward by dt using Velocity Verlet.

        Parameters
        ----------
        state : Any
            Cartesian state vector [x, y, z, vx, vy, vz] of shape (..., 6).
            Supports PyTorch tensors and NumPy arrays.
        dt : Any
            Step time [seconds] of shape (...) or float.
        acceleration_function : Callable
            A function mapping position (..., 3) to acceleration (..., 3).

        Returns
        -------
        Any
            Next state vector [x, y, z, vx, vy, vz] of shape (..., 6) after dt.
        """
        q = state[..., :3]
        v = state[..., 3:6]

        # Align dt shape
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

        acc_0 = acceleration_function(q)
        v_half = v + 0.5 * dt_aligned * acc_0
        q_next = q + dt_aligned * v_half
        acc_next = acceleration_function(q_next)
        v_next = v_half + 0.5 * dt_aligned * acc_next

        if is_tensor(state):
            return torch.cat([q_next, v_next], dim=-1)
        else:
            return np.concatenate([q_next, v_next], axis=-1)
