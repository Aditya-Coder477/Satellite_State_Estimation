"""
src/losses/orbital_losses.py
============================
Differentiable conservation and orbital element losses for the PI-STN model.

Implements:
1. Specific Mechanical Energy Conservation Loss (L_energy)
2. Specific Angular Momentum Conservation Loss (L_angular)
3. Keplerian Orbital Element Loss (L_orbital: a, e, i)

All calculations are fully differentiable, batch-compatible, and GPU-efficient.
Uses safe operations to prevent NaN gradients (e.g. at zero eccentricity/inclination).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from src.physics.constants import MU as MU_EARTH_KM3S2

log: logging.Logger = logging.getLogger(__name__)


class EnergyConservationLoss(nn.Module):
    """
    Computes specific mechanical energy conservation loss:
        epsilon = v^2 / 2 - mu / r
        L_energy = MSE(epsilon_pred, epsilon_true)
    """

    def __init__(self, mu: float = MU_EARTH_KM3S2) -> None:
        super().__init__()
        self.mu = mu
        self.mse = nn.MSELoss()

    def compute_energy(self, state: torch.Tensor) -> torch.Tensor:
        """
        Computes specific mechanical energy for a batch of states.
        state: (B, 6) - [x, y, z, vx, vy, vz] in km and km/s.
        """
        pos = state[:, 0:3]
        vel = state[:, 3:6]
        
        r = torch.norm(pos, dim=-1).clamp(min=1e-6)
        v_sq = torch.sum(vel ** 2, dim=-1)
        
        return 0.5 * v_sq - self.mu / r

    def forward(self, state_pred: torch.Tensor, state_true: torch.Tensor) -> torch.Tensor:
        energy_pred = self.compute_energy(state_pred)
        energy_true = self.compute_energy(state_true)
        return self.mse(energy_pred, energy_true)


class AngularMomentumLoss(nn.Module):
    """
    Computes specific angular momentum vector conservation loss:
        h = r x v
        L_h = MSE(h_pred, h_true)
    Enforces vector alignment (magnitude + direction) to preserve the orbital plane.
    """

    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss()

    def compute_h(self, state: torch.Tensor) -> torch.Tensor:
        """
        Computes angular momentum vector: h = r x v.
        state: (B, 6) - [x, y, z, vx, vy, vz]
        """
        pos = state[:, 0:3]
        vel = state[:, 3:6]
        return torch.cross(pos, vel, dim=-1)

    def forward(self, state_pred: torch.Tensor, state_true: torch.Tensor) -> torch.Tensor:
        h_pred = self.compute_h(state_pred)
        h_true = self.compute_h(state_true)
        return self.mse(h_pred, h_true)


class OrbitalElementLoss(nn.Module):
    """
    Computes differentiable Keplerian orbital element losses (semi-major axis, eccentricity, inclination).
    Only utilizes these three elements.
    """

    def __init__(
        self,
        mu: float = MU_EARTH_KM3S2,
        w_a: float = 1.0,
        w_e: float = 1.0,
        w_i: float = 1.0,
    ) -> None:
        super().__init__()
        self.mu = mu
        self.w_a = w_a
        self.w_e = w_e
        self.w_i = w_i
        self.mse = nn.MSELoss()

    def compute_elements(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Differentiable Cartesian to Keplerian conversion (a, e, i only).
        Uses safe gradients to prevent NaN values.
        """
        pos = state[:, 0:3]
        vel = state[:, 3:6]

        r = torch.norm(pos, dim=-1).clamp(min=1e-6)
        v_sq = torch.sum(vel ** 2, dim=-1)
        v = torch.sqrt(v_sq).clamp(min=1e-8)

        # 1. Semi-major axis (a) via Specific Mechanical Energy
        energy = 0.5 * v_sq - self.mu / r
        # GEO orbits are bound (energy < 0). Clamp to prevent division by zero or positive a.
        energy_safe = torch.clamp(energy, max=-1e-4)
        a = -self.mu / (2.0 * energy_safe)

        # 2. Eccentricity (e)
        r_dot_v = torch.sum(pos * vel, dim=-1, keepdim=True)
        # e_vec = ((v^2 - mu/r)*r - (r.v)*v) / mu
        e_vec = ((v_sq.unsqueeze(-1) - self.mu / r.unsqueeze(-1)) * pos - r_dot_v * vel) / self.mu
        # Safe norm to prevent NaN gradient at e = 0
        e = torch.sqrt(torch.sum(e_vec ** 2, dim=-1) + 1e-12)

        # 3. Inclination (i)
        h_vec = torch.cross(pos, vel, dim=-1)
        h_norm = torch.sqrt(torch.sum(h_vec ** 2, dim=-1) + 1e-12)
        
        cos_i = h_vec[:, 2] / h_norm
        cos_i_safe = torch.clamp(cos_i, min=-0.999999, max=0.999999)
        i = torch.acos(cos_i_safe)

        return a, e, i

    def forward(
        self, state_pred: torch.Tensor, state_true: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes individual and weighted Keplerian element losses.
        Returns (total_loss, a_loss, e_loss, i_loss).
        """
        a_pred, e_pred, i_pred = self.compute_elements(state_pred)
        a_true, e_true, i_true = self.compute_elements(state_true)

        # Compute MSE
        a_loss = self.mse(a_pred, a_true)
        e_loss = self.mse(e_pred, e_true)
        i_loss = self.mse(i_pred, i_true)

        # Normalize elements internal scale comparison
        a_true_mean = torch.mean(torch.abs(a_true)).clamp(min=1.0)
        e_true_mean = torch.mean(torch.abs(e_true)).clamp(min=1e-6)
        i_true_mean = torch.mean(torch.abs(i_true)).clamp(min=1e-4)

        # Normalized MSE
        a_loss_norm = a_loss / (a_true_mean ** 2)
        e_loss_norm = e_loss / (e_true_mean ** 2)
        i_loss_norm = i_loss / (max(float(i_true_mean.item()), 1e-3) ** 2)

        total_loss = (
            self.w_a * a_loss_norm +
            self.w_e * e_loss_norm +
            self.w_i * i_loss_norm
        )

        return total_loss, a_loss, e_loss, i_loss
