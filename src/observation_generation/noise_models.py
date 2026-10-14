"""
src/observation_generation/noise_models.py
===========================================
Noise Injection Models.

Implements modular and vectorized noise models: Gaussian, Bias, and Random Walk.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)


class NoiseModelFactory:
    """Injects various noise models into Right Ascension and Declination measurements."""

    @staticmethod
    def inject_noise(
        ra_deg: np.ndarray,
        dec_deg: np.ndarray,
        cfg: dict[str, Any],
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Inject enabled noise models vectorially.

        Parameters
        ----------
        ra_deg : np.ndarray, shape (N,)
            True Right Ascension in degrees.
        dec_deg : np.ndarray, shape (N,)
            True Declination in degrees.
        cfg : dict[str, Any]
            The `observation_generation` configuration block.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        ra_noisy : np.ndarray, shape (N,)
            Noisy Right Ascension in degrees, mapped back to [0, 360).
        dec_noisy : np.ndarray, shape (N,)
            Noisy Declination in degrees, clipped to [-90, 90].
        noise_ra : np.ndarray, shape (N,)
            Delta RA noise injected.
        noise_dec : np.ndarray, shape (N,)
            Delta Dec noise injected.
        """
        n = len(ra_deg)
        noise_ra = np.zeros(n, dtype=np.float64)
        noise_dec = np.zeros(n, dtype=np.float64)

        noise_cfg = cfg.get("noise", {})
        rng = np.random.default_rng(seed)

        # ── Model 1: Gaussian Noise ────────────────────────────────────
        g_cfg = noise_cfg.get("gaussian", {})
        if g_cfg.get("enabled", True):
            ra_std = float(g_cfg.get("ra_std_deg", 0.005))
            dec_std = float(g_cfg.get("dec_std_deg", 0.005))
            
            noise_ra += rng.normal(loc=0.0, scale=ra_std, size=n)
            noise_dec += rng.normal(loc=0.0, scale=dec_std, size=n)
            log.debug("Gaussian noise applied: RA_std=%.4f°, Dec_std=%.4f°", ra_std, dec_std)

        # ── Model 2: Pointing Bias ─────────────────────────────────────
        b_cfg = noise_cfg.get("bias", {})
        if b_cfg.get("enabled", False):
            ra_bias = float(b_cfg.get("ra_bias_deg", 0.0))
            dec_bias = float(b_cfg.get("dec_bias_deg", 0.0))
            
            noise_ra += ra_bias
            noise_dec += dec_bias
            log.debug("Pointing bias applied: RA_bias=%.4f°, Dec_bias=%.4f°", ra_bias, dec_bias)

        # ── Model 3: Random Walk ───────────────────────────────────────
        rw_cfg = noise_cfg.get("random_walk", {})
        if rw_cfg.get("enabled", False):
            rw_std = float(rw_cfg.get("std_deg", 0.0001))
            
            # Cumulative sum represents random walk drift
            noise_ra += np.cumsum(rng.normal(loc=0.0, scale=rw_std, size=n))
            noise_dec += np.cumsum(rng.normal(loc=0.0, scale=rw_std, size=n))
            log.debug("Random walk drift applied: std=%.5f°/step", rw_std)

        # Apply noise to true observations
        ra_noisy = (ra_deg + noise_ra) % 360.0
        dec_noisy = np.clip(dec_deg + noise_dec, -90.0, 90.0)

        return ra_noisy, dec_noisy, noise_ra, noise_dec
class ConstantPointingBias:
    """Helper representing a constant sensor pointing offset."""
    pass
class RandomWalkDrift:
    """Helper representing sensor drift over time."""
    pass
