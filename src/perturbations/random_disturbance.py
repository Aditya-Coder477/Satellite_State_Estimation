"""
src/perturbations/random_disturbance.py
=========================================
Model 3 — Gaussian Random Disturbance.

Represents unmodeled dynamics such as:
  - Thruster noise and micro-vibrations
  - Thermal recoil effects
  - Atmospheric neutral wind (at GEO altitudes, negligible but non-zero)
  - Sensor noise in attitude control feedback loops

Each component (ax, ay, az) is drawn independently from a Gaussian
distribution N(mean, std) using a configurable seed for reproducibility.

All accelerations are in km/s².
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.perturbations.perturbation_models import BasePerturbation
from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class RandomDisturbance(BasePerturbation):
    """
    Zero-mean (or biased) Gaussian random acceleration disturbance.

    Parameters
    ----------
    mean : float
        Mean of the Gaussian distribution [km/s²]. Default: 0.0.
    std : float
        Standard deviation of the Gaussian distribution [km/s²]. Default: 5e-10.
    seed : int
        Random seed for reproducibility. Default: 42.
    enabled : bool
        Whether this model is active.

    Notes
    -----
    The same seed always produces the same random sequence for a given
    number of samples. This ensures experiment reproducibility.

    Examples
    --------
    >>> rd = RandomDisturbance(mean=0.0, std=5e-10, seed=42)
    >>> acc = rd.compute(times, positions)   # shape (N, 3), km/s²
    """

    def __init__(
        self,
        mean: float = 0.0,
        std: float = 5e-10,
        seed: int = 42,
        enabled: bool = True,
    ) -> None:
        super().__init__(name="random_disturbance", enabled=enabled)
        self._mean = float(mean)
        self._std = float(std)
        self._seed = int(seed)

        log.debug(
            "RandomDisturbance initialised: mean=%.2e, std=%.2e km/s², seed=%d",
            mean, std, seed,
        )

    def compute(self, times: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """
        Generate reproducible Gaussian random accelerations.

        Parameters
        ----------
        times : np.ndarray, shape (N,)
            Elapsed seconds from epoch (used only for length).
        positions : np.ndarray, shape (N, 3)
            TEME positions [km] (not used, retained for API consistency).

        Returns
        -------
        np.ndarray, shape (N, 3)
            Random acceleration [km/s²].
        """
        n = len(times)
        if not self.enabled:
            return self._zeros(n)

        rng = np.random.default_rng(self._seed)
        acc = rng.normal(loc=self._mean, scale=self._std, size=(n, 3))

        mag = float(np.mean(np.linalg.norm(acc, axis=-1)))
        log.debug(
            "RandomDisturbance: %d samples | mean_mag=%.3e km/s²", n, mag
        )
        return acc

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "RandomDisturbance":
        """Construct from the ``perturbation.random`` config block."""
        rand_cfg = cfg.get("perturbation", {}).get("random", {})
        models_cfg = cfg.get("perturbation", {}).get("models", {})
        enabled = models_cfg.get("random", True)
        return cls(
            mean=float(rand_cfg.get("mean", 0.0)),
            std=float(rand_cfg.get("std", 5e-10)),
            seed=int(rand_cfg.get("seed", 42)),
            enabled=enabled,
        )
