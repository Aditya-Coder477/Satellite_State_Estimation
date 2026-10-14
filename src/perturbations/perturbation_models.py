"""
src/perturbations/perturbation_models.py
=========================================
Base class for all perturbation models and the ConstantBias implementation.

Design
------
All perturbation models inherit from ``BasePerturbation`` and implement a single
method ``compute(times, positions)``.  This strategy pattern allows the
``PerturbationOrchestrator`` to sum contributions from any number of models
without knowing their implementation details.

New perturbation models (e.g. J2 mismatch, eclipse thermal, atmospheric drag)
can be added by subclassing ``BasePerturbation`` without modifying any existing
module.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class BasePerturbation(ABC):
    """
    Abstract base class for all perturbation models.

    Subclasses must implement :meth:`compute`.

    Parameters
    ----------
    name : str
        Human-readable model identifier used in logs and outputs.
    enabled : bool
        When ``False``, :meth:`compute` returns a zero array without
        executing any computation.
    """

    def __init__(self, name: str, enabled: bool = True) -> None:
        self.name = name
        self.enabled = enabled

    @abstractmethod
    def compute(
        self,
        times: np.ndarray,
        positions: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the acceleration contribution of this perturbation.

        Parameters
        ----------
        times : np.ndarray, shape (N,)
            Elapsed seconds from epoch for each sample.
        positions : np.ndarray, shape (N, 3)
            TEME position vectors [km] at each sample.

        Returns
        -------
        np.ndarray, shape (N, 3)
            Acceleration [km/s²] at each sample: ``[ax, ay, az]``.
        """

    def _zeros(self, n: int) -> np.ndarray:
        """Return an (N, 3) zero array."""
        return np.zeros((n, 3), dtype=np.float64)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, enabled={self.enabled})"


# ---------------------------------------------------------------------------
# Model 4 — Constant Bias
# ---------------------------------------------------------------------------

class ConstantBias(BasePerturbation):
    """
    Constant acceleration bias in TEME frame.

    Represents a systematic sensor offset or unmodeled constant force.
    The acceleration is constant for all time steps.

    Parameters
    ----------
    ax : float
        X-component of the bias acceleration [km/s²].
    ay : float
        Y-component of the bias acceleration [km/s²].
    az : float
        Z-component of the bias acceleration [km/s²].
    enabled : bool
        Whether this model is active.

    Examples
    --------
    >>> bias = ConstantBias(ax=2e-10, ay=1e-10, az=0.0)
    >>> acc = bias.compute(times, positions)   # shape (N, 3)
    """

    def __init__(
        self,
        ax: float = 0.0,
        ay: float = 0.0,
        az: float = 0.0,
        enabled: bool = True,
    ) -> None:
        super().__init__(name="constant_bias", enabled=enabled)
        self._bias = np.array([ax, ay, az], dtype=np.float64)
        log.debug(
            "ConstantBias initialised: ax=%.2e, ay=%.2e, az=%.2e km/s²",
            ax, ay, az,
        )

    def compute(self, times: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Return constant bias acceleration for every sample."""
        n = len(times)
        if not self.enabled:
            return self._zeros(n)

        acc = np.tile(self._bias, (n, 1))
        log.debug("ConstantBias: computed %d samples | bias_mag=%.2e km/s²", n, float(np.linalg.norm(self._bias)))
        return acc

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "ConstantBias":
        """Construct from the ``perturbation.bias`` config block."""
        bias_cfg = cfg.get("perturbation", {}).get("bias", {})
        models_cfg = cfg.get("perturbation", {}).get("models", {})
        enabled = models_cfg.get("constant_bias", True)
        return cls(
            ax=float(bias_cfg.get("ax", 2e-10)),
            ay=float(bias_cfg.get("ay", 1e-10)),
            az=float(bias_cfg.get("az", 0.0)),
            enabled=enabled,
        )
