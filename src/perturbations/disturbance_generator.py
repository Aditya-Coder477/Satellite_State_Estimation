"""
src/perturbations/disturbance_generator.py
============================================
PerturbationOrchestrator — Combines all perturbation models into the
total residual acceleration.

The orchestrator:
  1. Instantiates models from config (or accepts pre-built models)
  2. Calls each model's ``compute(times, positions)``
  3. Sums contributions: a_res = a_SRP + a_maneuver + a_random + a_bias
  4. Returns per-model breakdown and total residual array

This design makes it trivial to add a new model: subclass BasePerturbation,
register it in ``_build_from_config``, done.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.perturbations.perturbation_models import BasePerturbation, ConstantBias
from src.perturbations.solar_radiation_pressure import SolarRadiationPressure
from src.perturbations.maneuver_model import StationKeepingManeuver
from src.perturbations.random_disturbance import RandomDisturbance
from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


@dataclass
class PerturbationResult:
    """
    Container for all perturbation acceleration arrays.

    All arrays have shape (N, 3) and units km/s².
    """
    times: np.ndarray                                   # (N,) elapsed seconds
    acc_srp: np.ndarray                                 # (N, 3)
    acc_maneuver: np.ndarray                            # (N, 3)
    acc_random: np.ndarray                              # (N, 3)
    acc_bias: np.ndarray                                # (N, 3)
    acc_total: np.ndarray                               # (N, 3)  = sum of above
    model_names: list[str] = field(default_factory=list)

    @property
    def perturbation_magnitude(self) -> np.ndarray:
        """Scalar magnitude of total residual acceleration [km/s²]. Shape (N,)."""
        return np.linalg.norm(self.acc_total, axis=-1)


class PerturbationOrchestrator:
    """
    Combines multiple perturbation models and returns the total residual acceleration.

    Parameters
    ----------
    models : list[BasePerturbation] | None
        Explicit list of pre-built model instances. If None, models are built
        from ``cfg`` via :meth:`from_config`.
    cfg : dict[str, Any] | None
        Project configuration. Used when ``models=None``.
    epoch_jd : float | None
        Julian date of the trajectory epoch (used by SRP for Sun direction).

    Examples
    --------
    >>> orch = PerturbationOrchestrator.from_config(cfg, epoch_jd=2461217.7)
    >>> result = orch.compute(times, positions)
    >>> result.acc_total   # shape (N, 3), km/s²
    """

    def __init__(
        self,
        models: list[BasePerturbation] | None = None,
        cfg: dict[str, Any] | None = None,
        epoch_jd: float | None = None,
    ) -> None:
        if models is not None:
            self._models = models
        elif cfg is not None:
            self._models = self._build_from_config(cfg, epoch_jd)
        else:
            raise ValueError("Either 'models' or 'cfg' must be provided.")

        enabled_names = [m.name for m in self._models if m.enabled]
        disabled_names = [m.name for m in self._models if not m.enabled]
        log.info("PerturbationOrchestrator — enabled : %s", enabled_names)
        if disabled_names:
            log.info("PerturbationOrchestrator — disabled: %s", disabled_names)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        times: np.ndarray,
        positions: np.ndarray,
    ) -> PerturbationResult:
        """
        Compute total residual acceleration at all time steps.

        Parameters
        ----------
        times : np.ndarray, shape (N,)
            Elapsed seconds from epoch.
        positions : np.ndarray, shape (N, 3)
            TEME position vectors [km].

        Returns
        -------
        PerturbationResult
            Named arrays for each model and total.
        """
        n = len(times)
        log.info("Computing perturbations for %d samples...", n)

        acc_srp = self._get_model_acc("solar_radiation_pressure", times, positions)
        acc_man = self._get_model_acc("maneuver", times, positions)
        acc_rnd = self._get_model_acc("random_disturbance", times, positions)
        acc_bias = self._get_model_acc("constant_bias", times, positions)

        acc_total = acc_srp + acc_man + acc_rnd + acc_bias

        result = PerturbationResult(
            times=times,
            acc_srp=acc_srp,
            acc_maneuver=acc_man,
            acc_random=acc_rnd,
            acc_bias=acc_bias,
            acc_total=acc_total,
            model_names=[m.name for m in self._models],
        )

        mag = result.perturbation_magnitude
        log.info(
            "Total residual: mean=%.3e, max=%.3e, min=%.3e km/s²",
            float(mag.mean()), float(mag.max()), float(mag.min()),
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_model_acc(
        self, name: str, times: np.ndarray, positions: np.ndarray
    ) -> np.ndarray:
        """Find a model by name and compute its acceleration. Returns zeros if not found."""
        n = len(times)
        for model in self._models:
            if model.name == name:
                return model.compute(times, positions)
        return np.zeros((n, 3), dtype=np.float64)

    @staticmethod
    def _build_from_config(
        cfg: dict[str, Any], epoch_jd: float | None
    ) -> list[BasePerturbation]:
        """Build model list from config.yaml ``perturbation`` block."""
        return [
            SolarRadiationPressure.from_config(cfg, epoch_jd=epoch_jd),
            StationKeepingManeuver.from_config(cfg),
            RandomDisturbance.from_config(cfg),
            ConstantBias.from_config(cfg),
        ]

    @classmethod
    def from_config(
        cls, cfg: dict[str, Any], epoch_jd: float | None = None
    ) -> "PerturbationOrchestrator":
        """Construct orchestrator entirely from config.yaml."""
        return cls(cfg=cfg, epoch_jd=epoch_jd)
