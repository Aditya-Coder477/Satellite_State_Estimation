"""
src/perturbations/maneuver_model.py
=====================================
Model 2 — Station-Keeping Maneuver.

Simulates a small continuous thrust event during a configurable time window.
The maneuver produces a constant acceleration vector in the TEME frame during
``[start_time_seconds, start_time_seconds + duration_seconds]``.

This represents realistic GEO North-South or East-West station-keeping burns.

All accelerations are in km/s².
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.perturbations.perturbation_models import BasePerturbation
from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class StationKeepingManeuver(BasePerturbation):
    """
    Station-keeping maneuver model: constant thrust during a time window.

    Parameters
    ----------
    start_time_seconds : float
        Maneuver start time [s] from epoch.
    duration_seconds : float
        Maneuver duration [s].
    magnitude_km_s2 : float
        Thrust acceleration magnitude [km/s²].
    direction : list[float] | None
        3-element unit vector [dx, dy, dz] in TEME. Normalised internally.
        Default: [1.0, 0.0, 0.0] (X direction).
    enabled : bool
        Whether this model is active.

    Examples
    --------
    >>> m = StationKeepingManeuver(start_time_seconds=259200,
    ...                            duration_seconds=1200,
    ...                            magnitude_km_s2=1e-8)
    >>> acc = m.compute(times, positions)   # shape (N, 3), km/s²
    """

    def __init__(
        self,
        start_time_seconds: float = 259200.0,
        duration_seconds: float = 1200.0,
        magnitude_km_s2: float = 1e-8,
        direction: list[float] | None = None,
        enabled: bool = True,
    ) -> None:
        super().__init__(name="maneuver", enabled=enabled)
        self._t_start = float(start_time_seconds)
        self._t_end = self._t_start + float(duration_seconds)
        self._magnitude = float(magnitude_km_s2)

        # Normalise direction vector
        dir_vec = np.array(direction if direction is not None else [1.0, 0.0, 0.0], dtype=np.float64)
        norm = np.linalg.norm(dir_vec)
        if norm < 1e-12:
            log.warning("Maneuver direction vector has zero magnitude, defaulting to [1,0,0].")
            dir_vec = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            dir_vec = dir_vec / norm

        self._thrust_acc = self._magnitude * dir_vec    # [km/s²]

        log.info(
            "StationKeepingManeuver: t=[%.0f, %.0f] s, mag=%.2e km/s², dir=%s",
            self._t_start, self._t_end, self._magnitude, dir_vec.tolist(),
        )

    def compute(self, times: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """
        Compute maneuver acceleration.

        Returns ``thrust_acc`` during the maneuver window, zeros elsewhere.

        Parameters
        ----------
        times : np.ndarray, shape (N,)
            Elapsed seconds from epoch.
        positions : np.ndarray, shape (N, 3)
            TEME positions [km] (not used, retained for API consistency).

        Returns
        -------
        np.ndarray, shape (N, 3)
            Acceleration [km/s²].
        """
        n = len(times)
        acc = self._zeros(n)

        if not self.enabled:
            return acc

        # Active samples: start <= t <= end
        active = (times >= self._t_start) & (times <= self._t_end)
        n_active = int(active.sum())

        if n_active > 0:
            acc[active] = self._thrust_acc
            log.debug(
                "Maneuver: %d active samples (t=[%.0f, %.0f] s) | acc=%.2e km/s²",
                n_active, self._t_start, self._t_end, self._magnitude,
            )

        return acc

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "StationKeepingManeuver":
        """Construct from the ``perturbation.maneuver`` config block."""
        man_cfg = cfg.get("perturbation", {}).get("maneuver", {})
        models_cfg = cfg.get("perturbation", {}).get("models", {})
        enabled = models_cfg.get("maneuver", True)
        return cls(
            start_time_seconds=float(man_cfg.get("start_time_seconds", 259200.0)),
            duration_seconds=float(man_cfg.get("duration_seconds", 1200.0)),
            magnitude_km_s2=float(man_cfg.get("magnitude_km_s2", 1e-8)),
            direction=man_cfg.get("direction", [1.0, 0.0, 0.0]),
            enabled=enabled,
        )
