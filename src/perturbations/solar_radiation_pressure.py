"""
src/perturbations/solar_radiation_pressure.py
===============================================
Model 1 — Solar Radiation Pressure (SRP).

SRP is the primary non-gravitational perturbation for GEO satellites.
The acceleration is computed as:

    a_SRP = -(P_SRP * C_R * A/m) * r̂_sun

where:
    P_SRP   = Solar radiation pressure constant [N/m²]
    C_R     = Reflectivity coefficient (1 = absorb, 2 = reflect)
    A/m     = Area-to-mass ratio [m²/kg]
    r̂_sun  = Unit vector from satellite toward the Sun (TEME frame)

Sun direction is computed using a simplified analytical solar longitude
formula (mean ecliptic + aberration approximation), accurate to within
~1° — sufficient for perturbation magnitudes in this application.

Unit conversions:
    P_SRP in N/m² = kg/(m·s²)
    A/m in m²/kg
    Combined: (N/m²) × (m²/kg) = m/s² = 1e-3 km/s²

All accelerations are returned in km/s².
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.perturbations.perturbation_models import BasePerturbation
from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)

# Solar radiation pressure constant at 1 AU [N/m²]
_P_SRP_DEFAULT = 4.56e-6   # N/m²

# Conversion: m/s² → km/s²
_M_S2_TO_KM_S2 = 1e-3


class SolarRadiationPressure(BasePerturbation):
    """
    Solar Radiation Pressure perturbation model for GEO satellites.

    Parameters
    ----------
    pressure_n_pa : float
        Solar radiation pressure constant [N/m²]. Default: 4.56e-6.
    reflectivity : float
        Reflectivity coefficient C_R (1=absorb, 1.3=partial, 2=mirror). Default: 1.3.
    area_mass_ratio : float
        Satellite area-to-mass ratio [m²/kg]. Default: 0.02.
    epoch_jd : float | None
        Epoch Julian date used to compute initial Sun longitude. If None,
        uses J2000 (JD 2451545.0).
    enabled : bool
        Whether this model is active.

    Examples
    --------
    >>> srp = SolarRadiationPressure(pressure_n_pa=4.56e-6, reflectivity=1.3,
    ...                               area_mass_ratio=0.02)
    >>> acc = srp.compute(times, positions)   # shape (N, 3), km/s²
    """

    def __init__(
        self,
        pressure_n_pa: float = _P_SRP_DEFAULT,
        reflectivity: float = 1.3,
        area_mass_ratio: float = 0.02,
        epoch_jd: float | None = None,
        enabled: bool = True,
    ) -> None:
        super().__init__(name="solar_radiation_pressure", enabled=enabled)
        self._pressure = pressure_n_pa       # N/m²
        self._cr = reflectivity
        self._am_ratio = area_mass_ratio     # m²/kg
        # Scale factor: (N/m²) × (m²/kg) → m/s² → km/s²
        self._scale = self._pressure * self._cr * self._am_ratio * _M_S2_TO_KM_S2
        self._epoch_jd = epoch_jd if epoch_jd is not None else 2451545.0  # J2000

        log.debug(
            "SRP initialised: P=%.3e N/m², C_R=%.2f, A/m=%.4f m²/kg → scale=%.3e km/s²",
            pressure_n_pa, reflectivity, area_mass_ratio, self._scale,
        )

    def compute(self, times: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """
        Compute SRP acceleration for all time steps.

        Parameters
        ----------
        times : np.ndarray, shape (N,)
            Elapsed seconds from epoch.
        positions : np.ndarray, shape (N, 3)
            TEME positions [km] (not used in simplified model, retained for API consistency).

        Returns
        -------
        np.ndarray, shape (N, 3)
            SRP acceleration [km/s²].
        """
        n = len(times)
        if not self.enabled:
            return self._zeros(n)

        sun_dirs = self._sun_unit_vectors(times)   # (N, 3)

        # a_SRP = -scale * r̂_sun (pushes satellite away from Sun)
        acc = -self._scale * sun_dirs              # (N, 3) km/s²

        mag = float(np.mean(np.linalg.norm(acc, axis=-1)))
        log.debug("SRP: computed %d samples | mean_acc=%.3e km/s²", n, mag)
        return acc

    # ------------------------------------------------------------------
    # Analytical Sun direction
    # ------------------------------------------------------------------

    def _sun_unit_vectors(self, times: np.ndarray) -> np.ndarray:
        """
        Compute unit vectors from Earth to Sun in TEME-approximate frame.

        Uses the mean solar longitude approximation (accurate to ~1°).

        Parameters
        ----------
        times : np.ndarray, shape (N,)
            Elapsed seconds from epoch.

        Returns
        -------
        np.ndarray, shape (N, 3)
            Unit vectors toward the Sun.
        """
        # Julian date at each time step
        jd = self._epoch_jd + times / 86400.0     # (N,)

        # Julian centuries from J2000
        T = (jd - 2451545.0) / 36525.0

        # Mean solar longitude [degrees]
        L0 = 280.46646 + 36000.76983 * T          # (N,)

        # Mean anomaly of the Sun [degrees]
        M = 357.52911 + 35999.05029 * T - 0.0001537 * T ** 2

        M_rad = np.radians(M)
        L0_rad = np.radians(L0)

        # Equation of the centre (first two terms)
        C = (1.914602 - 0.004817 * T - 0.000014 * T**2) * np.sin(M_rad)
        C += (0.019993 - 0.000101 * T) * np.sin(2 * M_rad)
        C += 0.000289 * np.sin(3 * M_rad)

        # Sun's true longitude [radians]
        sun_lon = L0_rad + np.radians(C)

        # Mean obliquity of ecliptic [radians] (IAU 2006, simplified)
        epsilon = np.radians(23.439291 - 0.013004 * T)

        # Sun direction in geocentric equatorial frame (≈ TEME for our purposes)
        sx = np.cos(sun_lon)
        sy = np.cos(epsilon) * np.sin(sun_lon)
        sz = np.sin(epsilon) * np.sin(sun_lon)

        sun_dirs = np.stack([sx, sy, sz], axis=-1)  # (N, 3)

        # Normalise (already unit vectors by construction, but guard against drift)
        norms = np.linalg.norm(sun_dirs, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return sun_dirs / norms

    @classmethod
    def from_config(cls, cfg: dict[str, Any], epoch_jd: float | None = None) -> "SolarRadiationPressure":
        """Construct from the ``perturbation.srp`` config block."""
        srp_cfg = cfg.get("perturbation", {}).get("srp", {})
        models_cfg = cfg.get("perturbation", {}).get("models", {})
        enabled = models_cfg.get("solar_radiation_pressure", True)
        return cls(
            pressure_n_pa=float(srp_cfg.get("pressure_n_pa", _P_SRP_DEFAULT)),
            reflectivity=float(srp_cfg.get("reflectivity", 1.3)),
            area_mass_ratio=float(srp_cfg.get("area_mass_ratio", 0.02)),
            epoch_jd=epoch_jd,
            enabled=enabled,
        )
