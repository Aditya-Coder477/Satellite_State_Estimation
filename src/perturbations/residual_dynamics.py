"""
src/perturbations/residual_dynamics.py
========================================
ResidualDynamicsIntegrator — Applies cumulative residual acceleration
to the nominal orbit to produce the disturbed trajectory.

Integration Method
------------------
The disturbed orbit is computed by cumulatively integrating the residual
acceleration using the trapezoidal rule:

    Δv(t_k) = Δv(t_{k-1}) + 0.5 * (a_res(t_{k-1}) + a_res(t_k)) * dt
    Δr(t_k) = Δr(t_{k-1}) + 0.5 * (Δv(t_{k-1}) + Δv(t_k)) * dt

Then:
    r_disturbed(t) = r_nominal(t) + Δr(t)
    v_disturbed(t) = v_nominal(t) + Δv(t)

This is physically correct for small perturbations and avoids the
re-propagation of a perturbed SGP4 state, keeping Phase 3 independent
of Phase 2's propagation engine.

The trapezoidal rule is preferred over simple Euler because it:
  - Is second-order accurate
  - Conserves energy better for oscillatory disturbances
  - Remains stable for the small perturbation magnitudes here
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.perturbations.disturbance_generator import PerturbationResult
from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)

# Column names used in truth_states.csv from Phase 2
_POS_COLS = ["x_teme_km", "y_teme_km", "z_teme_km"]
_VEL_COLS = ["vx_teme_kms", "vy_teme_kms", "vz_teme_kms"]


class ResidualDynamicsIntegrator:
    """
    Applies cumulative residual acceleration to produce the disturbed orbit.

    Parameters
    ----------
    step_seconds : float
        Propagation step size [s]. Default: 60.0.

    Examples
    --------
    >>> integrator = ResidualDynamicsIntegrator(step_seconds=60.0)
    >>> df_disturbed = integrator.apply(df_nominal, perturbation_result)
    """

    def __init__(self, step_seconds: float = 60.0) -> None:
        self._dt = float(step_seconds)
        log.debug("ResidualDynamicsIntegrator: dt=%.1f s", self._dt)

    def apply(
        self,
        df_nominal: pd.DataFrame,
        result: PerturbationResult,
    ) -> pd.DataFrame:
        """
        Create the disturbed trajectory DataFrame.

        Parameters
        ----------
        df_nominal : pd.DataFrame
            Nominal orbit from Phase 2 ``truth_states.csv``.
            Must contain: utc_time, elapsed_seconds, satellite_name, norad_id,
            x_teme_km, y_teme_km, z_teme_km, vx_teme_kms, vy_teme_kms, vz_teme_kms.
        result : PerturbationResult
            Output from :class:`~src.perturbations.disturbance_generator.PerturbationOrchestrator`.

        Returns
        -------
        pd.DataFrame
            Disturbed trajectory with additional columns:
            ax_res, ay_res, az_res, perturbation_magnitude.
            The position/velocity column names match Phase 2.
        """
        n = len(df_nominal)
        log.info("Integrating residual dynamics over %d samples (dt=%.1f s)...", n, self._dt)

        # Extract nominal state arrays
        r_nom = df_nominal[_POS_COLS].to_numpy(dtype=np.float64)   # (N, 3) km
        v_nom = df_nominal[_VEL_COLS].to_numpy(dtype=np.float64)   # (N, 3) km/s
        a_res = result.acc_total                                     # (N, 3) km/s²

        # Cumulative trapezoidal integration
        delta_v, delta_r = self._trapezoidal_cumsum(a_res, self._dt)

        # Apply perturbations
        r_disturbed = r_nom + delta_r   # (N, 3)
        v_disturbed = v_nom + delta_v   # (N, 3)

        # Log statistics
        dr_mag = np.linalg.norm(delta_r, axis=-1)
        dv_mag = np.linalg.norm(delta_v, axis=-1)
        log.info(
            "Position perturbation: max=%.4f km, mean=%.4f km",
            float(dr_mag.max()), float(dr_mag.mean()),
        )
        log.info(
            "Velocity perturbation: max=%.4e km/s, mean=%.4e km/s",
            float(dv_mag.max()), float(dv_mag.mean()),
        )

        # Assemble disturbed DataFrame
        df_dist = pd.DataFrame({
            "utc_time": df_nominal["utc_time"].values,
            "elapsed_seconds": df_nominal["elapsed_seconds"].values,
            "satellite_name": df_nominal["satellite_name"].values,
            "norad_id": df_nominal["norad_id"].values,
            # Phase 2 style (TEME) columns
            "x_teme_km": r_disturbed[:, 0],
            "y_teme_km": r_disturbed[:, 1],
            "z_teme_km": r_disturbed[:, 2],
            "vx_teme_kms": v_disturbed[:, 0],
            "vy_teme_kms": v_disturbed[:, 1],
            "vz_teme_kms": v_disturbed[:, 2],
            # Phase 3 style (km, km/s) columns
            "x_km": r_disturbed[:, 0],
            "y_km": r_disturbed[:, 1],
            "z_km": r_disturbed[:, 2],
            "vx_km_s": v_disturbed[:, 0],
            "vy_km_s": v_disturbed[:, 1],
            "vz_km_s": v_disturbed[:, 2],
            "ax_res": a_res[:, 0],
            "ay_res": a_res[:, 1],
            "az_res": a_res[:, 2],
            "perturbation_magnitude": result.perturbation_magnitude,
        })

        log.info("Disturbed orbit generated: %d rows", len(df_dist))
        return df_dist

    # ------------------------------------------------------------------
    # Numerical integration
    # ------------------------------------------------------------------

    @staticmethod
    def _trapezoidal_cumsum(
        acc: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute cumulative Δv and Δr from residual acceleration using the
        trapezoidal rule.

        Parameters
        ----------
        acc : np.ndarray, shape (N, 3)
            Residual acceleration [km/s²].
        dt : float
            Time step [s].

        Returns
        -------
        delta_v : np.ndarray, shape (N, 3)
            Cumulative velocity perturbation [km/s].
        delta_r : np.ndarray, shape (N, 3)
            Cumulative position perturbation [km].
        """
        n = acc.shape[0]
        delta_v = np.zeros_like(acc)
        delta_r = np.zeros_like(acc)

        for k in range(1, n):
            # Trapezoidal velocity update
            delta_v[k] = delta_v[k - 1] + 0.5 * (acc[k - 1] + acc[k]) * dt
            # Trapezoidal position update
            delta_r[k] = delta_r[k - 1] + 0.5 * (delta_v[k - 1] + delta_v[k]) * dt

        return delta_v, delta_r
