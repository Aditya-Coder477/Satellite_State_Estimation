"""
src/evaluation/orbital_elements.py
=====================================
Classical orbital element conversion and drift analysis for Phase 9.

Converts GCRS Cartesian states [x, y, z, vx, vy, vz] to the classical
Keplerian orbital elements:
  a  — semi-major axis (km)
  e  — eccentricity
  i  — inclination (deg)
  Ω  — right ascension of ascending node (RAAN) (deg)
  ω  — argument of perigee (deg)
  ν  — true anomaly (deg)

All computations are pure NumPy.

Earth gravitational parameter:
  μ = 398600.4418 km³/s²
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np

from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)

# WGS-84 Earth gravitational parameter (km³/s²)
_MU_KM3_S2 = 398600.4418


class OrbitalElementsCalculator:
    """
    Converts Cartesian GCRS states to Keplerian orbital elements and
    computes orbital drift metrics over a prediction window.
    """

    def __init__(self) -> None:
        log.info("OrbitalElementsCalculator initialised.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def state_to_elements(self, state: np.ndarray) -> Dict[str, float]:
        """
        Convert a single Cartesian state to classical orbital elements.

        Parameters
        ----------
        state : np.ndarray, shape (6,)
            [x, y, z, vx, vy, vz] in km, km/s.

        Returns
        -------
        Dict[str, float]
            Keys: ``a_km``, ``e``, ``i_deg``, ``raan_deg``, ``argp_deg``, ``nu_deg``.
        """
        r_vec = state[:3]
        v_vec = state[3:]
        return self._rv_to_elements(r_vec, v_vec)

    def batch_to_elements(self, states: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Convert a batch of Cartesian states to orbital element arrays.

        Parameters
        ----------
        states : np.ndarray, shape (N, 6)

        Returns
        -------
        Dict[str, np.ndarray]
            Each key maps to an array of shape (N,).
        """
        n = len(states)
        results = {k: np.zeros(n) for k in ("a_km", "e", "i_deg", "raan_deg", "argp_deg", "nu_deg")}

        for idx, state in enumerate(states):
            try:
                elem = self._rv_to_elements(state[:3], state[3:])
                for k in results:
                    results[k][idx] = elem[k]
            except Exception:
                for k in results:
                    results[k][idx] = np.nan

        return results

    def compute_drift(
        self,
        predicted_states: np.ndarray,
        truth_states: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute orbital element drift between predicted and truth trajectories.

        Parameters
        ----------
        predicted_states : np.ndarray, shape (N, 6)
        truth_states : np.ndarray, shape (N, 6)

        Returns
        -------
        Dict[str, float]
            Mean absolute drift per orbital element.
        """
        pred_elems = self.batch_to_elements(predicted_states)
        true_elems = self.batch_to_elements(truth_states)

        drift: Dict[str, float] = {}

        for key in ("a_km", "e", "i_deg", "raan_deg", "argp_deg"):
            p = pred_elems[key]
            t = true_elems[key]
            # Mask NaN entries
            valid = np.isfinite(p) & np.isfinite(t)
            if valid.sum() == 0:
                drift[f"{key}_mae"] = float("nan")
                drift[f"{key}_rmse"] = float("nan")
                continue

            diff = p[valid] - t[valid]
            drift[f"{key}_mae"] = float(np.mean(np.abs(diff)))
            drift[f"{key}_rmse"] = float(np.sqrt(np.mean(diff ** 2)))

        log.info(
            "Orbital drift — a RMSE: %.4f km | e RMSE: %.6f | i RMSE: %.4f deg",
            drift.get("a_km_rmse", float("nan")),
            drift.get("e_rmse", float("nan")),
            drift.get("i_deg_rmse", float("nan")),
        )
        return drift

    # ------------------------------------------------------------------
    # Core Keplerian conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _rv_to_elements(r_vec: np.ndarray, v_vec: np.ndarray) -> Dict[str, float]:
        """
        Convert position/velocity vectors to classical orbital elements.

        Reference: Bate, Mueller, White — "Fundamentals of Astrodynamics"
        """
        mu = _MU_KM3_S2
        eps = 1e-12   # numerical guard

        r = np.linalg.norm(r_vec)
        v = np.linalg.norm(v_vec)

        # ── Specific angular momentum ────────────────────────────────
        h_vec = np.cross(r_vec, v_vec)
        h = np.linalg.norm(h_vec)

        # ── Node vector ──────────────────────────────────────────────
        K = np.array([0.0, 0.0, 1.0])
        n_vec = np.cross(K, h_vec)
        n = np.linalg.norm(n_vec)

        # ── Eccentricity vector ──────────────────────────────────────
        e_vec = ((v ** 2 - mu / r) * r_vec - np.dot(r_vec, v_vec) * v_vec) / mu
        e = np.linalg.norm(e_vec)

        # ── Specific mechanical energy ───────────────────────────────
        xi = v ** 2 / 2.0 - mu / r

        # ── Semi-major axis ──────────────────────────────────────────
        if abs(1.0 - e) < eps:
            a = float("inf")  # parabolic
        else:
            a = -mu / (2.0 * xi)

        # ── Inclination ──────────────────────────────────────────────
        i_rad = np.arccos(np.clip(h_vec[2] / (h + eps), -1.0, 1.0))

        # ── RAAN ─────────────────────────────────────────────────────
        if n < eps:
            raan_rad = 0.0
        else:
            raan_rad = np.arccos(np.clip(n_vec[0] / n, -1.0, 1.0))
            if n_vec[1] < 0:
                raan_rad = 2 * np.pi - raan_rad

        # ── Argument of perigee ──────────────────────────────────────
        if n < eps or e < eps:
            argp_rad = 0.0
        else:
            argp_rad = np.arccos(np.clip(np.dot(n_vec, e_vec) / (n * e), -1.0, 1.0))
            if e_vec[2] < 0:
                argp_rad = 2 * np.pi - argp_rad

        # ── True anomaly ─────────────────────────────────────────────
        if e < eps:
            nu_rad = 0.0
        else:
            nu_cos = np.dot(e_vec, r_vec) / (e * r + eps)
            nu_rad = np.arccos(np.clip(nu_cos, -1.0, 1.0))
            if np.dot(r_vec, v_vec) < 0:
                nu_rad = 2 * np.pi - nu_rad

        return {
            "a_km":     float(a),
            "e":        float(e),
            "i_deg":    float(np.degrees(i_rad)),
            "raan_deg": float(np.degrees(raan_rad)),
            "argp_deg": float(np.degrees(argp_rad)),
            "nu_deg":   float(np.degrees(nu_rad)),
        }
