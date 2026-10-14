"""
src/evaluation/INSAT3D/orbit_analysis.py
========================================
Performs orbital elements conversion and evaluates orbital parameter differences.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple
import numpy as np

from src.evaluation.orbital_elements import OrbitalElementsCalculator
from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class KeplerianElementsAnalyzer:
    """
    Converts Cartesian states to Keplerian elements and evaluates deviations.
    """

    def __init__(self) -> None:
        self.calculator = OrbitalElementsCalculator()

    def _angle_diff(self, pred: np.ndarray, true: np.ndarray) -> np.ndarray:
        """Computes shortest angular difference in degrees (wrapping between -180 and 180)."""
        diff = (pred - true) % 360.0
        diff = np.where(diff > 180.0, diff - 360.0, diff)
        diff = np.where(diff < -180.0, diff + 360.0, diff)
        return diff

    def analyze_orbital_elements(
        self,
        true_states: np.ndarray,
        pred_states: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """
        Converts batches and returns element-wise differences and stats.
        """
        true_elems = self.calculator.batch_to_elements(true_states)
        pred_elems = self.calculator.batch_to_elements(pred_states)

        # Compute deviations
        da = pred_elems["a_km"] - true_elems["a_km"]
        de = pred_elems["e"] - true_elems["e"]
        di = self._angle_diff(pred_elems["i_deg"], true_elems["i_deg"])
        draan = self._angle_diff(pred_elems["raan_deg"], true_elems["raan_deg"])
        dargp = self._angle_diff(pred_elems["argp_deg"], true_elems["argp_deg"])
        dnu = self._angle_diff(pred_elems["nu_deg"], true_elems["nu_deg"])

        deviations = {
            "delta_a": da,
            "delta_e": de,
            "delta_i": di,
            "delta_raan": draan,
            "delta_omega": dargp,
            "delta_nu": dnu,
        }

        # Filter out NaNs for stats computation
        valid_mask = np.isfinite(da) & np.isfinite(de) & np.isfinite(di)

        def get_stats(val: np.ndarray) -> Tuple[float, float, float]:
            if valid_mask.sum() == 0:
                return 0.0, 0.0, 0.0
            v = val[valid_mask]
            return float(np.mean(np.abs(v))), float(np.std(v)), float(np.max(np.abs(v)))

        mae_a, std_a, max_a = get_stats(da)
        mae_e, std_e, max_e = get_stats(de)
        mae_i, std_i, max_i = get_stats(di)
        mae_raan, std_raan, max_raan = get_stats(draan)
        mae_omega, std_omega, max_omega = get_stats(dargp)
        mae_nu, std_nu, max_nu = get_stats(dnu)

        summary = {
            "mean_delta_a": mae_a,
            "std_delta_a": std_a,
            "max_delta_a": max_a,
            
            "mean_delta_e": mae_e,
            "std_delta_e": std_e,
            "max_delta_e": max_e,
            
            "mean_delta_i": mae_i,
            "std_delta_i": std_i,
            "max_delta_i": max_i,
            
            "mean_delta_raan": mae_raan,
            "std_delta_raan": std_raan,
            "max_delta_raan": max_raan,
            
            "mean_delta_omega": mae_omega,
            "std_delta_omega": std_omega,
            "max_delta_omega": max_omega,
            
            "mean_delta_nu": mae_nu,
            "std_delta_nu": std_nu,
            "max_delta_nu": max_nu,
        }

        log.info(
            "Orbital Elements — Mean Δa: %.4f km | Mean Δe: %.6f | Mean Δi: %.4f deg",
            mae_a,
            mae_e,
            mae_i,
        )

        return deviations, summary
