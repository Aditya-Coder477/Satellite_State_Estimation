"""
src/observation_generation/validator.py
========================================
ObservationValidator.

Performs schema, range, visibility, and noise reproducibility validation checks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class ObservationValidationResult:
    """Outcome of validation checks."""

    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.passed = False
        log.error("OBSERVATION VALIDATION ERROR: %s", msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)
        log.warning("OBSERVATION VALIDATION WARNING: %s", msg)

    def summary(self) -> str:
        lines = [
            f"Observation Dataset Validation {'PASSED ✓' if self.passed else 'FAILED ✗'}",
            f"  Errors  : {len(self.errors)}",
            f"  Warnings: {len(self.warnings)}",
        ]
        for e in self.errors:
            lines.append(f"  [ERROR]   {e}")
        for w in self.warnings:
            lines.append(f"  [WARNING] {w}")
        return "\n".join(lines)


class ObservationValidator:
    """Validates generated observation datasets."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.ob_cfg = cfg.get("observation_generation", {})

    def validate(
        self,
        df_clean: pd.DataFrame,
        df_noisy: pd.DataFrame,
        df_quality: pd.DataFrame,
    ) -> ObservationValidationResult:
        """Run all validation checks."""
        res = ObservationValidationResult()

        if len(df_clean) == 0 or len(df_noisy) == 0:
            res.add_error("Empty datasets generated.")
            return res

        # ── 1. NaN and Inf check ───────────────────────────────────────
        self._check_nan_inf(df_clean, "clean", res)
        self._check_nan_inf(df_noisy, "noisy", res)

        # ── 2. Boundary and range checks ────────────────────────────────
        self._check_bounds(df_clean, "clean", res)
        self._check_bounds(df_noisy, "noisy", res)

        # ── 3. Quality score limits ────────────────────────────────────
        q_scores = df_quality["quality_score"].to_numpy()
        if np.any((q_scores < 0.0) | (q_scores > 1.0)):
            res.add_error("quality_score contains values outside [0, 1].")

        # ── 4. Scenario label validation ───────────────────────────────
        scenarios = df_noisy["scenario"].unique()
        valid_scenarios = {
            "Scenario A (Ideal)",
            "Scenario B (Low Noise)",
            "Scenario C (High Noise)",
            "Scenario D (Missing Obs)",
            "Scenario E (Variable Sampling)",
            "Scenario F (Station Outage)",
        }
        invalid = [s for s in scenarios if s not in valid_scenarios]
        if invalid:
            res.add_error(f"Invalid scenario labels detected: {invalid}")

        # ── 5. Noise reproducibility check ─────────────────────────────
        self._check_noise_reproducibility(df_noisy, res)

        log.info(res.summary())
        return res

    def _check_nan_inf(
        self, df: pd.DataFrame, label: str, res: ObservationValidationResult
    ) -> None:
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            n_nan = df[col].isna().sum()
            n_inf = np.isinf(df[col]).sum()
            if n_nan > 0:
                res.add_error(f"[{label}] Column '{col}' contains {n_nan} NaN values.")
            if n_inf > 0:
                res.add_error(f"[{label}] Column '{col}' contains {n_inf} infinite values.")

    def _check_bounds(
        self, df: pd.DataFrame, label: str, res: ObservationValidationResult
    ) -> None:
        if label == "clean":
            ra = df["ra_deg"].to_numpy()
            dec = df["dec_deg"].to_numpy()
        else:
            ra = df["ra_observed_deg"].to_numpy()
            dec = df["dec_observed_deg"].to_numpy()

        if np.any((ra < 0.0) | (ra >= 360.0)):
            res.add_error(f"[{label}] Right Ascension contains values outside [0, 360) degrees.")
        if np.any((dec < -90.0) | (dec > 90.0)):
            res.add_error(f"[{label}] Declination contains values outside [-90, 90] degrees.")

    def _check_noise_reproducibility(
        self, df_noisy: pd.DataFrame, res: ObservationValidationResult
    ) -> None:
        # Check Scenario B (Low Noise) std dev
        df_b = df_noisy[df_noisy["scenario"] == "Scenario B (Low Noise)"]
        if len(df_b) > 10:
            std_ra = float(np.std(df_b["noise_ra_deg"]))
            std_dec = float(np.std(df_b["noise_dec_deg"]))
            
            # Expected is 0.005 degrees
            if not np.isclose(std_ra, 0.005, rtol=0.2):
                res.add_warning(
                    f"Low Noise Scenario (RA) standard deviation {std_ra:.5f}° "
                    f"deviates from expected 0.005°"
                )
            if not np.isclose(std_dec, 0.005, rtol=0.2):
                res.add_warning(
                    f"Low Noise Scenario (Dec) standard deviation {std_dec:.5f}° "
                    f"deviates from expected 0.005°"
                )
