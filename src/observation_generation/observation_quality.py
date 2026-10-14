"""
src/observation_generation/observation_quality.py
==================================================
Observation Quality Evaluator.

Computes a quality score between 0.0 and 1.0 for each observation based on
visibility, missing flags, and sensor noise levels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class QualityEvaluator:
    """Calculates observation quality scores based on noise and visibility."""

    @staticmethod
    def evaluate_quality(
        visible: np.ndarray | pd.Series,
        missing: np.ndarray | pd.Series,
        noise_ra: np.ndarray,
        noise_dec: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate observation quality scores vectorially.

        Parameters
        ----------
        visible : np.ndarray
            Boolean array indicating visibility.
        missing : np.ndarray
            Boolean array indicating if observation is missing.
        noise_ra : np.ndarray
            RA noise values.
        noise_dec : np.ndarray
            Dec noise values.

        Returns
        -------
        quality_score : np.ndarray, shape (N,)
            Quality scores between 0.0 and 1.0.
        noise_level : np.ndarray, shape (N,)
            Combined noise magnitude in degrees.
        """
        n = len(visible)
        
        # Calculate combined noise magnitude
        noise_level = np.sqrt(noise_ra**2 + noise_dec**2)
        
        # Base quality: 1.0 for zero noise, decreases with higher noise
        # Scale: noise of 0.05 degrees reduces quality to 0.1
        quality_score = 1.0 - np.clip(noise_level / 0.05, 0.0, 0.9)

        # Force quality score to 0.0 for invisible or missing observations
        quality_score = np.where(visible, quality_score, 0.0)
        quality_score = np.where(~missing, quality_score, 0.0)

        return quality_score, noise_level
