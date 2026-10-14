"""
src/observation_generation/scenario_generator.py
=================================================
Observation Scenarios Generator.

Generates the six configured optical tracking scenarios:
  - Scenario A: Ideal (no noise, uniform)
  - Scenario B: Low Gaussian Noise (0.005 deg)
  - Scenario C: High Gaussian Noise (0.020 deg)
  - Scenario D: Missing Observations (random dropouts)
  - Scenario E: Variable Sampling (non-uniform cadence)
  - Scenario F: Station Outage (temporary outage at a station)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.observation_generation.ra_dec_generator import RADecGenerator
from src.observation_generation.noise_models import NoiseModelFactory
from src.observation_generation.sampling_models import ObservationSampler
from src.observation_generation.observation_quality import QualityEvaluator
from src.utils.logger import get_logger

log = get_logger(__name__)


class ScenarioGenerator:
    """Generates clean and noisy observation datasets across multiple scenarios."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.sc_cfg = cfg.get("observation_generation", {}).get("scenarios", {})

    def generate_all(
        self,
        df_geom: pd.DataFrame,
        df_states: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate datasets for all scenarios.

        Parameters
        ----------
        df_geom : pd.DataFrame
        df_states : pd.DataFrame

        Returns
        -------
        df_clean : pd.DataFrame
            The clean observations dataset (Scenario A).
        df_noisy : pd.DataFrame
            The combined noisy observations dataset (Scenarios A-F).
        df_quality : pd.DataFrame
            Data quality metrics dataset.
        df_visibility_summary : pd.DataFrame
            Station visibility summary dataset.
        """
        # Filter for visible observations only
        df_vis = df_geom[df_geom["visible"]].copy().reset_index(drop=True)
        if len(df_vis) == 0:
            log.warning("No visible samples found in ground station geometry.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # Compute true/ideal RA and Dec vectorially
        ra_true, dec_true = RADecGenerator.compute_vectorized(df_vis, df_states)
        df_vis["ra_true_deg"] = ra_true
        df_vis["dec_true_deg"] = dec_true

        noisy_list = []
        quality_list = []

        # ── Scenario A: Ideal (No noise, uniform) ───────────────────────
        log.info("Generating Scenario A (Ideal)...")
        df_a = df_vis.copy()
        df_a["ra_observed_deg"] = df_a["ra_true_deg"]
        df_a["dec_observed_deg"] = df_a["dec_true_deg"]
        df_a["noise_ra_deg"] = 0.0
        df_a["noise_dec_deg"] = 0.0
        df_a["quality_score"] = 1.0
        df_a["scenario"] = "Scenario A (Ideal)"
        
        noisy_list.append(df_a)
        
        # Build quality record for A
        df_q_a = self._build_quality_record(df_a, "Scenario A (Ideal)", visible=True, missing=False)
        quality_list.append(df_q_a)

        # ── Scenario B: Low Gaussian Noise ──────────────────────────────
        if self.sc_cfg.get("low_noise", True):
            log.info("Generating Scenario B (Low Gaussian Noise)...")
            df_b = df_vis.copy()
            # Construct temporary config for low noise
            cfg_b = {"noise": {"gaussian": {"enabled": True, "ra_std_deg": 0.005, "dec_std_deg": 0.005}}}
            ra_n, dec_n, n_ra, n_dec = NoiseModelFactory.inject_noise(
                df_b["ra_true_deg"].to_numpy(), df_b["dec_true_deg"].to_numpy(), cfg_b, seed=101
            )
            df_b["ra_observed_deg"] = ra_n
            df_b["dec_observed_deg"] = dec_n
            df_b["noise_ra_deg"] = n_ra
            df_b["noise_dec_deg"] = n_dec
            df_b["quality_score"], _ = QualityEvaluator.evaluate_quality(
                np.ones(len(df_b), dtype=bool), np.zeros(len(df_b), dtype=bool), n_ra, n_dec
            )
            df_b["scenario"] = "Scenario B (Low Noise)"
            noisy_list.append(df_b)
            quality_list.append(self._build_quality_record(df_b, "Scenario B (Low Noise)", visible=True, missing=False))

        # ── Scenario C: High Gaussian Noise ─────────────────────────────
        if self.sc_cfg.get("high_noise", True):
            log.info("Generating Scenario C (High Gaussian Noise)...")
            df_c = df_vis.copy()
            cfg_c = {"noise": {"gaussian": {"enabled": True, "ra_std_deg": 0.02, "dec_std_deg": 0.02}}}
            ra_n, dec_n, n_ra, n_dec = NoiseModelFactory.inject_noise(
                df_c["ra_true_deg"].to_numpy(), df_c["dec_true_deg"].to_numpy(), cfg_c, seed=102
            )
            df_c["ra_observed_deg"] = ra_n
            df_c["dec_observed_deg"] = dec_n
            df_c["noise_ra_deg"] = n_ra
            df_c["noise_dec_deg"] = n_dec
            df_c["quality_score"], _ = QualityEvaluator.evaluate_quality(
                np.ones(len(df_c), dtype=bool), np.zeros(len(df_c), dtype=bool), n_ra, n_dec
            )
            df_c["scenario"] = "Scenario C (High Noise)"
            noisy_list.append(df_c)
            quality_list.append(self._build_quality_record(df_c, "Scenario C (High Noise)", visible=True, missing=False))

        # ── Scenario D: Missing Observations (random dropouts) ──────────
        if self.sc_cfg.get("missing_observations", True):
            log.info("Generating Scenario D (Missing Observations)...")
            df_d_full = df_vis.copy()
            
            # Apply 20% random dropouts
            df_d = ObservationSampler.apply_missing_dropout(df_d_full, missing_fraction=0.20, seed=103)
            
            # Inject standard low noise
            cfg_d = {"noise": {"gaussian": {"enabled": True, "ra_std_deg": 0.005, "dec_std_deg": 0.005}}}
            ra_n, dec_n, n_ra, n_dec = NoiseModelFactory.inject_noise(
                df_d["ra_true_deg"].to_numpy(), df_d["dec_true_deg"].to_numpy(), cfg_d, seed=103
            )
            df_d["ra_observed_deg"] = ra_n
            df_d["dec_observed_deg"] = dec_n
            df_d["noise_ra_deg"] = n_ra
            df_d["noise_dec_deg"] = n_dec
            df_d["quality_score"], _ = QualityEvaluator.evaluate_quality(
                np.ones(len(df_d), dtype=bool), np.zeros(len(df_d), dtype=bool), n_ra, n_dec
            )
            df_d["scenario"] = "Scenario D (Missing Obs)"
            noisy_list.append(df_d)
            
            # Record quality metrics including missing flag
            missing_mask = ~df_d_full["elapsed_seconds"].isin(df_d["elapsed_seconds"])
            df_q_d = self._build_quality_record(
                df_d_full, "Scenario D (Missing Obs)", visible=True, missing=missing_mask
            )
            quality_list.append(df_q_d)

        # ── Scenario E: Variable Sampling ───────────────────────────────
        if self.sc_cfg.get("variable_sampling", True):
            log.info("Generating Scenario E (Variable Sampling)...")
            df_e_full = df_vis.copy()
            
            # Vectorized selection: keep only samples matching variable cadences (e.g. 120s or 300s)
            # Or keep randomly select ~33% of samples to represent variable observation intervals
            rng_e = np.random.default_rng(104)
            keep_idx = rng_e.choice([True, False], size=len(df_e_full), p=[0.33, 0.67])
            df_e = df_e_full[keep_idx].copy()
            
            cfg_e = {"noise": {"gaussian": {"enabled": True, "ra_std_deg": 0.005, "dec_std_deg": 0.005}}}
            ra_n, dec_n, n_ra, n_dec = NoiseModelFactory.inject_noise(
                df_e["ra_true_deg"].to_numpy(), df_e["dec_true_deg"].to_numpy(), cfg_e, seed=104
            )
            df_e["ra_observed_deg"] = ra_n
            df_e["dec_observed_deg"] = dec_n
            df_e["noise_ra_deg"] = n_ra
            df_e["noise_dec_deg"] = n_dec
            df_e["quality_score"], _ = QualityEvaluator.evaluate_quality(
                np.ones(len(df_e), dtype=bool), np.zeros(len(df_e), dtype=bool), n_ra, n_dec
            )
            df_e["scenario"] = "Scenario E (Variable Sampling)"
            noisy_list.append(df_e)
            
            missing_mask_e = ~keep_idx
            df_q_e = self._build_quality_record(
                df_e_full, "Scenario E (Variable Sampling)", visible=True, missing=missing_mask_e, sampling_int=180
            )
            quality_list.append(df_q_e)

        # ── Scenario F: Station Outage ──────────────────────────────────
        if self.sc_cfg.get("station_outage", True):
            log.info("Generating Scenario F (Station Outage)...")
            df_f_full = df_vis.copy()
            
            outage_id = int(self.sc_cfg.get("outage_station_id", 1))
            t_start = float(self.sc_cfg.get("outage_start_s", 259200))
            t_end = float(self.sc_cfg.get("outage_end_s", 345600))
            
            # Mask representing outage interval
            outage_mask = (df_f_full["station_id"] == outage_id) & \
                          (df_f_full["elapsed_seconds"] >= t_start) & \
                          (df_f_full["elapsed_seconds"] <= t_end)
            
            df_f = df_f_full[~outage_mask].copy()
            
            cfg_f = {"noise": {"gaussian": {"enabled": True, "ra_std_deg": 0.005, "dec_std_deg": 0.005}}}
            ra_n, dec_n, n_ra, n_dec = NoiseModelFactory.inject_noise(
                df_f["ra_true_deg"].to_numpy(), df_f["dec_true_deg"].to_numpy(), cfg_f, seed=105
            )
            df_f["ra_observed_deg"] = ra_n
            df_f["dec_observed_deg"] = dec_n
            df_f["noise_ra_deg"] = n_ra
            df_f["noise_dec_deg"] = n_dec
            df_f["quality_score"], _ = QualityEvaluator.evaluate_quality(
                np.ones(len(df_f), dtype=bool), np.zeros(len(df_f), dtype=bool), n_ra, n_dec
            )
            df_f["scenario"] = "Scenario F (Station Outage)"
            noisy_list.append(df_f)
            
            df_q_f = self._build_quality_record(
                df_f_full, "Scenario F (Station Outage)", visible=True, missing=outage_mask
            )
            quality_list.append(df_q_f)

        # ── 1. Create clean observations.csv ───────────────────────────
        df_clean = df_a[[
            "utc_time", "elapsed_seconds", "station_id", "station_name",
            "ra_true_deg", "dec_true_deg", "azimuth_deg", "elevation_deg",
            "range_km", "range_rate_km_s", "quality_score", "scenario"
        ]].rename(columns={"ra_true_deg": "ra_deg", "dec_true_deg": "dec_deg"})

        # ── 2. Create noisy_observations.csv ───────────────────────────
        df_noisy_all = pd.concat(noisy_list, ignore_index=True)
        df_noisy = df_noisy_all[[
            "utc_time", "elapsed_seconds", "station_id", "station_name",
            "ra_true_deg", "dec_true_deg", "ra_observed_deg", "dec_observed_deg",
            "noise_ra_deg", "noise_dec_deg", "quality_score", "scenario"
        ]]

        # ── 3. Create observation_quality.csv ──────────────────────────
        df_quality = pd.concat(quality_list, ignore_index=True)

        # ── 4. Create station_visibility_summary.csv ───────────────────
        df_visibility_summary = self._build_visibility_summary(df_geom)

        return df_clean, df_noisy, df_quality, df_visibility_summary

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_quality_record(
        df: pd.DataFrame,
        scenario_name: str,
        visible: bool,
        missing: bool | np.ndarray,
        sampling_int: int = 60,
    ) -> pd.DataFrame:
        """Construct a data quality metrics log DataFrame."""
        n = len(df)
        
        # Resolve quality scores
        noise_ra = df["noise_ra_deg"].to_numpy() if "noise_ra_deg" in df.columns else np.zeros(n)
        noise_dec = df["noise_dec_deg"].to_numpy() if "noise_dec_deg" in df.columns else np.zeros(n)
        
        if isinstance(missing, bool):
            missing_arr = np.full(n, missing, dtype=bool)
        else:
            missing_arr = np.array(missing, dtype=bool)

        visible_arr = np.full(n, visible, dtype=bool)
        
        q_scores, n_levels = QualityEvaluator.evaluate_quality(
            visible_arr, missing_arr, noise_ra, noise_dec
        )

        return pd.DataFrame({
            "utc_time": df["utc_time"].values,
            "station_id": df["station_id"].values,
            "quality_score": q_scores,
            "noise_level": n_levels,
            "sampling_interval": sampling_int,
            "visible": visible_arr,
            "missing": missing_arr,
            "scenario": scenario_name,
        })

    @staticmethod
    def _build_visibility_summary(df_geom: pd.DataFrame) -> pd.DataFrame:
        """Group by station to compute aggregate visibility statistics."""
        summary = []
        for name, df_s in df_geom.groupby("station_name"):
            vis = df_s["visible"].to_numpy()
            total_samples = len(vis)
            visible_samples = int(np.sum(vis))
            
            summary.append({
                "station_name": name,
                "total_samples": total_samples,
                "visible_samples": visible_samples,
                "visibility_percentage": float(visible_samples / total_samples * 100.0),
            })
        return pd.DataFrame(summary)
