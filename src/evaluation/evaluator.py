"""
src/evaluation/evaluator.py
============================
Unified public API for Phase 9 (Evaluation Framework).

This is the **only** class external code should instantiate.
It orchestrates:
  1. FuturePredictionEngine  — generates predicted states
  2. MetricsCalculator       — computes regression + physics metrics
  3. OrbitalElementsCalculator — Keplerian orbit drift analysis
  4. ReportGenerator         — produces Markdown + JSON reports
  5. EvaluationVisualizer    — generates all diagnostic figures

Design principles:
  - Never loads CSV files directly; delegates all I/O to DatasetManager.
  - Accepts any model exposing ``predict(times: np.ndarray) -> np.ndarray``.
  - All outputs (metrics, reports, figures, CSVs) are written to paths
    configured in config.yaml under the ``evaluation.output`` block.
  - A root-level "CSV Outputs" folder receives the prediction comparison CSV.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.dataset.dataset_manager import DatasetManager
from src.evaluation.future_prediction import FuturePredictionEngine
from src.evaluation.metrics_calculator import MetricsCalculator
from src.evaluation.orbital_elements import OrbitalElementsCalculator
from src.evaluation.report_generator import ReportGenerator
from src.evaluation.visualizer import EvaluationVisualizer
from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)

# Optional PyTorch for physics residual time-series
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class EvaluationEngine:
    """
    Orchestrates the full Phase 9 evaluation pipeline.

    Parameters
    ----------
    cfg : Dict[str, Any]
        Master project configuration (from ConfigLoader).
    satellite_key : str
        Satellite identifier (e.g. ``"GOES16"``).
    physics_engine : optional
        PhysicsEngine instance (Phase 7). Required for physics residuals.
        Pass ``None`` to skip physics consistency validation.
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        satellite_key: str,
        physics_engine: Optional[Any] = None,
    ) -> None:
        self._cfg = cfg
        self._satellite_key = satellite_key
        self._eval_cfg = cfg.get("evaluation", {})
        self._root = Path(cfg["_root"])

        # Resolve output directories
        out_cfg = self._eval_cfg.get("output", {})
        self._metrics_dir  = self._root / out_cfg.get("metrics_dir",  "results/evaluation/metrics")
        self._figures_dir  = self._root / out_cfg.get("figures_dir",  "results/evaluation/figures")
        self._reports_dir  = self._root / out_cfg.get("reports_dir",  "results/evaluation/reports")
        self._csv_out_dir  = self._root / out_cfg.get("csv_outputs_dir", "CSV Outputs")

        # Create all output directories
        for d in (self._metrics_dir, self._figures_dir, self._reports_dir, self._csv_out_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Instantiate sub-components
        self._dataset_manager = DatasetManager(cfg, satellite_key)
        self._metrics_calc    = MetricsCalculator(physics_engine)
        self._orbital_calc    = OrbitalElementsCalculator()
        self._report_gen      = ReportGenerator(cfg, self._reports_dir)
        self._visualizer      = EvaluationVisualizer(cfg, self._figures_dir)

        log.info(
            "EvaluationEngine initialised for satellite '%s'.", satellite_key
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, predictor: Any) -> Dict[str, Any]:
        """
        Execute the full evaluation pipeline.

        Parameters
        ----------
        predictor : any object with ``predict(times: np.ndarray) -> np.ndarray``
            The trained model used for inference during the 2-day window.

        Returns
        -------
        Dict[str, Any]
            Full evaluation report dictionary.
        """
        log.info("=" * 72)
        log.info("Phase 9 — Future 2-Day Prediction Evaluation")
        log.info("Satellite : %s", self._satellite_key)
        log.info("=" * 72)

        # ── Step 1: Future prediction ─────────────────────────────────
        log.info("[1/5] Running future prediction (Day 5 → Day 7)...")
        pred_engine = FuturePredictionEngine(
            cfg=self._cfg,
            dataset_manager=self._dataset_manager,
            predictor=predictor,
        )
        times, predicted_states, truth_states = pred_engine.run()

        # ── Step 2: Build comparison DataFrame ───────────────────────
        log.info("[2/5] Building prediction comparison DataFrame...")
        comparison_df = pred_engine.build_comparison_dataframe(
            times, predicted_states, truth_states
        )

        # ── Step 3: Compute metrics ───────────────────────────────────
        log.info("[3/5] Computing evaluation metrics...")
        metrics = self._metrics_calc.compute_all(times, predicted_states, truth_states)

        # ── Step 4: Orbital elements drift ───────────────────────────
        log.info("[4/5] Computing orbital element drift...")
        orbital_drift = self._orbital_calc.compute_drift(predicted_states, truth_states)

        # Build per-step orbital element arrays for visualization
        pred_elems  = self._orbital_calc.batch_to_elements(predicted_states)
        truth_elems = self._orbital_calc.batch_to_elements(truth_states)
        orbital_vis_data = {
            **{f"{k}_pred": v for k, v in pred_elems.items()},
            **{f"{k}_true": v for k, v in truth_elems.items()},
        }

        # ── Step 5: Physics residual time-series ─────────────────────
        physics_residuals_ts: Optional[np.ndarray] = None
        phys_metrics = metrics.get("physics", {})
        if phys_metrics.get("physics_available", False) and TORCH_AVAILABLE:
            physics_residuals_ts = self._build_physics_residual_ts(times, predicted_states)

        # ── Step 6: Save CSVs ─────────────────────────────────────────
        log.info("[5/5] Saving outputs...")
        self._save_outputs(comparison_df, metrics, orbital_drift)

        # ── Step 7: Generate visualizations ──────────────────────────
        self._visualizer.generate_all(
            times=times,
            predicted=predicted_states,
            truth=truth_states,
            metrics=metrics,
            orbital_drift_data=orbital_vis_data,
            physics_residuals_ts=physics_residuals_ts,
        )

        # ── Step 8: Generate report ───────────────────────────────────
        report = self._report_gen.generate(
            satellite_key=self._satellite_key,
            metrics=metrics,
            orbital_drift=orbital_drift,
        )

        log.info("=" * 72)
        log.info("Evaluation complete. Overall grade: %s",
                 report["grades"]["overall"])
        log.info("Reports  : %s", self._reports_dir)
        log.info("Figures  : %s", self._figures_dir)
        log.info("CSV Outputs: %s", self._csv_out_dir)
        log.info("=" * 72)

        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save_outputs(
        self,
        comparison_df: pd.DataFrame,
        metrics: Dict[str, Any],
        orbital_drift: Dict[str, float],
    ) -> None:
        """Save all CSV outputs to both metrics_dir and root CSV Outputs folder."""
        import json

        # 1. Prediction comparison CSV (primary deliverable)
        comp_csv = self._csv_out_dir / f"{self._satellite_key}_prediction_comparison.csv"
        comparison_df.to_csv(comp_csv, index=False)
        log.info("Saved prediction comparison CSV: %s", comp_csv)

        # Also save a copy inside the metrics directory for archive
        comp_csv_metrics = self._metrics_dir / f"{self._satellite_key}_prediction_comparison.csv"
        comparison_df.to_csv(comp_csv_metrics, index=False)

        # 2. Metrics summary CSV
        flat_metrics = self._flatten_dict(metrics)
        metrics_df = pd.DataFrame([flat_metrics])
        metrics_csv = self._csv_out_dir / f"{self._satellite_key}_metrics_summary.csv"
        metrics_df.to_csv(metrics_csv, index=False)
        log.info("Saved metrics summary CSV: %s", metrics_csv)

        metrics_csv_arch = self._metrics_dir / f"{self._satellite_key}_metrics_summary.csv"
        metrics_df.to_csv(metrics_csv_arch, index=False)

        # 3. Orbital drift CSV
        if orbital_drift:
            drift_df = pd.DataFrame([orbital_drift])
            drift_csv = self._csv_out_dir / f"{self._satellite_key}_orbital_drift.csv"
            drift_df.to_csv(drift_csv, index=False)
            log.info("Saved orbital drift CSV: %s", drift_csv)

            drift_csv_arch = self._metrics_dir / f"{self._satellite_key}_orbital_drift.csv"
            drift_df.to_csv(drift_csv_arch, index=False)

    def _build_physics_residual_ts(
        self, times: np.ndarray, predicted: np.ndarray
    ) -> Optional[np.ndarray]:
        """Build per-step physics residual norms for visualization."""
        try:
            dt = np.diff(times)
            dS = np.diff(predicted, axis=0)
            state_mid = 0.5 * (predicted[:-1] + predicted[1:])
            deriv_fd  = dS / dt[:, None]

            state_t = torch.tensor(state_mid, dtype=torch.float32)
            deriv_t = torch.tensor(deriv_fd, dtype=torch.float32)

            from src.physics.physics_engine import PhysicsEngine  # avoid circular at module level
            # Access from metrics calculator
            pe = self._metrics_calc._physics_engine
            if pe is None:
                return None

            residuals = pe.compute_residuals(state_t, deriv_t)
            return torch.norm(residuals, dim=-1).detach().cpu().numpy()
        except Exception as exc:
            log.warning("Could not build physics residual time-series: %s", exc)
            return None

    @staticmethod
    def _flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Flatten nested dictionary to a single-level dict for CSV export."""
        flat: Dict[str, Any] = {}
        for k, v in d.items():
            key = f"{prefix}{k}" if not prefix else f"{prefix}_{k}"
            if isinstance(v, dict):
                flat.update(EvaluationEngine._flatten_dict(v, prefix=key))
            else:
                flat[key] = v
        return flat
