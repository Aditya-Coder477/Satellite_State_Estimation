"""
src/evaluation/report_generator.py
=====================================
Generates the final evaluation report for Phase 9.

Produces:
  - evaluation_report.md  (human-readable Markdown with performance grading)
  - evaluation_report.json (machine-readable JSON for downstream pipelines)

Performance grades are computed from configurable thresholds in config.yaml:
  Excellent / Good / Acceptable / Poor

The report includes:
  1. Model identification
  2. Prediction window summary
  3. Regression metrics (position, velocity)
  4. Orbital element drift
  5. Physics consistency (residuals)
  6. Overall grade and recommendation
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class ReportGenerator:
    """
    Generates Markdown and JSON evaluation reports.

    Parameters
    ----------
    cfg : Dict[str, Any]
        Master project configuration.
    output_dir : Path
        Directory where reports are written.
    """

    # Grade lookup — index 0 = best
    _GRADES = ("Excellent", "Good", "Acceptable", "Poor")

    def __init__(self, cfg: Dict[str, Any], output_dir: Path) -> None:
        self._cfg = cfg
        self._eval_cfg = cfg.get("evaluation", {})
        self._thresholds = self._eval_cfg.get("thresholds", {})
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log.info("ReportGenerator initialised. Output dir: %s", self.output_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        satellite_key: str,
        metrics: Dict[str, Any],
        orbital_drift: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Generate and save the full evaluation report.

        Parameters
        ----------
        satellite_key : str
            Satellite identifier (e.g. "GOES16").
        metrics : Dict[str, Any]
            Output of MetricsCalculator.compute_all().
        orbital_drift : Dict[str, float], optional
            Output of OrbitalElementsCalculator.compute_drift().

        Returns
        -------
        Dict[str, Any]
            The assembled report dictionary (also saved as JSON).
        """
        report_cfg = self._eval_cfg.get("report", {})

        grade_pos   = self._grade_position(metrics)
        grade_vel   = self._grade_velocity(metrics)
        grade_phys  = self._grade_physics(metrics)
        overall     = self._overall_grade(grade_pos, grade_vel, grade_phys)

        report = {
            "metadata": {
                "satellite":        satellite_key,
                "generated_at":     datetime.now(timezone.utc).isoformat(),
                "training_window":  self._eval_cfg.get("training_window", {}),
                "prediction_window": self._eval_cfg.get("prediction_window", {}),
                "project":          self._cfg.get("project", {}).get("name", ""),
                "version":          self._cfg.get("project", {}).get("version", ""),
            },
            "grades": {
                "position":  grade_pos,
                "velocity":  grade_vel,
                "physics":   grade_phys,
                "overall":   overall,
            },
            "metrics": metrics,
            "orbital_drift": orbital_drift or {},
        }

        # Write JSON
        if report_cfg.get("generate_json", True):
            json_path = self.output_dir / "evaluation_report.json"
            json_path.write_text(
                json.dumps(report, indent=4, default=str), encoding="utf-8"
            )
            log.info("Saved evaluation JSON to: %s", json_path)

        # Write Markdown
        if report_cfg.get("generate_markdown", True):
            md_path = self.output_dir / "evaluation_report.md"
            md_path.write_text(
                self._build_markdown(satellite_key, report), encoding="utf-8"
            )
            log.info("Saved evaluation Markdown to: %s", md_path)

        log.info("Overall grade: %s", overall)
        return report

    # ------------------------------------------------------------------
    # Grading logic
    # ------------------------------------------------------------------

    def _grade_position(self, metrics: Dict[str, Any]) -> str:
        rmse = metrics.get("position", {}).get("rmse_km", float("inf"))
        t = self._thresholds.get("position_rmse_km", {})
        return self._threshold_grade(rmse, t)

    def _grade_velocity(self, metrics: Dict[str, Any]) -> str:
        rmse = metrics.get("velocity", {}).get("rmse_kms", float("inf"))
        t = self._thresholds.get("velocity_rmse_kms", {})
        return self._threshold_grade(rmse, t)

    def _grade_physics(self, metrics: Dict[str, Any]) -> str:
        residual = metrics.get("physics", {}).get("physics_residual_mean", float("nan"))
        if residual != residual:  # NaN check
            return "N/A"
        t = self._thresholds.get("physics_residual_mean", {})
        return self._threshold_grade(residual, t)

    @staticmethod
    def _threshold_grade(value: float, thresholds: Dict[str, float]) -> str:
        """Map a scalar value to a grade using monotonic thresholds."""
        if value <= thresholds.get("excellent", float("inf")):
            return "Excellent"
        if value <= thresholds.get("good", float("inf")):
            return "Good"
        if value <= thresholds.get("acceptable", float("inf")):
            return "Acceptable"
        return "Poor"

    @staticmethod
    def _overall_grade(*grades: str) -> str:
        """Overall grade is the worst individual grade."""
        order = {"Excellent": 0, "Good": 1, "Acceptable": 2, "Poor": 3, "N/A": 0}
        worst = max((g for g in grades if g != "N/A"),
                    key=lambda g: order.get(g, 3),
                    default="N/A")
        return worst

    # ------------------------------------------------------------------
    # Markdown builder
    # ------------------------------------------------------------------

    def _build_markdown(self, satellite_key: str, report: Dict[str, Any]) -> str:
        m  = report["metadata"]
        g  = report["grades"]
        mt = report["metrics"]
        od = report.get("orbital_drift", {})
        pos = mt.get("position", {})
        vel = mt.get("velocity", {})
        phy = mt.get("physics", {})
        trj = mt.get("trajectory", {})

        pred_win = m.get("prediction_window", {})
        train_win = m.get("training_window", {})

        grade_emoji = {
            "Excellent": "🟢", "Good": "🟡", "Acceptable": "🟠", "Poor": "🔴", "N/A": "⚪"
        }

        lines = [
            f"# Satellite PINN State Estimation — Evaluation Report",
            f"",
            f"**Satellite**: `{satellite_key}`  ",
            f"**Generated**: {m.get('generated_at', '')}  ",
            f"**Project**: {m.get('project', '')} v{m.get('version', '')}  ",
            f"",
            f"---",
            f"",
            f"## Prediction Window",
            f"",
            f"| Parameter | Value |",
            f"|-----------|-------|",
            f"| Training window | Day {train_win.get('start_day', 0)} → Day {train_win.get('end_day', 5)} |",
            f"| Prediction window | Day {pred_win.get('start_day', 5)} → Day {pred_win.get('end_day', 7)} (2-day future horizon) |",
            f"| Physics model | {self._cfg.get('physics', {}).get('gravity_model', 'two_body')} |",
            f"",
            f"---",
            f"",
            f"## Overall Performance Grade",
            f"",
            f"| Dimension | Grade |",
            f"|-----------|-------|",
            f"| Position Accuracy | {grade_emoji.get(g['position'], '⚪')} **{g['position']}** |",
            f"| Velocity Accuracy | {grade_emoji.get(g['velocity'], '⚪')} **{g['velocity']}** |",
            f"| Physics Consistency | {grade_emoji.get(g['physics'], '⚪')} **{g['physics']}** |",
            f"| **Overall** | {grade_emoji.get(g['overall'], '⚪')} **{g['overall']}** |",
            f"",
            f"---",
            f"",
            f"## Position Metrics (GCRS, km)",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| RMSE (total) | `{pos.get('rmse_km', float('nan')):.4f}` km |",
            f"| MAE (total)  | `{pos.get('mae_km', float('nan')):.4f}` km |",
            f"| Max error    | `{pos.get('max_error_km', float('nan')):.4f}` km |",
            f"| Final error  | `{pos.get('final_error_km', float('nan')):.4f}` km |",
            f"| RMSE X       | `{pos.get('rmse_x_km', float('nan')):.4f}` km |",
            f"| RMSE Y       | `{pos.get('rmse_y_km', float('nan')):.4f}` km |",
            f"| RMSE Z       | `{pos.get('rmse_z_km', float('nan')):.4f}` km |",
            f"",
            f"---",
            f"",
            f"## Velocity Metrics (GCRS, km/s)",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| RMSE (total) | `{vel.get('rmse_kms', float('nan')):.6f}` km/s |",
            f"| MAE (total)  | `{vel.get('mae_kms', float('nan')):.6f}` km/s |",
            f"| Max error    | `{vel.get('max_error_kms', float('nan')):.6f}` km/s |",
            f"| Final error  | `{vel.get('final_error_kms', float('nan')):.6f}` km/s |",
            f"| RMSE Vx      | `{vel.get('rmse_vx_kms', float('nan')):.6f}` km/s |",
            f"| RMSE Vy      | `{vel.get('rmse_vy_kms', float('nan')):.6f}` km/s |",
            f"| RMSE Vz      | `{vel.get('rmse_vz_kms', float('nan')):.6f}` km/s |",
            f"",
            f"---",
            f"",
            f"## Trajectory Drift",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Mean position error | `{trj.get('mean_position_error_km', float('nan')):.4f}` km |",
            f"| Std position error  | `{trj.get('std_position_error_km', float('nan')):.4f}` km |",
            f"| Cumulative drift    | `{trj.get('cumulative_position_drift_km', float('nan')):.1f}` km·s |",
            f"| Error growth rate   | `{trj.get('error_growth_rate_kmps', float('nan')):.6f}` km/s |",
            f"",
        ]

        # ── Orbital elements ──────────────────────────────────────────
        if od:
            lines += [
                f"---",
                f"",
                f"## Orbital Element Drift",
                f"",
                f"| Element | MAE | RMSE |",
                f"|---------|-----|------|",
                f"| Semi-major axis (a) | `{od.get('a_km_mae', float('nan')):.4f}` km | `{od.get('a_km_rmse', float('nan')):.4f}` km |",
                f"| Eccentricity (e) | `{od.get('e_mae', float('nan')):.6f}` | `{od.get('e_rmse', float('nan')):.6f}` |",
                f"| Inclination (i) | `{od.get('i_deg_mae', float('nan')):.4f}` deg | `{od.get('i_deg_rmse', float('nan')):.4f}` deg |",
                f"| RAAN (Ω) | `{od.get('raan_deg_mae', float('nan')):.4f}` deg | `{od.get('raan_deg_rmse', float('nan')):.4f}` deg |",
                f"| Arg. of perigee (ω) | `{od.get('argp_deg_mae', float('nan')):.4f}` deg | `{od.get('argp_deg_rmse', float('nan')):.4f}` deg |",
                f"",
            ]

        # ── Physics residuals ─────────────────────────────────────────
        phys_avail = phy.get("physics_available", False)
        if phys_avail:
            lines += [
                f"---",
                f"",
                f"## Physics Consistency (ODE Residuals)",
                f"",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Mean residual | `{phy.get('physics_residual_mean', float('nan')):.4e}` |",
                f"| Max residual  | `{phy.get('physics_residual_max', float('nan')):.4e}` |",
                f"| Std residual  | `{phy.get('physics_residual_std', float('nan')):.4e}` |",
                f"",
            ]

        # ── Thresholds used ───────────────────────────────────────────
        t = self._thresholds
        lines += [
            f"---",
            f"",
            f"## Grading Thresholds",
            f"",
            f"| Grade | Position RMSE | Velocity RMSE | Physics Residual |",
            f"|-------|--------------|---------------|-----------------|",
            f"| 🟢 Excellent  | ≤ {t.get('position_rmse_km', {}).get('excellent', '—')} km | ≤ {t.get('velocity_rmse_kms', {}).get('excellent', '—')} km/s | ≤ {t.get('physics_residual_mean', {}).get('excellent', '—')} |",
            f"| 🟡 Good       | ≤ {t.get('position_rmse_km', {}).get('good', '—')} km | ≤ {t.get('velocity_rmse_kms', {}).get('good', '—')} km/s | ≤ {t.get('physics_residual_mean', {}).get('good', '—')} |",
            f"| 🟠 Acceptable | ≤ {t.get('position_rmse_km', {}).get('acceptable', '—')} km | ≤ {t.get('velocity_rmse_kms', {}).get('acceptable', '—')} km/s | ≤ {t.get('physics_residual_mean', {}).get('acceptable', '—')} |",
            f"| 🔴 Poor       | > {t.get('position_rmse_km', {}).get('acceptable', '—')} km | > {t.get('velocity_rmse_kms', {}).get('acceptable', '—')} km/s | > {t.get('physics_residual_mean', {}).get('acceptable', '—')} |",
            f"",
            f"---",
            f"",
            f"*Report generated automatically by Phase 9 — Satellite PINN State Estimation.*",
        ]

        return "\n".join(lines)
