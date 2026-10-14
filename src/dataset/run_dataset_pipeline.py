"""
src/dataset/run_dataset_pipeline.py
=====================================
Entry-point script for **Phase 7: State Transition Dataset Engineering**.

Builds the (St, t, Δt) → (ΔS, a_res) machine-learning dataset for the
Physics-Informed State Transition Network (PI-STN).

Usage
-----
From the project root::

    python -m src.dataset.run_dataset_pipeline --sat INSAT3D
    python -m src.dataset.run_dataset_pipeline --sat INSAT3D --no-plots
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from src.dataset.transition_generator import TransitionGenerator
from src.dataset.feature_engineering import FeatureEngineer
from src.dataset.normalization import Normalizer
from src.dataset.dataset_splitter import SplitStrategyFactory
from src.dataset.collocation_generator import CollocationGenerator, CollocationStrategyFactory
from src.dataset.statistics import DatasetStatistics
from src.dataset.visualizer import DatasetVisualizer
from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger, setup_logging


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _resolve_paths(cfg: dict[str, Any], satellite_key: str) -> dict[str, Path]:
    root = Path(cfg["_root"])
    processed_base = root / cfg["paths"]["data"]["processed"]["base"]
    sat_dir = processed_base / cfg["satellites"][satellite_key]["processed_dir"]
    ds_cfg = cfg.get("dataset", {})
    out_dir = sat_dir / ds_cfg.get("output_dir", "dataset")

    return {
        "sat_dir": sat_dir,
        "input_states": sat_dir / ds_cfg.get("input_states", "truth_states_transformed.csv"),
        "input_observations": sat_dir / ds_cfg.get("input_observations", "noisy_observations.csv"),
        "output_dir": out_dir,
        "log_dir": root / "logs" / "dataset" / satellite_key,
        "results_dir": root / "results" / "dataset" / satellite_key,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(satellite_key: str, config_path: str | None = None, no_plots: bool = False) -> int:
    cfg: dict[str, Any] = ConfigLoader.load(config_path)

    if satellite_key not in cfg.get("satellites", {}):
        import logging
        logging.basicConfig(level=logging.ERROR)
        logging.getLogger(__name__).error(
            "Satellite '%s' not found in config.", satellite_key)
        return 1

    paths = _resolve_paths(cfg, satellite_key)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    paths["log_dir"].mkdir(parents=True, exist_ok=True)

    log_file = paths["log_dir"] / "dataset_pipeline.log"
    setup_logging(cfg, log_file=log_file)
    log = get_logger(__name__)

    log.info("=" * 80)
    log.info("Satellite PINN State Estimation — Phase 7: Dataset Engineering")
    log.info("=" * 80)
    log.info("Satellite   : %s", satellite_key)
    log.info("States input: %s", paths["input_states"])
    log.info("Output dir  : %s", paths["output_dir"])
    log.info("Log file    : %s", log_file)

    wall_start = time.perf_counter()
    ds_cfg = cfg.get("dataset", {})
    out_dir = paths["output_dir"]

    try:
        # ── STEP 1: Load states ────────────────────────────────────────
        log.info("-" * 60)
        log.info("STEP 1/9: Load trajectory states")
        if not paths["input_states"].is_file():
            log.error("States file not found: %s", paths["input_states"])
            return 1

        df_states = pd.read_csv(paths["input_states"])
        log.info("  Loaded %d state rows.", len(df_states))

        # ── STEP 2: Generate transition pairs ──────────────────────────
        log.info("-" * 60)
        log.info("STEP 2/9: Generate (St, Δt) → (ΔS, a_res) transition pairs")
        gen = TransitionGenerator(cfg)
        df_pairs = gen.generate(df_states)
        if len(df_pairs) == 0:
            log.error("No transition pairs generated.")
            return 1
        log.info("  %d transition pairs generated.", len(df_pairs))

        # ── STEP 3: Feature engineering ────────────────────────────────
        log.info("-" * 60)
        log.info("STEP 3/9: Feature engineering (radius, v_mag, energy, L)")
        engineer = FeatureEngineer(cfg)
        df_featured = engineer.engineer_transition_features(df_pairs)

        # ── STEP 4: Save raw (un-normalised) dataset ───────────────────
        log.info("-" * 60)
        log.info("STEP 4/9: Save raw transition dataset")
        raw_path = out_dir / "transition_dataset_raw.csv"
        df_featured.to_csv(raw_path, index=False, float_format="%.10f")
        log.info("  Saved raw dataset: %d rows → %s", len(df_featured), raw_path)

        # ── STEP 5: Normalisation ──────────────────────────────────────
        log.info("-" * 60)
        norm_method = ds_cfg.get("normalization", {}).get("method", "standard")
        log.info("STEP 5/9: Normalise features (method=%s)", norm_method)

        # Identify input and target feature columns
        input_features, target_features = _get_feature_columns(df_featured, ds_cfg)
        all_feature_cols = input_features + target_features

        normalizer = Normalizer.from_cfg(cfg)
        df_norm_features = normalizer.fit_transform(df_featured[all_feature_cols])

        # Build normalised full dataset (identity columns + normalised)
        id_cols = [c for c in ["utc_time", "elapsed_seconds", "delta_time"] if c not in df_norm_features.columns]
        df_norm = pd.concat([df_featured[id_cols].reset_index(drop=True),
                             df_norm_features.reset_index(drop=True)], axis=1)

        # Save normalisation params
        norm_params = normalizer.get_params()
        with open(out_dir / "normalization.json", "w") as f:
            json.dump(norm_params, f, indent=2)
        log.info("  Saved normalization.json")

        # ── STEP 6: Save normalised dataset ───────────────────────────
        df_norm.to_csv(out_dir / "transition_dataset.csv", index=False, float_format="%.10f")
        log.info("  Saved normalised dataset: %s", out_dir / "transition_dataset.csv")

        # ── STEP 7: Train / Val / Test split ──────────────────────────
        log.info("-" * 60)
        log.info("STEP 6/9: Chronological train/val/test split")
        split_cfg = ds_cfg.get("split", {})
        # Default to temporal (chronological) strategy
        strategy_name = "temporal" if split_cfg.get("chronological", True) else "random"
        strategy = SplitStrategyFactory.create(strategy_name)
        train_ratio = float(split_cfg.get("train", 0.70))
        val_ratio   = float(split_cfg.get("validation", 0.15))
        test_ratio  = float(split_cfg.get("test", 0.15))
        train_df, val_df, test_df = strategy.split(
            df_norm, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio
        )
        # Also split raw featured dataset (for debugging/raw outputs)
        train_raw, val_raw, test_raw = strategy.split(
            df_featured, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio
        )

        train_df.to_csv(out_dir / "train.csv", index=False, float_format="%.10f")
        val_df.to_csv(out_dir / "validation.csv", index=False, float_format="%.10f")
        test_df.to_csv(out_dir / "test.csv", index=False, float_format="%.10f")
        log.info(
            "  Split sizes → train: %d | val: %d | test: %d",
            len(train_df), len(val_df), len(test_df),
        )

        # ── STEP 8: Collocation points ─────────────────────────────────
        log.info("-" * 60)
        log.info("STEP 7/9: Generate collocation points")
        coll_cfg = ds_cfg.get("collocation", {})
        if coll_cfg.get("enabled", True):
            t_min = float(df_pairs["elapsed_seconds"].min())
            t_max = float(df_pairs["elapsed_seconds"].max())
            n_pts = int(coll_cfg.get("number_of_points", 10000))
            samp   = coll_cfg.get("sampling", "uniform")

            cgen_strategy = CollocationStrategyFactory.create(samp)
            coll_gen = CollocationGenerator(cgen_strategy)
            colloc_df = coll_gen.generate(
                min_time=t_min, max_time=t_max,
                num_points=n_pts,
            )
            colloc_df.to_csv(out_dir / "collocation_points.csv", index=False, float_format="%.10f")
            log.info("  Generated %d collocation points.", len(colloc_df))
        else:
            colloc_df = pd.DataFrame(columns=["time", "delta_time"])

        # ── STEP 9: Metadata JSONs ─────────────────────────────────────
        log.info("-" * 60)
        log.info("STEP 8/9: Write metadata JSONs")

        # feature_mapping.json
        feature_map = {"inputs": input_features, "targets": target_features}
        with open(out_dir / "feature_mapping.json", "w") as f:
            json.dump(feature_map, f, indent=2)

        # feature_statistics.json
        feat_stats = DatasetStatistics.compute_feature_statistics(
            df_featured[all_feature_cols], all_feature_cols
        )
        with open(out_dir / "feature_statistics.json", "w") as f:
            json.dump(feat_stats, f, indent=2)

        # transition_pairs.json
        tp_summary = {
            "total_pairs": len(df_pairs),
            "delta_time_seconds": float(df_pairs["delta_time"].median()),
            "time_span_seconds": float(df_pairs["elapsed_seconds"].max() - df_pairs["elapsed_seconds"].min()),
        }
        with open(out_dir / "transition_pairs.json", "w") as f:
            json.dump(tp_summary, f, indent=2)

        # pytorch_metadata.json
        pytorch_meta = {
            "input_dimension": len(input_features),
            "output_dimension": len(target_features),
            "normalization_method": norm_method,
            "feature_names": input_features,
            "target_names": target_features,
            "dataset_sizes": {
                "total": len(df_norm),
                "train": len(train_df),
                "validation": len(val_df),
                "test": len(test_df),
            },
        }
        with open(out_dir / "pytorch_metadata.json", "w") as f:
            json.dump(pytorch_meta, f, indent=2)

        # dataset_summary.json
        split_r = split_cfg
        summary = {
            "phase": "Phase 7 — State Transition Dataset Engineering",
            "satellite": satellite_key,
            "total_samples": len(df_norm),
            "training_samples": len(train_df),
            "validation_samples": len(val_df),
            "test_samples": len(test_df),
            "number_of_input_features": len(input_features),
            "number_of_target_features": len(target_features),
            "time_span_hours": float(df_pairs["elapsed_seconds"].max()) / 3600.0,
            "sampling_interval_seconds": float(df_pairs["delta_time"].median()),
            "transition_interval_seconds": float(ds_cfg.get("transition", {}).get("delta_time_seconds", 60)),
            "normalization_method": norm_method,
            "status": "SUCCESS",
        }
        with open(out_dir / "dataset_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # dataset_config_used.json
        with open(out_dir / "dataset_config_used.json", "w") as f:
            json.dump(ds_cfg, f, indent=2)

        log.info("  All metadata JSONs saved.")

        # ── Validation (lightweight inline) ───────────────────────────
        log.info("-" * 60)
        log.info("STEP 9/9: Validate dataset")
        val_errors: list[str] = []
        # 1. No NaN/Inf in normalised feature columns
        num_cols = df_norm.select_dtypes(include=[np.number]).columns
        nan_cols = [c for c in num_cols if df_norm[c].isna().any()]
        inf_cols = [c for c in num_cols if np.isinf(df_norm[c]).any()]
        if nan_cols:
            val_errors.append(f"NaN found in: {nan_cols}")
        if inf_cols:
            val_errors.append(f"Inf found in: {inf_cols}")
        # 2. Split sizes approximate expected ratios (±2%)
        total = len(train_df) + len(val_df) + len(test_df)
        for name_, actual_, expected_ in [
            ("train", len(train_df) / total, train_ratio),
            ("val",   len(val_df)   / total, val_ratio),
            ("test",  len(test_df)  / total, test_ratio),
        ]:
            if abs(actual_ - expected_) > 0.02:
                val_errors.append(f"{name_} ratio {actual_:.3f} deviates from expected {expected_:.3f}")
        # 3. Chronological order inside splits
        if "elapsed_seconds" in train_df.columns:
            for sname_, sdf_ in [("train", train_df), ("val", val_df), ("test", test_df)]:
                if not sdf_["elapsed_seconds"].is_monotonic_increasing:
                    val_errors.append(f"{sname_} split is not monotonically increasing in time.")

        if val_errors:
            for e in val_errors:
                log.error("  Validation error: %s", e)
            log.error("Dataset validation FAILED (%d issues).", len(val_errors))
            return 1
        log.info("  Dataset validation PASSED.")

        # ── Visualisations ─────────────────────────────────────────────
        if not no_plots:
            log.info("-" * 60)
            log.info("Generating visualisation figures …")
            paths["results_dir"].mkdir(parents=True, exist_ok=True)
            viz = DatasetVisualizer(cfg, output_dir=paths["results_dir"])
            viz.plot_all(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                colloc_df=colloc_df,
                raw_df=df_featured,
                raw_train_df=train_raw,
            )
            # Additional Phase 7 specific plots
            _plot_phase7_figures(
                df_raw=df_featured,
                df_norm=df_norm,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                input_features=input_features,
                target_features=target_features,
                output_dir=paths["results_dir"],
            )

        wall_elapsed = time.perf_counter() - wall_start
        log.info("=" * 80)
        log.info("Phase 7 Dataset Pipeline COMPLETE in %.3f s", wall_elapsed)
        log.info("  Train: %d | Val: %d | Test: %d", len(train_df), len(val_df), len(test_df))
        log.info("=" * 80)
        return 0

    except Exception as exc:
        log.exception("Unexpected error in dataset pipeline: %s", exc)
        return 1


# ---------------------------------------------------------------------------
# Feature column helpers
# ---------------------------------------------------------------------------

def _get_feature_columns(df: pd.DataFrame, ds_cfg: dict) -> tuple[list[str], list[str]]:
    """Return (input_features, target_features) lists based on columns present."""
    base_inputs = ["elapsed_seconds", "delta_time", "x", "y", "z", "vx", "vy", "vz"]
    engineered = ["radius", "velocity_magnitude", "orbital_energy", "angular_momentum"]
    targets = ["delta_x", "delta_y", "delta_z", "delta_vx", "delta_vy", "delta_vz",
               "ax_res", "ay_res", "az_res"]

    fe_cfg = ds_cfg.get("feature_engineering", {})
    opt_inputs = [f for f in engineered if fe_cfg.get(f, True) and f in df.columns]
    input_features = [c for c in (base_inputs + opt_inputs) if c in df.columns]
    target_features = [c for c in targets if c in df.columns]

    return input_features, target_features


# ---------------------------------------------------------------------------
# Phase 7 additional figures
# ---------------------------------------------------------------------------

def _plot_phase7_figures(
    df_raw: pd.DataFrame,
    df_norm: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    input_features: list[str],
    target_features: list[str],
    output_dir: Path,
) -> None:
    """Render Phase 7 specific diagnostic figures (orbital mechanics + targets)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("dark_background")
    output_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger(__name__)

    try:
        # --- Figure 7: Orbital energy and angular momentum over time ---
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), facecolor="#0d0d0d")
        t = df_raw["elapsed_seconds"].to_numpy() / 3600.0
        for ax, col, label, color in [
            (axes[0], "orbital_energy", "Orbital Energy [km²/s²]", "#00D4FF"),
            (axes[1], "angular_momentum", "|L| [km²/s]", "#FFE66D"),
        ]:
            if col in df_raw.columns:
                ax.plot(t, df_raw[col], color=color, lw=0.8, alpha=0.85)
            ax.set_ylabel(label, color="white")
            ax.set_xlabel("Elapsed Time [hr]", color="white")
            ax.set_facecolor("#121212")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#333333")
        fig.suptitle("Orbital Energy and Angular Momentum (Raw)", color="white", fontsize=13)
        fig.tight_layout()
        fig.savefig(output_dir / "fig07_orbital_energy_momentum.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # --- Figure 8: Target distribution (delta states + residuals) ---
        fig, axes = plt.subplots(3, 3, figsize=(15, 10), facecolor="#0d0d0d")
        axes_flat = axes.flatten()
        colors8 = ["#00D4FF", "#FFE66D", "#FF6B6B",
                   "#4ECDC4", "#9B5DE5", "#F7B731",
                   "#FC5C65", "#45AAF2", "#26de81"]
        for idx, col in enumerate(target_features[:9]):
            ax = axes_flat[idx]
            if col in df_raw.columns:
                ax.hist(df_raw[col], bins=60, color=colors8[idx % len(colors8)], edgecolor="none", alpha=0.85)
            ax.set_title(col, color="white", fontsize=9)
            ax.set_facecolor("#121212")
            ax.tick_params(colors="white", labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#333333")
        for idx in range(len(target_features), 9):
            axes_flat[idx].set_visible(False)
        fig.suptitle("Target Feature Distributions (ΔS and a_res)", color="white", fontsize=13)
        fig.tight_layout()
        fig.savefig(output_dir / "fig08_target_distributions.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        log.info("Phase 7 extra figures saved to: %s", output_dir)

    except Exception as exc:
        log.warning("Failed to generate Phase 7 extra figures: %s", exc, exc_info=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_dataset_pipeline",
        description="Phase 7: State Transition Dataset Engineering.",
    )
    p.add_argument("--sat", type=str, default="INSAT3D",
                   metavar="SATELLITE_KEY", help="Satellite key. Default: INSAT3D")
    p.add_argument("--config", type=str, default=None,
                   metavar="PATH", help="Path to config.yaml.")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip figure generation.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sys.exit(run(satellite_key=args.sat, config_path=args.config, no_plots=args.no_plots))


if __name__ == "__main__":
    main()
