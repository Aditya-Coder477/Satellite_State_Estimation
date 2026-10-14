"""
src/dataset/transition_generator.py
=====================================
Transition Pair Generator for the PI-STN Dataset.

Converts a chronological state trajectory into (St, t, Δt) → (ΔS, a_res)
training pairs required by the Physics-Informed State Transition Network.

Design
------
- Input  : [time, x, y, z, vx, vy, vz] at step  i
- Target : [Δx, Δy, Δz, Δvx, Δvy, Δvz, ax_res, ay_res, az_res] from i → i+1
- Never shuffles data; preserves chronological order.
- Fully vectorised using NumPy (no Python loops over samples).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)

# Physical constants
_MU_KM3_S2 = 398600.4418   # Earth gravitational parameter [km³/s²]


class TransitionGenerator:
    """Generates state-transition pairs from a chronological trajectory."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        ds_cfg = cfg.get("dataset", {})
        tr_cfg = ds_cfg.get("transition", {})
        self.delta_time_s = float(tr_cfg.get("delta_time_seconds", 60.0))
        self.enabled = bool(tr_cfg.get("enabled", True))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """
        Build transition-pair dataset from a state trajectory DataFrame.

        Parameters
        ----------
        df_states : pd.DataFrame
            Trajectory with columns:
            utc_time, elapsed_seconds,
            x_km, y_km, z_km, vx_km_s, vy_km_s, vz_km_s,
            ax_res, ay_res, az_res   (residual accelerations from Phase 3)

        Returns
        -------
        pd.DataFrame
            Transition pairs.  N-1 rows (one per consecutive pair).
        """
        if not self.enabled:
            log.warning("Transition generation disabled in config; returning empty DataFrame.")
            return pd.DataFrame()

        log.info(
            "Building transition pairs from %d states (Δt = %.1f s) …",
            len(df_states), self.delta_time_s,
        )

        df = df_states.sort_values("elapsed_seconds").reset_index(drop=True)

        # ── Extract numpy arrays for speed ────────────────────────────
        t    = df["elapsed_seconds"].to_numpy(dtype=np.float64)
        x    = df["x_km"].to_numpy(dtype=np.float64)
        y    = df["y_km"].to_numpy(dtype=np.float64)
        z    = df["z_km"].to_numpy(dtype=np.float64)
        vx   = df["vx_km_s"].to_numpy(dtype=np.float64)
        vy   = df["vy_km_s"].to_numpy(dtype=np.float64)
        vz   = df["vz_km_s"].to_numpy(dtype=np.float64)
        ax   = df["ax_res"].to_numpy(dtype=np.float64)
        ay   = df["ay_res"].to_numpy(dtype=np.float64)
        az   = df["az_res"].to_numpy(dtype=np.float64)

        utc_times = df["utc_time"].to_numpy()

        # ── Vectorised delta computation (current index vs next) ───────
        i  = np.arange(len(df) - 1)       # current sample indices
        i1 = i + 1                          # next sample indices

        dt      = t[i1] - t[i]
        delta_x  = x[i1]  - x[i]
        delta_y  = y[i1]  - y[i]
        delta_z  = z[i1]  - z[i]
        delta_vx = vx[i1] - vx[i]
        delta_vy = vy[i1] - vy[i]
        delta_vz = vz[i1] - vz[i]

        # Residual accelerations are taken from current sample
        # (they represent the instantaneous disturbance at that epoch)
        ax_res = ax[i]
        ay_res = ay[i]
        az_res = az[i]

        pairs = pd.DataFrame({
            # --- Identity ------------------------------------------------
            "utc_time":        utc_times[i],
            "elapsed_seconds": t[i],
            "delta_time":      dt,
            # --- Current state -------------------------------------------
            "x":  x[i],  "y":  y[i],  "z":  z[i],
            "vx": vx[i], "vy": vy[i], "vz": vz[i],
            # --- Targets -------------------------------------------------
            "delta_x":  delta_x,  "delta_y":  delta_y,  "delta_z":  delta_z,
            "delta_vx": delta_vx, "delta_vy": delta_vy, "delta_vz": delta_vz,
            # --- Residual dynamics ---------------------------------------
            "ax_res": ax_res, "ay_res": ay_res, "az_res": az_res,
        })

        log.info("Generated %d transition pairs.", len(pairs))
        return pairs
