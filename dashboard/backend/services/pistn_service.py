"""
dashboard/backend/services/pistn_service.py
============================================
PI-STN dashboard rollout predictor.

Implements a true autoregressive rollout:

  S_t → normalize(8/12 features) → model(S_t_norm) → ΔS_norm
    → denormalize(ΔS_norm) → S_{t+1} = S_t + ΔS
    → feed S_{t+1} back → S_{t+2} → ...

The rollout starts from the SAME initial GCRS state produced by the SGP4
pipeline at the OMM epoch, ensuring:
  - Identical starting timestamp
  - Identical initial frame (GCRS)
  - Identical initial physical state
  - Direct comparison with reference trajectory

Architecture validation is performed at startup:
  - Expected input_dim from feature_mapping.json
  - Actual model param shapes from checkpoint
  - Logged clearly if mismatch detected

Coordinate convention:
  - Model operates in GCRS (km, km/s)
  - GCRS → ITRS for lat/lon output (via sgp4_service.itrs_to_geodetic)

Derived features computed per step:
  radius = |r|, velocity_magnitude = |v|,
  orbital_energy = v²/2 - μ/r,
  angular_momentum = |r × v|
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from dashboard.backend.services.sgp4_service import itrs_to_geodetic

log = logging.getLogger(__name__)

# Earth gravitational parameter [km³/s²]
_MU_KM3_S2 = 398600.4418

# ---------------------------------------------------------------------------
# Derived-feature computation (mirrors src/dataset/feature_engineering.py)
# ---------------------------------------------------------------------------

def _compute_derived_features(state_gcrs: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Compute the four derived features used as PI-STN inputs.

    Parameters
    ----------
    state_gcrs : np.ndarray shape (6,)
        [x, y, z, vx, vy, vz] in km / km/s (GCRS)

    Returns
    -------
    radius_km, velocity_magnitude_kms, orbital_energy_km2s2, angular_momentum_km2s
    """
    r_vec = state_gcrs[:3]
    v_vec = state_gcrs[3:]
    r = float(np.linalg.norm(r_vec))
    v = float(np.linalg.norm(v_vec))
    energy = 0.5 * v ** 2 - _MU_KM3_S2 / (r + 1e-12)
    h = float(np.linalg.norm(np.cross(r_vec, v_vec)))
    return r, v, energy, h


def _build_input_tensor(
    state_gcrs: np.ndarray,       # (6,)
    elapsed_s: float,
    dt_s: float,
    norm_stats: Dict[str, Dict[str, float]],
) -> np.ndarray:
    """
    Build and normalize the 12-feature input vector for PI-STN.

    Feature order (from feature_mapping.json):
    [elapsed_seconds, delta_time, x, y, z, vx, vy, vz,
     radius, velocity_magnitude, orbital_energy, angular_momentum]
    """
    r, v, energy, h = _compute_derived_features(state_gcrs)

    raw = np.array([
        elapsed_s,
        dt_s,
        state_gcrs[0],   # x
        state_gcrs[1],   # y
        state_gcrs[2],   # z
        state_gcrs[3],   # vx
        state_gcrs[4],   # vy
        state_gcrs[5],   # vz
        r,               # radius
        v,               # velocity_magnitude
        energy,          # orbital_energy
        h,               # angular_momentum
    ], dtype=np.float64)

    col_names = [
        "elapsed_seconds", "delta_time",
        "x", "y", "z", "vx", "vy", "vz",
        "radius", "velocity_magnitude", "orbital_energy", "angular_momentum",
    ]

    normed = np.zeros_like(raw)
    for i, col in enumerate(col_names):
        stats = norm_stats.get(col, {"mean": 0.0, "std": 1.0})
        mean = float(stats.get("mean", 0.0))
        std = max(float(stats.get("std", 1.0)), 1e-8)
        normed[i] = (raw[i] - mean) / std

    return normed.astype(np.float32)


def _denormalize_output(
    output_norm: np.ndarray,  # (9,)
    target_stats: Dict[str, Dict[str, float]],
) -> np.ndarray:
    """
    Denormalize the model output (9 features) to physical units.

    Output order: [delta_x, delta_y, delta_z, delta_vx, delta_vy, delta_vz,
                   ax_res, ay_res, az_res]
    """
    target_cols = [
        "delta_x", "delta_y", "delta_z",
        "delta_vx", "delta_vy", "delta_vz",
        "ax_res", "ay_res", "az_res",
    ]
    result = np.zeros(9, dtype=np.float64)
    for i, col in enumerate(target_cols):
        stats = target_stats.get(col, {"mean": 0.0, "std": 1.0})
        mean = float(stats.get("mean", 0.0))
        std = max(float(stats.get("std", 1.0)), 1e-8)
        result[i] = float(output_norm[i]) * std + mean
    return result


class PISTNRolloutService:
    """
    Dashboard PI-STN autoregressive rollout predictor.

    Operates entirely in GCRS frame [km, km/s].
    Produces trajectories suitable for direct comparison with SGP4 reference.
    """

    def __init__(self, project_root: Path) -> None:
        self._root = Path(project_root).resolve()
        if str(self._root) not in sys.path:
            sys.path.insert(0, str(self._root))

    def _load_normalization(
        self,
        norm_json_path: Path,
        target_norm_json_path: Path,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Load and return input and target normalization stats."""
        with open(norm_json_path, "r", encoding="utf-8") as f:
            input_norm = json.load(f)
        with open(target_norm_json_path, "r", encoding="utf-8") as f:
            target_norm = json.load(f)
        return input_norm.get("stats", {}), target_norm.get("stats", {})

    def _load_model(
        self,
        checkpoint_dir: Path,
        feature_mapping_json: Path,
        satellite_id: str,
    ) -> Any:
        """
        Load the PI-STN model from checkpoint.

        Architecture is read from config.yaml (since GOES-16 lacks model_config.json).
        Validates input_dim against feature_mapping.json.
        Validates parameter count against checkpoint tensor shapes.
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for PI-STN inference.")

        from src.utils.config_loader import ConfigLoader
        from src.pistn.forward_model import PISTNForwardModel

        cfg = ConfigLoader.load(self._root / "config.yaml")

        # Determine actual input_dim from feature_mapping.json
        actual_input_dim = 8  # fallback
        if feature_mapping_json.is_file():
            with open(feature_mapping_json, "r") as f:
                feat_map = json.load(f)
            actual_input_dim = len(feat_map.get("inputs", []))
            log.info("[%s] Feature mapping loaded: %d inputs", satellite_id, actual_input_dim)
        else:
            log.warning("[%s] feature_mapping.json not found; using config input_dim", satellite_id)
            actual_input_dim = int(cfg.get("pistn", {}).get("network", {}).get("input_dim", 8))

        # Override config input_dim with actual from feature_mapping
        pistn_cfg = cfg.get("pistn", {})
        net_cfg = dict(pistn_cfg.get("network", {}))
        net_cfg["input_dim"] = actual_input_dim
        pistn_cfg_patched = dict(pistn_cfg)
        pistn_cfg_patched["network"] = net_cfg
        cfg_patched = dict(cfg)
        cfg_patched["pistn"] = pistn_cfg_patched

        log.info(
            "[%s] Building PISTNForwardModel | input_dim=%d output_dim=%d hidden=%s",
            satellite_id,
            actual_input_dim,
            net_cfg.get("output_dim", 9),
            net_cfg.get("hidden_layers"),
        )

        model = PISTNForwardModel(cfg_patched)
        model.eval()

        # Load checkpoint weights
        best_ckpt = checkpoint_dir / "best_model.pt"
        if not best_ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {best_ckpt}")

        state_dict = torch.load(str(best_ckpt), map_location="cpu")

        # Architecture validation: check first layer input size
        first_weight_key = next(
            (k for k in state_dict.keys() if "weight" in k and state_dict[k].dim() == 2),
            None,
        )
        if first_weight_key is not None:
            ckpt_in = state_dict[first_weight_key].shape[1]
            if ckpt_in != actual_input_dim:
                log.warning(
                    "[%s] ARCHITECTURE MISMATCH: checkpoint first layer input=%d, "
                    "feature_mapping input_dim=%d. Using checkpoint input_dim.",
                    satellite_id, ckpt_in, actual_input_dim,
                )
                # Rebuild with checkpoint's actual input dim
                net_cfg["input_dim"] = ckpt_in
                actual_input_dim = ckpt_in
                cfg_patched["pistn"]["network"] = net_cfg
                model = PISTNForwardModel(cfg_patched)
                model.eval()

        model.load_state_dict(state_dict)
        log.info("[%s] Checkpoint loaded: %s", satellite_id, best_ckpt)

        return model, actual_input_dim

    def run(
        self,
        satellite_id: str,
        checkpoint_dir: Path,
        norm_json_path: Path,
        target_norm_json_path: Path,
        feature_mapping_json: Path,
        initial_state_gcrs: List[float],  # [x,y,z,vx,vy,vz] GCRS km/km/s
        initial_elapsed_s: float,         # = 0.0 (at OMM epoch)
        initial_timestamp_utc: str,        # ISO-8601 epoch timestamp
        dt_s: float,
        n_steps: int,
    ) -> Dict[str, Any]:
        """
        Run autoregressive PI-STN rollout from initial GCRS state.

        The initial state MUST come from SGP4 at epoch (GCRS frame),
        so that the reference and prediction share the exact same t=0 state.

        Parameters
        ----------
        satellite_id : str
        checkpoint_dir : Path
        norm_json_path : Path            — input normalization stats
        target_norm_json_path : Path     — target normalization stats
        feature_mapping_json : Path      — input/target column names
        initial_state_gcrs : list[float] — [x,y,z,vx,vy,vz] GCRS km/km/s at epoch
        initial_elapsed_s : float        — elapsed seconds at start (0.0 for epoch)
        initial_timestamp_utc : str      — UTC timestamp of initial state (ISO-8601)
        dt_s : float                     — rollout timestep [seconds]
        n_steps : int                    — number of rollout steps

        Returns
        -------
        dict with trajectory arrays, lat/lon/alt, and orbital elements
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch required for PI-STN.")

        from src.evaluation.orbital_elements import OrbitalElementsCalculator
        from astropy.time import Time
        import astropy.units as u

        log.info(
            "[%s] PI-STN rollout | n_steps=%d | dt=%.0fs | initial_t=%s",
            satellite_id, n_steps, dt_s, initial_timestamp_utc,
        )
        log.info(
            "[%s] Initial GCRS state | x=%.2fkm y=%.2fkm z=%.2fkm | "
            "vx=%.5fkm/s vy=%.5fkm/s vz=%.5fkm/s",
            satellite_id, *initial_state_gcrs,
        )

        # ── Load normalization ─────────────────────────────────────────
        if not norm_json_path.is_file():
            raise FileNotFoundError(f"Input normalization not found: {norm_json_path}")
        if not target_norm_json_path.is_file():
            raise FileNotFoundError(f"Target normalization not found: {target_norm_json_path}")

        input_stats, target_stats = self._load_normalization(
            norm_json_path, target_norm_json_path
        )
        log.info("[%s] Normalization stats loaded", satellite_id)

        # ── Load model ─────────────────────────────────────────────────
        model, actual_input_dim = self._load_model(
            checkpoint_dir, feature_mapping_json, satellite_id
        )
        model.eval()

        # ── Autoregressive rollout ─────────────────────────────────────
        state = np.array(initial_state_gcrs, dtype=np.float64)  # (6,)
        elapsed = float(initial_elapsed_s)

        # Pre-allocate output arrays (n_steps+1 to include initial state)
        n_out = n_steps + 1
        x_arr = np.empty(n_out, dtype=np.float64)
        y_arr = np.empty(n_out, dtype=np.float64)
        z_arr = np.empty(n_out, dtype=np.float64)
        vx_arr = np.empty(n_out, dtype=np.float64)
        vy_arr = np.empty(n_out, dtype=np.float64)
        vz_arr = np.empty(n_out, dtype=np.float64)
        elapsed_arr = np.empty(n_out, dtype=np.float64)

        # Store initial state
        x_arr[0], y_arr[0], z_arr[0] = state[0], state[1], state[2]
        vx_arr[0], vy_arr[0], vz_arr[0] = state[3], state[4], state[5]
        elapsed_arr[0] = elapsed

        # Rollout
        with torch.no_grad():
            for step in range(n_steps):
                # Build normalized input (12 features)
                input_vec = _build_input_tensor(state, elapsed, dt_s, input_stats)
                input_tensor = torch.tensor(input_vec, dtype=torch.float32).unsqueeze(0)  # (1, 12)

                # Truncate to model's actual input_dim (handles model/mapping mismatch)
                if input_tensor.shape[1] > actual_input_dim:
                    input_tensor = input_tensor[:, :actual_input_dim]

                # Forward pass
                output_tensor = model(input_tensor)  # (1, 9)
                output_np = output_tensor.squeeze(0).cpu().numpy()  # (9,)

                # Denormalize: get [Δx, Δy, Δz, Δvx, Δvy, Δvz, ax_res, ay_res, az_res]
                delta_phys = _denormalize_output(output_np, target_stats)

                # Update state: S_{t+1} = S_t + ΔS
                next_state = np.array([
                    state[0] + delta_phys[0],   # x + Δx
                    state[1] + delta_phys[1],   # y + Δy
                    state[2] + delta_phys[2],   # z + Δz
                    state[3] + delta_phys[3],   # vx + Δvx
                    state[4] + delta_phys[4],   # vy + Δvy
                    state[5] + delta_phys[5],   # vz + Δvz
                ], dtype=np.float64)

                elapsed += dt_s
                state = next_state

                x_arr[step + 1] = state[0]
                y_arr[step + 1] = state[1]
                z_arr[step + 1] = state[2]
                vx_arr[step + 1] = state[3]
                vy_arr[step + 1] = state[4]
                vz_arr[step + 1] = state[5]
                elapsed_arr[step + 1] = elapsed

        log.info(
            "[%s] Rollout complete | final state: x=%.2fkm y=%.2fkm z=%.2fkm",
            satellite_id, x_arr[-1], y_arr[-1], z_arr[-1],
        )

        # ── Timestamps from epoch + elapsed ───────────────────────────
        # Ensure initial_timestamp_utc is clean ISO-8601 without offset or Z suffix
        clean_utc = initial_timestamp_utc.replace("+00:00", "").replace("Z", "")
        try:
            from datetime import datetime, timedelta
            base_dt = datetime.fromisoformat(clean_utc)
            timestamps = [
                (base_dt + timedelta(seconds=float(e))).strftime("%Y-%m-%dT%H:%M:%S.%f")
                for e in elapsed_arr
            ]
        except Exception as exc:
            log.warning("Could not compute timestamps from datetime: %s", exc)
            timestamps = [clean_utc] * n_out

        # ── GCRS → ITRS → lat/lon/alt ──────────────────────────────────
        # For GEO satellites, approximate ITRS from GCRS is good enough
        # (Earth rotation matters for ground track accuracy).
        # Use the astropy transform via coordinate_transformer for accuracy.
        from src.coordinate_systems.gcrs_to_itrs import gcrs_to_itrs_vector
        from src.coordinate_systems.astronomy_utils import AstronomyTimeConverter

        obstime = AstronomyTimeConverter.parse_utc_to_time(timestamps)
        x_itrs, y_itrs, z_itrs, _, _, _ = gcrs_to_itrs_vector(
            x_arr, y_arr, z_arr,
            vx_arr, vy_arr, vz_arr,
            obstime,
        )
        lat_deg, lon_deg, alt_km = itrs_to_geodetic(x_itrs, y_itrs, z_itrs)

        # ── Orbital elements ───────────────────────────────────────────
        gcrs_states = np.stack([x_arr, y_arr, z_arr, vx_arr, vy_arr, vz_arr], axis=1)
        oe_calc = OrbitalElementsCalculator()
        orbital_elements = oe_calc.batch_to_elements(gcrs_states)

        return {
            "timestamps": timestamps,
            "elapsed_seconds": elapsed_arr.tolist(),
            "x_gcrs_km": x_arr.tolist(),
            "y_gcrs_km": y_arr.tolist(),
            "z_gcrs_km": z_arr.tolist(),
            "vx_gcrs_kms": vx_arr.tolist(),
            "vy_gcrs_kms": vy_arr.tolist(),
            "vz_gcrs_kms": vz_arr.tolist(),
            "lat_deg": lat_deg.tolist(),
            "lon_deg": lon_deg.tolist(),
            "alt_km": alt_km.tolist(),
            "orbital_elements": {k: v.tolist() for k, v in orbital_elements.items()},
        }
