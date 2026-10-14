"""
src/utils/path_manager.py
=========================
Resolves all project-level filesystem paths from the loaded configuration.

Usage
-----
    from src.utils.config_loader import ConfigLoader
    from src.utils.path_manager import PathManager

    cfg = ConfigLoader.load()
    pm  = PathManager(cfg, satellite_key="GOES16")

    pm.omm_file          # Path to GOES16.txt
    pm.processed_dir     # data/processed/GOES16/
    pm.truth_csv         # data/processed/GOES16/truth_states.csv
    pm.propagation_log   # logs/propagation/propagation.log
    pm.results_dir       # results/propagation/
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class PathManager:
    """
    Resolves and owns all filesystem paths for a single propagation run.

    Parameters
    ----------
    cfg : dict[str, Any]
        Project configuration loaded by :class:`~src.utils.config_loader.ConfigLoader`.
    satellite_key : str
        Key into ``cfg["satellites"]`` (e.g. ``"GOES16"``).

    Attributes
    ----------
    root : Path
        Absolute project root directory.
    satellite_key : str
        Canonical satellite identifier.
    satellite_cfg : dict[str, Any]
        Per-satellite config block.
    omm_file : Path
        Full path to the OMM text file.
    processed_dir : Path
        Output directory for processed data (auto-created on access).
    truth_csv : Path
        Full path to the output ``truth_states.csv``.
    log_dir : Path
        Directory for propagation log files.
    propagation_log : Path
        Full path to ``propagation.log``.
    results_dir : Path
        Directory for result plots.

    Raises
    ------
    KeyError
        If *satellite_key* is not found in the config.
    FileNotFoundError
        If the resolved OMM file does not exist on disk.
    """

    def __init__(self, cfg: dict[str, Any], satellite_key: str) -> None:
        self._cfg = cfg
        self.satellite_key = satellite_key

        if satellite_key not in cfg.get("satellites", {}):
            raise KeyError(
                f"Satellite '{satellite_key}' not found in config. "
                f"Available: {list(cfg.get('satellites', {}).keys())}"
            )

        self.satellite_cfg: dict[str, Any] = cfg["satellites"][satellite_key]
        self.root = Path(cfg["_root"]).resolve()
        self._build_paths()

    # ------------------------------------------------------------------
    # Public path attributes
    # ------------------------------------------------------------------

    @property
    def omm_file(self) -> Path:
        """Absolute path to the OMM source file."""
        return self._omm_file

    @property
    def processed_dir(self) -> Path:
        """Absolute path to the processed output directory (created if absent)."""
        self._processed_dir.mkdir(parents=True, exist_ok=True)
        return self._processed_dir

    @property
    def truth_csv(self) -> Path:
        """Absolute path to ``truth_states.csv`` inside :attr:`processed_dir`."""
        return self.processed_dir / self._cfg["propagation"]["output_csv"]

    @property
    def log_dir(self) -> Path:
        """Absolute path to the propagation log directory (created if absent)."""
        self._log_dir.mkdir(parents=True, exist_ok=True)
        return self._log_dir

    @property
    def propagation_log(self) -> Path:
        """Absolute path to ``propagation.log``."""
        return self.log_dir / "propagation.log"

    @property
    def results_dir(self) -> Path:
        """Absolute path to the propagation results directory (created if absent)."""
        self._results_dir.mkdir(parents=True, exist_ok=True)
        return self._results_dir

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_paths(self) -> None:
        """Resolve and validate all path objects from config."""
        paths_cfg = self._cfg["paths"]

        # Raw OMM
        omm_dir = self.root / paths_cfg["data"]["raw"]["omm"]
        self._omm_file = omm_dir / self.satellite_cfg["omm_file"]
        if not self._omm_file.is_file():
            raise FileNotFoundError(
                f"OMM file not found: {self._omm_file}\n"
                f"Expected satellite_key='{self.satellite_key}' → "
                f"omm_file='{self.satellite_cfg['omm_file']}'"
            )

        # Processed output
        processed_base = self.root / paths_cfg["data"]["processed"]["base"]
        self._processed_dir = processed_base / self.satellite_cfg["processed_dir"]

        # Logs — each satellite gets its own subdirectory under logs/propagation/
        self._log_dir = self.root / paths_cfg["logs"]["propagation"] / self.satellite_key

        # Results — each satellite gets its own subdirectory under results/propagation/
        self._results_dir = self.root / paths_cfg["results"]["propagation"] / self.satellite_key

    def __repr__(self) -> str:
        return (
            f"PathManager(satellite={self.satellite_key!r}, "
            f"root={str(self.root)!r})"
        )
