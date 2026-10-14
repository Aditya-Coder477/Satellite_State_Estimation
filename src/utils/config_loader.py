"""
src/utils/config_loader.py
==========================
Loads and validates the project-wide YAML configuration.

Usage
-----
    from src.utils.config_loader import ConfigLoader

    cfg = ConfigLoader.load()          # uses default config.yaml at project root
    cfg = ConfigLoader.load("my.yaml") # explicit path
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class ConfigLoader:
    """
    Loads the project YAML configuration and resolves the project root path.

    The project root is defined as the directory containing `config.yaml`.
    All relative paths stored in the config are resolved against this root
    at load-time and stored under ``cfg["_root"]``.
    """

    _DEFAULT_FILENAME = "config.yaml"

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> dict[str, Any]:
        """
        Load and return the YAML configuration as a nested dictionary.

        Parameters
        ----------
        config_path : str | Path | None
            Explicit path to ``config.yaml``.  When *None* the loader
            searches upward from the current working directory until it
            finds the file or reaches the filesystem root.

        Returns
        -------
        dict[str, Any]
            Parsed configuration with an extra ``_root`` key set to the
            resolved absolute project root directory.

        Raises
        ------
        FileNotFoundError
            If the configuration file cannot be located.
        yaml.YAMLError
            If the file contains invalid YAML.
        """
        path = cls._resolve_path(config_path)
        with open(path, "r", encoding="utf-8") as fh:
            cfg: dict[str, Any] = yaml.safe_load(fh)

        cfg["_root"] = str(path.parent.resolve())
        return cfg

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _resolve_path(cls, config_path: str | Path | None) -> Path:
        """
        Return an absolute ``Path`` to the config file.

        If *config_path* is given it is used directly (after validation).
        Otherwise, the method walks from ``cwd`` up to the filesystem root,
        returning the first directory that contains ``config.yaml``.
        """
        if config_path is not None:
            p = Path(config_path).resolve()
            if not p.is_file():
                raise FileNotFoundError(f"Config file not found: {p}")
            return p

        # Walk up from cwd
        current = Path.cwd().resolve()
        while True:
            candidate = current / cls._DEFAULT_FILENAME
            if candidate.is_file():
                return candidate
            parent = current.parent
            if parent == current:
                raise FileNotFoundError(
                    f"'{cls._DEFAULT_FILENAME}' not found in '{Path.cwd()}' or "
                    "any of its parent directories."
                )
            current = parent

    @staticmethod
    def abs_path(cfg: dict[str, Any], *relative_parts: str) -> Path:
        """
        Resolve a relative config path against the project root.

        Parameters
        ----------
        cfg : dict[str, Any]
            The config dict returned by :meth:`load`.
        *relative_parts : str
            Path segments (joined with ``os.path.join``) relative to root.

        Returns
        -------
        Path
            Absolute ``Path`` object.

        Examples
        --------
        >>> p = ConfigLoader.abs_path(cfg, "data/raw/omm", "GOES16.txt")
        """
        root = Path(cfg["_root"])
        return root.joinpath(os.path.join(*relative_parts))
