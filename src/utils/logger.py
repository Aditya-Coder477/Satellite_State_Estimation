"""
src/utils/logger.py
===================
Centralised logging factory for the project.

Every module obtains its logger via::

    from src.utils.logger import get_logger
    log = get_logger(__name__)

Call :func:`setup_logging` once at application startup (e.g. in the entry
point script) to configure handlers.  Subsequent calls to :func:`get_logger`
will inherit the configured root handler(s).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any


def setup_logging(
    cfg: dict[str, Any],
    log_file: Path | str | None = None,
) -> None:
    """
    Configure the root logger from the project configuration.

    Parameters
    ----------
    cfg : dict[str, Any]
        Full project config (as returned by :class:`~src.utils.config_loader.ConfigLoader`).
    log_file : Path | str | None
        Absolute path to the log file.  When *None* no file handler is added.

    Notes
    -----
    * Calling this function multiple times is safe — existing handlers are
      cleared and replaced so that duplicate log lines are avoided.
    * A ``StreamHandler`` writing to *stderr* is added when
      ``cfg["logging"]["console"]`` is ``True``.
    """
    log_cfg: dict[str, Any] = cfg.get("logging", {})
    level_str: str = log_cfg.get("level", "INFO").upper()
    level: int = getattr(logging, level_str, logging.INFO)
    fmt: str = log_cfg.get(
        "format",
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    datefmt: str = log_cfg.get("datefmt", "%Y-%m-%dT%H:%M:%S")
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    root = logging.getLogger()
    root.setLevel(level)

    # Remove any pre-existing handlers to avoid duplicates
    root.handlers.clear()

    # ── Console handler ────────────────────────────────────────────────
    if log_cfg.get("console", True):
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        root.addHandler(ch)

    # ── File handler ───────────────────────────────────────────────────
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        root.addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    """
    Return a named :class:`logging.Logger`.

    Parameters
    ----------
    name : str
        Logger name — conventionally ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
    """
    return logging.getLogger(name)
