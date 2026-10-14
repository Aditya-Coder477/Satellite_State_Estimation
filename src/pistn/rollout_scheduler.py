"""
src/pistn/rollout_scheduler.py
================================
Curriculum rollout length scheduler for multi-step rollout training.

Provides configurable, epoch-aware rollout step counts following a user-defined
curriculum schedule from config.yaml (pistn.rollout_training.schedule).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class RolloutScheduler:
    """
    Manages curriculum-based multi-step rollout schedule.

    Reads the ``pistn.rollout_training`` block from config and returns the
    appropriate rollout step count K for any given training epoch.

    Config format::

        pistn:
          rollout_training:
            enabled: true
            curriculum: true
            schedule:
              - epochs: [1, 20]
                rollout_steps: 1
              - epochs: [21, 40]
                rollout_steps: 2
              - epochs: [41, 60]
                rollout_steps: 5
              - epochs: [61, 100]
                rollout_steps: 10
              - epochs: [101, 9999]
                rollout_steps: 20
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        rollout_cfg = cfg.get("pistn", {}).get("rollout_training", {})

        self._enabled: bool = bool(rollout_cfg.get("enabled", False))
        self._curriculum: bool = bool(rollout_cfg.get("curriculum", True))

        # Parse schedule entries into (start_epoch, end_epoch, rollout_steps) tuples
        schedule_raw: List[Dict[str, Any]] = rollout_cfg.get("schedule", [])
        self._schedule: List[Tuple[int, int, int]] = []

        for entry in schedule_raw:
            epoch_range = entry.get("epochs", [1, 9999])
            steps = int(entry.get("rollout_steps", 1))
            start = int(epoch_range[0])
            end = int(epoch_range[1])
            self._schedule.append((start, end, steps))

        # Fallback if schedule is empty
        if not self._schedule:
            self._schedule = [(1, 9999, 1)]

        if self._enabled:
            log.info(
                "RolloutScheduler initialized: enabled=%s, curriculum=%s, %d schedule entries.",
                self._enabled, self._curriculum, len(self._schedule)
            )
            for start, end, steps in self._schedule:
                log.info("  Epochs [%d – %d] → K = %d rollout steps", start, end, steps)
        else:
            log.info("RolloutScheduler: rollout training is disabled (K=1 always).")

    def is_enabled(self) -> bool:
        """Returns True if multi-step rollout training is active."""
        return self._enabled

    def get_rollout_steps(self, epoch: int) -> int:
        """
        Returns the number of rollout steps K for the given epoch.

        - If disabled: always returns 1 (one-step teacher-forced training).
        - If enabled without curriculum: returns the last schedule entry's steps.
        - If enabled with curriculum: returns the K defined for this epoch range.
        """
        if not self._enabled:
            return 1

        if not self._curriculum:
            # No curriculum — use max steps from last schedule entry
            return self._schedule[-1][2] if self._schedule else 1

        for start, end, steps in self._schedule:
            if start <= epoch <= end:
                return steps

        # Past the last defined range — use last entry's value
        return self._schedule[-1][2] if self._schedule else 1
