"""
src/losses/scheduled_sampler.py
==============================
Scheduled sampling probability manager for PI-STN rollout training.

Controls the probability of using teacher forcing (ground-truth next state)
versus autonomous rollout (predicted next state) as training progresses.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

log: logging.Logger = logging.getLogger(__name__)


class ScheduledSampler:
    """
    Manages the teacher-forcing probability decay schedule.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        ss_cfg = cfg.get("pistn", {}).get("scheduled_sampling", {})
        
        self.enabled = bool(ss_cfg.get("enabled", False))
        self.decay_type = ss_cfg.get("decay_type", "linear").lower()
        self.p_init = float(ss_cfg.get("initial_probability", 1.0))
        self.p_min = float(ss_cfg.get("minimum_probability", 0.05))
        self.decay_epochs = int(ss_cfg.get("decay_epochs", 100))

        if self.enabled:
            log.info(
                "ScheduledSampler initialized: enabled=True, type=%s, "
                "p_init=%.2f, p_min=%.2f, decay_epochs=%d",
                self.decay_type, self.p_init, self.p_min, self.decay_epochs
            )
        else:
            log.info("ScheduledSampler: scheduled sampling is disabled (using K-step rollout settings).")

    def get_probability(self, epoch: int) -> float:
        """
        Returns the teacher-forcing probability p for a given training epoch.
        epoch is 1-indexed.
        """
        if not self.enabled:
            # If scheduled sampling is disabled, default to 0.0 teacher forcing
            # (which means fully autonomous rollout after step 1).
            return 0.0

        if epoch <= 1:
            return self.p_init

        if self.decay_type == "linear":
            # Linear decay from p_init to p_min over decay_epochs
            decay_rate = (self.p_init - self.p_min) / max(self.decay_epochs, 1)
            p = self.p_init - (epoch - 1) * decay_rate
            return max(self.p_min, min(self.p_init, p))
            
        elif self.decay_type == "exponential":
            # Exponential decay factor to reach p_min at decay_epochs
            # p = p_init * factor^(epoch-1)
            ratio = self.p_min / max(self.p_init, 1e-8)
            factor = ratio ** (1.0 / max(self.decay_epochs, 1))
            p = self.p_init * (factor ** (epoch - 1))
            return max(self.p_min, min(self.p_init, p))
            
        else:
            log.warning("Unknown decay_type '%s' in ScheduledSampler, using linear.", self.decay_type)
            decay_rate = (self.p_init - self.p_min) / max(self.decay_epochs, 1)
            p = self.p_init - (epoch - 1) * decay_rate
            return max(self.p_min, min(self.p_init, p))
