"""
src/pistn/checkpoint.py
========================
Checkpoint saving and loading utilities for Phase 9 (PI-STN).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

# PyTorch import handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class CheckpointManager:
    """
    Saves and loads training state checkpoints.
    """

    def __init__(self, checkpoint_dir: Path) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        model: Any,
        optimizer: Any,
        scheduler: Any,
        epoch: int,
        history: Dict[str, Any],
        config: Dict[str, Any],
        is_best: bool = False,
    ) -> None:
        """
        Saves all training elements as files under checkpoints/.
        """
        if not TORCH_AVAILABLE:
            log.error("PyTorch is required to save checkpoints.")
            return

        try:
            # Save model config
            with open(self.checkpoint_dir / "model_config.json", "w") as f:
                json.dump(config, f, indent=4)

            # Save training history
            with open(self.checkpoint_dir / "training_history.json", "w") as f:
                json.dump(history, f, indent=4)

            # PyTorch states dict
            torch.save(model.state_dict(), self.checkpoint_dir / "last_model.pt")
            torch.save(optimizer.state_dict(), self.checkpoint_dir / "optimizer.pt")
            if scheduler is not None:
                torch.save(scheduler.state_dict(), self.checkpoint_dir / "scheduler.pt")

            # Save best if requested
            if is_best:
                torch.save(model.state_dict(), self.checkpoint_dir / "best_model.pt")
                log.info("Saved new best model checkpoint at epoch %d.", epoch)

        except Exception as e:
            log.error("Failed to save checkpoint: %s", str(e), exc_info=True)

    def load_checkpoint(self, model: Any, load_best: bool = True) -> int:
        """
        Loads weight states back into the model. Returns last completed epoch.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to load checkpoints.")

        filename = "best_model.pt" if load_best else "last_model.pt"
        path = self.checkpoint_dir / filename

        if not path.is_file():
            log.warning("No checkpoint file found at %s. Model remains uninitialized.", path)
            return 0

        try:
            log.info("Loading model weights from checkpoint: %s", path)
            model.load_state_dict(torch.load(path, map_location="cpu"))
            
            # Retrieve epoch count from history
            history_path = self.checkpoint_dir / "training_history.json"
            if history_path.is_file():
                with open(history_path, "r") as f:
                    history = json.load(f)
                return len(history.get("train_loss_history", []))
            
            return 0
        except Exception as e:
            log.error("Failed to load model checkpoint: %s", str(e), exc_info=True)
            return 0
