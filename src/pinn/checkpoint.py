"""
src/pinn/checkpoint.py
======================
Checkpoint manager for Phase 8 (PINN Framework).

Handles serialization and deserialization of network weights, optimizer states,
scheduler parameters, and training metrics (loss history) to allow training runs
to resume seamlessly from failures.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

# PyTorch import handling
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class CheckpointManager:
    """
    Manages saving and loading model training state checkpoints.

    Parameters
    ----------
    checkpoint_dir : Path or str
        Directory where checkpoints are stored.
    """

    def __init__(self, checkpoint_dir: Path | str) -> None:
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log.info("CheckpointManager initialised at: %s", self.checkpoint_dir)

    def save(
        self,
        network: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        best_loss: float,
        loss_history: Dict[str, List[float]],
        is_best: bool = False,
    ) -> None:
        """
        Saves the current training state to disk.

        Saves model weights, optimizer/scheduler parameters, and a JSON
        containing state metadata.

        Parameters
        ----------
        network : nn.Module
            The neural network.
        optimizer : torch.optim.Optimizer
            The active optimizer.
        scheduler : Any
            The learning rate scheduler.
        epoch : int
            Current epoch index.
        best_loss : float
            Best validation loss recorded so far.
        loss_history : Dict[str, List[float]]
            Accumulated loss history.
        is_best : bool, optional
            If True, saves as the 'best_model.pt' checkpoint, by default False.
        """
        if not TORCH_AVAILABLE:
            log.warning("PyTorch not available. Skipping checkpoint saving.")
            return

        prefix = "best" if is_best else "last"
        
        # Paths
        model_path = self.checkpoint_dir / f"{prefix}_model.pt"
        optimizer_path = self.checkpoint_dir / f"{prefix}_optimizer.pt"
        scheduler_path = self.checkpoint_dir / f"{prefix}_scheduler.pt"
        state_path = self.checkpoint_dir / f"{prefix}_training_state.json"

        try:
            # 1. Save Network State Dict
            torch.save(network.state_dict(), model_path)
            
            # 2. Save Optimizer State Dict
            torch.save(optimizer.state_dict(), optimizer_path)

            # 3. Save Scheduler State Dict
            if scheduler is not None:
                torch.save(scheduler.state_dict(), scheduler_path)

            # 4. Save JSON Training State
            training_state = {
                "epoch": epoch,
                "best_loss": best_loss,
                "loss_history": loss_history,
            }
            with open(state_path, "w", encoding="utf-8") as fh:
                json.dump(training_state, fh, indent=4)

            log.debug("Saved %s checkpoint to: %s", prefix, self.checkpoint_dir)
        except Exception as e:
            log.error("Failed to save training checkpoint: %s", str(e), exc_info=True)

    def load_latest(
        self,
        network: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any = None,
    ) -> Tuple[int, float, Dict[str, List[float]]] | None:
        """
        Loads the latest ('last') saved checkpoint if it exists.

        Parameters
        ----------
        network : nn.Module
            The neural network.
        optimizer : torch.optim.Optimizer
            The active optimizer.
        scheduler : Any, optional
            The learning rate scheduler, by default None.

        Returns
        -------
        Tuple[int, float, Dict[str, List[float]]] | None
            A tuple containing (last_epoch, best_loss, loss_history), or None
            if no checkpoint exists.
        """
        if not TORCH_AVAILABLE:
            log.warning("PyTorch not available. Skipping checkpoint loading.")
            return None

        # Check for last checkpoint metadata
        state_path = self.checkpoint_dir / "last_training_state.json"
        model_path = self.checkpoint_dir / "last_model.pt"
        optimizer_path = self.checkpoint_dir / "last_optimizer.pt"
        scheduler_path = self.checkpoint_dir / "last_scheduler.pt"

        if not state_path.is_file() or not model_path.is_file():
            log.info("No checkpoint found at %s. Starting training from scratch.", self.checkpoint_dir)
            return None

        try:
            log.info("Loading latest checkpoint from: %s", self.checkpoint_dir)
            
            # Load weights
            network.load_state_dict(torch.load(model_path, map_location="cpu"))
            
            # Load optimizer state
            if optimizer_path.is_file():
                optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu"))

            # Load scheduler state
            if scheduler is not None and scheduler_path.is_file():
                scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu"))

            # Load training metadata
            with open(state_path, "r", encoding="utf-8") as fh:
                state_data = json.load(fh)

            epoch = int(state_data.get("epoch", 0))
            best_loss = float(state_data.get("best_loss", float("inf")))
            loss_history = state_data.get("loss_history", {})

            log.info(
                "Checkpoint loaded successfully: Resuming at epoch %d, previous best loss: %.6e",
                epoch,
                best_loss,
            )
            return epoch, best_loss, loss_history

        except Exception as e:
            log.error("Failed to load checkpoint: %s. Starting from scratch.", str(e), exc_info=True)
            return None
