"""
src/pinn/callbacks.py
=====================
Reusable callback utilities for Phase 8 (PINN Framework).

Defines the Callback base class and standard subclasses:
- CallbackManager (coordinates callback lifecycle calls)
- EarlyStopping (stops training if validation loss plateaus)
- ModelCheckpoint (automatically saves best/last model states)
- LearningRateMonitor (logs optimizer learning rate changes)
- LossHistory (accumulates epoch loss metrics for plotting)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class Callback:
    """Base class for all PINN training callbacks."""

    def on_train_begin(self, trainer: Any) -> None:
        """Called at the start of training."""
        pass

    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]) -> None:
        """Called at the end of every epoch."""
        pass

    def on_train_end(self, trainer: Any) -> None:
        """Called at the end of training."""
        pass


class CallbackManager:
    """Coordinates a list of Callbacks, calling lifecycle methods in order."""

    def __init__(self, callbacks: List[Callback]) -> None:
        self.callbacks = callbacks or []

    def on_train_begin(self, trainer: Any) -> None:
        for cb in self.callbacks:
            cb.on_train_begin(trainer)

    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(trainer, epoch, logs)

    def on_train_end(self, trainer: Any) -> None:
        for cb in self.callbacks:
            cb.on_train_end(trainer)


class EarlyStopping(Callback):
    """
    Stops training early if validation loss does not improve.

    Parameters
    ----------
    patience : int, optional
        Number of epochs to wait for improvement, by default 50.
    min_delta : float, optional
        Minimum change in monitored value to qualify as improvement, by default 1e-6.
    """

    def __init__(self, patience: int = 50, min_delta: float = 1e-6) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.wait = 0

    def on_train_begin(self, trainer: Any) -> None:
        self.best_loss = float("inf")
        self.wait = 0

    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]) -> None:
        val_loss = logs.get("val_total_loss")
        if val_loss is None:
            val_loss = logs.get("total_loss", float("inf"))

        if val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                log.warning("Early stopping triggered at epoch %d. Best loss: %.6e", epoch, self.best_loss)
                trainer.stop_training = True


class ModelCheckpoint(Callback):
    """
    Saves model checkpoints automatically when performance improves.

    Parameters
    ----------
    checkpoint_manager : CheckpointManager
        The checkpoint saving manager instance.
    save_best_only : bool, optional
        If True, only saves when validation loss improves, by default True.
    """

    def __init__(self, checkpoint_manager: Any, save_best_only: bool = True) -> None:
        self.checkpoint_manager = checkpoint_manager
        self.save_best_only = save_best_only
        self.best_loss = float("inf")

    def on_train_begin(self, trainer: Any) -> None:
        self.best_loss = float("inf")

    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]) -> None:
        # Save last checkpoint regardless
        self.checkpoint_manager.save(
            network=trainer.forward_model.network,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            epoch=epoch,
            best_loss=self.best_loss,
            loss_history=trainer.loss_history.history,
            is_best=False,
        )

        # Monitored loss
        val_loss = logs.get("val_total_loss")
        if val_loss is None:
            val_loss = logs.get("total_loss", float("inf"))

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            log.info("Validation loss improved to %.6e. Saving best model checkpoint.", self.best_loss)
            self.checkpoint_manager.save(
                network=trainer.forward_model.network,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                epoch=epoch,
                best_loss=self.best_loss,
                loss_history=trainer.loss_history.history,
                is_best=True,
            )


class LearningRateMonitor(Callback):
    """Logs learning rate changes at the end of each epoch."""

    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]) -> None:
        # Get active learning rate from optimizer
        lrs = [param_group["lr"] for param_group in trainer.optimizer.param_groups]
        if lrs:
            logs["lr"] = lrs[0]
            if epoch % 100 == 0 or epoch == 1:
                log.info("Epoch %5d | LR: %.4e | Total Loss: %.5e", epoch, lrs[0], logs.get("total_loss", 0.0))


class LossHistory(Callback):
    """Accumulates losses over training epochs for diagnostic plotting."""

    def __init__(self) -> None:
        self.history: Dict[str, List[float]] = {}

    def on_epoch_end(self, trainer: Any, epoch: int, logs: Dict[str, float]) -> None:
        for k, v in logs.items():
            if k not in self.history:
                self.history[k] = []
            self.history[k].append(float(v))
