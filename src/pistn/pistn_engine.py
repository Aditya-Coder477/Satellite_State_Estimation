"""
src/pistn/pistn_engine.py
=========================
Public API for Phase 9 (PI-STN Framework).

Coordinates dataset loaders, optimizer, model state, de-normalizer, and trainers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

# PyTorch import handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.dataset.pytorch_dataset import ObservationDataset as PyTorchDataset
from src.physics.physics_engine import PhysicsEngine
from src.pistn.forward_model import PISTNForwardModel
from src.pistn.losses import PISTNLoss
from src.pistn.normalization import DeNormalizer
from src.pistn.optimizer_factory import OptimizerFactory
from src.pistn.scheduler_factory import SchedulerFactory
from src.pistn.trainer import PISTNTrainer
from src.pistn.predictor import PISTNPredictor
from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class PISTNEngine:
    """
    Main coordinate gateway for training and executing PI-STN state transitions.
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        checkpoint_dir: Path,
        norm_json_path: Path,
        target_norm_json_path: Path,
    ) -> None:
        self.cfg = cfg
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize de-normalizer
        self.denormalizer = DeNormalizer(norm_json_path, target_norm_json_path)

        # Initialize core components
        self.physics_engine = PhysicsEngine(cfg)
        self.model = PISTNForwardModel(cfg)
        self.loss_fn = PISTNLoss(cfg)

        # Setup Optimizer and Scheduler
        opt_cfg = cfg.get("pistn", {}).get("optimizer", {})
        opt_name = opt_cfg.get("name", "adamw")
        lr = float(opt_cfg.get("lr", 1e-3))
        weight_decay = float(opt_cfg.get("weight_decay", 1e-5))

        self.optimizer = OptimizerFactory.create(
            name=opt_name,
            params=self.model.parameters() if TORCH_AVAILABLE else [],
            lr=lr,
            weight_decay=weight_decay,
        )

        sched_cfg = cfg.get("pistn", {}).get("scheduler", {})
        sched_name = sched_cfg.get("name", "cosine")
        epochs = int(cfg.get("pistn", {}).get("training", {}).get("epochs", 500))

        self.scheduler = SchedulerFactory.create(
            name=sched_name,
            optimizer=self.optimizer,
            epochs=epochs,
        )

        self.trainer = PISTNTrainer(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            loss_fn=self.loss_fn,
            physics_engine=self.physics_engine,
            denormalizer=self.denormalizer,
            checkpoint_dir=self.checkpoint_dir,
            cfg=self.cfg,
        )

        self.checkpoint_manager = self.trainer.checkpoint_manager
        self.predictor = PISTNPredictor(self.model, self.physics_engine, self.denormalizer)

    def train_model(self, train_loader: Any, val_loader: Any) -> Dict[str, Any]:
        """Runs model training."""
        return self.trainer.train(train_loader, val_loader, device=self.device)

    def predict(self, loader: Any, output_path: Path) -> None:
        """Runs predictions."""
        self.predictor.predict_to_csv(loader, output_path, device=self.device)
