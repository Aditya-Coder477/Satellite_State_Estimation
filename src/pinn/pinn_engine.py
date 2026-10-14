"""
src/pinn/pinn_engine.py
=======================
Unified public API for Phase 8 (PINN Framework).

Exposes the ``PINNEngine`` class orchestrating neural networks, forward models,
loss calculators, trainers, evaluators, and checkpoint managers.
Other modules should interact with the pinn package solely through this class.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# PyTorch import handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.physics.physics_engine import PhysicsEngine
from src.pinn.callbacks import EarlyStopping, ModelCheckpoint
from src.pinn.checkpoint import CheckpointManager
from src.pinn.evaluator import PINNEvaluator
from src.pinn.forward_model import ForwardModel
from src.pinn.losses import PINNLossCalculator
from src.pinn.model_factory import PINNModelFactory
from src.pinn.predictor import PINNPredictor
from src.pinn.trainer import PINNTrainer
from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class PINNEngine:
    """
    Main PINN execution engine.

    Wraps networks, optimizers, and helpers to expose simplified APIs.

    Parameters
    ----------
    cfg : Dict[str, Any]
        Project configuration dictionary.
    physics_engine : PhysicsEngine
        PhysicsEngine instance.
    """

    def __init__(self, cfg: Dict[str, Any], physics_engine: PhysicsEngine) -> None:
        self._cfg = cfg
        self.physics_engine = physics_engine
        
        pinn_cfg = cfg.get("pinn", {})
        train_cfg = pinn_cfg.get("training", {})
        
        # 1. Device Selection
        self.device = self._resolve_device(train_cfg.get("device", "auto"))

        # 2. Build Model Components
        if TORCH_AVAILABLE:
            self.network, self.optimizer, self.scheduler = PINNModelFactory.create_model(cfg)
            self.forward_model = ForwardModel(self.network, self.device)
        else:
            self.network, self.optimizer, self.scheduler = None, None, None
            self.forward_model = None
            log.warning("PyTorch not found. PINNEngine running in NumPy fallback mode (predict/evaluate only).")

        # 3. Instantiate Helper Components
        checkpoint_dir = Path(pinn_cfg.get("checkpoint", {}).get("dir", "checkpoints"))
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        
        if TORCH_AVAILABLE:
            self.loss_calculator = PINNLossCalculator(cfg, physics_engine)
            self.trainer = PINNTrainer(
                forward_model=self.forward_model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                loss_calculator=self.loss_calculator,
                checkpoint_manager=self.checkpoint_manager,
                cfg=cfg,
            )
            self.evaluator = PINNEvaluator(self.forward_model, physics_engine)
            self.predictor = PINNPredictor(self.forward_model)
        else:
            self.loss_calculator = None
            self.trainer = None
            self.evaluator = None
            self.predictor = None

        log.info("PINNEngine initialised on device: %s", self.device)

    def train(self, train_df: Any, val_df: Any, colloc_df: Any) -> Dict[str, List[float]]:
        """
        Loads dataset dataframes and trains the PINN model.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training observations split.
        val_df : pd.DataFrame
            Validation observations split.
        colloc_df : pd.DataFrame
            Collocation points.

        Returns
        -------
        Dict[str, List[float]]
            Training loss histories.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to train the PINN model.")

        log.info("Preparing training tensors...")
        
        # Extract observations
        t_obs_np = train_df["elapsed_seconds"].values.astype(np.float32)
        pos_cols = ["x_gcrs_km", "y_gcrs_km", "z_gcrs_km"]
        vel_cols = ["vx_gcrs_kms", "vy_gcrs_kms", "vz_gcrs_kms"]
        state_obs_np = train_df[pos_cols + vel_cols].values.astype(np.float32)

        # Extract collocation
        t_colloc_np = colloc_df["time"].values.astype(np.float32)

        # Place tensors on device
        t_obs = torch.tensor(t_obs_np, dtype=torch.float32, device=self.device).unsqueeze(1)
        state_obs_true = torch.tensor(state_obs_np, dtype=torch.float32, device=self.device)
        t_colloc = torch.tensor(t_colloc_np, dtype=torch.float32, device=self.device).unsqueeze(1).requires_grad_(True)

        # Set up callbacks
        # EarlyStopping: stop training if validation total loss does not improve for 100 epochs
        early_stop = EarlyStopping(patience=100)
        checkpoint = ModelCheckpoint(self.checkpoint_manager, save_best_only=True)
        self.trainer.set_callbacks([early_stop, checkpoint])

        # Try to resume from checkpoint
        resume_state = self.checkpoint_manager.load_latest(self.network, self.optimizer, self.scheduler)
        if resume_state is not None:
            # Checkpoint loaded successfully
            last_epoch, best_loss, loss_history = resume_state
            self.trainer.loss_history.history = loss_history
            log.info("Resuming training from epoch %d.", last_epoch)

        log.info("Starting PINN training...")
        return self.trainer.train(t_obs, state_obs_true, t_colloc, val_df)

    def validate(self, val_df: Any) -> Dict[str, float]:
        """Runs evaluation on the validation split."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for validation.")
        return self.evaluator.evaluate(val_df)

    def evaluate(self, test_df: Any) -> Dict[str, float]:
        """Runs evaluation on the test split."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for testing.")
        return self.evaluator.evaluate(test_df)

    def predict(self, times: np.ndarray) -> np.ndarray:
        """Generates satellite Cartesian state predictions for the given time stamps."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for prediction.")
        return self.predictor.predict_trajectory(times)

    def save(self, is_best: bool = True) -> None:
        """Saves current network weights."""
        if not TORCH_AVAILABLE:
            return
        prefix = "best" if is_best else "last"
        model_path = self.checkpoint_manager.checkpoint_dir / f"{prefix}_model.pt"
        torch.save(self.network.state_dict(), model_path)
        log.info("Saved model state dict to: %s", model_path)

    def load(self, is_best: bool = True) -> None:
        """Loads saved network weights."""
        if not TORCH_AVAILABLE:
            return
        prefix = "best" if is_best else "last"
        model_path = self.checkpoint_manager.checkpoint_dir / f"{prefix}_model.pt"
        if not model_path.is_file():
            raise FileNotFoundError(f"Saved weights not found at: {model_path}")
        self.network.load_state_dict(torch.load(model_path, map_location=self.device))
        log.info("Loaded model state dict from: %s", model_path)

    def _resolve_device(self, device_cfg: str) -> str:
        """Helper to resolve CPU/GPU device selection based on config."""
        if not TORCH_AVAILABLE:
            return "cpu"
            
        device_lower = device_cfg.lower()
        if device_lower == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device_lower == "cuda":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                log.warning("CUDA requested in config, but GPU is not available. Falling back to CPU.")
        else:
            device = "cpu"
            
        return device
