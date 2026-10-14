"""
src/pinn/__init__.py
====================
Phase 8: Physics-Informed Neural Network (PINN) Framework.

Exposes network layouts, activation factories, optimizer factories, loss calculators,
checkpointers, validation checkpoints, and orchestrator engines.

Public API
----------
- PINNNetwork            : Configurable PyTorch feed-forward MLP.
- ActivationFactory      : Custom sine and standard activation modules.
- OptimizerFactory       : Resolves Adam, SGD, and L-BFGS.
- PINNModelFactory       : Jointly instantiates network, optimizer, and scheduler.
- ForwardModel           : Wrapped execution layer with automatic device placement.
- PINNLossCalculator     : Computes observation, physics, boundary, and initial losses.
- PINNEngine             : Main API gateway orchestrating training, prediction, and evaluation.
- PINNValidator          : Quality control checker (finiteness, gradients, structures).
- PINNVisualizer         : Heads-up plotting utilities for loss trends and coordinates.
"""

from __future__ import annotations

from src.pinn.activations import ActivationFactory, Sin, Swish
from src.pinn.callbacks import (
    Callback,
    CallbackManager,
    EarlyStopping,
    LearningRateMonitor,
    LossHistory,
    ModelCheckpoint,
)
from src.pinn.checkpoint import CheckpointManager
from src.pinn.evaluator import PINNEvaluator
from src.pinn.forward_model import ForwardModel
from src.pinn.losses import PINNLossCalculator
from src.pinn.metrics import compute_metrics
from src.pinn.model_factory import PINNModelFactory
from src.pinn.network import PINNNetwork
from src.pinn.optimizer_factory import OptimizerFactory
from src.pinn.pinn_engine import PINNEngine
from src.pinn.predictor import PINNPredictor
from src.pinn.scheduler_factory import SchedulerFactory
from src.pinn.validator import PINNValidator
from src.pinn.visualizer import PINNVisualizer

__all__ = [
    "ActivationFactory",
    "Sin",
    "Swish",
    "Callback",
    "CallbackManager",
    "EarlyStopping",
    "LearningRateMonitor",
    "LossHistory",
    "ModelCheckpoint",
    "CheckpointManager",
    "PINNEvaluator",
    "ForwardModel",
    "PINNLossCalculator",
    "compute_metrics",
    "PINNModelFactory",
    "PINNNetwork",
    "OptimizerFactory",
    "PINNEngine",
    "PINNPredictor",
    "SchedulerFactory",
    "PINNValidator",
    "PINNVisualizer",
]
