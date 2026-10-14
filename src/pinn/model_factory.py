"""
src/pinn/model_factory.py
=========================
Orchestrates model instantiation for Phase 8 (PINN Framework).

Builds the entire training package: instantiates the network, connects the
configured optimizer, and links the learning rate scheduler.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

# PyTorch import handling
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.pinn.network import PINNNetwork
from src.pinn.optimizer_factory import OptimizerFactory
from src.pinn.scheduler_factory import SchedulerFactory
from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class PINNModelFactory:
    """Instantiates the joint neural network, optimizer, and scheduler components."""

    @classmethod
    def create_model(
        cls,
        cfg: Dict[str, Any],
    ) -> Tuple[nn.Module, torch.optim.Optimizer, Any]:
        """
        Creates the complete PINN model training components.

        Parameters
        ----------
        cfg : Dict[str, Any]
            Project configuration.

        Returns
        -------
        Tuple[nn.Module, torch.optim.Optimizer, Any]
            A tuple of (network, optimizer, scheduler).

        Raises
        ------
        ImportError
            If PyTorch is not installed.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to construct the model components.")

        pinn_cfg = cfg.get("pinn", {})
        net_cfg = pinn_cfg.get("network", {})
        opt_cfg = pinn_cfg.get("optimizer", {})
        sched_cfg = pinn_cfg.get("scheduler", {})

        # 1. Create Network
        network = PINNNetwork(
            input_dim=int(net_cfg.get("input_dim", 1)),
            output_dim=int(net_cfg.get("output_dim", 6)),
            hidden_layers=int(net_cfg.get("hidden_layers", 8)),
            neurons=int(net_cfg.get("neurons", 128)),
            activation=str(net_cfg.get("activation", "tanh")),
            dropout=float(net_cfg.get("dropout", 0.0)),
            initializer=str(net_cfg.get("initializer", "xavier")),
        )

        # 2. Create Optimizer
        # Pass extra options for L-BFGS (max_iter, history_size)
        opt_kwargs = {
            "max_iter": opt_cfg.get("max_iter", 20),
            "history_size": opt_cfg.get("history_size", 100),
        }
        optimizer = OptimizerFactory.create(
            name=str(opt_cfg.get("name", "adam")),
            params=network.parameters(),
            lr=float(opt_cfg.get("learning_rate", 0.001)),
            weight_decay=float(opt_cfg.get("weight_decay", 1e-5)),
            **opt_kwargs,
        )

        # 3. Create Scheduler
        sched_kwargs = {
            "step_size": sched_cfg.get("step_size", 1000),
            "gamma": sched_cfg.get("gamma", 0.5),
            "patience": sched_cfg.get("patience", 10),
        }
        scheduler = SchedulerFactory.create(
            name=str(sched_cfg.get("name", "cosine")),
            optimizer=optimizer,
            **sched_kwargs,
        )

        return network, optimizer, scheduler
