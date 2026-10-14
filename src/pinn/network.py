"""
src/pinn/network.py
===================
Configurable Multi-Layer Perceptron (MLP) architecture for Phase 8 (PINN Framework).

Implements the standard feed-forward network with configurable inputs, outputs,
layers, neuron counts, dropouts, activations, and weight initialization strategies.
"""

from __future__ import annotations

import logging
from typing import Any

# PyTorch import handling
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    # Stub base class for compilation safety
    class nn:  # type: ignore
        Module = object
    TORCH_AVAILABLE = False

from src.pinn.activations import ActivationFactory
from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


if TORCH_AVAILABLE:
    class PINNNetwork(nn.Module):
        """
        Multi-Layer Perceptron (MLP) for Physics-Informed Neural Networks.

        Parameters
        ----------
        input_dim : int
            Dimensionality of the input (typically 1 for time 't').
        output_dim : int
            Dimensionality of the output (typically 6 for orbital states [x,y,z,vx,vy,vz]).
        hidden_layers : int
            Number of hidden Linear layers in the network.
        neurons : int
            Number of neurons (width) in each hidden layer.
        activation : str
            Name of the activation function (e.g. "tanh", "sin").
        dropout : float, optional
            Dropout probability, by default 0.0.
        initializer : str, optional
            Weight initialization strategy: "xavier", "kaiming", or "normal", by default "xavier".
        """

        def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_layers: int,
            neurons: int,
            activation: str,
            dropout: float = 0.0,
            initializer: str = "xavier",
        ) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.hidden_layers = hidden_layers
            self.neurons = neurons
            self.initializer = initializer

            layers = []

            # 1. Input layer
            layers.append(nn.Linear(input_dim, neurons))
            layers.append(ActivationFactory.create(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

            # 2. Hidden layers
            for _ in range(hidden_layers - 1):
                layers.append(nn.Linear(neurons, neurons))
                layers.append(ActivationFactory.create(activation))
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))

            # 3. Output layer
            layers.append(nn.Linear(neurons, output_dim))

            self.net = nn.Sequential(*layers)
            self._init_weights()

            log.info(
                "PINNNetwork initialised: layers=%d, width=%d, activation='%s', initializer='%s', parameters=%d",
                hidden_layers,
                neurons,
                activation,
                initializer,
                sum(p.numel() for p in self.parameters() if p.requires_grad),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Evaluates the network for the input tensor x."""
            return self.net(x)

        def _init_weights(self) -> None:
            """Applies weight initialization to all linear layers."""
            init_name = self.initializer.lower()
            
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    if init_name == "xavier":
                        nn.init.xavier_normal_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    elif init_name == "kaiming":
                        # tanh/sine work well with Xavier, relu works well with Kaiming
                        nn.init.kaiming_normal_(m.weight, nonlinearity="tanh")
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    elif init_name == "normal":
                        nn.init.normal_(m.weight, mean=0.0, std=0.02)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    else:
                        # Fallback to standard PyTorch default
                        pass
else:
    # Stub if PyTorch is not available
    class PINNNetwork:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None: pass
