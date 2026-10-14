"""
src/physics/gradient_debugger.py
=================================
Step 5 — Gradient Diagnostics for Physics-Informed Neural Networks.

Provides ``GradientDebugger``, a utility class that inspects every parameter
in a PyTorch ``nn.Module`` and reports:

  • Gradient norm (L2)
  • Max absolute gradient
  • Min absolute gradient
  • Number of NaN gradients
  • Number of Inf gradients
  • Number of disconnected parameters (grad is None)

Usage
-----
    from src.physics.gradient_debugger import GradientDebugger

    debugger = GradientDebugger(log_file="logs/training/gradient_debug.log")

    # After loss.backward() — call inside training loop
    report = debugger.diagnose(network, epoch=epoch, batch_id=batch_id)

    # Generate gradient flow PNG (Step 11)
    debugger.plot_gradient_flow(network, save_path="results/training/gradient_flow.png")
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class GradientDebugger:
    """
    Diagnostic utility for monitoring gradient health during PINN training.

    Parameters
    ----------
    log_file : str or Path, optional
        Path to the gradient debug log file.
        Defaults to ``logs/training/gradient_debug.log``.
    log_every_n_epochs : int
        How frequently (in epochs) to emit full diagnostics. Default: 1.
    """

    def __init__(
        self,
        log_file: str | Path = "logs/training/gradient_debug.log",
        log_every_n_epochs: int = 1,
    ) -> None:
        self._log_every = max(1, log_every_n_epochs)
        self._log_file = Path(log_file)
        self._log_file.parent.mkdir(parents=True, exist_ok=True)

        # Dedicated file handler for gradient debug log
        self._file_handler = logging.FileHandler(self._log_file, mode="a", encoding="utf-8")
        self._file_handler.setLevel(logging.DEBUG)
        fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
        self._file_handler.setFormatter(fmt)
        log.addHandler(self._file_handler)

    def diagnose(
        self,
        network: "nn.Module",
        epoch: int = 0,
        batch_id: int = 0,
    ) -> Dict[str, Any]:
        """
        Inspect all named parameters in ``network`` for gradient health.

        Must be called **after** ``loss.backward()`` and **before** ``optimizer.step()``.

        Parameters
        ----------
        network : nn.Module
            The neural network to inspect.
        epoch : int
            Current training epoch (for logging).
        batch_id : int
            Current mini-batch index (for logging).

        Returns
        -------
        Dict[str, Any]
            Summary report with per-layer stats and overall flags.
        """
        if not TORCH_AVAILABLE:
            log.warning("PyTorch unavailable — GradientDebugger skipped.")
            return {}

        if epoch % self._log_every != 0:
            return {}

        layer_stats: List[Dict[str, Any]] = []
        total_nan   = 0
        total_inf   = 0
        disconnected = 0

        for name, param in network.named_parameters():
            if param.grad is None:
                disconnected += 1
                layer_stats.append({
                    "layer": name,
                    "grad_norm": float("nan"),
                    "grad_max":  float("nan"),
                    "grad_min":  float("nan"),
                    "n_nan":     0,
                    "n_inf":     0,
                    "connected": False,
                })
                log.debug(
                    "[Epoch %d | Batch %d] DISCONNECTED param: %s (grad=None)",
                    epoch, batch_id, name,
                )
                continue

            g = param.grad.detach()
            n_nan  = int(torch.isnan(g).sum().item())
            n_inf  = int(torch.isinf(g).sum().item())
            g_norm = float(torch.norm(g).item())
            g_max  = float(g.abs().max().item())
            g_min  = float(g.abs().min().item())

            total_nan += n_nan
            total_inf += n_inf

            layer_stats.append({
                "layer":     name,
                "grad_norm": g_norm,
                "grad_max":  g_max,
                "grad_min":  g_min,
                "n_nan":     n_nan,
                "n_inf":     n_inf,
                "connected": True,
            })

            if n_nan > 0 or n_inf > 0:
                log.error(
                    "[Epoch %d | Batch %d] BAD GRAD in '%s': "
                    "n_nan=%d  n_inf=%d  norm=%.4e  max=%.4e  min=%.4e",
                    epoch, batch_id, name, n_nan, n_inf, g_norm, g_max, g_min,
                )
            else:
                log.debug(
                    "[Epoch %d | Batch %d] '%s': norm=%.4e  max=%.4e  min=%.4e",
                    epoch, batch_id, name, g_norm, g_max, g_min,
                )

        report = {
            "epoch":             epoch,
            "batch_id":          batch_id,
            "total_nan":         total_nan,
            "total_inf":         total_inf,
            "disconnected_params": disconnected,
            "gradient_healthy":  (total_nan == 0 and total_inf == 0 and disconnected == 0),
            "layers":            layer_stats,
        }

        if not report["gradient_healthy"]:
            log.error(
                "[Epoch %d] Gradient health FAILED: total_nan=%d  total_inf=%d  disconnected=%d",
                epoch, total_nan, total_inf, disconnected,
            )
        else:
            # Compute overall norm for summary
            total_norm = math.sqrt(
                sum(s["grad_norm"] ** 2 for s in layer_stats if math.isfinite(s["grad_norm"]))
            )
            log.debug(
                "[Epoch %d] Gradient health OK: overall_norm=%.4e  layers=%d",
                epoch, total_norm, len(layer_stats),
            )

        return report

    def plot_gradient_flow(
        self,
        network: "nn.Module",
        save_path: str | Path = "results/training/gradient_flow.png",
    ) -> Optional[Path]:
        """
        Step 11 — Generate gradient flow visualization.

        Plots the mean absolute gradient for each named parameter layer.
        Immediately reveals dead layers (near-zero gradient) or exploding
        gradients (very large gradient).

        Parameters
        ----------
        network : nn.Module
            The trained/training network.
        save_path : str or Path
            Where to save the PNG figure.

        Returns
        -------
        Path or None
            The saved file path, or None if plotting failed.
        """
        if not TORCH_AVAILABLE or not MATPLOTLIB_AVAILABLE:
            log.warning("Cannot plot gradient flow — PyTorch or Matplotlib unavailable.")
            return None

        layers, mean_grads, max_grads = [], [], []

        for name, param in network.named_parameters():
            if param.grad is not None and param.requires_grad:
                g = param.grad.detach().abs()
                layers.append(name)
                mean_grads.append(float(g.mean().item()))
                max_grads.append(float(g.max().item()))

        if not layers:
            log.warning("No gradients available for plotting. Call after loss.backward().")
            return None

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(max(10, len(layers) * 0.6), 5))
        x = range(len(layers))
        ax.bar(x, max_grads,  alpha=0.5, label="Max |grad|",  color="#e74c3c")
        ax.bar(x, mean_grads, alpha=0.8, label="Mean |grad|", color="#2980b9")
        ax.set_xticks(list(x))
        ax.set_xticklabels(layers, rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Gradient magnitude")
        ax.set_title("Gradient Flow — Layer-wise Gradient Norms")
        ax.legend()
        ax.set_yscale("symlog", linthresh=1e-8)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

        log.info("Gradient flow figure saved: %s", save_path)
        return save_path
