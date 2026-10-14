"""
src/physics/autograd_utils.py
=============================
PyTorch automatic differentiation utilities for Phase 7 (Physics Engine).

Provides helpers for batched first and second derivatives of neural network
outputs with respect to inputs, bypassing row-by-row loops for fast training on GPU.

Bug Fixes (Phase 9 — Physics Residual NaN elimination):
  * Removed deprecated ``only_inputs=True`` (removed in PyTorch ≥ 2.0).
  * Added ``allow_unused=False`` — raises ``AutogradDisconnectedError`` instead of
    silently returning None (which propagated as NaN).
  * Added pre-condition diagnostic logging (requires_grad, grad_fn, device, dtype).
  * Added NaN guard on each computed gradient column.
"""

from __future__ import annotations

import logging

# PyTorch import handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.utils.logger import get_logger

log: logging.Logger = get_logger(__name__)


class AutogradDisconnectedError(RuntimeError):
    """
    Raised when ``torch.autograd.grad`` returns None, indicating that
    the computational graph is broken — the output is not a function of
    the input time tensor.

    Root cause is almost always one of:
      1. ``t_colloc`` was indexed/sliced without ``.clone().detach().requires_grad_(True)``
         before the forward pass (most common in mini-batch loops).
      2. ``torch.no_grad()`` context wraps the forward pass.
      3. The network was called with ``t.detach()`` before entering the model.
    """


class AutogradUtils:
    """
    Automatic differentiation utilities for PyTorch tensors.
    """

    @staticmethod
    def compute_first_derivative(t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes the first derivative dy/dt of state y with respect to time t.

        Parameters
        ----------
        t : torch.Tensor
            Time tensor of shape (Batch, 1). **Must** have ``requires_grad=True``
            and must be connected to ``y`` through the computational graph.
        y : torch.Tensor
            State tensor of shape (Batch, D).

        Returns
        -------
        torch.Tensor
            The derivative dy/dt of shape (Batch, D).

        Raises
        ------
        ImportError
            If PyTorch is not installed.
        ValueError
            If ``t`` does not require gradients or has wrong shape.
        AutogradDisconnectedError
            If the gradient of any output column w.r.t. ``t`` is None (disconnected
            computational graph).
        RuntimeError
            If any gradient column contains NaN or Inf values.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use AutogradUtils.")

        # ── Pre-condition diagnostics (Step 1) ─────────────────────────────
        log.debug(
            "compute_first_derivative | t.shape=%s  t.requires_grad=%s  "
            "t.device=%s  t.dtype=%s  y.grad_fn=%s",
            tuple(t.shape),
            t.requires_grad,
            t.device,
            t.dtype,
            y.grad_fn,
        )

        if not t.requires_grad:
            raise ValueError(
                "Input time tensor 't' must have requires_grad=True to compute derivatives. "
                f"Current state: shape={tuple(t.shape)}, device={t.device}, dtype={t.dtype}. "
                "Fix: use t = t.clone().detach().requires_grad_(True) before the forward pass. "
                "In mini-batch loops, ensure each sliced batch tensor is re-attached: "
                "t_batch = t_full[idx].clone().detach().requires_grad_(True)."
            )

        if y.grad_fn is None:
            raise ValueError(
                "State tensor 'y' has no grad_fn — it is not part of the computational graph. "
                "Ensure the network forward pass receives the requires_grad=True 't' tensor "
                "and is NOT wrapped inside torch.no_grad()."
            )

        dy_dt = []
        n_dims = y.shape[1]

        for i in range(n_dims):
            y_col = y[:, i : i + 1]  # shape (B, 1)

            # ── Step 2: allow_unused=False — never silently return None ─────
            grads = torch.autograd.grad(
                outputs=y_col,
                inputs=t,
                grad_outputs=torch.ones_like(y_col),
                create_graph=True,
                retain_graph=True,
                allow_unused=False,          # raises if disconnected
            )
            grad_col = grads[0]              # shape (B, 1)

            # ── Guard: grad must not be None after allow_unused=False ───────
            if grad_col is None:
                raise AutogradDisconnectedError(
                    f"torch.autograd.grad returned None for output dimension {i}. "
                    "The computational graph is broken between y and t. "
                    "Ensure t.requires_grad=True before the forward pass and that "
                    "no .detach() or torch.no_grad() interrupts the graph."
                )

            # ── Step 6 partial: NaN/Inf guard on each gradient column ───────
            if not torch.isfinite(grad_col).all():
                n_nan = torch.isnan(grad_col).sum().item()
                n_inf = torch.isinf(grad_col).sum().item()
                bad_idx = (~torch.isfinite(grad_col)).nonzero(as_tuple=False)
                log.error(
                    "NaN/Inf detected in gradient for state dimension %d: "
                    "n_nan=%d, n_inf=%d, first_bad_index=%s, value=%s",
                    i, n_nan, n_inf,
                    bad_idx[0].tolist() if len(bad_idx) > 0 else "none",
                    grad_col[bad_idx[0][0]].item() if len(bad_idx) > 0 else "n/a",
                )
                raise RuntimeError(
                    f"NaN/Inf detected in autograd gradient for state dim {i}. "
                    f"n_nan={n_nan}, n_inf={n_inf}. "
                    "Check for extreme input values or exploding gradients in the network."
                )

            dy_dt.append(grad_col)

        return torch.cat(dy_dt, dim=-1)  # shape (B, D)

    @classmethod
    def compute_second_derivative(cls, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes the second derivative d²y/dt² of state y with respect to time t.

        Parameters
        ----------
        t : torch.Tensor
            Time tensor of shape (Batch, 1). Must have requires_grad=True.
        y : torch.Tensor
            State tensor of shape (Batch, D).

        Returns
        -------
        torch.Tensor
            The second derivative d²y/dt² of shape (Batch, D).
        """
        dy_dt = cls.compute_first_derivative(t, y)
        d2y_dt2 = cls.compute_first_derivative(t, dy_dt)
        return d2y_dt2

    @staticmethod
    def compute_jacobian(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes the batched Jacobian dy/dx.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (Batch, N_in).
        y : torch.Tensor
            Output tensor of shape (Batch, N_out).

        Returns
        -------
        torch.Tensor
            Jacobian tensor of shape (Batch, N_out, N_in).
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use AutogradUtils.")

        batch_size = y.shape[0]
        n_out = y.shape[1]
        n_in = x.shape[1]

        jacobian = torch.zeros(batch_size, n_out, n_in, dtype=y.dtype, device=y.device)

        for i in range(n_out):
            y_col = y[:, i : i + 1]
            grads = torch.autograd.grad(
                outputs=y_col,
                inputs=x,
                grad_outputs=torch.ones_like(y_col),
                create_graph=True,
                retain_graph=True,
                allow_unused=False,
            )
            if grads[0] is None:
                raise AutogradDisconnectedError(
                    f"Jacobian: gradient is None for output dim {i}. "
                    "Ensure x.requires_grad=True and is connected to y."
                )
            jacobian[:, i, :] = grads[0]

        return jacobian
