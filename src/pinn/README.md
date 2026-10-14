# Phase 8 — Physics-Informed Neural Network (PINN) Framework

This sub-package contains the complete **Physics-Informed Neural Network Framework** for the **Satellite PINN State Estimation** project. It is written in PyTorch and provides the networks, activation functions, optimizers, loss functions, training loops, evaluators, and checkpoint managers to train Scientific Machine Learning models.

---

## Folder Structure

```text
src/
pinn/
    __init__.py
    activations.py             # Sin and standard activation factories
    network.py                 # Fully-connected MLP neural network layout
    optimizer_factory.py       # Resolves Adam, SGD, and L-BFGS optimizers
    scheduler_factory.py       # Resolves step, cosine, or plateau lr decay
    model_factory.py           # Jointly instantiates model, optimizer, scheduler
    forward_model.py           # Wrapped prediction pass with device mapping
    losses.py                  # Computes MSE observations and physics losses
    callbacks.py               # Early stopping, LR monitor, history logger
    checkpoint.py              # Serializes training parameters and resume states
    evaluator.py               # Runs validation checks and calculates RMSE/MAEs
    predictor.py               # High-level inference and trajectory extrapolation
    metrics.py                 # Trajectory error statistics (RMSE, MAE, MAPE)
    validator.py               # Instability validators (NaN gradients/losses)
    visualizer.py              # Headless matplotlib diagnostic plots
    trainer.py                 # coordinates training loops (Adam + L-BFGS)
    pinn_engine.py             # Main API gateway orchestrator class
    run_training.py            # CLI training pipeline entry-point
    README.md
```

---

## Architectural Principles

### 1. Unified public API (`PINNEngine`)
External packages (and CLI runner scripts) interact exclusively with the model via `PINNEngine`. This abstracts away the trainer, evaluator, predictor, and model factory, providing clean methods: `train()`, `validate()`, `predict()`, `evaluate()`, `save()`, and `load()`.

### 2. Multi-Stage Optimization
PINNs have highly complex loss landscapes. The trainer supports standard stochastic gradient descent (Adam/AdamW/SGD) to find a global region, followed by second-order optimization (L-BFGS) utilizing line searches (`strong_wolfe`) to converge to high-precision physical trajectory coordinates.

### 3. Separation of Concerns
All gravity field computations, state derivatives, and physical constraints (minimum radius, escape velocity) are handled by the Phase 7 `PhysicsEngine`. The neural network only learns the coordinate mapping:
$$\text{Time } t \to \text{State } [x, y, z, v_x, v_y, v_z]$$

---

## Mathematical Design

### 1. Unified Loss Formula
$$\mathcal{L}_{total} = w_{obs} \mathcal{L}_{obs} + w_{phys} \mathcal{L}_{physics} + w_{bound} \mathcal{L}_{bound} + w_{init} \mathcal{L}_{init} + w_{const} \mathcal{L}_{const}$$
Where:
- $\mathcal{L}_{obs}$: Mean Squared Error (MSE) compared to observations.
- $\mathcal{L}_{physics}$: MSE of physics residuals evaluated at collocation points (via `PhysicsEngine`).
- $\mathcal{L}_{init}$: Initial condition state matching at $t=0$.
- $\mathcal{L}_{const}$: One-sided quadratic constraint penalties.

### 2. Custom Sin Activation
To resolve high-order derivatives cleanly without vanishing gradients during deep backpropagation:
$$\text{Sin}(x) = \sin(\omega_0 x)$$
where $\omega_0$ is a configurable scaling factor (default 1.0).

---

## Configuration

Add the following block to `config.yaml`:
```yaml
pinn:
  network:
    input_dim: 1
    output_dim: 6
    hidden_layers: 8
    neurons: 128
    activation: "tanh"         # tanh | relu | silu | gelu | sin
    dropout: 0.0
    initializer: "xavier"      # xavier | kaiming | normal
  optimizer:
    name: "adam"               # adam | adamw | lbfgs | sgd
    learning_rate: 0.001
    weight_decay: 1.0e-5
    # L-BFGS settings
    max_iter: 20
    history_size: 100
  scheduler:
    name: "cosine"             # cosine | step | plateau | none
    step_size: 1000
    gamma: 0.5
    patience: 10
  training:
    epochs: 500
    batch_size: 1024
    gradient_clip: 1.0
    mixed_precision: false
    device: "auto"             # auto | cuda | cpu
    pin_memory: true
    num_workers: 0
  losses:
    observation_weight: 1.0
    physics_weight: 10.0
    boundary_weight: 5.0
    initial_weight: 5.0
    constraint_weight: 1.0
  checkpoint:
    dir: "checkpoints"
    save_best_only: true
```

---

## Execution & Resuming

To start a new training run or resume an existing check-pointed training run:
```bash
python -m src.pinn.run_training --sat GOES16
```

To run a fast validation test (e.g. 5 epochs, no plots):
```bash
python -m src.pinn.run_training --sat GOES16 --epochs 5 --no-plots
```

Logs are saved to `logs/training/training.log`, checkpoints are written to `checkpoints/`, and figures are rendered under `results/training/`.
