# Phase 7 — Physics Engine

This sub-package contains the **Physics Engine** for the **Satellite PINN State Estimation** project. It provides the mathematical equations, automatic differentiation utilities, physical constraints, and loss component calculations required to train Physics-Informed Neural Networks (PINNs).

---

## Folder Structure

```text
src/
physics/
    __init__.py
    constants.py                 # Resolves astrodynamic constants from JSON
    coordinate_dynamics.py       # Position/velocity vector helper functions
    gravity_models.py            # Gravity models (TwoBody, J2) using Strategy Pattern
    orbital_equations.py         # Analytical state derivative formulation
    autograd_utils.py            # Vectorized PyTorch autograd gradients
    residuals.py                 # Calculates position/velocity residuals
    constraints.py               # Enforces minimum radius and escape velocities
    loss_components.py           # Evaluates loss terms and weighted sum
    physics_engine.py            # Orchestrator and unified public API
    validator.py                 # Dimensional and magnitude check suites
    visualizer.py                # Headless matplotlib diagnostic plots
    run_physics_validation.py    # CLI validation entry-point
    README.md
```

---

## Architectural Principles

### 1. Decoupled from Neural Networks
The Physics Engine does not define neural network layers, optimization parameters, or epoch loops. It operates purely on predicted states and time, making it reusable across PINNs, Neural ODEs, Hamiltonian Neural Networks, or classical numerical integrators.

### 2. Strategy Pattern for Gravity Models
Gravitational acceleration is computed using the Strategy Pattern. The active model can be changed in `config.yaml` to `two_body` or `j2` without modifying the core orbital equations or residual calculators.

### 3. Vectorized Automatic Differentiation
Using PyTorch's `torch.autograd.grad` with batch-summed gradients:
$$\frac{\partial S_j}{\partial t} = \text{Autograd}\left(S_j, \ t, \ \vec{1}\right)$$
This avoids row-by-row python loops, allowing highly efficient GPU-accelerated backpropagation during neural network training.

---

## Mathematical Formulations

### 1. Two-Body & J2 Gravity Acceleration
The gravitational acceleration vector $\vec{a}_{grav}$ at position $\vec{r} = [x, y, z]$ with norm $r = |\vec{r}|$ is:
$$\vec{a}_{grav} = \vec{a}_{2body} + \vec{a}_{J2}$$
$$\vec{a}_{2body} = -\mu \frac{\vec{r}}{r^3}$$
$$\vec{a}_{J2} = -\frac{3 J_2 \mu R_e^2}{2 r^5} \left[ x\left(1 - 5\frac{z^2}{r^2}\right)\hat{i} + y\left(1 - 5\frac{z^2}{r^2}\right)\hat{j} + z\left(3 - 5\frac{z^2}{r^2}\right)\hat{k} \right]$$

### 2. Differentiable Constraint Penalties
To guide the neural network away from unphysical regimes, we formulate one-sided quadratic boundary penalties:
$$\mathcal{L}_{radius} = \max\left(0, \ R_{min} - r\right)^2$$
$$\mathcal{L}_{velocity} = \max\left(0, \ v - V_{max}\right)^2$$
These penalties are continuous and differentiable, allowing direct optimization.

---

## Configuration

Add the following block to `config.yaml`:
```yaml
physics:
  gravity_model: "two_body"     # two_body | j2 | j4
  residual:
    reduction: "mean"
  constraints:
    radius_positive: true
    min_radius_km: 6378.137     # Earth radius limit
    max_velocity_kms: 12.0      # Earth escape velocity limit
    finite_velocity: true
    finite_acceleration: true
  loss:
    physics_weight: 1.0
    boundary_weight: 1.0
    initial_weight: 1.0
    constraint_weight: 0.1
```

---

## Verification

To run the validation suite from your project root:
```bash
python -m src.physics.run_physics_validation
```

This script:
1. Loads simulated orbital positions from `noisy_observations.csv`.
2. Generates an analytical circular Keplerian trajectory to create a valid PyTorch autograd graph.
3. Computes predicted derivatives and evaluates the position/velocity residuals.
4. Performs verification checks (dimensions, finiteness, realistic gravity magnitudes).
5. Generates diagnostic plots under `results/physics/` showing residuals, acceleration profiles, and loss weight distributions.
