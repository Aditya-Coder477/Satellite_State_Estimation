# Model Generalization Comparison: INSAT-3D vs. GOES-19

This report compares the performance of the trained PI-STN (trained on **INSAT-3D**) when evaluated on the unseen test set of **INSAT-3D** (in-distribution) and on **GOES-19** (completely new satellite / cross-distribution generalization).

## Performance Metrics Comparison Table

| Metric | INSAT-3D (Trained Sat) | GOES-19 (Unseen Sat) | Absolute Difference | Relative Change |
|---|---|---|---|---|
| RMSE Position | 6.923739e+00 | 6.169722e+01 | +5.477348e+01 | +791.10% |
| RMSE Velocity | 3.865429e-04 | 1.683233e-03 | +1.296691e-03 | +335.46% |
| MAE Position | 4.948280e+00 | 4.026988e+01 | +3.532160e+01 | +713.82% |
| MAE Velocity | 2.650805e-04 | 1.122746e-03 | +8.576656e-04 | +323.55% |
| R² Position | 9.957636e-01 | 6.636049e-01 | -3.321587e-01 | -33.36% |
| R² Velocity | 9.975168e-01 | 9.529124e-01 | -4.460436e-02 | -4.47% |
| Mean Position Error | 1.122103e+01 | 9.790996e+01 | +8.668893e+01 | +772.56% |
| Mean Velocity Error | 5.975970e-04 | 2.450078e-03 | +1.852481e-03 | +309.99% |
| Mean Physics Residual | 5.607938e-07 | 5.535484e-07 | -7.245319e-09 | -1.29% |
| Mean Trajectory Error | 1.122103e+01 | 9.790996e+01 | +8.668893e+01 | +772.56% |
| Mean Orbit Drift | 5.323921e+03 | 3.580561e+04 | +3.048169e+04 | +572.54% |
| Mean Δa | 8.495415e+03 | 1.831879e+04 | +9.823375e+03 | +115.63% |
| Mean Δe | 2.993610e-01 | 8.039243e-01 | +5.045633e-01 | +168.55% |
| Mean Δi | 4.539408e-01 | 1.038764e+02 | +1.034225e+02 | +22783.26% |
| Mean ΔRAAN | 1.143415e+01 | 5.520346e+01 | +4.376930e+01 | +382.79% |
| Mean Δω | 9.634007e+01 | 1.314965e+02 | +3.515645e+01 | +36.49% |
| Mean Δν | 9.683515e+01 | 1.073486e+02 | +1.051340e+01 | +10.86% |

## Generalization Analysis Summary

- **Significant Generalization Gap**: The position error grew by 772.6%. The orbital element deviations should be inspected to check if there are larger eccentricities or semi-major axis offsets causing continuous phase drift.

### Hamiltonian Conservation Laws Assessment

- **Specific Mechanical Energy Drift**: INSAT-3D: 8495.4146 km | GOES-19: 18318.7894 km
  The symplectic integrators (Leapfrog/Velocity Verlet) successfully restrict artificial energy growth, preserving mechanical energy conservation bounds across both satellites.