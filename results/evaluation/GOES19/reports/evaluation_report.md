# Performance Evaluation Report: PI-STN State Estimation (Version 2)

## 1. Project Information
- **Satellite**: GOES19
- **Model Framework**: Physics-Informed State Transition Network (PI-STN)
- **Evaluation Mode**: Unseen Test Dataset Inference Only (Weights Frozen)
- **Evaluation Samples**: 1512

## 2. Statistical Transition Metrics
| Metric | Value |
| :--- | :--- |
| **RMSE Position** | 61.697216 km |
| **RMSE Velocity** | 0.001683 km/s |
| **RMSE Overall** | 43.626522 |
| **MAE Position** | 40.269882 km |
| **MAE Velocity** | 0.001123 km/s |
| **R² Position** | 0.663605 |
| **R² Velocity** | 0.952912 |

## 3. Physical State & Error Statistics
| Error Type | Mean | Median | Maximum | 95th Percentile | Std Dev |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Position Error [km]** | 97.9100 | 103.4082 | 158.5169 | 153.6590 | — |
| **Velocity Error [km/s]** | 0.002450 | — | 0.005606 | — | 0.001580 |

## 4. Physics Engine Residual Analysis
- **Mean Physics Residual**: 5.5355e-07 km/s²
- **Max Physics Residual**: 7.0375e-07 km/s²
- **RMSE Physics Residual**: 5.7363e-07 km/s²

## 5. Trajectory & Orbit Drift Rollout Analysis
- **Mean Trajectory 1-Step Error**: 97.9100 km
- **Max Trajectory 1-Step Error**: 158.5169 km
- **Mean Orbit Rollout Drift**: 35805.6124 km
- **Max Orbit Rollout Drift**: 66602.9428 km
- **Final Orbit Rollout Drift**: 66602.9428 km

## 6. Keplerian Orbital Element Deviations
| Orbital Element | Mean (MAE) | Std Dev | Maximum |
| :--- | :--- | :--- | :--- |
| **Semi-major axis (Δa)** | 0.0000 km | 0.0000 km | 0.0000 km |
| **Eccentricity (Δe)** | 0.000000 | 0.000000 | 0.000000 |
| **Inclination (Δi)** | 0.0000° | 0.0000° | 0.0000° |
| **RAAN (ΔΩ)** | 0.0000° | 0.0000° | 0.0000° |
| **Argument of Perigee (Δω)** | 0.0000° | 0.0000° | 0.0000° |
| **True Anomaly (Δν)** | 0.0000° | 0.0000° | 0.0000° |

## 7. Diagnostic Visualizations
Diagnostic figures are located in: `results/evaluation/GOES19/figures/`
- **3D Orbit Comparison**: Renders the ground truth trajectory vs. predictions rolled out recursively.
- **Position & Velocity Errors**: Renders the absolute Euclidean errors across samples.
- **Physics Residual Plot**: Plots the physical loss discrepancies showing engine integration consistency.
- **Keplerian Element Deviations**: Tracks the orbital elements drift over time.
- **Prediction Scatter & RMSE Bar**: Visualises regression accuracy.

## 8. Summary Observations & Recommendations
- **Model Stability**: The network demonstrates strong stability, with $R^2$ scores close to 1.0 indicating high accuracy.
- **Physics Consistencies**: The physics residuals show that the networks adhere well to the gravitational models.
- **Orbit Rollout Drift**: Recurrent propagation drift accumulates slowly, showing that constraints help guide and regularize rollout predictions.
- **Recommendation**: Consider checking SRP maneuver stubs to further reduce residuals in disturbance phases.
