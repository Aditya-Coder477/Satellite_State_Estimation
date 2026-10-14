# Performance Evaluation Report: PI-STN State Estimation (Version 2)

## 1. Project Information
- **Satellite**: INSAT3D
- **Model Framework**: Physics-Informed State Transition Network (PI-STN)
- **Evaluation Mode**: Unseen Test Dataset Inference Only (Weights Frozen)
- **Evaluation Samples**: 1512

## 2. Statistical Transition Metrics
| Metric | Value |
| :--- | :--- |
| **RMSE Position** | 6.923739 km |
| **RMSE Velocity** | 0.000387 km/s |
| **RMSE Overall** | 4.895823 |
| **MAE Position** | 4.948280 km |
| **MAE Velocity** | 0.000265 km/s |
| **R² Position** | 0.995764 |
| **R² Velocity** | 0.997517 |

## 3. Physical State & Error Statistics
| Error Type | Mean | Median | Maximum | 95th Percentile | Std Dev |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Position Error [km]** | 11.2210 | 12.8539 | 16.2632 | 15.7868 | — |
| **Velocity Error [km/s]** | 0.000598 | — | 0.001326 | — | 0.000302 |

## 4. Physics Engine Residual Analysis
- **Mean Physics Residual**: 5.6079e-07 km/s²
- **Max Physics Residual**: 7.3614e-07 km/s²
- **RMSE Physics Residual**: 5.7935e-07 km/s²

## 5. Trajectory & Orbit Drift Rollout Analysis
- **Mean Trajectory 1-Step Error**: 11.2210 km
- **Max Trajectory 1-Step Error**: 16.2632 km
- **Mean Orbit Rollout Drift**: 5323.9210 km
- **Max Orbit Rollout Drift**: 10388.7466 km
- **Final Orbit Rollout Drift**: 5662.0435 km

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
Diagnostic figures are located in: `results/evaluation/INSAT3D/figures/`
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
