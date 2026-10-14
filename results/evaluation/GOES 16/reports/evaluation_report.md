# Satellite PINN State Estimation — Evaluation Report

**Satellite**: `GOES16`  
**Generated**: 2026-06-28T20:19:40.201549+00:00  
**Project**: Satellite PINN State Estimation v0.1.0  

---

## Prediction Window

| Parameter | Value |
|-----------|-------|
| Training window | Day 0 → Day 5 |
| Prediction window | Day 5 → Day 7 (2-day future horizon) |
| Physics model | two_body |

---

## Overall Performance Grade

| Dimension | Grade |
|-----------|-------|
| Position Accuracy | 🔴 **Poor** |
| Velocity Accuracy | 🔴 **Poor** |
| Physics Consistency | ⚪ **N/A** |
| **Overall** | 🔴 **Poor** |

---

## Position Metrics (GCRS, km)

| Metric | Value |
|--------|-------|
| RMSE (total) | `51.8100` km |
| MAE (total)  | `51.8061` km |
| Max error    | `52.6369` km |
| Final error  | `52.6050` km |
| RMSE X       | `29.6708` km |
| RMSE Y       | `29.0967` km |
| RMSE Z       | `30.9402` km |

---

## Velocity Metrics (GCRS, km/s)

| Metric | Value |
|--------|-------|
| RMSE (total) | `1.771824` km/s |
| MAE (total)  | `1.756737` km/s |
| Max error    | `2.092927` km/s |
| Final error  | `2.092927` km/s |
| RMSE Vx      | `1.024732` km/s |
| RMSE Vy      | `0.978350` km/s |
| RMSE Vz      | `1.064009` km/s |

---

## Trajectory Drift

| Metric | Value |
|--------|-------|
| Mean position error | `51.8061` km |
| Std position error  | `0.6320` km |
| Cumulative drift    | `4699840.5` km·s |
| Error growth rate   | `-0.000007` km/s |

---

## Orbital Element Drift

| Element | MAE | RMSE |
|---------|-----|------|
| Semi-major axis (a) | `25.0020` km | `25.0022` km |
| Eccentricity (e) | `0.000012` | `0.000012` |
| Inclination (i) | `28.2135` deg | `28.2136` deg |
| RAAN (Ω) | `52.2443` deg | `52.2444` deg |
| Arg. of perigee (ω) | `94.6690` deg | `109.2433` deg |

---

## Physics Consistency (ODE Residuals)

| Metric | Value |
|--------|-------|
| Mean residual | `nan` |
| Max residual  | `nan` |
| Std residual  | `nan` |

---

## Grading Thresholds

| Grade | Position RMSE | Velocity RMSE | Physics Residual |
|-------|--------------|---------------|-----------------|
| 🟢 Excellent  | ≤ 1.0 km | ≤ 0.001 km/s | ≤ 1e-06 |
| 🟡 Good       | ≤ 5.0 km | ≤ 0.005 km/s | ≤ 1e-05 |
| 🟠 Acceptable | ≤ 20.0 km | ≤ 0.02 km/s | ≤ 0.0001 |
| 🔴 Poor       | > 20.0 km | > 0.02 km/s | > 0.0001 |

---

*Report generated automatically by Phase 9 — Satellite PINN State Estimation.*