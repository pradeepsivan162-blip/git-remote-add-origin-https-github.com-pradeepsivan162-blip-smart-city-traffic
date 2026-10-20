# Academic Project Report Outline
## Smart City Traffic Analytics and AI-Based Congestion Prediction System

**Academic Degree:** Bachelor of Technology / Master of Science in Computer Science & Engineering / Data Science  
**Project Category:** Machine Learning, Intelligent Transportation Systems (ITS), Big Data Analytics

---

### Table of Contents

#### 1. Abstract
- Summary of the urban traffic congestion challenge.
- System scope: historical data analysis (PEMS-BAY), ML forecasting (15, 30, 60 minutes), and interactive dashboard analytics.
- Key outcomes: XGBoost and Random Forest forecasting achieving test MAE between 1.92 and 2.88 mph, congestion classification, and decision-support dashboard.

#### 2. Introduction
- Background: Rapid urbanization and vehicular growth.
- Role of Smart Cities and Intelligent Transportation Systems (ITS).
- Motivation: Proactive vs reactive traffic incident and congestion management.
- Research questions and target operational horizons (15, 30, 60 min).

#### 3. Problem Statement & Objectives
- Specific challenges in freeway speed forecasting without external telemetry.
- Formulation of the time-series regression and classification problem.
- Primary and secondary project objectives.

#### 4. Literature Review
- Classical statistical methods (ARIMA, SARIMA, Kalman Filters).
- Tree-based ensemble learning (Random Forest, Gradient Boosting, XGBoost, LightGBM).
- Deep learning architectures in spatial-temporal traffic forecasting (ST-GCN, DCRNN, LSTM).
- Comparative evaluation: trade-offs between computational overhead, interpretability, and predictive fidelity.

#### 5. Existing System vs. Proposed System
- Existing systems: Loop detector threshold alarms, static timing, high latency.
- Proposed system: Modular data pipeline, multi-horizon feature engineering, ML ensemble forecasting, congestion tiering, and dual-mode dashboard interface.

#### 6. System Architecture & Methodology
- End-to-end data flow: Raw PeMS loop detectors → Long-format transformation → Feature engineering → Model training → Inference & Analytics Engine → Presentation layer.
- Chronological train/validation/test partitioning (70% / 15% / 15%) strictly eliminating temporal data leakage.

#### 7. Dataset Description & Limitations
- Source: California Department of Transportation (Caltrans) PeMS (San Francisco Bay Area).
- Volume: 52,116 time slices at 5-minute granularity × 325 sensor stations (16,937,700 total observations).
- Characteristics, measurement units (mph), missingness profile (0.00% missing values in cleaned snapshot).
- Explicit documentation of constraints: Loop detector speed telemetry only; absence of native GPS coordinates, vehicle counts, weather telemetry.

#### 8. Data Cleaning & Feature Engineering
- Wide-to-long transformation.
- Outlier detection using Interquartile Range (IQR) bounds [45.9 mph, 83.7 mph] and physical limits.
- Lag features ($t-1$, $t-3$, $t-6$, $t-12$ representing 5, 15, 30, 60 min historic values).
- Rolling statistics (moving averages and moving standard deviations over 15, 30, and 60-minute windows).
- Cyclic temporal encoding: Sine/Cosine transformations of diurnal hour ($[0, 23]$) and day-of-week ($[0, 6]$).

#### 9. Database Design (PostgreSQL Schema)
- Entity-Relationship (ER) model: `sensors`, `traffic_readings`, `congestion_predictions`, `model_registry`.
- Indexing strategies (composite B-tree on `(sensor_id, recorded_at DESC)`, partial indexes for anomalous readings).
- Materialized views (`mv_sensor_summary`) and analytical queries for dashboard aggregation.

#### 10. Predictive Modeling & Forecasting
- Persistence baseline formulation: $\hat{y}_{t+h} = y_t$.
- Historical seasonal baseline: $\hat{y}_{t+h} = \mathbb{E}[y \mid \text{hour}, \text{day\_of\_week}]$.
- Random Forest Regressor architecture and hyperparameter configuration.
- XGBoost Regressor formulation with validation early stopping.

#### 11. Empirical Evaluation & Experimental Results
- Evaluation metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE).
- Test set performance across horizons:
  - **15-Minute Horizon:** XGBoost MAE = 1.923 mph, RMSE = 3.813 mph vs Persistence MAE = 2.124 mph.
  - **30-Minute Horizon:** XGBoost MAE = 2.380 mph, RMSE = 4.683 mph vs Persistence MAE = 2.695 mph.
  - **60-Minute Horizon:** XGBoost MAE = 2.885 mph, RMSE = 5.462 mph; RF MAE = 2.951 mph vs Persistence MAE = 3.612 mph.
- Error residual distributions and actual-versus-predicted regression traces.

#### 12. Congestion Classification Framework
- Demonstration classification taxonomy:
  - Free Flow: $\ge 60\text{ mph}$
  - Moderate: $45 - 59.9\text{ mph}$
  - Heavy: $30 - 44.9\text{ mph}$
  - Severe: $< 30\text{ mph}$
- Proportion of time spent in congestion states per sensor corridor.

#### 13. Dashboard & Visualization Implementation
- Modular Streamlit web architecture (7 specialized operational pages).
- Dynamic Plotly graphing, responsive dark-mode UI styling.
- Standalone execution capabilities (seamless fallback to packaged demo subsets when cloud-hosted).

#### 14. Verification & Reliability
- 36 automated unit and integration tests covering data integrity, leakage prevention, classification thresholds, and model inference.
- Handling of edge cases: missing sensor records, corrupt timestamps, cold-start forecasts.

#### 15. Limitations & Future Scope
- Integration with live Caltrans PeMS streaming APIs.
- Incorporation of post-incident dispatch records and micro-climate meteorological observations.
- Implementation of Spatial-Temporal Graph Neural Networks (ST-GNN) when physical adjacency matrices are available.

#### 16. References & Appendix
- Key literature citations (PeMS documentation, DCRNN, XGBoost, Streamlit architecture).
- SQL schema scripts, parameter definitions, reproducibility checklist.
