# Final Year Project Presentation Outline
## Smart City Traffic Analytics & AI-Based Congestion Prediction System

**Duration:** 15–20 minutes + 5 minutes Q&A  
**Audience:** Academic Evaluation Committee / Project Reviewers

---

### Slide 1: Title Slide
- **Project Title:** Smart City Traffic Analytics & AI-Based Congestion Prediction System
- **Candidate Name / Roll No:** [Your Name / Roll Number]
- **Supervisor / Guide:** [Guide Name, Designation]
- **Department & Institution:** Computer Science & Engineering / Data Science

### Slide 2: Context & Motivation
- The urban mobility challenge: Traffic congestion generates lost economic productivity, excess emissions, and transit unpredictability.
- Shift from **reactive monitoring** (detecting traffic jams after they form) to **predictive intelligence** (anticipating congestion 15–60 minutes in advance).

### Slide 3: Objectives & Deliverables
- Analyze historical highway sensor telemetry (PEMS-BAY).
- Engineer time-series features without data leakage.
- Build and benchmark multi-horizon forecasting models (15, 30, 60 minutes).
- Deploy an end-to-end interactive dashboard for traffic control operators.

### Slide 4: Dataset Overview (PEMS-BAY)
- **Source:** Caltrans PeMS (California Bay Area highway network).
- **Volume:** 52,116 timestamps × 325 sensors = 16,937,700 readings.
- **Granularity:** 5-minute intervals, January to June 2017.
- **Data Integrity:** Fully validated, 0.00% missing values in clean partition, IQR outlier boundaries.

### Slide 5: System Architecture
- Flow diagram: Raw Sensor Ingestion → Data Cleaning Pipeline → Chronological Feature Store → Baseline & ML Models → PostgreSQL/Analytics Cache → Streamlit UI.

### Slide 6: Feature Engineering (Preventing Leakage)
- Strict chronological train/validation/test split (70% / 15% / 15%).
- **Lags:** $t-1, t-3, t-6, t-12$ (5 to 60 minutes historical).
- **Rolling Stats:** 15, 30, and 60-minute rolling means and standard deviations.
- **Cyclic Temporal Signals:** Sinusoidal encodings for hour-of-day and day-of-week.

### Slide 7: Predictive Modeling Strategy
- **Baselines:**
  1. *Persistence Baseline:* Last observed reading $\hat{y}_{t+h} = y_t$.
  2. *Seasonal Baseline:* Historical mean for corresponding hour and day of week.
- **Machine Learning Models:**
  3. *Random Forest Regressor:* 200 estimators, max depth 15.
  4. *XGBoost Regressor:* Gradient boosting with early stopping on validation split.

### Slide 8: Model Performance & Benchmark Results
- Visual comparison bar chart (MAE across 15, 30, and 60 minutes):
  - **15-Min:** XGBoost achieves **1.923 mph MAE** (vs. Persistence 2.124 mph).
  - **30-Min:** XGBoost achieves **2.380 mph MAE** (vs. Persistence 2.695 mph).
  - **60-Min:** XGBoost achieves **2.885 mph MAE** (vs. Persistence 3.612 mph).
- Significant error reduction over baselines across all horizons.

### Slide 9: Congestion Classification & Decision Support
- Mapping continuous velocity forecasts to actionable four-tier states:
  - Free Flow ($\ge 60\text{ mph}$), Moderate ($45–59.9\text{ mph}$), Heavy ($30–44.9\text{ mph}$), Severe ($< 30\text{ mph}$).
- Identification of persistent bottleneck sensors and sustained congestion events.

### Slide 10: PostgreSQL Database Architecture
- Relational schema: `sensors`, `traffic_readings`, `congestion_predictions`, `model_registry`.
- Optimized indexing for fast range scans and dashboard aggregation views.

### Slide 11: Interactive Dashboard Demo (7 Operational Views)
- Executive Overview (Network-wide KPIs & trends).
- Traffic Explorer (Multi-sensor historical drilldown).
- AI Prediction (Live horizon selector, actual vs predicted curve).
- Congestion Analysis (Hotspot identification, duration metrics).
- Geospatial View (Location visualizer with comparison fallback).
- AI Traffic Insights (Rule-based and LLM-assisted summaries).
- Model Evaluation (Residual analysis and error distribution).

### Slide 12: Testing & Reliability Verification
- 36 unit and integration test cases covering edge boundaries, sensor preservation, chronological integrity, and error trapping.
- 100% test pass rate.

### Slide 13: Limitations & Future Enhancements
- Current: Loop detector speed readings only.
- Future Work: PostGIS live GPS integration, weather radar data, Graph Neural Networks (GCNs).

### Slide 14: Conclusion
- Successful realization of an end-to-end, production-structured Intelligent Transportation System.
- Reproducible, documented codebase ready for local execution and cloud deployment.

### Slide 15: Questions & Answers (Q&A)
- Thank you. Open for questions.
