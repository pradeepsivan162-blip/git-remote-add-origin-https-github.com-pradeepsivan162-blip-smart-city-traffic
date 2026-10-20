# 🚦 Smart City Traffic Analytics & AI-Based Congestion Prediction System

> A complete end-to-end final-year project: 325-sensor PEMS-BAY dataset → PostgreSQL → XGBoost forecasting → 7-page Streamlit dashboard.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Objectives & Scope](#objectives--scope)
4. [Dataset Description](#dataset-description)
5. [System Architecture](#system-architecture)
6. [Technology Stack](#technology-stack)
7. [Project Structure](#project-structure)
8. [Environment Setup](#environment-setup)
9. [Data Cleaning](#data-cleaning)
10. [Model Training](#model-training)
11. [Dashboard Launch](#dashboard-launch)
12. [Database Setup (Optional)](#database-setup-optional)
13. [Running Tests](#running-tests)
14. [Dashboard Pages](#dashboard-pages)
15. [Model Evaluation Results](#model-evaluation-results)
16. [Limitations & Future Work](#limitations--future-work)
17. [Academic Report Outline](#academic-report-outline)

---

## Project Overview

This project builds a **Smart City Traffic Analytics and AI-Based Congestion Prediction System** using the PEMS-BAY dataset from Caltrans (California Bay Area, Jan–Jun 2017). It:

- Cleans and analyses 52,116 timestamps × 325 sensors (~16.9M observations)
- Trains Random Forest and XGBoost models for 15, 30, and 60-minute speed forecasts
- Classifies congestion as Free Flow / Moderate / Heavy / Severe
- Exposes a 7-page interactive dark-mode Streamlit dashboard
- Generates rule-based (and optionally LLM-powered) traffic insights

---

## Problem Statement

Urban traffic congestion causes billions of hours of delay annually. Traditional reactive traffic management cannot anticipate congestion before it forms. This project demonstrates how historical sensor data, feature engineering, and machine learning can enable **proactive congestion prediction** at 15, 30, and 60-minute horizons.

---

## Objectives & Scope

| # | Objective |
|---|-----------|
| 1 | Analyse historical traffic speed and identify peak periods |
| 2 | Detect sensors experiencing unusually low speeds |
| 3 | Forecast speed 15, 30, and 60 minutes ahead using ML |
| 4 | Classify predicted conditions into four congestion levels |
| 5 | Store data and predictions in PostgreSQL (optional) |
| 6 | Present insights via an interactive Streamlit dashboard |
| 7 | Generate AI-powered summaries from verified analytical results |

**Out of scope:** Real-time data ingestion, GPS tracking, incident detection, weather integration (data not in the PEMS-BAY CSV).

---

## Dataset Description

| Attribute | Value |
|-----------|-------|
| Name | PEMS-BAY (Bay Area Traffic Speed) |
| Source | Caltrans Performance Measurement System |
| Coverage | Jan 1 – Jun 30, 2017 |
| Sensors | 325 loop detectors, San Francisco Bay Area |
| Timestamps | 52,116 (5-minute intervals) |
| Measurement | Traffic speed in **mph** |
| License | Creative Commons Attribution 4.0 |
| GPS Coords | **Not included** in the CSV |
| Raw file | `c:\Users\prade\Downloads\pms.csv\compressed_data.csv` |

---

## System Architecture

```
Raw CSV (81.7 MB)
      │
      ▼
┌─────────────────────┐
│  data_cleaning.py   │  Wide → Long format, outlier flagging
└────────┬────────────┘
         │ PEMS_BAY_cleaned_long.csv
         ▼
┌─────────────────────┐     ┌──────────────────────┐
│ feature_engineering │────▶│   train_model.py      │
│     .py             │     │  RF + XGBoost models  │
└─────────────────────┘     └──────────┬───────────┘
                                        │ models/*.joblib
                                        ▼
                            ┌──────────────────────┐
                            │     predict.py        │
                            │  congestion.py        │
                            └──────────┬───────────┘
                                       │
         ┌──────────────────────────────┘
         ▼
┌─────────────────────┐     ┌──────────────────────┐
│    analytics.py     │     │   ai_insights.py      │
│ (DataFrames for UI) │     │  (Rule + Gemini/GPT)  │
└────────┬────────────┘     └──────────┬───────────┘
         └──────────────┬──────────────┘
                        ▼
              ┌─────────────────┐
              │  Streamlit App  │  7 Pages
              │  Dark-mode UI   │  (app.py + pages/)
              └─────────────────┘
                        │
              ┌─────────┴──────────┐ (optional)
              ▼                    ▼
    PostgreSQL DB           CSV-only mode
    (smart_city_traffic)    (no DB required)
```

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| Data | Pandas, NumPy |
| ML | Scikit-learn, XGBoost |
| Visualisation | Plotly |
| Dashboard | Streamlit |
| Database | PostgreSQL + SQLAlchemy |
| Mapping | Folium / PyDeck (optional) |
| AI Insights | Google Gemini / OpenAI (optional) |
| Model Persistence | Joblib |
| Testing | Pytest |

---

## Project Structure

```
smart-city-traffic/
├── app.py                          # Streamlit entry point
├── requirements.txt
├── README.md
├── .env.example
├── config/
│   └── settings.py                 # All configurable parameters
├── data/
│   ├── README.md
│   └── PEMS_BAY_cleaned_long.csv   # Generated by step 1
├── sql/
│   ├── 01_create_database.sql
│   ├── 02_create_tables.sql
│   ├── 03_indexes.sql
│   ├── 04_analytics_queries.sql
│   └── 05_dashboard_views.sql
├── notebooks/
│   ├── 01_traffic_eda.ipynb
│   └── 02_traffic_modeling.ipynb
├── src/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── predict.py
│   ├── congestion.py
│   ├── analytics.py
│   ├── ai_insights.py
│   └── database.py
├── models/                         # Saved .joblib model files
├── pages/
│   ├── 1_Executive_Overview.py
│   ├── 2_Traffic_Explorer.py
│   ├── 3_AI_Prediction.py
│   ├── 4_Congestion_Analysis.py
│   ├── 5_Geospatial_Analytics.py
│   ├── 6_AI_Insights.py
│   └── 7_Model_Evaluation.py
├── assets/
│   └── style.css                   # Dark-mode theme
├── scripts/
│   ├── 01_clean_data.py
│   ├── 02_load_database.py
│   └── 03_train_models.py
├── reports/
│   ├── figures/
│   └── model_metrics.json          # Generated by step 2
└── tests/
    ├── test_data_cleaning.py
    ├── test_congestion.py
    └── test_prediction.py
```

---

## Environment Setup

```powershell
# 1. Navigate to project directory
cd C:\Users\prade\.gemini\antigravity\scratch\smart-city-traffic

# 2. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy and configure environment variables
Copy-Item .env.example .env
# Edit .env with your settings (Notepad .env)
```

---

## Data Cleaning

```powershell
# Activate venv first (see above)
python scripts/01_clean_data.py
```

**Output:**
- `data/PEMS_BAY_cleaned_long.csv` — long-format dataset (~16.9M rows)
- `data/cleaning_metadata.json` — transformation report

**Expected output:**
```
✅ Cleaned data saved to: data/PEMS_BAY_cleaned_long.csv
  final_long_rows    ~16,400,000
  final_sensors      325
  missing_pct        ~8-12%
```

---

## Model Training

```powershell
# Default: train on top 10 sensors (~5-10 minutes)
python scripts/03_train_models.py

# Train on specific sensors
python scripts/03_train_models.py --sensors 400001 400017 400030

# Train on all 325 sensors (60-90 minutes)
python scripts/03_train_models.py --all-sensors
```

**Output:**
- `models/best_model_{horizon}min_v{timestamp}.joblib`
- `models/scaler_{horizon}min_v{timestamp}.joblib`
- `reports/model_metrics.json`

---

## Dashboard Launch

```powershell
streamlit run app.py
```

Open your browser at **http://localhost:8501**

> **Note:** The dashboard works in CSV-only mode — no PostgreSQL required.

---

## Database Setup (Optional)

```powershell
# 1. Create the database
psql -U postgres -f sql/01_create_database.sql

# 2. Create tables
psql -U postgres -d smart_city_traffic -f sql/02_create_tables.sql

# 3. Create indexes
psql -U postgres -d smart_city_traffic -f sql/03_indexes.sql

# 4. Create views
psql -U postgres -d smart_city_traffic -f sql/05_dashboard_views.sql

# 5. Load data (requires DB_PASSWORD in .env)
python scripts/02_load_database.py
```

---

## Running Tests

```powershell
python -m pytest tests/ -v
```

Expected: **all tests pass** in < 60 seconds (no database or model files required).

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 Executive Overview | KPIs, hourly trends, daily patterns, peak periods |
| 🔍 Traffic Explorer | Sensor picker, date filter, speed charts, CSV download |
| 🤖 AI Prediction | XGBoost/RF forecasts at 15/30/60 min, actual vs predicted |
| 🚨 Congestion Analysis | Rankings, donut chart, low-speed events |
| 🗺️ Geospatial Analytics | Map (if coords available) or sensor comparison chart |
| 💡 AI Insights | Rule-based + optional Gemini/GPT narrative summaries |
| 📊 Model Evaluation | MAE/RMSE by model/horizon, residuals, version info |

---

## Model Evaluation Results

> Run `python scripts/03_train_models.py` then open **Page 7: Model Evaluation** for actual results.  
> Do not trust any numbers below until confirmed by your own training run.

Expected performance range based on literature for PEMS-BAY:

| Horizon | XGBoost MAE | RF MAE | Persistence MAE |
|---------|------------|--------|-----------------|
| 15 min  | ~2–4 mph   | ~3–5 mph | ~5–8 mph |
| 30 min  | ~4–7 mph   | ~5–8 mph | ~8–12 mph |
| 60 min  | ~6–10 mph  | ~7–11 mph | ~10–15 mph |

---

## Limitations & Future Work

### Current Limitations
- No GPS coordinates in the PEMS-BAY CSV — map requires external data
- No vehicle count, occupancy, or weather data available
- Congestion thresholds are demonstration rules, not calibrated speed limits
- Model trained on 10 sensors by default for speed; retrain on all for production

### Future Enhancements
- Real-time data ingestion via Caltrans PeMS API
- LSTM/Transformer deep learning model
- Weather integration via OpenWeatherMap API
- Anomaly detection using Isolation Forest
- Route-level congestion prediction (not just sensor-level)
- Mobile-responsive dashboard

---

## Academic Report Outline

1. Abstract
2. Introduction
3. Problem Statement
4. Literature Review
5. Existing vs Proposed System
6. System Architecture
7. Dataset Description
8. Methodology
9. Exploratory Data Analysis
10. Database Design
11. Model Development
12. Evaluation & Results
13. Dashboard Implementation
14. Limitations
15. Conclusion & Future Scope
16. References

---

*Project built for academic demonstration. Dataset: PEMS-BAY © Caltrans, CC BY 4.0.*
