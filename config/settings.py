"""
config/settings.py
──────────────────
Central configuration for the Smart City Traffic Analytics system.
All thresholds, paths, and model parameters are defined here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Project Paths ────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SQL_DIR = BASE_DIR / "sql"

# ── Raw Dataset ──────────────────────────────────────────────────
RAW_DATA_PATH = os.getenv(
    "RAW_DATA_PATH",
    r"c:\Users\prade\Downloads\pms.csv\compressed_data.csv",
)
CLEANED_DATA_PATH = DATA_DIR / "PEMS_BAY_cleaned_long.csv"
SAMPLE_DATA_PATH = DATA_DIR / "sample_demo.csv"  # fallback synthetic data

# ── Database ─────────────────────────────────────────────────────
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_NAME = os.getenv("DB_NAME", "smart_city_traffic")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DATABASE_URL = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)
USE_DATABASE = bool(DB_PASSWORD)  # auto-disable DB if no password set

# ── AI / LLM ─────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
USE_LLM = bool(GEMINI_API_KEY or OPENAI_API_KEY)
LLM_PROVIDER = "gemini" if GEMINI_API_KEY else ("openai" if OPENAI_API_KEY else "rule_based")

# ── Congestion Thresholds (mph) ──────────────────────────────────
# These are demonstration thresholds derived from general US freeway norms.
# Calibrate against verified speed limits when available.
CONGESTION_THRESHOLDS = {
    "free_flow":  60,   # ≥ 60 mph  → Free Flow (Low congestion)
    "moderate":   45,   # 45–59 mph → Moderate
    "heavy":      30,   # 30–44 mph → Heavy
    "severe":      0,   # < 30 mph  → Severe
}

CONGESTION_LABELS = {
    "free_flow": "🟢 Free Flow",
    "moderate":  "🟡 Moderate",
    "heavy":     "🟠 Heavy",
    "severe":    "🔴 Severe",
}

CONGESTION_COLORS = {
    "free_flow": "#00C851",
    "moderate":  "#FFD700",
    "heavy":     "#FF8C00",
    "severe":    "#FF1744",
}

# ── Machine Learning ─────────────────────────────────────────────
FORECAST_HORIZONS_MIN = [15, 30, 60]        # minutes ahead to predict
LAG_INTERVALS = [1, 3, 6, 12]               # 5-min intervals → 5, 15, 30, 60 min
ROLLING_WINDOWS = [3, 6, 12]                # 5-min intervals → 15, 30, 60 min rolling
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
TOP_N_SENSORS_FOR_TRAINING = 10             # train on N most complete sensors
RANDOM_STATE = 42

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_split": 5,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

XGB_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "early_stopping_rounds": 20,
    "random_state": RANDOM_STATE,
    "verbosity": 0,
}

# ── Dashboard ────────────────────────────────────────────────────
APP_TITLE = "Smart City Traffic Analytics"
APP_ICON  = "🚦"
PAGE_ICON  = "🏙️"
DEFAULT_SENSOR_COUNT = 10       # sensors shown in overview
CHART_HEIGHT = 420
MAP_DEFAULT_ZOOM = 10

# ── Logging ──────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
