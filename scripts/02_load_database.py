"""
scripts/02_load_database.py
────────────────────────────
Loads the cleaned long-format CSV into PostgreSQL.
Skipped automatically when DB_PASSWORD is not set (demo mode).

Usage (PowerShell):
  cd smart-city-traffic
  python scripts/02_load_database.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import USE_DATABASE, CLEANED_DATA_PATH
from src.database import bulk_insert_df, db_available
import pandas as pd


def main():
    if not USE_DATABASE:
        print("⚠️  DB_PASSWORD not set — skipping database load (demo/CSV mode).")
        print("   Set DB_PASSWORD in .env to enable PostgreSQL loading.")
        return

    if not db_available():
        print("❌ Cannot connect to database. Check your .env settings.")
        return

    print(f"Loading {CLEANED_DATA_PATH} …")
    df = pd.read_csv(CLEANED_DATA_PATH, parse_dates=["timestamp"])

    # Insert sensor IDs first
    sensors = pd.DataFrame({"sensor_id": df["sensor_id"].unique()})
    bulk_insert_df(sensors, "sensors", if_exists="append")
    print(f"✅ Inserted {len(sensors)} sensor records.")

    # Rename columns to match DB schema
    readings = df.rename(columns={
        "timestamp":      "recorded_at",
        "observed_speed": "observed_speed",
    })[["sensor_id", "recorded_at", "observed_speed", "outlier_flag"]]

    bulk_insert_df(readings, "traffic_readings", if_exists="append", chunksize=50_000)
    print(f"✅ Inserted {len(readings):,} traffic readings.")


if __name__ == "__main__":
    main()
