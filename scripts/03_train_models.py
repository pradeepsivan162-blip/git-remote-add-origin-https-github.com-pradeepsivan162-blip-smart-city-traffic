"""
scripts/03_train_models.py
───────────────────────────
Trains all ML models for all forecast horizons.

Usage (PowerShell):
  cd smart-city-traffic

  # Train on top 10 sensors (default, fast ~5-10 min):
  python scripts/03_train_models.py

  # Train on all 325 sensors (slow ~60-90 min):
  python scripts/03_train_models.py --all-sensors

  # Train on specific sensors:
  python scripts/03_train_models.py --sensors 400001 400017 400030
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.train_model import train_all
from config.settings import CLEANED_DATA_PATH
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Train traffic forecasting models.")
    parser.add_argument("--all-sensors", action="store_true",
                        help="Train on all 325 sensors (slow)")
    parser.add_argument("--sensors", nargs="+", default=None,
                        help="Specific sensor IDs to train on")
    args = parser.parse_args()

    sensor_subset = None
    if args.all_sensors:
        print("⚠️  Training on all sensors. This may take 60–90 minutes.")
    elif args.sensors:
        sensor_subset = args.sensors
        print(f"Training on specified sensors: {sensor_subset}")
    else:
        print("Training on top 10 sensors by data completeness (default).")

    results = train_all(sensor_subset=sensor_subset)

    print("\n── Final Results ─────────────────────────────────────────")
    for h, r in results["horizons"].items():
        print(f"\n  Horizon: {h} min  |  Best model: {r['best_model']}")
        for model_name, metrics in r["models"].items():
            print(f"    {model_name:<20}  MAE={metrics['test_mae']:.3f}  RMSE={metrics['test_rmse']:.3f}")
    print("\n✅ Models saved to models/")
    print("✅ Metrics saved to reports/model_metrics.json")


if __name__ == "__main__":
    main()
