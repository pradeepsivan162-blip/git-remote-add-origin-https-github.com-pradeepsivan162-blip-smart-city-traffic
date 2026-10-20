"""
scripts/01_clean_data.py
─────────────────────────
Runs the full data-cleaning pipeline and saves the long-format CSV.

Usage (PowerShell):
  cd smart-city-traffic
  python scripts/01_clean_data.py
  python scripts/01_clean_data.py --input "C:/path/to/PEMS-BAY.csv" --output data/cleaned.csv
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_cleaning import clean
from config.settings import RAW_DATA_PATH, CLEANED_DATA_PATH


def main():
    parser = argparse.ArgumentParser(description="Clean PEMS-BAY traffic data.")
    parser.add_argument("--input",  default=str(RAW_DATA_PATH),    help="Path to raw CSV")
    parser.add_argument("--output", default=str(CLEANED_DATA_PATH), help="Output path for cleaned CSV")
    args = parser.parse_args()

    print(f"Input  : {args.input}")
    print(f"Output : {args.output}")
    print("-" * 60)

    df, metadata = clean(raw_path=args.input, output_path=args.output)

    print("\n-- Cleaning Summary ------------------------------------------")
    for k, v in metadata.items():
        print(f"  {k:<40} {v}")

    # Save metadata
    meta_path = Path(args.output).parent / "cleaning_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"\nOK Cleaned data saved to: {args.output}")
    print(f"OK Metadata saved to:      {meta_path}")


if __name__ == "__main__":
    main()
