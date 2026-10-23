"""
train_baseline.py — Naive persistence baseline for daily streamflow forecasting.

Model : Predict tomorrow's discharge = today's observed discharge.
Data  : data/raw/streamflow.csv  (USGS-01646500, Potomac River at Little Falls)
Split : Chronological — first 80 % train, final 20 % test (no shuffling).
Output: data/processed/baseline_predictions.csv

Usage:
    python src/train_baseline.py
"""

import os
import sys

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────
# Paths (relative to project root)
# ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

CSV_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "streamflow.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "baseline_predictions.csv")


# ──────────────────────────────────────────────
# 1. Load & parse
# ──────────────────────────────────────────────
def load_data(csv_path: str) -> pd.DataFrame:
    """Load the CSV, parse the date column, and keep only what we need."""
    df = pd.read_csv(csv_path, parse_dates=["date"])
    print(f"Loaded {len(df)} rows from {csv_path}")
    return df


# ──────────────────────────────────────────────
# 2. Data-quality checks
# ──────────────────────────────────────────────
def quality_checks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run data-quality checks and fix issues if possible.
    Returns the cleaned DataFrame.  Exits on fatal problems.
    """
    issues_found = False

    # --- 2a. Sort by date (must come first) ---
    df = df.sort_values("date").reset_index(drop=True)
    print(f"\nSorted by date: {df['date'].iloc[0].date()} --> {df['date'].iloc[-1].date()}")

    # --- 2b. Duplicate dates ---
    dup_mask = df["date"].duplicated(keep=False)
    n_dups = dup_mask.sum()
    if n_dups > 0:
        print(f"[!]  {n_dups} rows share a duplicate date -- keeping first occurrence.")
        df = df.drop_duplicates(subset="date", keep="first").reset_index(drop=True)
        issues_found = True
    else:
        print("[OK]  No duplicate dates.")

    # --- 2c. Daily continuity (every day present?) ---
    full_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    missing_dates = full_range.difference(df["date"])
    if len(missing_dates) > 0:
        print(f"[!]  {len(missing_dates)} missing calendar date(s):")
        for d in missing_dates[:10]:
            print(f"     {d.date()}")
        if len(missing_dates) > 10:
            print(f"     ... and {len(missing_dates) - 10} more")
        issues_found = True
    else:
        print("[OK]  Dates are continuous (daily, no gaps).")

    # --- 2d. Unit consistency ---
    if "unit_of_measure" in df.columns:
        units = df["unit_of_measure"].dropna().unique()
        if len(units) == 1 and units[0] == "ft^3/s":
            print(f"[OK]  Units are consistent: {units[0]}")
        else:
            print(f"[!]  Unexpected unit(s): {units}")
            issues_found = True
    else:
        print("[!]  'unit_of_measure' column not found -- cannot verify units.")
        issues_found = True

    # --- 2e. Valid numeric discharge values ---
    n_missing = df["discharge_cfs"].isna().sum()
    if n_missing > 0:
        print(f"[!]  {n_missing} missing discharge value(s) -- dropping those rows.")
        df = df.dropna(subset=["discharge_cfs"]).reset_index(drop=True)
        issues_found = True
    else:
        print("[OK]  Zero missing discharge values.")

    n_negative = (df["discharge_cfs"] < 0).sum()
    if n_negative > 0:
        print(f"[!]  {n_negative} negative discharge value(s) -- unexpected for streamflow.")
        issues_found = True
    else:
        print("[OK]  All discharge values are non-negative.")

    # --- Summary ---
    if issues_found:
        print("\n[!]  Some data-quality issues were found and handled (see above).")
    else:
        print("\n[OK]  All data-quality checks passed -- dataset is clean.")

    print(f"   Rows after cleaning: {len(df)}")
    return df


# ──────────────────────────────────────────────
# 3. Chronological train / test split
# ──────────────────────────────────────────────
def split_data(df: pd.DataFrame, test_frac: float = 0.20):
    """
    Split chronologically (no shuffling).
    Returns (train_df, test_df).
    """
    n = len(df)
    split_idx = int(n * (1 - test_frac))

    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test


# ──────────────────────────────────────────────
# 4. Naive persistence baseline
# ──────────────────────────────────────────────
def persistence_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Naive baseline: predicted discharge for day t = observed discharge on day t-1.

    Operates on the FULL sorted DataFrame so that the first test-day prediction
    uses the last training-day observation (no information leak).
    Returns a DataFrame with columns: date, actual, predicted.
    """
    result = pd.DataFrame({
        "date": df["date"],
        "actual": df["discharge_cfs"],
        "predicted": df["discharge_cfs"].shift(1),   # previous day's value
    })
    # The very first row has no predecessor → drop it
    result = result.dropna(subset=["predicted"]).reset_index(drop=True)
    return result


# ──────────────────────────────────────────────
# 5. Evaluation metrics
# ──────────────────────────────────────────────
def evaluate(actual: np.ndarray, predicted: np.ndarray):
    """Compute MAE and RMSE."""
    errors = actual - predicted
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    return mae, rmse


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print("=" * 64)
    print("  HydroGuard AI -- Naive Persistence Baseline")
    print("=" * 64)

    # 1. Load
    df = load_data(CSV_PATH)

    # 2. Quality checks
    df = quality_checks(df)

    # 3. Build persistence predictions on the full (sorted) series
    #    so that the first test prediction uses the last train observation.
    preds = persistence_baseline(df)

    # 4. Chronological split  (applied to *predictions*, not raw rows)
    #    We want the test set to be the final 20 % of calendar days.
    n_total = len(df)
    split_idx = int(n_total * 0.80)
    test_start_date = df["date"].iloc[split_idx]

    # Slice predictions to the test window
    test_preds = preds[preds["date"] >= test_start_date].copy()

    # 5. Evaluate
    mae, rmse = evaluate(
        test_preds["actual"].values,
        test_preds["predicted"].values,
    )

    # 6. Print results
    train_start = df["date"].iloc[0].date()
    train_end = df["date"].iloc[split_idx - 1].date()
    test_start = test_start_date.date()
    test_end = df["date"].iloc[-1].date()

    print("\n" + "=" * 64)
    print("  TRAIN / TEST SPLIT")
    print("=" * 64)
    print(f"  Train period : {train_start}  -->  {train_end}  ({split_idx} days)")
    print(f"  Test  period : {test_start}  -->  {test_end}  ({n_total - split_idx} days)")
    print(f"  Test predictions evaluated : {len(test_preds)} rows")

    print("\n" + "=" * 64)
    print("  BASELINE METRICS  (Naive Persistence)")
    print("=" * 64)
    print(f"  MAE  = {mae:,.2f} cfs")
    print(f"  RMSE = {rmse:,.2f} cfs")
    print("=" * 64)

    # 7. Save predictions
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    test_preds.to_csv(OUTPUT_PATH, index=False)
    print(f"\n  Test predictions saved to: {OUTPUT_PATH}")
    print(f"  ({len(test_preds)} rows)")

    # 8. Quick peek
    print("\n  First 5 test predictions:")
    print(test_preds.head().to_string(index=False))
    print("\n  Last 5 test predictions:")
    print(test_preds.tail().to_string(index=False))
    print()


if __name__ == "__main__":
    main()
