"""
train_model.py -- RandomForestRegressor for next-day river discharge prediction.

s : lag_1, lag_2, lag_3, lag_7, rolling_mean_7  (all from past data only)
Target   : next-day discharge (discharge_cfs)
Split    : Chronological 80/20 -- identical test window as train_baseline.py
Output   : data/processed/model_comparison.csv

Usage:
    python src/train_model.py
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# ----------------------------------------------------------------
# Paths
# ----------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

FEATURE_COLS = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_7",
    "rolling_mean_7",
]


CSV_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "streamflow.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "model_comparison.csv")

RANDOM_STATE = 42


# ----------------------------------------------------------------
# 1. Load & prepare
# ----------------------------------------------------------------
def load_and_sort(csv_path: str) -> pd.DataFrame:
    """Load CSV, parse dates, sort chronologically."""
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"Loaded {len(df)} rows  ({df['date'].iloc[0].date()} --> {df['date'].iloc[-1].date()})")
    return df


# ----------------------------------------------------------------
# 2. Feature engineering  (only past information)
# ----------------------------------------------------------------
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build lag and rolling features from discharge_cfs.

    All features use ONLY past discharge values:
      lag_1          : discharge yesterday         (shift 1)
      lag_2          : discharge 2 days ago         (shift 2)
      lag_3          : discharge 3 days ago         (shift 3)
      lag_7          : discharge 7 days ago         (shift 7)
      rolling_mean_7 : mean of the PREVIOUS 7 days  (shift 1, then rolling 7)

    The rolling mean is shifted first so it never includes the current day's
    discharge (which would leak the target).
    """
    d = df[["date", "discharge_cfs"]].copy()

    d["lag_1"] = d["discharge_cfs"].shift(1)
    d["lag_2"] = d["discharge_cfs"].shift(2)
    d["lag_3"] = d["discharge_cfs"].shift(3)
    d["lag_7"] = d["discharge_cfs"].shift(7)

    # rolling_mean_7 = mean of days [t-7 .. t-1]  (7 past values, not today)
    d["rolling_mean_7"] = d["discharge_cfs"].shift(1).rolling(window=7).mean()

    return d


train_test_split = ["lag_1", "lag_2", "lag_3", "lag_7", "rolling_mean_7"]
TARGET_COL = "discharge_cfs"


# ----------------------------------------------------------------
# 3. Chronological split  (same logic as train_baseline.py)
# ----------------------------------------------------------------
def chronological_split(df: pd.DataFrame, test_frac: float = 0.20):
    """
    Split on the ORIGINAL row count (before dropping NaN feature rows)
    so the test start date matches the baseline exactly.

    Returns (train_df, test_df, split_idx_in_original).
    """
    n_original = len(df)
    split_idx = int(n_original * (1 - test_frac))  # = int(1826 * 0.80) = 1460
    test_start_date = df["date"].iloc[split_idx]

    # Drop rows where any feature is NaN (first 7 rows due to lag_7 / rolling)
  

    df_clean = df.dropna(
        subset=["lag_1", "lag_2", "lag_3", "lag_7", "rolling_mean_7"]
    ).reset_index(drop=True)




    train = df_clean[df_clean["date"] < test_start_date].copy()
    test = df_clean[df_clean["date"] >= test_start_date].copy()

    return train, test, split_idx, test_start_date

   



# ----------------------------------------------------------------
# 4. Naive persistence baseline  (replicate for comparison)
# ----------------------------------------------------------------
def naive_baseline_predictions(test_df: pd.DataFrame) -> np.ndarray:
    """
    Naive baseline on the test set: predicted = lag_1  (yesterday's discharge).
    Since lag_1 is already a column, just return it.
    """
    return test_df["lag_1"].values


# ----------------------------------------------------------------
# 5. Evaluation metrics
# ----------------------------------------------------------------
def evaluate(actual: np.ndarray, predicted: np.ndarray):
    """Return (MAE, RMSE)."""
    errors = actual - predicted
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    return mae, rmse


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main():
    print("=" * 68)
    print("  HydroGuard AI -- Random Forest vs Naive Baseline")
    print("=" * 68)

    # 1. Load
    df = load_and_sort(CSV_PATH)

    # 2. Feature engineering
    df = create_features(df)
    print(f"Features created: {FEATURE_COLS}")

    # 3. Chronological split
    train, test, split_idx, test_start_date = chronological_split(df)

    print(f"\n  Train rows (usable) : {len(train)}")
    print(f"  Test  rows          : {len(test)}")
    print(f"  Train period : {train['date'].iloc[0].date()}  -->  {train['date'].iloc[-1].date()}")
    print(f"  Test  period : {test['date'].iloc[0].date()}  -->  {test['date'].iloc[-1].date()}")

    # 4. Prepare arrays
    X_train = train[FEATURE_COLS].values
    y_train = train[TARGET_COL].values
    X_test = test[FEATURE_COLS].values
    y_test = test[TARGET_COL].values

    # 5. Train Random Forest
    print("\n  Training RandomForestRegressor ...")
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=4,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    print("  Training complete.")

    # 6. Predict
    rf_preds = rf.predict(X_test)

    # 7. Naive baseline predictions (= lag_1, same as train_baseline.py)
    naive_preds = naive_baseline_predictions(test)

    # 8. Evaluate both
    rf_mae, rf_rmse = evaluate(y_test, rf_preds)
    naive_mae, naive_rmse = evaluate(y_test, naive_preds)

    # 9. Improvement calculation
    mae_change = naive_mae - rf_mae          # positive = RF is better
    rmse_change = naive_rmse - rf_rmse
    mae_pct = (mae_change / naive_mae) * 100 if naive_mae != 0 else 0.0
    rmse_pct = (rmse_change / naive_rmse) * 100 if naive_rmse != 0 else 0.0

    # 10. Print results
    print("\n" + "=" * 68)
    print("  MODEL COMPARISON  (same test period)")
    print("=" * 68)
    print(f"  Test period : {test['date'].iloc[0].date()} --> {test['date'].iloc[-1].date()}")
    print(f"  Test rows   : {len(test)}")
    print("-" * 68)
    print(f"  {'Metric':<10}  {'Naive Baseline':>16}  {'Random Forest':>16}  {'Change':>12}")
    print("-" * 68)
    print(f"  {'MAE':<10}  {naive_mae:>14,.2f}  {rf_mae:>14,.2f}  {mae_change:>+10,.2f} cfs")
    print(f"  {'RMSE':<10}  {naive_rmse:>14,.2f}  {rf_rmse:>14,.2f}  {rmse_change:>+10,.2f} cfs")
    print("-" * 68)

    # Honest verdict
    if rf_mae < naive_mae and rf_rmse < naive_rmse:
        print(f"\n  Random Forest IMPROVES on both metrics.")
        print(f"    MAE  reduced by {mae_change:,.2f} cfs  ({mae_pct:.1f}%)")
        print(f"    RMSE reduced by {rmse_change:,.2f} cfs  ({rmse_pct:.1f}%)")
    elif rf_mae < naive_mae:
        print(f"\n  Random Forest improves MAE by {mae_change:,.2f} cfs ({mae_pct:.1f}%)")
        print(f"  but RMSE is worse by {-rmse_change:,.2f} cfs ({-rmse_pct:.1f}%).")
    elif rf_rmse < naive_rmse:
        print(f"\n  Random Forest improves RMSE by {rmse_change:,.2f} cfs ({rmse_pct:.1f}%)")
        print(f"  but MAE is worse by {-mae_change:,.2f} cfs ({-mae_pct:.1f}%).")
    else:
        print(f"\n  Random Forest does NOT beat the naive baseline on either metric.")

    # 11. Feature importances
    print("\n" + "=" * 68)
    print("  FEATURE IMPORTANCES")
    print("=" * 68)
    importances = rf.feature_importances_
    for feat, imp in sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1]):
        bar = "#" * int(imp * 50)
        print(f"  {feat:<16}  {imp:.4f}  {bar}")

    # 12. Save combined results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    comparison = pd.DataFrame({
        "date": test["date"].values,
        "actual": y_test,
        "naive_predicted": naive_preds,
        "rf_predicted": rf_preds,
    })
    comparison.to_csv(OUTPUT_PATH, index=False)
    print(f"\n  Results saved to: {OUTPUT_PATH}")
    print(f"  ({len(comparison)} rows)")

    # 13. Quick peek
    print("\n  First 5 rows:")
    print(comparison.head().to_string(index=False))
    print("\n  Last 5 rows:")
    print(comparison.tail().to_string(index=False))
    print()


if __name__ == "__main__":
    main()

