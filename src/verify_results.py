"""
verify_results.py -- Verify model_comparison.csv and produce an evaluation report.

Checks: file existence, schema, missing values, duplicate dates, metrics.
Output: reports/model_evaluation.txt

Usage:
    python src/verify_results.py
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# ----------------------------------------------------------------
# Paths
# ----------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "model_comparison.csv")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports")
REPORT_PATH = os.path.join(REPORT_DIR, "model_evaluation.txt")

EXPECTED_COLS = {"date", "actual", "naive_predicted", "rf_predicted"}


# ----------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------
def mae(actual, predicted):
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual, predicted):
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main():
    lines = []  # collect every printed line for the report file

    def out(text=""):
        print(text)
        lines.append(text)

    out("=" * 68)
    out("  HydroGuard AI -- Model Comparison Verification")
    out("=" * 68)

    # ----------------------------------------------------------
    # 1. File existence
    # ----------------------------------------------------------
    if not os.path.isfile(CSV_PATH):
        out(f"\n[FAIL]  File not found: {CSV_PATH}")
        out("        Run src/train_model.py first to generate it.")
        sys.exit(1)

    out(f"\n[OK]  File found: {CSV_PATH}")

    # ----------------------------------------------------------
    # 2. Load & column check
    # ----------------------------------------------------------
    df = pd.read_csv(CSV_PATH, parse_dates=["date"])
    actual_cols = set(df.columns)

    if actual_cols != EXPECTED_COLS:
        missing = EXPECTED_COLS - actual_cols
        extra = actual_cols - EXPECTED_COLS
        out(f"\n[FAIL]  Column mismatch!")
        if missing:
            out(f"        Missing columns : {missing}")
        if extra:
            out(f"        Extra columns   : {extra}")
        out("        Expected columns: date, actual, naive_predicted, rf_predicted")
        sys.exit(1)

    out(f"[OK]  Columns match: {sorted(df.columns)}")

    # ----------------------------------------------------------
    # 3. Basic shape & date range
    # ----------------------------------------------------------
    out(f"\n  Row count  : {len(df)}")
    out(f"  Columns    : {list(df.columns)}")
    out(f"  Date range : {df['date'].iloc[0].date()} --> {df['date'].iloc[-1].date()}")

    out("\n  First 5 rows:")
    out(df.head().to_string(index=False))

    # ----------------------------------------------------------
    # 4. Missing values
    # ----------------------------------------------------------
    out("\n" + "-" * 68)
    out("  MISSING VALUE CHECK")
    out("-" * 68)
    n_missing = df.isna().sum()
    any_missing = False
    for col in df.columns:
        n = n_missing[col]
        status = "[OK]" if n == 0 else "[!] "
        out(f"  {status}  {col:<20s}: {n} missing")
        if n > 0:
            any_missing = True

    if any_missing:
        out("\n  [!]  Some columns have missing values (see above).")
    else:
        out("\n  [OK]  No missing values in any column.")

    # ----------------------------------------------------------
    # 5. Duplicate dates
    # ----------------------------------------------------------
    dup_count = df["date"].duplicated().sum()
    if dup_count > 0:
        out(f"\n  [!]  {dup_count} duplicate date(s) found!")
    else:
        out(f"  [OK]  No duplicate dates.")

    # ----------------------------------------------------------
    # 6. Metrics on exactly the same rows
    # ----------------------------------------------------------
    y = df["actual"].values
    naive = df["naive_predicted"].values
    rf = df["rf_predicted"].values

    naive_mae = mae(y, naive)
    naive_rmse = rmse(y, naive)
    rf_mae = mae(y, rf)
    rf_rmse = rmse(y, rf)

    mae_abs = naive_mae - rf_mae       # positive => RF better
    rmse_abs = naive_rmse - rf_rmse
    mae_pct = (mae_abs / naive_mae * 100) if naive_mae != 0 else 0.0
    rmse_pct = (rmse_abs / naive_rmse * 100) if naive_rmse != 0 else 0.0

    out("\n" + "=" * 68)
    out("  METRICS  (computed on {0} identical test rows)".format(len(df)))
    out("=" * 68)
    out(f"  {'':10s}  {'Naive Baseline':>16s}  {'Random Forest':>16s}  {'Change':>14s}")
    out("-" * 68)
    out(f"  {'MAE':10s}  {naive_mae:>14,.2f}  {rf_mae:>14,.2f}  {mae_abs:>+12,.2f} cfs")
    out(f"  {'RMSE':10s}  {naive_rmse:>14,.2f}  {rf_rmse:>14,.2f}  {rmse_abs:>+12,.2f} cfs")
    out("-" * 68)
    out(f"  MAE  change : {mae_abs:>+,.2f} cfs  ({mae_pct:>+.1f}%)")
    out(f"  RMSE change : {rmse_abs:>+,.2f} cfs  ({rmse_pct:>+.1f}%)")

    # ----------------------------------------------------------
    # 7. Honest verdict
    # ----------------------------------------------------------
    out("\n" + "=" * 68)
    out("  VERDICT")
    out("=" * 68)

    rf_better_mae = rf_mae < naive_mae
    rf_better_rmse = rf_rmse < naive_rmse

    if rf_better_mae and rf_better_rmse:
        out("  Random Forest outperforms the naive baseline on BOTH metrics.")
    elif rf_better_mae:
        out("  Random Forest improves MAE but NOT RMSE.")
    elif rf_better_rmse:
        out("  Random Forest improves RMSE but NOT MAE.")
    else:
        out("  Random Forest does NOT beat the naive baseline on either metric.")

    out("")

    # ----------------------------------------------------------
    # 8. Save report
    # ----------------------------------------------------------
    os.makedirs(REPORT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(f"Generated: {timestamp}\n")
        f.write(f"Source   : {CSV_PATH}\n\n")
        for line in lines:
            f.write(line + "\n")

    print(f"  Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()

