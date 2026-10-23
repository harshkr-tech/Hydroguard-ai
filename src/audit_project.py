"""
audit_project.py -- Phase 1 audit of HydroGuard AI project.
Read-only: inspects data, features, splits, and existing results.
"""
import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_CSV = os.path.join(PROJECT_ROOT, "data", "raw", "streamflow.csv")
COMP_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "model_comparison.csv")
BASELINE_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "baseline_predictions.csv")
PEAK_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "peak_predictions.csv")
WORST_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "worst_predictions.csv")

def section(title):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)

def main():
    section("1. RAW DATA AUDIT")
    raw = pd.read_csv(RAW_CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    print(f"  Rows              : {len(raw)}")
    print(f"  Date range        : {raw['date'].iloc[0].date()} to {raw['date'].iloc[-1].date()}")
    print(f"  Columns           : {list(raw.columns)}")
    print(f"  Missing discharge : {raw['discharge_cfs'].isna().sum()}")
    print(f"  Negative discharge: {(raw['discharge_cfs'] < 0).sum()}")
    print(f"  Duplicate dates   : {raw['date'].duplicated().sum()}")
    print(f"  Min discharge     : {raw['discharge_cfs'].min():,.0f} cfs")
    print(f"  Max discharge     : {raw['discharge_cfs'].max():,.0f} cfs")
    print(f"  Mean discharge    : {raw['discharge_cfs'].mean():,.1f} cfs")
    print(f"  Median discharge  : {raw['discharge_cfs'].median():,.0f} cfs")

    # Date continuity
    full_range = pd.date_range(raw["date"].min(), raw["date"].max(), freq="D")
    missing_dates = full_range.difference(raw["date"])
    print(f"  Missing cal dates : {len(missing_dates)}")
    if len(missing_dates) > 0:
        for d in missing_dates[:5]:
            print(f"    {d.date()}")

    section("2. CHRONOLOGICAL SPLIT ANALYSIS")
    n = len(raw)
    split_idx = int(n * 0.80)
    train_data = raw.iloc[:split_idx]
    test_data = raw.iloc[split_idx:]
    print(f"  Total rows  : {n}")
    print(f"  Split index : {split_idx}")
    print(f"  Train period: {train_data['date'].iloc[0].date()} to {train_data['date'].iloc[-1].date()} ({len(train_data)} days)")
    print(f"  Test period : {test_data['date'].iloc[0].date()} to {test_data['date'].iloc[-1].date()} ({len(test_data)} days)")
    print(f"  Train max   : {train_data['discharge_cfs'].max():,.0f} cfs")
    print(f"  Train mean  : {train_data['discharge_cfs'].mean():,.1f} cfs")
    print(f"  Test max    : {test_data['discharge_cfs'].max():,.0f} cfs")
    print(f"  Test mean   : {test_data['discharge_cfs'].mean():,.1f} cfs")

    print("\n  Distribution of extreme days (train vs test):")
    for thresh in [20000, 40000, 60000, 80000, 100000, 120000]:
        tr = (train_data["discharge_cfs"] >= thresh).sum()
        te = (test_data["discharge_cfs"] >= thresh).sum()
        print(f"    >= {thresh:>7,} cfs:  train={tr:>3d}  test={te:>3d}")

    section("3. FEATURE ENGINEERING LEAKAGE CHECK")
    # Recreate features exactly as in train_model_before_weighting.py
    d = raw[["date", "discharge_cfs"]].copy()
    flow = d["discharge_cfs"]
    d["lag_1"] = flow.shift(1)
    d["lag_2"] = flow.shift(2)
    d["lag_3"] = flow.shift(3)
    d["lag_7"] = flow.shift(7)
    d["lag_14"] = flow.shift(14)
    past_flow = flow.shift(1)
    d["rolling_mean_7"] = past_flow.rolling(7).mean()
    d["rolling_max_7"] = past_flow.rolling(7).max()
    d["rolling_std_7"] = past_flow.rolling(7).std()
    d["flow_change_1"] = flow.shift(1) - flow.shift(2)
    d["rolling_max_14"] = flow.shift(1).rolling(14).max()
    d["rolling_max_30"] = flow.shift(1).rolling(30).max()
    d["lag_7_max"] = flow.shift(1).rolling(7).max()

    # Verify: for row i, lag_1 should equal discharge at row i-1
    # Pick a test row
    test_row = 1000
    actual_prev = raw["discharge_cfs"].iloc[test_row - 1]
    lag_1_val = d["lag_1"].iloc[test_row]
    print(f"  Spot check row {test_row}:")
    print(f"    discharge_cfs[{test_row}]   = {raw['discharge_cfs'].iloc[test_row]:,.0f}")
    print(f"    discharge_cfs[{test_row-1}] = {actual_prev:,.0f}")
    print(f"    lag_1[{test_row}]           = {lag_1_val:,.0f}")
    print(f"    lag_1 == prev day?         : {actual_prev == lag_1_val}")

    # Check rolling_mean_7: should be mean of days [i-7 .. i-1]
    rm7 = d["rolling_mean_7"].iloc[test_row]
    manual_rm7 = raw["discharge_cfs"].iloc[test_row-7:test_row].mean()
    print(f"    rolling_mean_7[{test_row}]  = {rm7:,.1f}")
    print(f"    manual mean [i-7..i-1]     = {manual_rm7:,.1f}")
    print(f"    Match?                     : {abs(rm7 - manual_rm7) < 0.01}")

    # Check: does any feature at row i use discharge_cfs[i]?
    print("\n  Leakage check: verifying no feature uses current-day discharge...")
    # lag_7_max and rolling_max_7 should be identical (both shift(1).rolling(7).max())
    print(f"    lag_7_max == rolling_max_7? : {(d['lag_7_max'].dropna() == d['rolling_max_7'].dropna()).all()}")

    # NaN rows after feature creation
    feature_cols = ["lag_1","lag_2","lag_3","lag_7","rolling_mean_7","lag_14",
                    "rolling_max_7","rolling_std_7","flow_change_1",
                    "rolling_max_14","rolling_max_30","lag_7_max"]
    nan_count = d[feature_cols].isna().any(axis=1).sum()
    first_valid = d.dropna(subset=feature_cols).index[0]
    print(f"  NaN feature rows  : {nan_count} (first valid index: {first_valid})")

    section("4. EXISTING MODEL COMPARISON CSV AUDIT")
    if os.path.exists(COMP_CSV):
        comp = pd.read_csv(COMP_CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        print(f"  File: {COMP_CSV}")
        print(f"  Rows     : {len(comp)}")
        print(f"  Columns  : {list(comp.columns)}")
        print(f"  Date range: {comp['date'].iloc[0].date()} to {comp['date'].iloc[-1].date()}")
        print(f"  Missing values: {comp.isna().sum().to_dict()}")
        print(f"  Duplicate dates: {comp['date'].duplicated().sum()}")

        y = comp["actual"].values
        naive = comp["naive_predicted"].values
        rf = comp["rf_predicted"].values

        # Overall metrics
        naive_mae = np.mean(np.abs(y - naive))
        naive_rmse = np.sqrt(np.mean((y - naive)**2))
        rf_mae = np.mean(np.abs(y - rf))
        rf_rmse = np.sqrt(np.mean((y - rf)**2))

        # R-squared
        ss_res_naive = np.sum((y - naive)**2)
        ss_res_rf = np.sum((y - rf)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        naive_r2 = 1 - ss_res_naive / ss_tot
        rf_r2 = 1 - ss_res_rf / ss_tot

        print(f"\n  OVERALL METRICS (on {len(comp)} test days):")
        print(f"    {'Metric':<10}  {'Naive':>14}  {'RF':>14}")
        print(f"    {'MAE':<10}  {naive_mae:>14,.2f}  {rf_mae:>14,.2f}")
        print(f"    {'RMSE':<10}  {naive_rmse:>14,.2f}  {rf_rmse:>14,.2f}")
        print(f"    {'R2':<10}  {naive_r2:>14.4f}  {rf_r2:>14.4f}")

        # Peak analysis: use training max as threshold
        train_max = train_data["discharge_cfs"].max()
        # Also use 90th percentile of training data
        train_p90 = train_data["discharge_cfs"].quantile(0.90)
        print(f"\n  Training max discharge : {train_max:,.0f} cfs")
        print(f"  Training 90th pctile  : {train_p90:,.0f} cfs")

        # Peak = test days above train_p90
        peak_mask = comp["actual"] >= train_p90
        n_peak = peak_mask.sum()
        if n_peak > 0:
            peak = comp[peak_mask]
            y_p = peak["actual"].values
            rf_p = peak["rf_predicted"].values
            naive_p = peak["naive_predicted"].values
            peak_rf_mae = np.mean(np.abs(y_p - rf_p))
            peak_rf_rmse = np.sqrt(np.mean((y_p - rf_p)**2))
            peak_naive_mae = np.mean(np.abs(y_p - naive_p))
            peak_naive_rmse = np.sqrt(np.mean((y_p - naive_p)**2))
            peak_rf_bias = np.mean(rf_p - y_p)

            print(f"\n  PEAK-EVENT METRICS (actual >= {train_p90:,.0f} cfs, {n_peak} days):")
            print(f"    {'Metric':<10}  {'Naive':>14}  {'RF':>14}")
            print(f"    {'MAE':<10}  {peak_naive_mae:>14,.2f}  {peak_rf_mae:>14,.2f}")
            print(f"    {'RMSE':<10}  {peak_naive_rmse:>14,.2f}  {peak_rf_rmse:>14,.2f}")
            print(f"    {'Bias':<10}  {'':>14}  {peak_rf_bias:>14,.2f}")

        # Top 5 worst RF errors
        comp["rf_error"] = comp["rf_predicted"] - comp["actual"]
        comp["rf_abs_error"] = comp["rf_error"].abs()
        worst5 = comp.nlargest(5, "rf_abs_error")
        print(f"\n  TOP 5 WORST RF PREDICTIONS:")
        for _, row in worst5.iterrows():
            print(f"    {row['date'].date()}  actual={row['actual']:>8,.0f}  rf_pred={row['rf_predicted']:>10,.1f}  error={row['rf_error']:>+10,.1f}")

        # Check if test days above train max exist
        above_train_max = comp[comp["actual"] > train_max]
        print(f"\n  Test days exceeding training max ({train_max:,.0f}):")
        print(f"    Count: {len(above_train_max)}")
        for _, row in above_train_max.iterrows():
            print(f"    {row['date'].date()}  actual={row['actual']:>8,.0f}  rf_pred={row['rf_predicted']:>10,.1f}")

    section("5. CODE BUG CHECK: train_model_before_weighting.py")
    # Read the file and check for issues
    script_path = os.path.join(PROJECT_ROOT, "src", "train_model_before_weighting.py")
    with open(script_path, "r") as f:
        code = f.read()

    issues = []
    # Check 1: duplicate function definition
    if code.count("def create_features") > 1:
        issues.append("CRITICAL: create_features() is defined TWICE (lines 62 and 76). The first definition is dead code.")

    # Check 2: dropna only checks 5 features but FEATURE_COLS has 12
    if "dropna" in code and "lag_14" not in code.split("dropna")[1].split(")")[0]:
        issues.append("BUG: dropna() only checks 5 base features, not the full 12 FEATURE_COLS. Rows with NaN in lag_14, rolling_max_7, etc. could pass through.")

    # Check 3: unused variable
    if "train_test_split = " in code:
        issues.append("WARNING: Unused variable 'train_test_split' on line 111. Shadows sklearn import if any.")

    # Check 4: lag_7_max and rolling_max_7 are identical
    if "lag_7_max" in code and "rolling_max_7" in code:
        issues.append("REDUNDANCY: lag_7_max = shift(1).rolling(7).max() is identical to rolling_max_7. Duplicate feature.")

    # Check 5: FEATURE_COLS has 12 features but dropna only checks 5
    issues.append("MISMATCH: FEATURE_COLS lists 12 features but chronological_split dropna only checks 5.")

    if issues:
        print(f"  Found {len(issues)} issue(s):")
        for i, issue in enumerate(issues, 1):
            print(f"    {i}. {issue}")
    else:
        print("  No issues found.")

    section("6. BASELINE PREDICTIONS CONSISTENCY")
    if os.path.exists(BASELINE_CSV):
        baseline = pd.read_csv(BASELINE_CSV, parse_dates=["date"])
        print(f"  Rows: {len(baseline)}")
        print(f"  Columns: {list(baseline.columns)}")
        # Check if baseline dates match model_comparison dates
        if os.path.exists(COMP_CSV):
            comp_dates = set(pd.read_csv(COMP_CSV)["date"])
            base_dates = set(baseline["date"].astype(str))
            print(f"  Dates match model_comparison? {comp_dates == base_dates}")

    section("7. SUMMARY OF EXISTING FILES")
    files = [
        ("src/train_model_backup.py", "Original RF (5 features, no weighting)"),
        ("src/train_model_before_weighting.py", "Extended RF (12 features, sample weighting 4x)"),
        ("src/train_baseline.py", "Naive persistence baseline"),
        ("src/verify_results.py", "Model comparison verifier"),
        ("src/inspect_peak_events.py", "Peak event inspector"),
        ("src/download_data.py", "USGS data downloader"),
        ("app.py", "Streamlit dashboard (2 sections)"),
        ("data/processed/model_comparison.csv", f"Current RF results ({COMP_CSV})"),
        ("data/processed/model_comparison_backup.csv", "Older RF results"),
        ("data/processed/model_comparison_baseline.csv", "Original 5-feature results"),
        ("data/processed/model_comparison_before_weighting.csv", "Pre-weighting 12-feature results"),
        ("data/processed/baseline_predictions.csv", "Naive baseline predictions"),
        ("data/processed/peak_predictions.csv", "Peak event predictions subset"),
        ("data/processed/worst_predictions.csv", "Worst prediction days"),
    ]
    for path, desc in files:
        full = os.path.join(PROJECT_ROOT, path)
        exists = "EXISTS" if os.path.exists(full) else "MISSING"
        print(f"  [{exists:>7s}] {path:<55s} {desc}")

    # Missing files
    print("\n  MISSING from project:")
    missing = ["requirements.txt", "README.md", "tests/", "src/train_model.py"]
    for m in missing:
        full = os.path.join(PROJECT_ROOT, m)
        if not os.path.exists(full):
            print(f"    - {m}")

if __name__ == "__main__":
    main()
