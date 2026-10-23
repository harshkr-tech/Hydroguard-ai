r"""
experiment_peak_improvement.py -- Systematic peak-prediction improvement experiments.

This script:
  1. Fixes bugs from the audit (duplicate features, NaN handling)
  2. Establishes a clean reproducible baseline
  3. Tests multiple model variants:
     - RF_base: original 5-feature RF (no weighting)
     - RF_extended: 11 cleaned features (no weighting)
     - RF_weighted: 11 features + training-based sample weighting
     - RF_log1p: 11 features + log1p target transform
     - RF_log1p_weighted: 11 features + log1p + weighting
     - GBR: Gradient Boosting Regressor (11 features)
     - GBR_log1p_weighted: GBR + log1p + weighting
  4. Compares all on identical test dates against the naive baseline
  5. Reports overall AND peak-specific metrics
  6. Saves results to new timestamped files (never overwrites raw data or previous models)

Usage:
    .\.venv\Scripts\python.exe src\experiment_peak_improvement.py
"""

import os
import sys
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------------------------------------------
# Paths
# ----------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_CSV = os.path.join(PROJECT_ROOT, "data", "raw", "streamflow.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports")

RANDOM_STATE = 42

# ----------------------------------------------------------------
# Feature definitions (fixed: removed duplicate lag_7_max)
# ----------------------------------------------------------------
BASE_FEATURES = ["lag_1", "lag_2", "lag_3", "lag_7", "rolling_mean_7"]

EXTENDED_FEATURES = [
    "lag_1", "lag_2", "lag_3", "lag_7", "lag_14",
    "rolling_mean_7", "rolling_max_7", "rolling_std_7",
    "flow_change_1",
    "rolling_max_14", "rolling_max_30",
]

TARGET_COL = "discharge_cfs"


# ================================================================
# Data loading and feature engineering
# ================================================================
def load_and_sort(csv_path):
    """Load CSV, parse dates, sort chronologically."""
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def create_features(df):
    """
    Build lag and rolling features from discharge_cfs.
    All features use ONLY past discharge values (shift >= 1).
    No target leakage.
    """
    d = df[["date", "discharge_cfs"]].copy()
    flow = d["discharge_cfs"]

    # Lag features (past only)
    d["lag_1"] = flow.shift(1)
    d["lag_2"] = flow.shift(2)
    d["lag_3"] = flow.shift(3)
    d["lag_7"] = flow.shift(7)
    d["lag_14"] = flow.shift(14)

    # Rolling statistics on past data only (shift(1) first)
    past_flow = flow.shift(1)
    d["rolling_mean_7"] = past_flow.rolling(7).mean()
    d["rolling_max_7"] = past_flow.rolling(7).max()
    d["rolling_std_7"] = past_flow.rolling(7).std()

    # Rate of change
    d["flow_change_1"] = flow.shift(1) - flow.shift(2)

    # Longer-term rolling peaks (past only)
    d["rolling_max_14"] = past_flow.rolling(14).max()
    d["rolling_max_30"] = past_flow.rolling(30).max()

    return d


# ================================================================
# Chronological split (fixed: dropna checks ALL feature columns)
# ================================================================
def chronological_split(df, feature_cols, test_frac=0.20):
    """
    Chronological train/test split.
    Split point computed on full dataset, then NaN rows dropped.
    Returns (train, test, split_date).
    """
    n = len(df)
    split_idx = int(n * (1 - test_frac))
    split_date = df["date"].iloc[split_idx]

    # Drop rows with NaN in ANY feature column (checks all features)
    df_clean = df.dropna(subset=feature_cols).reset_index(drop=True)

    train = df_clean[df_clean["date"] < split_date].copy()
    test = df_clean[df_clean["date"] >= split_date].copy()

    return train, test, split_date


# ================================================================
# Evaluation
# ================================================================
def compute_metrics(actual, predicted, label=""):
    """Compute MAE, RMSE, R2, and bias."""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    bias = float(np.mean(predicted - actual))
    return {
        "model": label,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "bias": bias,
    }


def compute_peak_metrics(actual, predicted, label=""):
    """Compute metrics on peak subset."""
    if len(actual) == 0:
        return {
            "model": label,
            "peak_mae": np.nan,
            "peak_rmse": np.nan,
            "peak_bias": np.nan,
            "peak_count": 0,
        }
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    bias = float(np.mean(predicted - actual))
    return {
        "model": label,
        "peak_mae": mae,
        "peak_rmse": rmse,
        "peak_bias": bias,
        "peak_count": len(actual),
    }


# ================================================================
# Model experiment functions
# ================================================================
def train_rf(X_train, y_train, sample_weight=None, **kwargs):
    """Train a RandomForestRegressor."""
    params = dict(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=4,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    params.update(kwargs)
    rf = RandomForestRegressor(**params)
    rf.fit(X_train, y_train, sample_weight=sample_weight)
    return rf


def train_gbr(X_train, y_train, sample_weight=None, **kwargs):
    """Train a GradientBoostingRegressor."""
    params = dict(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        subsample=0.8,
    )
    params.update(kwargs)
    gbr = GradientBoostingRegressor(**params)
    gbr.fit(X_train, y_train, sample_weight=sample_weight)
    return gbr


# ================================================================
# Main experiment
# ================================================================
def main():
    print("=" * 72)
    print("  HydroGuard AI -- Peak Prediction Improvement Experiments")
    print("=" * 72)

    # ----------------------------------------------------------
    # 1. Load and prepare data
    # ----------------------------------------------------------
    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(f"Expected input dataset not found at: {RAW_CSV}")

    raw = load_and_sort(RAW_CSV)
    print(f"\n  Loaded {len(raw)} rows ({raw['date'].iloc[0].date()} to {raw['date'].iloc[-1].date()})")

    df = create_features(raw)

    # Split using extended features (most demanding NaN requirement)
    train, test, split_date = chronological_split(df, EXTENDED_FEATURES)

    print(f"\n  Split date  : {split_date.date()}")
    print(f"  Train rows  : {len(train)} ({train['date'].iloc[0].date()} to {train['date'].iloc[-1].date()})")
    print(f"  Test rows   : {len(test)} ({test['date'].iloc[0].date()} to {test['date'].iloc[-1].date()})")

    # Training data statistics (for peak threshold)
    y_train_all = train[TARGET_COL].values
    train_max = y_train_all.max()
    train_p90 = np.percentile(y_train_all, 90)
    print(f"\n  Training max discharge  : {train_max:,.0f} cfs")
    print(f"  Training 90th percentile: {train_p90:,.0f} cfs")

    # Peak threshold: training 90th percentile
    PEAK_THRESHOLD = train_p90

    # Prepare arrays
    X_train_ext = train[EXTENDED_FEATURES].values
    X_test_ext = test[EXTENDED_FEATURES].values
    X_train_base = train[BASE_FEATURES].values
    X_test_base = test[BASE_FEATURES].values
    y_train = train[TARGET_COL].values
    y_test = test[TARGET_COL].values

    # Peak mask for test set
    peak_mask = y_test >= PEAK_THRESHOLD
    print(f"  Peak test days (>= {PEAK_THRESHOLD:,.0f} cfs): {peak_mask.sum()}")

    # Naive baseline
    naive_preds = test["lag_1"].values

    # ----------------------------------------------------------
    # 2. Sample weights (computed from TRAINING data only)
    # ----------------------------------------------------------
    weights_4x = np.ones(len(y_train))
    weights_4x[y_train >= PEAK_THRESHOLD] = 4.0

    # Progressive weighting: weight proportional to discharge magnitude
    weights_prop = np.where(
        y_train >= PEAK_THRESHOLD,
        1.0 + 3.0 * (y_train - PEAK_THRESHOLD) / (train_max - PEAK_THRESHOLD + 1),
        1.0
    )

    # ----------------------------------------------------------
    # 3. Run experiments
    # ----------------------------------------------------------
    experiments = {}
    all_preds = {"date": test["date"].values, "actual": y_test, "naive_predicted": naive_preds}

    # --- Experiment 0: Naive Baseline ---
    experiments["Naive"] = naive_preds

    # --- Experiment 1: RF base (5 features, no weighting) ---
    print("\n  [1/7] Training RF_base (5 features, no weighting)...")
    rf_base = train_rf(X_train_base, y_train)
    preds_base = rf_base.predict(X_test_base)
    experiments["RF_base_5feat"] = preds_base
    all_preds["rf_base_5feat"] = preds_base

    # --- Experiment 2: RF extended (11 features, no weighting) ---
    print("  [2/7] Training RF_extended (11 features, no weighting)...")
    rf_ext = train_rf(X_train_ext, y_train)
    preds_ext = rf_ext.predict(X_test_ext)
    experiments["RF_ext_11feat"] = preds_ext
    all_preds["rf_ext_11feat"] = preds_ext

    # --- Experiment 3: RF weighted (11 features, 4x peak weight) ---
    print("  [3/7] Training RF_weighted (11 features, 4x peak weight)...")
    rf_wt = train_rf(X_train_ext, y_train, sample_weight=weights_4x)
    preds_wt = rf_wt.predict(X_test_ext)
    experiments["RF_weighted_4x"] = preds_wt
    all_preds["rf_weighted_4x"] = preds_wt

    # --- Experiment 4: RF log1p target (11 features) ---
    print("  [4/7] Training RF_log1p (11 features, log1p target)...")
    y_train_log = np.log1p(y_train)
    rf_log = train_rf(X_train_ext, y_train_log)
    preds_log_raw = rf_log.predict(X_test_ext)
    preds_log = np.expm1(preds_log_raw)  # convert back to cfs
    preds_log = np.maximum(preds_log, 0)  # floor at 0
    experiments["RF_log1p"] = preds_log
    all_preds["rf_log1p"] = preds_log

    # --- Experiment 5: RF log1p + progressive weighting ---
    print("  [5/7] Training RF_log1p_weighted (11 features, log1p + progressive weight)...")
    rf_log_wt = train_rf(X_train_ext, y_train_log, sample_weight=weights_prop)
    preds_log_wt_raw = rf_log_wt.predict(X_test_ext)
    preds_log_wt = np.expm1(preds_log_wt_raw)
    preds_log_wt = np.maximum(preds_log_wt, 0)
    experiments["RF_log1p_weighted"] = preds_log_wt
    all_preds["rf_log1p_weighted"] = preds_log_wt

    # --- Experiment 6: Gradient Boosting (11 features) ---
    print("  [6/7] Training GBR (11 features)...")
    gbr = train_gbr(X_train_ext, y_train)
    preds_gbr = gbr.predict(X_test_ext)
    experiments["GBR"] = preds_gbr
    all_preds["gbr_predicted"] = preds_gbr

    # --- Experiment 7: GBR log1p + progressive weighting ---
    print("  [7/7] Training GBR_log1p_weighted (11 features, log1p + progressive weight)...")
    gbr_log_wt = train_gbr(X_train_ext, y_train_log, sample_weight=weights_prop)
    preds_gbr_log_wt_raw = gbr_log_wt.predict(X_test_ext)
    preds_gbr_log_wt = np.expm1(preds_gbr_log_wt_raw)
    preds_gbr_log_wt = np.maximum(preds_gbr_log_wt, 0)
    experiments["GBR_log1p_weighted"] = preds_gbr_log_wt
    all_preds["gbr_log1p_weighted"] = preds_gbr_log_wt

    # ----------------------------------------------------------
    # 4. Compute metrics
    # ----------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"  OVERALL METRICS (all {len(y_test)} test days)")
    print("=" * 72)

    overall_results = []
    peak_results = []

    for name, preds in experiments.items():
        overall = compute_metrics(y_test, preds, label=name)
        overall_results.append(overall)

        peak_m = compute_peak_metrics(
            y_test[peak_mask], preds[peak_mask], label=name
        )
        peak_results.append(peak_m)

    # Print overall
    print(f"  {'Model':<22s}  {'MAE':>10s}  {'RMSE':>10s}  {'R2':>8s}  {'Bias':>10s}")
    print("  " + "-" * 66)
    for r in overall_results:
        print(f"  {r['model']:<22s}  {r['mae']:>10,.1f}  {r['rmse']:>10,.1f}  {r['r2']:>8.4f}  {r['bias']:>+10,.1f}")

    # Print peak
    print(f"\n{'=' * 72}")
    print(f"  PEAK-EVENT METRICS (actual >= {PEAK_THRESHOLD:,.0f} cfs, {peak_mask.sum()} days)")
    print(f"{'=' * 72}")
    print(f"  {'Model':<22s}  {'Peak MAE':>10s}  {'Peak RMSE':>10s}  {'Peak Bias':>10s}")
    print("  " + "-" * 56)
    for r in peak_results:
        print(f"  {r['model']:<22s}  {r['peak_mae']:>10,.1f}  {r['peak_rmse']:>10,.1f}  {r['peak_bias']:>+10,.1f}")

    # ----------------------------------------------------------
    # 5. Detailed worst predictions for key models
    # ----------------------------------------------------------
    best_peak = min(
        [r for r in peak_results if r["model"] != "Naive"],
        key=lambda x: x["peak_mae"]
    )
    best_overall = min(
        [r for r in overall_results if r["model"] != "Naive"],
        key=lambda x: x["mae"]
    )

    print(f"\n  Best overall MAE : {best_overall['model']} ({best_overall['mae']:,.1f} cfs)")
    print(f"  Best peak MAE    : {best_peak['model']} ({best_peak['peak_mae']:,.1f} cfs)")

    # Out-of-distribution analysis
    print(f"\n{'=' * 72}")
    print(f"  OUT-OF-DISTRIBUTION ANALYSIS")
    print(f"{'=' * 72}")
    print(f"  Training max: {train_max:,.0f} cfs")
    ood_mask = y_test > train_max
    ood_count = ood_mask.sum()
    print(f"  Test days exceeding training max: {ood_count}")
    if ood_count > 0:
        for name, preds in experiments.items():
            if name == "Naive":
                continue
            ood_mae = mean_absolute_error(y_test[ood_mask], preds[ood_mask])
            print(f"    {name:<22s}  OOD MAE: {ood_mae:>12,.1f} cfs")

    # Top 5 worst days for key models
    print(f"\n{'=' * 72}")
    print(f"  TOP 5 WORST PREDICTIONS PER KEY MODEL")
    print(f"{'=' * 72}")
    for name in ["RF_base_5feat", "RF_log1p_weighted", "GBR_log1p_weighted"]:
        if name not in experiments:
            continue
        preds = experiments[name]
        errors = np.abs(y_test - preds)
        worst_idx = np.argsort(errors)[::-1][:5]
        print(f"\n  {name}:")
        print(f"    {'Date':<12s}  {'Actual':>8s}  {'Predicted':>10s}  {'Error':>10s}")
        for idx in worst_idx:
            dt = test["date"].iloc[idx]
            print(f"    {str(dt.date()):<12s}  {y_test[idx]:>8,.0f}  {preds[idx]:>10,.1f}  {preds[idx]-y_test[idx]:>+10,.1f}")

    # ----------------------------------------------------------
    # 6. Feature importances for best RF model
    # ----------------------------------------------------------
    print(f"\n{'=' * 72}")
    print("  FEATURE IMPORTANCES (RF_log1p_weighted)")
    print(f"{'=' * 72}")

    importances = rf_log_wt.feature_importances_

    for feat, imp in sorted(
        zip(EXTENDED_FEATURES, importances),
        key=lambda x: -x[1]
    ):
        bar = "#" * int(imp * 50)
        print(f"  {feat:<16s}  {imp:.4f}  {bar}")

    # ----------------------------------------------------------
    # 7. Save results (unique filenames for every run)
    # ----------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 7a. Full comparison CSV
    comp_df = pd.DataFrame(all_preds)
    comp_path = os.path.join(
        OUTPUT_DIR, f"experiment_comparison_{run_id}.csv"
    )
    comp_df.to_csv(comp_path, index=False)
    print(f"\nSaved: {comp_path} ({len(comp_df)} rows)")

    # 7b. Metrics CSV
    metrics_data = []
    for ov, pk in zip(overall_results, peak_results):
        row = {**ov, **{k: v for k, v in pk.items() if k != "model"}}
        metrics_data.append(row)

    metrics_df = pd.DataFrame(metrics_data)
    metrics_path = os.path.join(
        OUTPUT_DIR, f"experiment_metrics_{run_id}.csv"
    )
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved: {metrics_path}")

    # 7c. Peak predictions CSV
    peak_comp = comp_df[
        comp_df["actual"] >= PEAK_THRESHOLD
    ].copy()
    peak_path = os.path.join(
        OUTPUT_DIR, f"experiment_peak_predictions_{run_id}.csv"
    )
    peak_comp.to_csv(peak_path, index=False)
    print(f"Saved: {peak_path} ({len(peak_comp)} rows)")

    # 7d. Out-of-distribution predictions CSV
    ood_comp = comp_df[
        comp_df["actual"] > train_max
    ].copy()

    if not ood_comp.empty:
        ood_path = os.path.join(
            OUTPUT_DIR, f"experiment_ood_predictions_{run_id}.csv"
        )
        ood_comp.to_csv(ood_path, index=False)
        print(f"Saved: {ood_path} ({len(ood_comp)} rows)")
    else:
        print("No test days exceeded the training maximum.")

    # 7e. Summary report
    report_path = os.path.join(
        REPORT_DIR, f"experiment_report_{run_id}.txt"
    )
    latest_report_path = os.path.join(REPORT_DIR, "experiment_report.txt")

    report_lines = []
    report_lines.append("HydroGuard AI - Peak Improvement Experiment Report")
    report_lines.append(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    report_lines.append(
        f"Dataset: {len(raw)} days, "
        f"{raw['date'].iloc[0].date()} to {raw['date'].iloc[-1].date()}"
    )
    report_lines.append(f"Train days: {len(train)}")
    report_lines.append(f"Test days: {len(test)}")
    report_lines.append(f"Split date: {split_date.date()}")
    report_lines.append(f"Training maximum: {train_max:,.0f} cfs")
    report_lines.append(f"Peak threshold: {PEAK_THRESHOLD:,.0f} cfs")
    report_lines.append(f"Peak test days: {int(peak_mask.sum())}")
    report_lines.append(f"OOD test days: {int(ood_count)}")
    report_lines.append("")
    report_lines.append(
        f"{'Model':<22} {'MAE':>10} {'RMSE':>10} {'R2':>8} "
        f"{'Peak MAE':>10} {'Peak RMSE':>12} {'Peak Bias':>10}"
    )
    report_lines.append("-" * 100)

    for ov, pk in zip(overall_results, peak_results):
        report_lines.append(
            f"{ov['model']:<22} "
            f"{ov['mae']:>10,.1f} "
            f"{ov['rmse']:>10,.1f} "
            f"{ov['r2']:>8.4f} "
            f"{pk['peak_mae']:>10,.1f} "
            f"{pk['peak_rmse']:>12,.1f} "
            f"{pk['peak_bias']:>+10,.1f}"
        )

    report_lines.extend([
        "",
        "Notes:",
        "- Tree-based models generally cannot extrapolate reliably beyond the training target range.",
        "- Peak weighting and log1p transformation are experimental approaches, not guaranteed improvements.",
        "- Features use past discharge values only."
    ])

    report_content = "\n".join(report_lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    with open(latest_report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"Saved report: {report_path}")
    print(f"Updated latest report: {latest_report_path}")
    print("\n" + "=" * 72)
    print("EXPERIMENT COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()

