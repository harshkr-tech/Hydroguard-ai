
from pathlib import Path
from datetime import datetime

import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# HydroGuard AI - River Discharge Prediction
# Chronological evaluation with leakage-safe features
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "raw" / "streamflow.csv"
MODELS_DIR = ROOT / "models"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

# Keep the established project evaluation period.
TEST_START = pd.Timestamp("2024-12-31")

TARGET = "discharge_cfs"

# Best overall feature set from the latest experiment.
FEATURES = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_7",
    "rolling_mean_7",
]

RANDOM_STATE = 42


def load_data():
    """Load and validate the daily discharge dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Check that data/raw/streamflow.csv exists."
        )

    df = pd.read_csv(DATA_PATH)

    required = {"date", TARGET}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

    df = df.dropna(subset=["date", TARGET])
    df = df.sort_values("date").drop_duplicates("date")
    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError("The dataset contains no valid observations.")

    if (df[TARGET] < 0).any():
        raise ValueError(
            "Negative discharge values found. Inspect the dataset "
            "before training."
        )

    return df


def create_features(df):
    """Create features using only earlier discharge observations."""
    df = df.copy()

    df["lag_1"] = df[TARGET].shift(1)
    df["lag_2"] = df[TARGET].shift(2)
    df["lag_3"] = df[TARGET].shift(3)
    df["lag_7"] = df[TARGET].shift(7)

    # Shift first so the current day's actual discharge is not used.
    df["rolling_mean_7"] = (
        df[TARGET].shift(1).rolling(window=7, min_periods=7).mean()
    )

    df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
    return df


def calculate_metrics(actual, predicted):
    """Calculate standard regression metrics."""
    return {
        "MAE_cfs": float(mean_absolute_error(actual, predicted)),
        "RMSE_cfs": float(
            np.sqrt(mean_squared_error(actual, predicted))
        ),
        "R2": float(r2_score(actual, predicted)),
        "Bias_cfs": float(np.mean(predicted - actual)),
    }


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("HYDROGUARD AI - MODEL TRAINING")
    print("=" * 65)

    df = load_data()

    print(f"Dataset: {DATA_PATH}")
    print(f"Valid observations: {len(df)}")
    print(f"Date range: {df['date'].min().date()} to "
          f"{df['date'].max().date()}")

    data = create_features(df)

    train = data[data["date"] < TEST_START].copy()
    test = data[data["date"] >= TEST_START].copy()

    if train.empty or test.empty:
        raise ValueError(
            f"Empty train or test set. Training dates must be before "
            f"{TEST_START.date()} and test dates must be on/after it."
        )

    X_train = train[FEATURES]
    y_train = train[TARGET]
    X_test = test[FEATURES]
    y_test = test[TARGET]

    print(f"\nTraining rows: {len(train)}")
    print(f"Testing rows:  {len(test)}")
    print(f"Test begins:   {test['date'].min().date()}")
    print(f"Features:      {FEATURES}")
    print(f"Training maximum discharge: {y_train.max():,.0f} cfs")

    model = RandomForestRegressor(
        n_estimators=400,
        max_features=1.0,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Naive baseline: predict today's discharge using yesterday's value.
    naive_predictions = test["lag_1"].to_numpy()

    rf_metrics = calculate_metrics(y_test, predictions)
    naive_metrics = calculate_metrics(y_test, naive_predictions)

    print("\nOVERALL TEST RESULTS")
    print("-" * 65)

    for name, metrics in [
        ("Random Forest", rf_metrics),
        ("Naive baseline", naive_metrics),
    ]:
        print(f"\n{name}")
        print(f"  MAE:  {metrics['MAE_cfs']:,.2f} cfs")
        print(f"  RMSE: {metrics['RMSE_cfs']:,.2f} cfs")
        print(f"  R2:   {metrics['R2']:.4f}")
        print(f"  Bias: {metrics['Bias_cfs']:,.2f} cfs")

    # Evaluate high-flow days using a threshold calculated from training
    # data only, avoiding information from the test period.
    peak_threshold = float(y_train.quantile(0.90))
    peak_mask = y_test.to_numpy() >= peak_threshold
    peak_count = int(peak_mask.sum())

    if peak_count:
        peak_metrics = calculate_metrics(
            y_test.to_numpy()[peak_mask],
            predictions[peak_mask],
        )
    else:
        peak_metrics = {
            "MAE_cfs": float("nan"),
            "RMSE_cfs": float("nan"),
            "R2": float("nan"),
            "Bias_cfs": float("nan"),
        }

    print("\nHIGH-FLOW TEST RESULTS")
    print("-" * 65)
    print(f"Training 90th-percentile threshold: {peak_threshold:,.2f} cfs")
    print(f"High-flow test days: {peak_count}")

    if peak_count:
        print(f"Peak MAE:  {peak_metrics['MAE_cfs']:,.2f} cfs")
        print(f"Peak RMSE: {peak_metrics['RMSE_cfs']:,.2f} cfs")
        print(f"Peak bias: {peak_metrics['Bias_cfs']:,.2f} cfs")
    else:
        print("No test observations exceeded the high-flow threshold.")

    # Save results under timestamped names so previous outputs are preserved.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = pd.DataFrame({
        "date": test["date"].to_numpy(),
        "actual": y_test.to_numpy(),
        "naive_predicted": naive_predictions,
        "rf_predicted": predictions,
        "rf_error": predictions - y_test.to_numpy(),
        "rf_abs_error": np.abs(predictions - y_test.to_numpy()),
    })

    results_path = (
        PROCESSED_DIR / f"training_predictions_{timestamp}.csv"
    )
    results.to_csv(results_path, index=False)

    metrics_path = PROCESSED_DIR / f"training_metrics_{timestamp}.json"
    metrics = {
        "model": "RandomForestRegressor",
        "features": FEATURES,
        "test_start": str(test["date"].min().date()),
        "test_end": str(test["date"].max().date()),
        "training_rows": len(train),
        "testing_rows": len(test),
        "training_max_discharge_cfs": float(y_train.max()),
        "peak_threshold_cfs": peak_threshold,
        "peak_test_days": peak_count,
        "random_forest": rf_metrics,
        "naive_baseline": naive_metrics,
        "peak_metrics": peak_metrics,
    }

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, allow_nan=True)

    model_path = MODELS_DIR / f"hydroguard_rf_{timestamp}.joblib"
    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "target": TARGET,
            "peak_threshold_cfs": peak_threshold,
            "training_max_discharge_cfs": float(y_train.max()),
        },
        model_path,
    )

    report_path = REPORTS_DIR / f"training_report_{timestamp}.txt"
    with report_path.open("w", encoding="utf-8") as file:
        file.write("HydroGuard AI - Training Report\n")
        file.write("=" * 45 + "\n")
        file.write(f"Dataset: {DATA_PATH}\n")
        file.write(f"Train rows: {len(train)}\n")
        file.write(f"Test rows: {len(test)}\n")
        file.write(f"Features: {FEATURES}\n")
        file.write(f"Training maximum: {y_train.max():,.2f} cfs\n")
        file.write(f"Peak threshold: {peak_threshold:,.2f} cfs\n")
        file.write(f"Peak test days: {peak_count}\n\n")
        file.write(f"Random Forest: {rf_metrics}\n")
        file.write(f"Naive baseline: {naive_metrics}\n")
        file.write(f"Peak metrics: {peak_metrics}\n")

    print("\nSAVED OUTPUTS")
    print("-" * 65)
    print(f"Predictions: {results_path}")
    print(f"Metrics:     {metrics_path}")
    print(f"Model:       {model_path}")
    print(f"Report:      {report_path}")

    print("\nIMPORTANT")
    print(
        "This script saves a timestamped model. It does not overwrite "
        "your backup scripts or an existing model file."
    )
    print(
        "Random Forest predictions may underestimate flows beyond the "
        "maximum discharge observed during training."
    )
    print("\nTraining and evaluation completed.")


if __name__ == "__main__":
    main()
