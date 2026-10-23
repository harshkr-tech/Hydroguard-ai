
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "model_comparison.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "peak_evaluation.csv"

# Load and validate the existing model comparison results.
if not CSV_PATH.exists():
    raise FileNotFoundError(f"Input CSV not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH, parse_dates=["date"])

required = ["date", "actual", "naive_predicted", "rf_predicted"]
missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(f"Missing required CSV columns: {missing}")

for col in ["actual", "naive_predicted", "rf_predicted"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = (
    df.dropna(subset=required)
    .sort_values("date")
    .reset_index(drop=True)
)

if df.empty:
    raise ValueError("No valid rows remain in model_comparison.csv")

# Retrospective peak evaluation: top 10% of actual test-period discharge.
threshold = df["actual"].quantile(0.90)
peaks = df.loc[df["actual"] >= threshold].copy()

if peaks.empty:
    raise ValueError("No peak rows found.")

peaks["rf_error"] = peaks["rf_predicted"] - peaks["actual"]
peaks["rf_abs_error"] = peaks["rf_error"].abs()
peaks["naive_error"] = peaks["naive_predicted"] - peaks["actual"]
peaks["naive_abs_error"] = peaks["naive_error"].abs()

rf_mae = mean_absolute_error(peaks["actual"], peaks["rf_predicted"])
rf_rmse = mean_squared_error(
    peaks["actual"], peaks["rf_predicted"]
) ** 0.5
naive_mae = mean_absolute_error(
    peaks["actual"], peaks["naive_predicted"]
)
naive_rmse = mean_squared_error(
    peaks["actual"], peaks["naive_predicted"]
) ** 0.5

print(f"Test rows: {len(df)}")
print(f"Test dates: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Peak threshold (90th percentile): {threshold:,.2f} cfs")
print(f"Peak rows: {len(peaks)}")

print("\nPeak metrics (lower is better):")
print(f"Random Forest MAE:  {rf_mae:,.2f} cfs")
print(f"Random Forest RMSE: {rf_rmse:,.2f} cfs")
print(f"Naive baseline MAE: {naive_mae:,.2f} cfs")
print(f"Naive baseline RMSE: {naive_rmse:,.2f} cfs")

print("\nLargest Random Forest errors:")
print(
    peaks.sort_values("rf_abs_error", ascending=False)[
        [
            "date",
            "actual",
            "naive_predicted",
            "rf_predicted",
            "rf_error",
            "rf_abs_error",
        ]
    ].head(10).to_string(index=False)
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
peaks.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved peak evaluation to: {OUTPUT_PATH}")
