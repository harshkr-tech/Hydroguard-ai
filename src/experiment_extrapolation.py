
"""HydroGuard AI: safe extrapolation experiment.
Creates new timestamped outputs and preserves existing results.
"""

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "streamflow.csv"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# Load and validate daily discharge data
df = pd.read_csv(DATA, parse_dates=["date"])
df["discharge_cfs"] = pd.to_numeric(
    df["discharge_cfs"], errors="coerce"
)
df = (
    df.dropna(subset=["date", "discharge_cfs"])
    .sort_values("date")
    .drop_duplicates(subset="date", keep="last")
    .reset_index(drop=True)
)

# Features use past discharge only
flow = df["discharge_cfs"]
df["lag_1"] = flow.shift(1)
df["lag_2"] = flow.shift(2)
df["lag_3"] = flow.shift(3)
df["lag_7"] = flow.shift(7)
df["flow_change_1"] = flow.shift(1) - flow.shift(2)

features = ["lag_1", "lag_2", "lag_3", "lag_7", "flow_change_1"]
df = df.dropna(subset=features + ["discharge_cfs"]).reset_index(drop=True)

# Preserve the existing project's chronological 80/20 split approach
split_idx = int(len(df) * 0.80)
train = df.iloc[:split_idx].copy()
test = df.iloc[split_idx:].copy()

X_train = train[features]
y_train = train["discharge_cfs"]
X_test = test[features]
y_test = test["discharge_cfs"]

# Linear regression can extrapolate, unlike tree models,
# but may still fail on abrupt flood spikes.
model = LinearRegression()
model.fit(X_train, y_train)

pred = np.maximum(model.predict(X_test), 0)
naive = test["lag_1"].to_numpy()
actual = y_test.to_numpy()

# Use training-only threshold to define peaks
threshold = y_train.quantile(0.90)
peak_mask = actual >= threshold

def metrics(y, p):
    return {
        "MAE": mean_absolute_error(y, p),
        "RMSE": np.sqrt(mean_squared_error(y, p)),
        "Bias_pred_minus_actual": np.mean(p - y),
    }

rows = []
for name, prediction in [
    ("Naive_lag1", naive),
    ("LinearRegression", pred),
]:
    overall = metrics(actual, prediction)
    peak = metrics(actual[peak_mask], prediction[peak_mask]) if peak_mask.any() else {}
    rows.append({
        "model": name,
        **{f"overall_{k}": v for k, v in overall.items()},
        **{f"peak_{k}": v for k, v in peak.items()},
        "peak_days": int(peak_mask.sum()),
        "peak_threshold_cfs": float(threshold),
    })

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
results_path = OUT / f"extrapolation_metrics_{run_id}.csv"
pred_path = OUT / f"extrapolation_predictions_{run_id}.csv"

predictions = pd.DataFrame({
    "date": test["date"].to_numpy(),
    "actual": actual,
    "naive_predicted": naive,
    "linear_regression_predicted": pred,
    "linear_abs_error": np.abs(actual - pred),
    "linear_bias_pred_minus_actual": pred - actual,
    "above_training_max": actual > y_train.max(),
})

pd.DataFrame(rows).to_csv(results_path, index=False)
predictions.to_csv(pred_path, index=False)

print(f"Train dates: {train['date'].min().date()} to {train['date'].max().date()}")
print(f"Test dates:  {test['date'].min().date()} to {test['date'].max().date()}")
print(f"Training max discharge: {y_train.max():,.0f} cfs")
print(f"Peak threshold (training P90): {threshold:,.0f} cfs")
print(f"Test peak days: {int(peak_mask.sum())}")
print("\nMETRICS")
print(pd.DataFrame(rows).round(2).to_string(index=False))
print("\nSaved new files:")
print(results_path)
print(pred_path)
