"""
app.py -- HydroGuard AI Streamlit Dashboard

Displays historical streamflow data and model comparison results.
Does NOT generate forecasts or claim to be an operational flood-warning system.

Usage:
    streamlit run app.py
"""

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------
# Paths (relative to this file's location)
# ----------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_CSV = os.path.join(SCRIPT_DIR, "data", "raw", "streamflow.csv")
COMPARISON_CSV = os.path.join(SCRIPT_DIR, "data", "processed", "model_comparison.csv")


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------
def mae(actual, predicted):
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual, predicted):
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


@st.cache_data
def load_raw_data(path):
    """Load the raw streamflow CSV."""
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


@st.cache_data
def load_comparison_data(path):
    """Load model comparison CSV."""
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ----------------------------------------------------------------
# Page config
# ----------------------------------------------------------------
st.set_page_config(
    page_title="HydroGuard AI",
    page_icon=":droplet:",
    layout="wide",
)

# ----------------------------------------------------------------
# Header
# ----------------------------------------------------------------
st.title(":droplet: HydroGuard AI")
st.caption("Potomac River at Little Falls (USGS-01646500) -- Daily Mean Discharge")

st.warning(
    "**Disclaimer:** This dashboard shows experimental model comparisons on "
    "historical data. It is **not** an operational flood-warning system. "
    "Do not use these predictions for real-world safety decisions.",
    icon="\u26A0\uFE0F",
)

# ================================================================
# SECTION 1 -- Historical Streamflow
# ================================================================
st.header("1. Historical Streamflow")

# --- Load raw data ---
if not os.path.isfile(RAW_CSV):
    st.error(
        f"Raw data file not found: `{RAW_CSV}`\n\n"
        "Run `python src/download_data.py` first to download the dataset."
    )
    st.stop()

try:
    raw = load_raw_data(RAW_CSV)
except Exception as e:
    st.error(f"Could not read raw data: {e}")
    st.stop()

# Check expected column
if "discharge_cfs" not in raw.columns or "date" not in raw.columns:
    st.error(
        f"Unexpected columns in raw CSV. Expected `date` and `discharge_cfs`, "
        f"got: {list(raw.columns)}"
    )
    st.stop()

# --- Dataset summary metrics ---
date_min = raw["date"].min()
date_max = raw["date"].max()
n_obs = len(raw)
n_missing = int(raw["discharge_cfs"].isna().sum())

col1, col2, col3 = st.columns(3)
col1.metric("Date Range", f"{date_min.date()}  to  {date_max.date()}")
col2.metric("Observations", f"{n_obs:,}")
col3.metric("Missing Discharge Values", n_missing)

# --- Date-range filter ---
st.subheader("Date-Range Filter")

filter_start, filter_end = st.slider(
    "Select date range for the historical chart:",
    min_value=date_min.to_pydatetime(),
    
    max_value=date_max.to_pydatetime(),
    value=(date_min.to_pydatetime(), date_max.to_pydatetime()),
    format="YYYY-MM-DD",
)

mask = (raw["date"] >= pd.Timestamp(filter_start)) & (raw["date"] <= pd.Timestamp(filter_end))
filtered = raw.loc[mask]

st.caption(f"Showing {len(filtered):,} of {n_obs:,} observations")

# --- Historical chart ---
chart_data = filtered.set_index("date")[["discharge_cfs"]].rename(
    columns={"discharge_cfs": "Discharge (cfs)"}
)
st.line_chart(chart_data, y="Discharge (cfs)", width="stretch")

# ================================================================
# SECTION 2 -- Model Comparison
# ================================================================
st.header("2. Model Comparison (Test Period)")

# --- Load comparison data ---
if not os.path.isfile(COMPARISON_CSV):
    st.error(
        f"Comparison file not found: `{COMPARISON_CSV}`\n\n"
        "Run `python src/train_model.py` first to generate predictions."
    )
    st.stop()

try:
    comp = load_comparison_data(COMPARISON_CSV)
except Exception as e:
    st.error(f"Could not read comparison data: {e}")
    st.stop()

# Check expected columns
expected = {"date", "actual", "naive_predicted", "rf_predicted"}

if not expected.issubset(set(comp.columns)):
    missing_cols = expected - set(comp.columns)
    st.error(
        f"Comparison CSV is missing columns: {missing_cols}\n\n"
        f"Found columns: {list(comp.columns)}"
    )
    st.stop()

# --- Comparison chart ---
st.subheader("Actual vs Predicted Discharge")
comp["date"] = pd.to_datetime(comp["date"], errors="coerce")
comp = comp.dropna(subset=["date"]).sort_values("date")

plot_df = comp.set_index("date")[["actual", "naive_predicted", "rf_predicted"]].rename(
    columns={
        "actual": "Actual (cfs)",
        "naive_predicted": "Naive Baseline (cfs)",
        "rf_predicted": "Random Forest (cfs)",
    }
)
st.line_chart(plot_df, width="stretch")

st.caption(
    f"Test period: {comp['date'].iloc[0].date()} to {comp['date'].iloc[-1].date()} "
    f"({len(comp)} days)"
)

# --- Metrics ---
st.subheader("Error Metrics")

y = comp["actual"].values
naive_pred = comp["naive_predicted"].values
rf_pred = comp["rf_predicted"].values

naive_mae = mae(y, naive_pred)
naive_rmse = rmse(y, naive_pred)
rf_mae = mae(y, rf_pred)
rf_rmse = rmse(y, rf_pred)

mae_delta = naive_mae - rf_mae
rmse_delta = naive_rmse - rf_rmse

m1, m2, m3, m4 = st.columns(4)
m1.metric("Naive MAE (cfs)", f"{naive_mae:,.0f}")
m2.metric("Naive RMSE (cfs)", f"{naive_rmse:,.0f}")
m3.metric(
    "RF MAE (cfs)",
    f"{rf_mae:,.0f}",
    delta=f"{mae_delta:+,.0f} cfs",
    delta_color="normal",
)
m4.metric(
    "RF RMSE (cfs)",
    f"{rf_rmse:,.0f}",
    delta=f"{rmse_delta:+,.0f} cfs",
    delta_color="normal",
)

# --- Honest interpretation ---
st.markdown("---")
if rf_mae < naive_mae and rf_rmse < naive_rmse:
    st.success(
        f"Random Forest improves over the naive baseline on **both** metrics: "
        f"MAE reduced by {mae_delta:,.0f} cfs ({mae_delta/naive_mae*100:.1f}%), "
        f"RMSE reduced by {rmse_delta:,.0f} cfs ({rmse_delta/naive_rmse*100:.1f}%)."
    )
elif rf_mae < naive_mae:
    st.info(
        f"Random Forest improves MAE by {mae_delta:,.0f} cfs "
        f"but RMSE is worse by {-rmse_delta:,.0f} cfs."
    )
elif rf_rmse < naive_rmse:
    st.info(
        f"Random Forest improves RMSE by {rmse_delta:,.0f} cfs "
        f"but MAE is worse by {-mae_delta:,.0f} cfs."
    )
else:
    st.error("Random Forest does **not** beat the naive baseline on either metric.")

# ================================================================
# Footer
# ================================================================
st.markdown("---")
st.caption(
    "HydroGuard AI | Data source: USGS Water Data API | "
    "Station: USGS-01646500 (Potomac River at Little Falls) | "
    "Discharge in cfs (cubic feet per second)"
)

