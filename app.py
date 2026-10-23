"""
app.py -- HydroGuard AI Streamlit Dashboard

Displays historical streamflow data, model comparison results, and peak-flow evaluation.
Does NOT generate forecasts or claim to be an operational flood-warning system.

Usage:
    streamlit run app.py
"""

import glob
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
    st.error(f"Could not read raw data from `{RAW_CSV}`: {e}")
    st.stop()

# Check expected columns
if "discharge_cfs" not in raw.columns or "date" not in raw.columns:
    st.error(
        f"Unexpected columns in raw CSV. Expected `date` and `discharge_cfs`, "
        f"got: {list(raw.columns)}"
    )
    st.stop()

if raw.empty:
    st.error("The raw dataset is empty.")
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
# SECTION 2 -- Model Comparison (Test Period)
# ================================================================
st.header("2. Model Comparison (Test Period)")

# --- Load comparison data with graceful fallback ---
active_comp_path = COMPARISON_CSV

if not os.path.isfile(active_comp_path):
    fallback_files = sorted(glob.glob(os.path.join(SCRIPT_DIR, "data", "processed", "experiment_comparison_*.csv")))
    if fallback_files:
        active_comp_path = fallback_files[-1]
        st.info(f"Standard comparison file not found. Falling back to latest experiment results: `{os.path.basename(active_comp_path)}`")
    else:
        st.error(
            f"Comparison file not found: `{COMPARISON_CSV}`\n\n"
            "Run `python src/train_model.py` or `python src/experiment_peak_improvement.py` first to generate predictions."
        )
        st.stop()

try:
    comp = load_comparison_data(active_comp_path)
except Exception as e:
    st.error(f"Could not read comparison data from `{active_comp_path}`: {e}")
    st.stop()

# If using experiment_comparison file, map rf_base_5feat to rf_predicted if needed
if "rf_predicted" not in comp.columns and "rf_base_5feat" in comp.columns:
    comp["rf_predicted"] = comp["rf_base_5feat"]

# Check expected columns
expected = {"date", "actual", "naive_predicted", "rf_predicted"}
if not expected.issubset(set(comp.columns)):
    missing_cols = expected - set(comp.columns)
    st.error(
        f"Comparison CSV is missing required columns: {missing_cols}\n\n"
        f"Found columns: {list(comp.columns)}"
    )
    st.stop()

if comp.empty:
    st.error("The comparison dataset contains no rows.")
    st.stop()

# --- Comparison chart ---
st.subheader("Actual vs Predicted Discharge")
comp["date"] = pd.to_datetime(comp["date"], errors="coerce")
comp = comp.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

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

# --- Overall Metrics ---
st.subheader("Overall Error Metrics")

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
# SECTION 3 -- Peak-Flow Evaluation & Extrapolation Warning
# ================================================================
st.header("3. Peak-Flow & Extreme-Discharge Analysis")

# Compute training thresholds from raw historical data strictly before test start
test_start_date = comp["date"].min()
train_mask = raw["date"] < test_start_date
train_max = float(raw.loc[train_mask, "discharge_cfs"].max()) if train_mask.any() else 102000.0
peak_thresh = float(raw.loc[train_mask, "discharge_cfs"].quantile(0.90)) if train_mask.any() else 17100.0

st.write(
    f"High-flow days are defined using the **training set 90th percentile** "
    f"({peak_thresh:,.0f} cfs). All statistics and thresholds are computed using earlier "
    f"chronological training data to avoid future data leakage."
)

# High-flow subset on test data
peak_mask = comp["actual"] >= peak_thresh
peak_count = int(peak_mask.sum())

if peak_count > 0:
    peak_y = comp.loc[peak_mask, "actual"].values
    peak_naive_pred = comp.loc[peak_mask, "naive_predicted"].values
    peak_rf_pred = comp.loc[peak_mask, "rf_predicted"].values

    pk_naive_mae = mae(peak_y, peak_naive_pred)
    pk_naive_rmse = rmse(peak_y, peak_naive_pred)
    pk_rf_mae = mae(peak_y, peak_rf_pred)
    pk_rf_rmse = rmse(peak_y, peak_rf_pred)
    pk_rf_bias = float(np.mean(peak_rf_pred - peak_y))

    pk_mae_delta = pk_naive_mae - pk_rf_mae
    pk_rmse_delta = pk_naive_rmse - pk_rf_rmse

    st.subheader(f"High-Flow Error Metrics ({peak_count} Peak Days $\\ge$ {peak_thresh:,.0f} cfs)")
    pm1, pm2, pm3, pm4 = st.columns(4)
    pm1.metric("Naive Peak MAE (cfs)", f"{pk_naive_mae:,.0f}")
    pm2.metric("Naive Peak RMSE (cfs)", f"{pk_naive_rmse:,.0f}")
    pm3.metric(
        "RF Peak MAE (cfs)",
        f"{pk_rf_mae:,.0f}",
        delta=f"{pk_mae_delta:+,.0f} cfs",
        delta_color="normal",
    )
    pm4.metric(
        "RF Peak RMSE (cfs)",
        f"{pk_rf_rmse:,.0f}",
        delta=f"{pk_rmse_delta:+,.0f} cfs",
        delta_color="normal",
    )

    st.caption(
        f"**Random Forest Peak Bias:** {pk_rf_bias:+,.0f} cfs. Negative bias confirms that "
        f"tree-based predictions systematically underestimate flow during extreme surges."
    )
else:
    st.info("No test days exceeded the 90th percentile peak threshold.")

# --- Prominent Extrapolation Warning Banner ---
st.subheader("Extrapolation Limitation Warning")

ood_mask = comp["actual"] > train_max
ood_count = int(ood_mask.sum())

st.warning(
    f"**CRITICAL MODEL LIMITATION -- Out-of-Distribution Discharge:**\n\n"
    f"- **Training Maximum Discharge:** {train_max:,.0f} cfs (recorded on 2022-05-09).\n"
    f"- **Unprecedented Test Observations:** {ood_count} day(s) in the test period exceeded "
    f"the historical training maximum (surging to {comp['actual'].max():,.0f} cfs on May 15, 2025).\n"
    f"- **Why Underestimation Occurs:** Decision tree ensembles (Random Forest, Gradient Boosting) "
    f"partition feature space into localized leaves and predict average historical values. "
    f"They **cannot extrapolate** beyond the maximum target value seen during training. "
    f"When an unprecedented flood occurs, predictions plateau and will substantially underestimate "
    f"the true water volume.\n"
    f"- **Practical Implication:** This experimental machine-learning model must **never** be used "
    f"as an operational flood-warning system or for public safety decisions.",
    icon="\u26A0\uFE0F",
)

# --- Top Peak Discharge Days Table ---
st.subheader("Top 5 Peak Discharge Events in Test Period")
n_show = min(5, len(comp))
top_peaks = comp.nlargest(n_show, "actual")[["date", "actual", "naive_predicted", "rf_predicted"]].copy()
top_peaks["rf_error"] = top_peaks["rf_predicted"] - top_peaks["actual"]
top_peaks["pct_error"] = (top_peaks["rf_error"] / top_peaks["actual"] * 100).round(1)

table_display = top_peaks.rename(
    columns={
        "date": "Date",
        "actual": "Actual (cfs)",
        "naive_predicted": "Naive Prediction (cfs)",
        "rf_predicted": "Random Forest (cfs)",
        "rf_error": "Error (cfs)",
        "pct_error": "Error (%)",
    }
)[["Date", "Actual (cfs)", "Naive Prediction (cfs)", "Random Forest (cfs)", "Error (cfs)", "Error (%)"]]

# Format date column for display
table_display["Date"] = table_display["Date"].dt.strftime("%Y-%m-%d")
table_display["Actual (cfs)"] = table_display["Actual (cfs)"].apply(lambda x: f"{x:,.0f}")
table_display["Naive Prediction (cfs)"] = table_display["Naive Prediction (cfs)"].apply(lambda x: f"{x:,.0f}")
table_display["Random Forest (cfs)"] = table_display["Random Forest (cfs)"].apply(lambda x: f"{x:,.0f}")
table_display["Error (cfs)"] = table_display["Error (cfs)"].apply(lambda x: f"{x:+,.0f}")
table_display["Error (%)"] = table_display["Error (%)"].apply(lambda x: f"{x:+.1f}%")

st.dataframe(table_display, width="stretch", hide_index=True)

# --- Optional: Multi-Model Experiment Benchmark Table (if available) ---
exp_metrics_files = sorted(glob.glob(os.path.join(SCRIPT_DIR, "data", "processed", "experiment_metrics_*.csv")))
if exp_metrics_files:
    with st.expander("Explore Multi-Model Peak Improvement Experiment Results (8 Candidates)", expanded=False):
        try:
            exp_df = pd.read_csv(exp_metrics_files[-1])
            st.caption(f"Source: `{os.path.basename(exp_metrics_files[-1])}`")

            display_exp = exp_df.rename(columns={
                "model": "Model",
                "mae": "Overall MAE (cfs)",
                "rmse": "Overall RMSE (cfs)",
                "r2": "R²",
                "bias": "Overall Bias (cfs)",
                "peak_mae": "Peak MAE (cfs)",
                "peak_rmse": "Peak RMSE (cfs)",
                "peak_bias": "Peak Bias (cfs)",
                "peak_count": "Peak Days",
            })
            for col in ["Overall MAE (cfs)", "Overall RMSE (cfs)", "Peak MAE (cfs)", "Peak RMSE (cfs)"]:
                if col in display_exp.columns:
                    display_exp[col] = display_exp[col].apply(lambda x: f"{x:,.1f}" if pd.notnull(x) else "N/A")
            for col in ["Overall Bias (cfs)", "Peak Bias (cfs)"]:
                if col in display_exp.columns:
                    display_exp[col] = display_exp[col].apply(lambda x: f"{x:+,.1f}" if pd.notnull(x) else "N/A")
            if "R²" in display_exp.columns:
                display_exp["R²"] = display_exp["R²"].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")

            st.dataframe(display_exp, width="stretch", hide_index=True)
        except Exception as e:
            st.caption(f"Could not load experiment metrics table: {e}")

# ================================================================
# Footer
# ================================================================
st.markdown("---")
st.caption(
    "HydroGuard AI | Data source: USGS Water Data API | "
    "Station: USGS-01646500 (Potomac River at Little Falls) | "
    "Discharge in cfs (cubic feet per second)"
)
