"""
generate_plots.py -- Generate evaluation plots for HydroGuard AI.
Saves plots to reports/ directory.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
COMP_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "experiment_comparison.csv")
METRICS_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "experiment_metrics.csv")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports")

os.makedirs(REPORT_DIR, exist_ok=True)

def main():
    comp = pd.read_csv(COMP_CSV, parse_dates=["date"]).sort_values("date")
    metrics = pd.read_csv(METRICS_CSV)

    # ---- Plot 1: Actual vs Best RF vs Naive (full test period) ----
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(comp["date"], comp["actual"], label="Actual", color="black", linewidth=1.2)
    ax.plot(comp["date"], comp["naive_predicted"], label="Naive Baseline", color="gray", linewidth=0.8, alpha=0.7)
    ax.plot(comp["date"], comp["rf_base_5feat"], label="RF (5 features)", color="blue", linewidth=0.8, alpha=0.8)
    ax.plot(comp["date"], comp["rf_log1p_weighted"], label="RF log1p+weighted", color="red", linewidth=0.8, alpha=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Discharge (cfs)")
    ax.set_title("HydroGuard AI: Actual vs Predicted Discharge (Test Period 2025)")
    ax.legend(loc="upper left", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "actual_vs_predicted.png"), dpi=150)
    plt.close(fig)
    print("  Saved: actual_vs_predicted.png")

    # ---- Plot 2: Peak event zoom (May 2025) ----
    may_mask = (comp["date"] >= "2025-05-01") & (comp["date"] <= "2025-06-30")
    may = comp[may_mask]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(may["date"], may["actual"], label="Actual", color="black", linewidth=1.5, marker="o", markersize=3)
    ax.plot(may["date"], may["naive_predicted"], label="Naive", color="gray", linewidth=1, alpha=0.7)
    ax.plot(may["date"], may["rf_base_5feat"], label="RF (5 feat)", color="blue", linewidth=1, marker="s", markersize=2)
    ax.plot(may["date"], may["rf_log1p_weighted"], label="RF log1p+wt", color="red", linewidth=1, marker="^", markersize=2)
    ax.plot(may["date"], may["gbr_log1p_weighted"], label="GBR log1p+wt", color="green", linewidth=1, marker="d", markersize=2)
    ax.axhline(y=102000, color="orange", linestyle="--", alpha=0.7, label="Training max (102K)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Discharge (cfs)")
    ax.set_title("Peak Flood Events: May-June 2025 (Zoom)")
    ax.legend(loc="upper right", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "peak_events_zoom.png"), dpi=150)
    plt.close(fig)
    print("  Saved: peak_events_zoom.png")

    # ---- Plot 3: Model comparison bar chart ----
    models = metrics["model"].tolist()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Overall MAE
    colors = ["gray" if m == "Naive" else "steelblue" for m in models]
    best_idx = metrics.loc[metrics["model"] != "Naive", "mae"].idxmin()
    colors[best_idx] = "darkgreen"
    axes[0].barh(models, metrics["mae"], color=colors)
    axes[0].set_xlabel("MAE (cfs)")
    axes[0].set_title("Overall MAE")
    axes[0].invert_yaxis()

    # Peak MAE
    colors2 = ["gray" if m == "Naive" else "steelblue" for m in models]
    best_peak_idx = metrics.loc[metrics["model"] != "Naive", "peak_mae"].idxmin()
    colors2[best_peak_idx] = "darkgreen"
    axes[1].barh(models, metrics["peak_mae"], color=colors2)
    axes[1].set_xlabel("Peak MAE (cfs)")
    axes[1].set_title("Peak-Event MAE")
    axes[1].invert_yaxis()

    # R-squared
    colors3 = ["gray" if m == "Naive" else "steelblue" for m in models]
    best_r2_idx = metrics.loc[metrics["model"] != "Naive", "r2"].idxmax()
    colors3[best_r2_idx] = "darkgreen"
    axes[2].barh(models, metrics["r2"], color=colors3)
    axes[2].set_xlabel("R-squared")
    axes[2].set_title("Overall R-squared")
    axes[2].invert_yaxis()

    fig.suptitle("HydroGuard AI: Model Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "model_comparison_chart.png"), dpi=150)
    plt.close(fig)
    print("  Saved: model_comparison_chart.png")

    # ---- Plot 4: Residual analysis for RF_base ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    residuals = comp["rf_base_5feat"] - comp["actual"]
    axes[0].scatter(comp["actual"], residuals, alpha=0.5, s=10, c="steelblue")
    axes[0].axhline(y=0, color="red", linestyle="--")
    axes[0].set_xlabel("Actual Discharge (cfs)")
    axes[0].set_ylabel("Residual (Predicted - Actual)")
    axes[0].set_title("RF Base: Residuals vs Actual")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(comp["actual"], comp["rf_base_5feat"], alpha=0.5, s=10, c="steelblue")
    max_val = max(comp["actual"].max(), comp["rf_base_5feat"].max())
    axes[1].plot([0, max_val], [0, max_val], "r--", label="Perfect prediction")
    axes[1].set_xlabel("Actual Discharge (cfs)")
    axes[1].set_ylabel("Predicted Discharge (cfs)")
    axes[1].set_title("RF Base: Predicted vs Actual")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "residual_analysis.png"), dpi=150)
    plt.close(fig)
    print("  Saved: residual_analysis.png")

    print("\n  All plots saved to reports/")

if __name__ == "__main__":
    main()
