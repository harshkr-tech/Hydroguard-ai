# HydroGuard AI — Presentation Outline

A slide-by-slide presentation guide for defending or demonstrating the HydroGuard AI daily river discharge prediction system.

---

### Slide 1: Title & Introduction
- **Title:** HydroGuard AI: Daily River Discharge Prediction & Peak-Flow Machine Learning Audit
- **Subtitle:** Autoregressive Time-Series Forecasting on USGS Potomac River Streamflow Data
- **Presenter / Team:** HydroGuard AI Team
- **Core Message:** Evaluating machine learning capability, persistence baselines, and mathematical extrapolation boundaries in river discharge forecasting.

---

### Slide 2: Problem Statement & Motivation
- **Hydrological Importance:** River discharge governs municipal water supply, dam operations, and flood risk mitigation.
- **The ML Challenge:** Machine learning models are often deployed on time-series without rigorous leakage prevention or peak-flow audits.
- **The Extreme Event Dilemma:** Decision tree ensembles cannot predict values higher than historical training maximums, leading to systematic underpredictions during major flood events.

---

### Slide 3: Project Objectives
- Ingest and validate 5 years of verified daily USGS streamflow data (`2021-01-01` to `2025-12-31`).
- Implement a leak-free chronological 80/20 train/test evaluation framework.
- Establish a baseline benchmark (Naive persistence: today predicts tomorrow).
- Train a Random Forest Regressor and conduct systematic peak-flow experiments.
- Quantify extrapolation limits on out-of-distribution flood events ($> 102,000$ cfs).
- Deploy an interactive Streamlit dashboard for real-time model comparison and risk communication.

---

### Slide 4: Dataset & Quality Audit
- **Station:** Potomac River at Little Falls near Washington, D.C. (`USGS-01646500`).
- **Basin Area:** $\approx 11,560\text{ sq mi}$.
- **Data Period:** 5 complete calendar years (`2021-01-01` to `2025-12-31`).
- **Data Cleanliness:**
  - 1,826 daily mean observations (`parameter 00060`, `statistic 00003`).
  - Zero missing calendar days, zero missing discharge values, zero negative values.
  - Consistent units: cubic feet per second (cfs).
  - Minimum flow: 450 cfs | Mean: 8,538 cfs | Maximum: 136,000 cfs.

---

### Slide 5: Methodology & Leakage Prevention
- **Chronological Split:**
  - Training Period: `2021-01-01` to `2024-12-30` (1,460 days, 80%).
  - Testing Period: `2024-12-31` to `2025-12-31` (366 days, 20%).
  - No random shuffling to maintain temporal causality.
- **Strict Leakage Prevention:**
  - Lags $\text{lag}_1, \text{lag}_2, \text{lag}_3, \text{lag}_7$ use strictly past flow.
  - Rolling mean ($\text{rolling\_mean}_7$) shifted by 1 to exclude the prediction day.
  - High-flow threshold ($17,080$ cfs) computed strictly from training data.

---

### Slide 6: System Architecture & Workflow
- **Data Ingestion:** Automated USGS OGC API downloader (`download_data.py`).
- **Persistence Benchmark:** 1-day lag benchmark (`train_baseline.py`).
- **ML Training Pipeline:** Random Forest trainer (`train_model.py`) with artifact serialization.
- **Testing & Verification:** 21 automated unit tests (`tests/test_core.py`) passing.
- **Interactive UI:** Streamlit application (`app.py`) with dynamic filtering and extrapolation warnings.

---

### Slide 7: Model Performance: Naive Baseline vs. Random Forest
- **Overall Test Period (366 Days):**
  - **Naive Persistence Baseline:** MAE = 1,805.3 cfs | RMSE = 7,237.9 cfs | $R^2 = 0.6586$
  - **Random Forest (Production):** MAE = **1,518.0 cfs** | RMSE = **6,513.3 cfs** | $R^2 = \mathbf{0.7235}$
  - **Overall Improvement:** MAE reduced by **$287.3\text{ cfs}$ (15.9%)**, RMSE reduced by **$724.6\text{ cfs}$ (10.0%)**.
- **Feature Importance:**
  - `lag_1`: 91.3% (dominant persistence factor).
  - `lag_2`, `lag_3`, `rolling_mean_7`: provide incremental autoregressive smoothing.

---

### Slide 8: Peak-Flow Improvement Experiments (8 Candidates)
- Evaluated 8 model variations on the 39 high-flow test days ($\ge 17,080$ cfs):
  - Naive Peak MAE: 11,842.6 cfs | Peak RMSE: 21,964.6 cfs
  - `RF_base_5feat`: Peak MAE: **9,386.6 cfs** | Peak RMSE: **19,682.7 cfs**
  - `RF_weighted_4x`: Peak MAE: **9,384.1 cfs** | Peak RMSE: 20,757.9 cfs
- **The Core Trade-off:**
  - 4x sample weighting improved peak MAE by a negligible 2.5 cfs (0.02%), but degraded overall MAE by **+204.6 cfs (+13.5%)** and worsened peak RMSE.
  - Log-transformation and Gradient Boosting also failed to outperform base Random Forest.
  - **Verdict:** `RF_base_5feat` preserved as the optimal production model.

---

### Slide 9: Extrapolation Boundaries & The May 2025 Flood Event
- **Mathematical Law of Decision Trees:** $\hat{y} \le \max(y_{\text{train}})$.
- **The Test Event:**
  - Training historical maximum: **102,000 cfs** (May 9, 2022).
  - On May 15, 2025: Flow surged from 25,900 cfs to **136,000 cfs** (+110,100 cfs in 24 hours).
  - Random Forest predicted: **33,018 cfs** (Error: -102,982 cfs).
- **Key Scientific Insight:** Without upstream precipitation grids or hydraulic physics, univariate ML cannot foresee sudden flood waves before the gauge itself rises.

---

### Slide 10: Interactive Streamlit Dashboard Demo
- **Section 1: Historical Streamflow:**
  - Dynamic date-range slider across all 1,826 days.
  - Summary cards for total observations, dates, and missing counts.
- **Section 2: Test Model Comparison:**
  - Real-time line chart comparing Actual vs. Naive vs. Random Forest.
  - Overall MAE / RMSE metric cards with delta indicators.
- **Section 3: Peak-Flow Analytics & Extrapolation Warnings:**
  - High-flow error metrics (39 peak days).
  - Prominent warning banner explaining physical and algorithmic extrapolation bounds.
  - Top 5 peak events error inspection table.
  - Collapsible 8-model experimental candidate benchmark expander.

---

### Slide 11: Limitations & Safety Considerations
- **Non-Operational Status:** Educational/research model only. Not certified for flood warnings or public evacuation decisions.
- **Lead-Time Constraint:** 1-day ahead forecasts are reactive to upstream rainfall rather than proactive.
- **Gauge-Specific:** Trained specifically on USGS 01646500; transfer to other river basins requires retraining.

---

### Slide 12: Future Scope & Roadmap
- **Precipitation Radar Integration:** Ingest NOAA HRRR / MRMS watershed rainfall grids.
- **Physics-Informed Neural Networks / Hybrids:** Combine 1D hydrodynamic routing (HEC-RAS) with ML residual learning.
- **Probabilistic / Quantile Regression:** Output 10th-to-90th percentile prediction bands.

---

### Slide 13: Summary Checklist & Repository Resources
- **Automated Tests:** 21/21 pytest tests passing in 0.6s.
- **Dashboard Command:** `streamlit run app.py` (Localhost:8501).
- **Core Documentation:** `README.md`, `requirements.txt`, `reports/final_project_report.md`.

