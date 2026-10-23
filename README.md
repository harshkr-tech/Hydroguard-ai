# HydroGuard AI 🌊

[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.65-red.svg)](https://streamlit.io/)
[![pytest](https://img.shields.io/badge/pytest-21%20passed-brightgreen.svg)](https://pytest.org/)

Machine learning system for daily river discharge prediction using USGS historical streamflow data, autoregressive lag features, and strict chronological time-series evaluation.

---

## 1. Project Overview
- **Station:** Potomac River at Little Falls, near Washington, D.C. (`USGS-01646500`)
- **Parameter:** Discharge (Streamflow), Daily Mean (`00060`, Statistic `00003`)
- **Units:** cfs (cubic feet per second)
- **Dataset Period:** `2021-01-01` to `2025-12-31` (1,826 daily observations, zero missing values)
- **Chronological Split:**
  - Training Period: `2021-01-01` to `2024-12-30` (1,460 calendar days, 80%)
  - Testing Period: `2024-12-31` to `2025-12-31` (366 calendar days, 20%)

---

## 2. Project Architecture & Pipeline

```text
[USGS Water Data OGC API]
            │
            ▼ (src/download_data.py)
   [data/raw/streamflow.csv]
            │
            ├─────────────────────────────────────────┐
            ▼ (src/train_baseline.py)                 ▼ (src/train_model.py)
 [Naive Persistence Baseline]                 [Feature Engineering: Lags & Rolling]
  (Predicted_t = Actual_t-1)                          │
            │                                         ▼ (RandomForestRegressor)
            │                                  [models/rf_model.joblib]
            │                                         │
            └────────────────────┬────────────────────┘
                                 ▼
                 [data/processed/model_comparison.csv]
                                 │
            ┌────────────────────┼────────────────────┐
            ▼                    ▼                    ▼
   [tests/test_core.py]   [reports/*.png]       [app.py (Streamlit)]
   - 21 automated unit    - Comparison charts   - Interactive slider
     and schema tests     - Peak flow zoom      - High-flow analytics
                          - Residual plots      - Extrapolation warnings
```

### Directory Structure
```text
hydroguard-ai/
├── app.py                             # Interactive Streamlit dashboard
├── requirements.txt                   # Pinned project dependencies
├── README.md                          # Comprehensive project documentation
├── data/
│   ├── raw/
│   │   ├── streamflow.csv             # Verified USGS daily streamflow observations
│   │   └── metadata.json              # USGS station and parameter metadata
│   └── processed/
│       ├── model_comparison.csv       # Standard actual vs naive vs RF test results
│       ├── baseline_predictions.csv   # Naive persistence baseline test predictions
│       └── experiment_*.csv           # Multi-model peak improvement experiment data
├── models/
│   ├── rf_model.joblib                # Production Random Forest model artifact
│   └── hydroguard_rf_*.joblib         # Timestamped model artifacts
├── reports/
│   ├── final_project_report.md        # Comprehensive technical report
│   ├── presentation_outline.md        # Presentation slide-by-slide outline
│   ├── experiment_report.txt          # Multi-model experiment benchmark summary
│   ├── model_evaluation.txt           # Model verification report
│   └── *.png                          # Evaluation, zoom, and residual charts
├── src/
│   ├── download_data.py               # USGS OGC API data downloader
│   ├── train_baseline.py              # Naive persistence baseline benchmark
│   ├── train_model.py                 # Production Random Forest trainer
│   ├── experiment_peak_improvement.py # Multi-model peak improvement experiment
│   ├── verify_results.py              # Automated results and metrics verifier
│   └── generate_plots.py              # Evaluation and residual plot generator
└── tests/
    └── test_core.py                   # Automated test suite (21 unit tests)
```

---

## 3. Installation & Environment Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.14.3)
- Windows PowerShell, macOS Terminal, or Linux Shell

### Setup Steps
```powershell
# 1. Clone repository and navigate to root
cd hydroguard-ai

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On macOS / Linux:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

---

## 4. Execution Commands

### Run Automated Unit Tests (21 Tests)
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_core.py -v
```

### Train Baseline & Production Model
```powershell
# Run naive persistence baseline benchmark
.\.venv\Scripts\python.exe src\train_baseline.py

# Train production Random Forest model
.\.venv\Scripts\python.exe src\train_model.py

# Verify results and schema integrity
.\.venv\Scripts\python.exe src\verify_results.py
```

### Run Peak-Flow Improvement Experiments (8 Candidates)
```powershell
.\.venv\Scripts\python.exe src\experiment_peak_improvement.py
```

### Launch Interactive Streamlit Dashboard
```powershell
.\.venv\Scripts\streamlit.exe run app.py
```
Open **http://localhost:8501** in your web browser.

---

## 5. Methodology

1. **Chronological Splitting (No Random Shuffling):**
   - Time-series observations violate sample independence. An 80/20 chronological split is strictly enforced: `2021-01-01` to `2024-12-30` for training (1,460 days) and `2024-12-31` to `2025-12-31` for testing (366 days).
2. **Leakage-Free Autoregressive Features:**
   - Features use strictly earlier observations:
     - $\text{lag}_1 = y_{t-1}$ (yesterday's flow)
     - $\text{lag}_2 = y_{t-2}$, $\text{lag}_3 = y_{t-3}$, $\text{lag}_7 = y_{t-7}$
     - $\text{rolling\_mean}_7$: 7-day rolling average shifted by 1 day so day $t$ is excluded.
3. **Reproducible Baselines:**
   - Every machine learning candidate is evaluated against the naive persistence baseline ($\hat{y}_t = y_{t-1}$) on the exact same 366 test dates.
4. **Multi-Tier Peak Evaluation:**
   - Evaluates overall performance, high-flow days ($\ge 90\text{th percentile}$, $17,080\text{ cfs}$), and out-of-distribution flood events ($> 102,000\text{ cfs}$).

---

## 6. Results Summary

All metrics evaluated on the 366 identical test days (`2024-12-31` to `2025-12-31`):

| Model Candidate | Overall MAE (cfs) | Overall RMSE (cfs) | Overall $R^2$ | Peak MAE (cfs) ($\ge 17,100$) | Peak RMSE (cfs) | Peak Bias (cfs) | OOD MAE (cfs) ($> 102,000$) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Naive Baseline (persistence)** | 1,805.3 | 7,237.9 | 0.6586 | 11,842.6 | 21,964.6 | -1,191.3 | — |
| **`RF_base_5feat` (Production)** | **1,518.0** | **6,513.3** | **0.7235** | 9,386.6 | **19,682.7** | -4,393.2 | **75,406.7** |
| **`RF_ext_11feat`** | 1,560.5 | 6,796.0 | 0.6990 | 9,602.6 | 20,538.6 | -5,023.8 | 81,202.0 |
| **`RF_weighted_4x`** | 1,722.6 | 7,013.4 | 0.6794 | **9,384.1** | 20,757.9 | -4,338.4 | 82,950.8 |
| **`RF_log1p`** | 1,552.2 | 7,120.1 | 0.6696 | 10,079.7 | 21,613.4 | -6,485.0 | 86,379.7 |
| **`RF_log1p_weighted`** | 1,551.6 | 6,948.4 | 0.6853 | 9,941.0 | 21,063.3 | -5,140.8 | 84,897.8 |
| **`GBR` (Gradient Boosting)** | 1,661.5 | 6,902.5 | 0.6895 | 9,817.5 | 20,789.6 | -4,800.9 | 83,421.7 |
| **`GBR_log1p_weighted`** | 1,618.3 | 7,093.7 | 0.6720 | 9,683.8 | 21,449.0 | -6,299.4 | 88,189.8 |

---

## 7. Limitations & Mathematical Extrapolation Boundary

1. **Extrapolation Limit of Decision Tree Models:**
   - Decision tree ensembles predict the mean of samples in leaf nodes:
     $$\hat{y}(\mathbf{x}) \le \max_{i \in \text{train}} y_i$$
   - In our training data (2021–2024), the historical maximum was **102,000 cfs** (recorded on May 9, 2022).
   - In the test period, a record flood occurred on May 15, 2025 reaching **136,000 cfs**.
   - Tree models mathematically cannot predict 136,000 cfs, causing predictions to plateau and underestimate unprecedented surges.
2. **Missing Precipitation Predictors:**
   - River discharge is physically driven by precipitation across the upstream watershed ($\approx 11,560\text{ sq mi}$). A univariate model relying only on gauge flow cannot detect a sudden storm until the flood wave reaches the station.
3. **Safety Notice & Disclaimer:**
   - HydroGuard AI is an educational and exploratory research tool. It is **not an operational flood-warning system** and must not be used for public safety or evacuation decisions.

---

## 8. License & Acknowledgements
- **Data Source:** USGS Water Data API (U.S. Geological Survey).
- **License:** MIT License.

1. **Chronological Splitting (No Random Shuffling):**
   - Time-series observations violate sample independence. An 80/20 chronological split is strictly enforced: `2021-01-01` to `2024-12-30` for training (1,460 days) and `2024-12-31` to `2025-12-31` for testing (366 days).
2. **Leakage-Free Autoregressive Features:**
   - Features use strictly earlier observations:
     - $\text{lag}_1 = y_{t-1}$ (yesterday's flow)
     - $\text{lag}_2 = y_{t-2}$, $\text{lag}_3 = y_{t-3}$, $\text{lag}_7 = y_{t-7}$
     - $\text{rolling\_mean}_7$: 7-day rolling average shifted by 1 day so day $t$ is excluded.
3. **Reproducible Baselines:**
   - Every machine learning candidate is evaluated against the naive persistence baseline ($\hat{y}_t = y_{t-1}$) on the exact same 366 test dates.
4. **Multi-Tier Peak Evaluation:**
   - Evaluates overall performance, high-flow days ($\ge 90\text{th percentile}$, $17,080\text{ cfs}$), and out-of-distribution flood events ($> 102,000\text{ cfs}$).

---

## 6. Results Summary

All metrics evaluated on the 366 identical test days (`2024-12-31` to `2025-12-31`):

| Model Candidate | Overall MAE (cfs) | Overall RMSE (cfs) | Overall $R^2$ | Peak MAE (cfs) ($\ge 17,100$) | Peak RMSE (cfs) | Peak Bias (cfs) | OOD MAE (cfs) ($> 102,000$) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Naive Baseline (persistence)** | 1,805.3 | 7,237.9 | 0.6586 | 11,842.6 | 21,964.6 | -1,191.3 | — |
| **`RF_base_5feat` (Production)** | **1,518.0** | **6,513.3** | **0.7235** | 9,386.6 | **19,682.7** | -4,393.2 | **75,406.7** |
| **`RF_ext_11feat`** | 1,560.5 | 6,796.0 | 0.6990 | 9,602.6 | 20,538.6 | -5,023.8 | 81,202.0 |
| **`RF_weighted_4x`** | 1,722.6 | 7,013.4 | 0.6794 | **9,384.1** | 20,757.9 | -4,338.4 | 82,950.8 |
| **`RF_log1p`** | 1,552.2 | 7,120.1 | 0.6696 | 10,079.7 | 21,613.4 | -6,485.0 | 86,379.7 |
| **`RF_log1p_weighted`** | 1,551.6 | 6,948.4 | 0.6853 | 9,941.0 | 21,063.3 | -5,140.8 | 84,897.8 |
| **`GBR` (Gradient Boosting)** | 1,661.5 | 6,902.5 | 0.6895 | 9,817.5 | 20,789.6 | -4,800.9 | 83,421.7 |
| **`GBR_log1p_weighted`** | 1,618.3 | 7,093.7 | 0.6720 | 9,683.8 | 21,449.0 | -6,299.4 | 88,189.8 |

---

## 7. Limitations & Mathematical Extrapolation Boundary

1. **Extrapolation Limit of Decision Tree Models:**
   - Decision tree ensembles predict the mean of samples in leaf nodes:
     $$\hat{y}(\mathbf{x}) \le \max_{i \in \text{train}} y_i$$
   - In our training data (2021–2024), the historical maximum was **102,000 cfs** (recorded on May 9, 2022).
   - In the test period, a record flood occurred on May 15, 2025 reaching **136,000 cfs**.
   - Tree models mathematically cannot predict 136,000 cfs, causing predictions to plateau and underestimate unprecedented surges.
2. **Missing Precipitation Predictors:**
   - River discharge is physically driven by precipitation across the upstream watershed ($\approx 11,560\text{ sq mi}$). A univariate model relying only on gauge flow cannot detect a sudden storm until the flood wave reaches the station.
3. **Safety Notice & Disclaimer:**
   - HydroGuard AI is an educational and exploratory research tool. It is **not an operational flood-warning system** and must not be used for public safety or evacuation decisions.

---

## 8. License & Acknowledgements
- **Data Source:** USGS Water Data API (U.S. Geological Survey).
- **License:** MIT License.

