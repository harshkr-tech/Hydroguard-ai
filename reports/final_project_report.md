# HydroGuard AI — Final Project Report

**Project Title:** HydroGuard AI: Daily River Discharge Forecasting & Machine Learning Peak-Flow Audit  
**Station:** USGS 01646500 (Potomac River at Little Falls, near Washington, D.C.)  
**Target Variable:** Daily Mean Discharge (`discharge_cfs`)  
**Evaluation Window:** 2024-12-31 to 2025-12-31 (366 days)  
**Primary Frameworks:** Python, Scikit-Learn, Pandas, NumPy, Streamlit  

---

## 1. Problem Statement
River discharge forecasting is essential for water supply management, civil infrastructure protection, and flood hazard mitigation. However, traditional hydrological modeling often requires extensive basin parameters, soil moisture grids, and upstream precipitation data that may be unavailable in data-scarce settings. 

When machine learning models (such as decision tree ensembles) are applied to river flow time-series, two critical challenges arise:
1. **Autoregressive Delay & Surge Lag:** Univariate streamflow models predict tomorrow's discharge primarily based on recent observations. When sudden storm surges or flash floods occur, the river gauge itself shows high water only *after* the precipitation has reached the river.
2. **Physical & Algorithmic Extrapolation Boundary:** Decision trees partition feature space into orthogonal regions and predict the leaf-node mean. Consequently, a tree-based ensemble mathematically cannot predict a discharge value higher than the maximum target observed during training, leading to severe underprediction during unprecedented historical floods.

---

## 2. Project Objectives
1. **Automate Data Ingestion:** Download and validate 5 years (2021–2025) of daily mean streamflow data from the official USGS Water Data OGC API for Station `01646500`.
2. **Establish a Reproducible Baseline:** Build a naive persistence benchmark (predict tomorrow = today) evaluated strictly on the final 20% chronological test period without data leakage.
3. **Train Machine Learning Models:** Engineer leakage-free autoregressive lag features (`lag_1`, `lag_2`, `lag_3`, `lag_7`) and rolling statistics (`rolling_mean_7`), and train a Random Forest Regressor.
4. **Conduct Peak-Improvement Experiments:** Systematically evaluate alternative feature sets, progressive sample weighting, log-transforms, and Gradient Boosting on high-flow and out-of-distribution events.
5. **Analyze Extrapolation Limits:** Quantify model performance during extreme floods exceeding the historical training record ($> 102,000$ cfs).
6. **Deploy an Interactive Dashboard:** Build a robust Streamlit dashboard presenting historical observations, model comparison metrics, high-flow analytics, and explicit flood warning disclaimers.

---

## 3. Dataset Description & Audit
- **Data Source:** USGS Water Data REST API / OGC API collection for Daily Streamflow.
- **Monitoring Station:** USGS 01646500 (Potomac River at Little Falls, DC/MD).
  - Drainage Area: $\approx 11,560\text{ sq mi}$
  - Latitude: `38.949778° N`, Longitude: `-77.127639° W`
- **Parameter & Statistic:** Parameter Code `00060` (Discharge), Statistic `00003` (Mean daily flow).
- **Timeframe:** `2021-01-01` to `2025-12-31` (5 continuous calendar years).
- **Quality & Integrity Metrics:**
  - Total records: **1,826 days** (zero missing days in calendar sequence).
  - Missing values: **0** across all 10 columns.
  - Negative values: **0** (Minimum discharge: $450\text{ cfs}$; Maximum: $136,000\text{ cfs}$; Mean: $8,537.9\text{ cfs}$).
  - Units of measure: strictly consistent `ft^3/s` (cubic feet per second).

---

## 4. Methodology

### 4.1 Chronological Split (No Shuffling)
Time-series data violates the independence assumption of random k-fold cross-validation. To prevent future information leakage:
- **Training Period:** `2021-01-01` to `2024-12-30` (1,460 calendar days, 80% split). Usable training rows: 1,453 (after dropping initial lag NaNs).
- **Testing Period:** `2024-12-31` to `2025-12-31` (366 calendar days, 20% split).
- **Historical Maximums:**
  - Training Period Maximum: **$102,000\text{ cfs}$** (recorded on 2022-05-09).
  - Testing Period Maximum: **$136,000\text{ cfs}$** (recorded on 2025-05-15).

### 4.2 Leakage-Free Feature Engineering
All features are constructed using strictly past observations:
- $\text{lag}_1 = y_{t-1}$ (yesterday's flow)
- $\text{lag}_2 = y_{t-2}$
- $\text{lag}_3 = y_{t-3}$
- $\text{lag}_7 = y_{t-7}$
- $\text{rolling\_mean}_7 = \frac{1}{7} \sum_{i=1}^{7} y_{t-i}$ (shifted by 1 so day $t$ is excluded).
- Extended features (for experimentation): $\text{lag}_{14}$, $\text{rolling\_max}_7$, $\text{rolling\_std}_7$, $\Delta\text{flow}_1 = y_{t-1} - y_{t-2}$, $\text{rolling\_max}_{14}$, $\text{rolling\_max}_{30}$.

### 4.3 Evaluation Metrics
- **Mean Absolute Error (MAE):** $\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$
- **Root Mean Squared Error (RMSE):** $\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$
- **Coefficient of Determination ($R^2$):** $R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$
- **Mean Prediction Bias:** $\text{Bias} = \frac{1}{n} \sum_{i=1}^n (\hat{y}_i - y_i)$

---

## 5. System Architecture

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

---

## 6. Experimental Results & Benchmark Comparison

All models were evaluated on the exact same 366 test days (`2024-12-31` to `2025-12-31`):

| Model | Overall MAE (cfs) | Overall RMSE (cfs) | Overall $R^2$ | Peak MAE (cfs) ($\ge 17,100$) | Peak RMSE (cfs) | Peak Bias (cfs) | OOD MAE (cfs) ($> 102,000$) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Naive Baseline (persistence)** | 1,805.3 | 7,237.9 | 0.6586 | 11,842.6 | 21,964.6 | -1,191.3 | — |
| **`RF_base_5feat` (Production)** | **1,518.0** | **6,513.3** | **0.7235** | 9,386.6 | **19,682.7** | -4,393.2 | **75,406.7** |
| **`RF_ext_11feat`** | 1,560.5 | 6,796.0 | 0.6990 | 9,602.6 | 20,538.6 | -5,023.8 | 81,202.0 |
| **`RF_weighted_4x`** | 1,722.6 | 7,013.4 | 0.6794 | **9,384.1** | 20,757.9 | -4,338.4 | 82,950.8 |
| **`RF_log1p`** | 1,552.2 | 7,120.1 | 0.6696 | 10,079.7 | 21,613.4 | -6,485.0 | 86,379.7 |
| **`RF_log1p_weighted`** | 1,551.6 | 6,948.4 | 0.6853 | 9,941.0 | 21,063.3 | -5,140.8 | 84,897.8 |
| **`GBR` (Gradient Boosting)** | 1,661.5 | 6,902.5 | 0.6895 | 9,817.5 | 20,789.6 | -4,800.9 | 83,421.7 |
| **`GBR_log1p_weighted`** | 1,618.3 | 7,093.7 | 0.6720 | 9,683.8 | 21,449.0 | -6,299.4 | 88,189.8 |

### Key Findings:
1. **Random Forest vs. Baseline:** `RF_base_5feat` reduces overall MAE by **$287.3\text{ cfs}$ ($-15.9\%$)** and RMSE by **$724.6\text{ cfs}$ ($-10.0\%$)** relative to naive persistence.
2. **Sample Weighting Trade-off:** While 4x sample weighting nominally reduced peak MAE by 2.5 cfs (from 9,386.6 to 9,384.1 cfs), it increased overall MAE by **$+204.6\text{ cfs}$ ($+13.5\%$)** and worsened peak RMSE (20,758 vs 19,683 cfs).
3. **Feature Importance:** `lag_1` dominates decision tree splits ($> 90\%$ importance), demonstrating the strong persistence characteristic of daily river discharge.

---

## 7. Model Limitations & Extrapolation Boundary
1. **Mathematical Upper Bound:** An unconstrained Random Forest or Gradient Boosting Regressor predicts:
   $$\hat{y}(\mathbf{x}) \le \max_{i \in \text{train}} y_i$$
   Because the training maximum was $102,000\text{ cfs}$, the model cannot predict $136,000\text{ cfs}$. On 2025-05-15, the actual flow was $136,000\text{ cfs}$ and the model predicted $33,018\text{ cfs}$ (error: $-102,982\text{ cfs}$).
2. **Missing Precipitation Inputs:** Without meteorological rainfall radar or upstream tributary gauges, an autoregressive model has no advance warning of rainfall until the surge reaches the station.
3. **Safety Notice:** HydroGuard AI is an educational and exploratory ML project. It is **not** certified as an operational flood-warning system.

---

## 8. Future Scope
1. **Multivariate Meteorological Integration:** Incorporate NOAA watershed precipitation grids (QPE / HRRR) and upstream gauge observations.
2. **Hybrid Physics-Informed ML:** Couple a 1D hydraulic routing model with machine learning error correction.
3. **Extreme Value Theory (EVT):** Use generalized Pareto distributions for tail extrapolation beyond historical training bounds.
4. **Quantile Regression:** Predict 10th, 50th, and 90th percentile bounds rather than point forecasts to quantify surge uncertainty.

