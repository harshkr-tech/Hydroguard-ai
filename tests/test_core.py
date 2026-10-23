"""
test_core.py -- Basic tests for HydroGuard AI.

Tests data loading, feature generation, evaluation, and prediction consistency.

Usage:
    .venv\\Scripts\\python.exe -m pytest tests/test_core.py -v
"""
import os
import sys
import numpy as np
import pandas as pd
import pytest

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

RAW_CSV = os.path.join(PROJECT_ROOT, "data", "raw", "streamflow.csv")
COMP_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "model_comparison.csv")


# ================================================================
# Test: Data loading
# ================================================================
class TestDataLoading:
    def test_raw_csv_exists(self):
        assert os.path.isfile(RAW_CSV), f"Raw CSV not found: {RAW_CSV}"

    def test_raw_csv_columns(self):
        df = pd.read_csv(RAW_CSV, nrows=5)
        assert "date" in df.columns
        assert "discharge_cfs" in df.columns

    def test_raw_csv_row_count(self):
        df = pd.read_csv(RAW_CSV)
        assert len(df) == 1826, f"Expected 1826 rows, got {len(df)}"

    def test_no_missing_discharge(self):
        df = pd.read_csv(RAW_CSV)
        assert df["discharge_cfs"].isna().sum() == 0

    def test_no_negative_discharge(self):
        df = pd.read_csv(RAW_CSV)
        assert (df["discharge_cfs"] < 0).sum() == 0

    def test_no_duplicate_dates(self):
        df = pd.read_csv(RAW_CSV, parse_dates=["date"])
        assert df["date"].duplicated().sum() == 0

    def test_chronological_order(self):
        df = pd.read_csv(RAW_CSV, parse_dates=["date"])
        df_sorted = df.sort_values("date")
        assert (df["date"].values == df_sorted["date"].values).all()


# ================================================================
# Test: Feature engineering
# ================================================================
class TestFeatureEngineering:
    @pytest.fixture
    def sample_df(self):
        """Create a small test dataframe."""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        discharge = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
                     1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
        return pd.DataFrame({"date": dates, "discharge_cfs": discharge})

    def test_lag_1_uses_previous_day(self, sample_df):
        d = sample_df.copy()
        d["lag_1"] = d["discharge_cfs"].shift(1)
        # lag_1 at index 1 should be value at index 0
        assert d["lag_1"].iloc[1] == d["discharge_cfs"].iloc[0]

    def test_lag_1_first_row_is_nan(self, sample_df):
        d = sample_df.copy()
        d["lag_1"] = d["discharge_cfs"].shift(1)
        assert pd.isna(d["lag_1"].iloc[0])

    def test_rolling_mean_7_excludes_current_day(self, sample_df):
        d = sample_df.copy()
        flow = d["discharge_cfs"]
        d["rolling_mean_7"] = flow.shift(1).rolling(7).mean()
        # At index 8, rolling_mean_7 should be mean of indices 1..7
        expected = np.mean([200, 300, 400, 500, 600, 700, 800])
        assert abs(d["rolling_mean_7"].iloc[8] - expected) < 0.01

    def test_no_target_leakage(self, sample_df):
        """Verify features at row i don't use discharge_cfs[i]."""
        d = sample_df.copy()
        flow = d["discharge_cfs"]
        d["lag_1"] = flow.shift(1)
        d["lag_7"] = flow.shift(7)
        d["rolling_mean_7"] = flow.shift(1).rolling(7).mean()

        # At index 10 (discharge=1100), no feature should contain 1100
        row = d.iloc[10]
        assert row["lag_1"] != 1100  # lag_1 = 1000
        assert row["lag_7"] != 1100  # lag_7 = 400


# ================================================================
# Test: Model comparison CSV
# ================================================================
class TestModelComparison:
    @pytest.fixture
    def comp_df(self):
        if not os.path.isfile(COMP_CSV):
            pytest.skip("model_comparison.csv not found")
        return pd.read_csv(COMP_CSV, parse_dates=["date"])

    def test_expected_columns(self, comp_df):
        expected = {"date", "actual", "naive_predicted", "rf_predicted"}
        assert expected.issubset(set(comp_df.columns))

    def test_no_missing_values(self, comp_df):
        assert comp_df.isna().sum().sum() == 0

    def test_no_duplicate_dates(self, comp_df):
        assert comp_df["date"].duplicated().sum() == 0

    def test_positive_actuals(self, comp_df):
        assert (comp_df["actual"] > 0).all()

    def test_positive_predictions(self, comp_df):
        assert (comp_df["rf_predicted"] > 0).all()
        assert (comp_df["naive_predicted"] > 0).all()

    def test_naive_is_lagged_actual(self, comp_df):
        """Naive prediction should be previous day actual (shift by 1)."""
        # The naive_predicted at row i should equal actual at row i-1
        # (since naive = lag_1 = yesterday's discharge)
        for i in range(1, min(10, len(comp_df))):
            assert comp_df["naive_predicted"].iloc[i] == comp_df["actual"].iloc[i - 1]

    def test_rf_beats_naive_mae(self, comp_df):
        y = comp_df["actual"].values
        naive = comp_df["naive_predicted"].values
        rf = comp_df["rf_predicted"].values
        naive_mae = np.mean(np.abs(y - naive))
        rf_mae = np.mean(np.abs(y - rf))
        assert rf_mae < naive_mae, f"RF MAE ({rf_mae:.1f}) should be less than Naive MAE ({naive_mae:.1f})"


# ================================================================
# Test: Evaluation metrics
# ================================================================
class TestMetrics:
    def test_mae_known_values(self):
        actual = np.array([10, 20, 30])
        predicted = np.array([12, 18, 33])
        mae = np.mean(np.abs(actual - predicted))
        assert abs(mae - 2.333) < 0.01

    def test_rmse_known_values(self):
        actual = np.array([10, 20, 30])
        predicted = np.array([12, 18, 33])
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        assert abs(rmse - 2.380) < 0.01

    def test_perfect_predictions(self):
        actual = np.array([100, 200, 300])
        mae = np.mean(np.abs(actual - actual))
        rmse = np.sqrt(np.mean((actual - actual) ** 2))
        assert mae == 0.0
        assert rmse == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

