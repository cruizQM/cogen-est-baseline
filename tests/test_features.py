"""Tests for feature engineering."""

import numpy as np
import pandas as pd

from cogen_est_baseline.models.features import FeatureConfig, build_features, get_feature_columns


def _make_sample_df(n=96):
    """Build a minimal DataFrame with DatetimeIndex and pmd column."""
    idx = pd.date_range("2025-01-15 00:00", periods=n, freq="15min", tz="Europe/Madrid")
    rng = np.random.default_rng(42)
    return pd.DataFrame({"pmd": rng.uniform(30, 80, n)}, index=idx)


class TestBuildFeatures:
    def test_output_columns_match_get_feature_columns(self):
        df = _make_sample_df()
        config = FeatureConfig()
        features = build_features(df, config)
        expected = get_feature_columns(config)
        assert list(features.columns) == expected

    def test_no_nans_in_core_features(self):
        df = _make_sample_df()
        features = build_features(df)
        # Core features (pmd, hour, month, etc.) should have no NaNs
        core = ["pmd", "hour", "month", "day_of_week", "quarter", "is_weekend"]
        assert features[core].notna().all().all()

    def test_cyclical_range(self):
        df = _make_sample_df(n=96 * 365)  # full year to cover all hours/months
        features = build_features(df)
        for col in ["hour_sin", "hour_cos", "month_sin", "month_cos"]:
            assert features[col].min() >= -1.0
            assert features[col].max() <= 1.0

    def test_is_weekend(self):
        # 2025-01-18 is a Saturday, 2025-01-19 is Sunday
        idx = pd.date_range("2025-01-17 12:00", periods=3, freq="D", tz="Europe/Madrid")
        df = pd.DataFrame({"pmd": [50, 50, 50]}, index=idx)
        features = build_features(df)
        # Fri=0, Sat=1, Sun=1
        assert features["is_weekend"].tolist() == [0, 1, 1]

    def test_no_cyclical(self):
        config = FeatureConfig(cyclical_encoding=False, pmd_rolling_windows=[])
        df = _make_sample_df()
        features = build_features(df, config)
        assert "hour_sin" not in features.columns

    def test_rolling_features(self):
        config = FeatureConfig(pmd_rolling_windows=[4])  # 1-hour window
        df = _make_sample_df(n=96)
        features = build_features(df, config)
        assert "pmd_rmean_4" in features.columns
        assert "pmd_rstd_4" in features.columns
