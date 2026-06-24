"""Tests for PMD degradation."""

import numpy as np
import pandas as pd
import pytest

from cogen_est_baseline.models.pmd_degradation import (
    PmdDegradationConfig,
    PmdDegradationMode,
    degrade_pmd,
)


def _make_df(n=2880, seed=42):
    """Build a DataFrame with volatile PMD over ~30 days."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="15min", tz="Europe/Madrid")
    # Sine wave (daily pattern) + noise
    hours = np.arange(n) / 4
    pmd = 50 + 20 * np.sin(2 * np.pi * hours / 24) + rng.normal(0, 10, n)
    return pd.DataFrame({"pmd": pmd, "other_col": 1.0}, index=idx)


class TestDegradePmd:
    def test_none_returns_same(self):
        df = _make_df()
        cfg = PmdDegradationConfig(mode=PmdDegradationMode.NONE)
        result = degrade_pmd(df, cfg)
        pd.testing.assert_frame_equal(result, df)

    def test_rolling_weekly_is_smoother(self):
        df = _make_df()
        cfg = PmdDegradationConfig(mode=PmdDegradationMode.ROLLING_WEEKLY)
        result = degrade_pmd(df, cfg)
        # Smoothed PMD should have lower std than original
        assert result["pmd"].std() < df["pmd"].std()

    def test_rolling_monthly_is_even_smoother(self):
        df = _make_df(n=5760)  # ~60 days to have enough data
        weekly = degrade_pmd(
            df, PmdDegradationConfig(mode=PmdDegradationMode.ROLLING_WEEKLY)
        )
        monthly = degrade_pmd(
            df, PmdDegradationConfig(mode=PmdDegradationMode.ROLLING_MONTHLY)
        )
        assert monthly["pmd"].std() < weekly["pmd"].std()

    def test_noisy_adds_some_variance_back(self):
        df = _make_df()
        weekly = degrade_pmd(
            df, PmdDegradationConfig(mode=PmdDegradationMode.ROLLING_WEEKLY)
        )
        noisy = degrade_pmd(
            df, PmdDegradationConfig(
                mode=PmdDegradationMode.ROLLING_WEEKLY_NOISY,
                noise_fraction=0.5,
            )
        )
        # Noisy should have more variance than pure weekly smooth
        assert noisy["pmd"].std() > weekly["pmd"].std()
        # But less than the original
        assert noisy["pmd"].std() < df["pmd"].std()

    def test_does_not_modify_other_columns(self):
        df = _make_df()
        cfg = PmdDegradationConfig(mode=PmdDegradationMode.ROLLING_WEEKLY)
        result = degrade_pmd(df, cfg)
        pd.testing.assert_series_equal(result["other_col"], df["other_col"])

    def test_returns_copy(self):
        df = _make_df()
        cfg = PmdDegradationConfig(mode=PmdDegradationMode.ROLLING_WEEKLY)
        result = degrade_pmd(df, cfg)
        # Modifying result should not affect original
        result["pmd"].iloc[0] = -999
        assert df["pmd"].iloc[0] != -999

    def test_reproducible_noise(self):
        df = _make_df()
        cfg = PmdDegradationConfig(
            mode=PmdDegradationMode.ROLLING_WEEKLY_NOISY,
            noise_seed=42,
        )
        r1 = degrade_pmd(df, cfg)
        r2 = degrade_pmd(df, cfg)
        pd.testing.assert_series_equal(r1["pmd"], r2["pmd"])
