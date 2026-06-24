"""Tests for stitching logic: 634 cutoff and 2197 backfill."""

import numpy as np
import pandas as pd
import pytest

from cogen_est_baseline.data.indicators import REGIME_634_START, REGIME_2197_START
from cogen_est_baseline.data.stitching import apply_634_cutoff, backfill_2197

TZ = "Europe/Madrid"


def _make_df(dates, values, ind_id=0):
    """Helper: build a minimal indicator DataFrame."""
    return pd.DataFrame(
        {
            "datetime": pd.to_datetime(dates, utc=True).tz_convert(TZ),
            "value": values,
            "id": float(ind_id),
        }
    )


# ── 634 cutoff ────────────────────────────────────────────────────────────────


class TestApply634Cutoff:
    def test_drops_rows_before_regime_date(self):
        dates = pd.date_range("2024-11-18", periods=5, freq="D", tz=TZ)
        df = _make_df(dates, [1, 2, 3, 4, 5], ind_id=634)
        result = apply_634_cutoff(df)
        # 18, 19 Nov dropped; 20, 21, 22 Nov kept
        assert len(result) == 3
        assert result["value"].tolist() == [3, 4, 5]

    def test_keeps_all_if_already_after_regime(self):
        dates = pd.date_range("2025-01-01", periods=3, freq="D", tz=TZ)
        df = _make_df(dates, [10, 20, 30], ind_id=634)
        result = apply_634_cutoff(df)
        assert len(result) == 3

    def test_empty_if_all_before_regime(self):
        dates = pd.date_range("2024-01-01", periods=3, freq="D", tz=TZ)
        df = _make_df(dates, [1, 2, 3], ind_id=634)
        result = apply_634_cutoff(df)
        assert len(result) == 0


# ── 2197 backfill ─────────────────────────────────────────────────────────────


def _make_backfill_inputs(
    dates, vol_values, up_values, down_values, native_2197_dates=None, native_2197_values=None
):
    """Helper: build all four DataFrames needed for backfill_2197."""
    df_10250 = _make_df(dates, vol_values, ind_id=10250)
    df_677 = _make_df(dates, up_values, ind_id=677)
    df_676 = _make_df(dates, down_values, ind_id=676)

    if native_2197_dates is not None:
        df_2197 = _make_df(native_2197_dates, native_2197_values, ind_id=2197)
    else:
        df_2197 = _make_df([], [], ind_id=2197)

    return df_676, df_677, df_2197, df_10250


class TestBackfill2197:
    def test_positive_volume_uses_677_up(self):
        """When 10250 > 0, backfilled 2197 should equal 677 (up price)."""
        dates = pd.date_range("2024-12-01", periods=3, freq="15min", tz=TZ)
        df_676, df_677, df_2197, df_10250 = _make_backfill_inputs(
            dates,
            vol_values=[100, 200, 50],
            up_values=[80, 90, 85],
            down_values=[40, 45, 42],
        )
        result = backfill_2197(df_676, df_677, df_2197, df_10250)
        assert result["value"].tolist() == [80, 90, 85]

    def test_negative_volume_uses_676_down(self):
        """When 10250 < 0, backfilled 2197 should equal 676 (down price)."""
        dates = pd.date_range("2024-12-01", periods=3, freq="15min", tz=TZ)
        df_676, df_677, df_2197, df_10250 = _make_backfill_inputs(
            dates,
            vol_values=[-100, -200, -50],
            up_values=[80, 90, 85],
            down_values=[40, 45, 42],
        )
        result = backfill_2197(df_676, df_677, df_2197, df_10250)
        assert result["value"].tolist() == [40, 45, 42]

    def test_zero_volume_yields_nan(self):
        """When 10250 == 0, backfilled 2197 should be NaN."""
        dates = pd.date_range("2024-12-01", periods=3, freq="15min", tz=TZ)
        df_676, df_677, df_2197, df_10250 = _make_backfill_inputs(
            dates,
            vol_values=[0, 100, 0],
            up_values=[80, 90, 85],
            down_values=[40, 45, 42],
        )
        result = backfill_2197(df_676, df_677, df_2197, df_10250)
        assert np.isnan(result["value"].iloc[0])
        assert result["value"].iloc[1] == 90  # positive → up
        assert np.isnan(result["value"].iloc[2])

    def test_nan_volume_yields_nan(self):
        """When 10250 is NaN, backfilled 2197 should be NaN."""
        dates = pd.date_range("2024-12-01", periods=2, freq="15min", tz=TZ)
        df_676, df_677, df_2197, df_10250 = _make_backfill_inputs(
            dates,
            vol_values=[np.nan, 100],
            up_values=[80, 90],
            down_values=[40, 45],
        )
        result = backfill_2197(df_676, df_677, df_2197, df_10250)
        assert np.isnan(result["value"].iloc[0])
        assert result["value"].iloc[1] == 90

    def test_mixed_directions(self):
        """Interleaved positive/negative/zero volumes."""
        dates = pd.date_range("2024-12-01", periods=5, freq="15min", tz=TZ)
        df_676, df_677, df_2197, df_10250 = _make_backfill_inputs(
            dates,
            vol_values=[100, -50, 0, 200, -10],
            up_values=[80, 81, 82, 83, 84],
            down_values=[40, 41, 42, 43, 44],
        )
        result = backfill_2197(df_676, df_677, df_2197, df_10250)
        expected = [80, 41, np.nan, 83, 44]
        for i, exp in enumerate(expected):
            if np.isnan(exp):
                assert np.isnan(result["value"].iloc[i])
            else:
                assert result["value"].iloc[i] == exp

    def test_native_post_regime_preserved(self):
        """Native 2197 values after the regime date are kept as-is."""
        # Pre-regime dates
        pre_dates = pd.date_range("2024-12-09", periods=2, freq="15min", tz=TZ)
        # Post-regime dates
        post_dates = pd.date_range("2024-12-10 12:00", periods=2, freq="15min", tz=TZ)

        df_676, df_677, df_2197, df_10250 = _make_backfill_inputs(
            pre_dates,
            vol_values=[100, -50],
            up_values=[80, 81],
            down_values=[40, 41],
            native_2197_dates=post_dates,
            native_2197_values=[99.0, 101.0],
        )
        result = backfill_2197(df_676, df_677, df_2197, df_10250)
        # Should have 4 rows: 2 backfilled + 2 native
        assert len(result) == 4
        # Pre-regime: 100 > 0 → 80, -50 < 0 → 41
        assert result["value"].iloc[0] == 80
        assert result["value"].iloc[1] == 41
        # Post-regime: native values preserved
        assert result["value"].iloc[2] == 99.0
        assert result["value"].iloc[3] == 101.0

    def test_all_ids_are_2197(self):
        """Output id column should always be 2197."""
        dates = pd.date_range("2024-12-01", periods=2, freq="15min", tz=TZ)
        df_676, df_677, df_2197, df_10250 = _make_backfill_inputs(
            dates,
            vol_values=[100, -50],
            up_values=[80, 81],
            down_values=[40, 41],
        )
        result = backfill_2197(df_676, df_677, df_2197, df_10250)
        assert (result["id"] == 2197.0).all()
