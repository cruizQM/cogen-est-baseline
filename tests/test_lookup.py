"""Tests for P50 lookup table construction and prediction."""

import numpy as np
import pandas as pd

from cogen_est_baseline.baseline.lookup import build_lookup_table, lookup_predict
from cogen_est_baseline.config import BaselineConfig, BandConfig


def _make_historical(n=200, seed=42):
    """Generate synthetic historical data for testing."""
    rng = np.random.default_rng(seed)
    hours = np.tile(np.arange(24), n // 24 + 1)[:n]
    pmd = rng.uniform(20, 80, size=n)
    # Target roughly = PMD * 0.5 + hour-dependent offset + noise
    target = pmd * 0.5 + hours * 0.3 + rng.normal(0, 2, size=n)
    return pd.DataFrame({"pmd": pmd, "value": target, "hour": hours})


class TestBuildLookupTable:
    def test_output_shape(self):
        df = _make_historical()
        cfg = BaselineConfig(
            band=BandConfig(band_width=20, band_min=20, band_max=80),
            quantiles=[0.1, 0.5, 0.9],
        )
        table = build_lookup_table(df, config=cfg)
        # Should have columns for each quantile
        assert set(table.columns) == {"q0.1", "q0.5", "q0.9"}
        # Index should be (hour, pmd_band)
        assert table.index.names == ["hour", "pmd_band"]

    def test_p50_is_median(self):
        """P50 should match pandas median for a single (hour, band) group."""
        df = pd.DataFrame({
            "pmd": [31, 32, 34, 31, 33],  # all in (30, 40] band
            "value": [10, 20, 30, 40, 50],
            "hour": [0, 0, 0, 0, 0],
        })
        cfg = BaselineConfig(
            band=BandConfig(band_width=10, band_min=30, band_max=40),
            quantiles=[0.5],
        )
        table = build_lookup_table(df, config=cfg)
        assert table["q0.5"].iloc[0] == 30.0  # median of [10,20,30,40,50]

    def test_spread_mode(self):
        """With use_spread=True, lookup target should be (value - pmd)."""
        df = pd.DataFrame({
            "pmd": [50, 50, 50],
            "value": [60, 70, 80],  # spreads: 10, 20, 30
            "hour": [0, 0, 0],
        })
        cfg = BaselineConfig(
            band=BandConfig(band_width=10, band_min=45, band_max=55),
            quantiles=[0.5],
        )
        table = build_lookup_table(df, config=cfg, use_spread=True)
        assert table["q0.5"].iloc[0] == 20.0  # median of [10, 20, 30]


class TestLookupPredict:
    def test_round_trip(self):
        """Prediction on training data should use the correct lookup values."""
        # Single hour, single band, known median
        hist = pd.DataFrame({
            "pmd": [50, 50, 50],
            "value": [10, 20, 30],
            "hour": [0, 0, 0],
        })
        cfg = BaselineConfig(
            band=BandConfig(band_width=10, band_min=45, band_max=55),
            quantiles=[0.5],
        )
        table = build_lookup_table(hist, config=cfg)

        forecast = pd.DataFrame({"pmd": [50], "hour": [0]})
        pred = lookup_predict(table, forecast, config=cfg)
        assert pred.iloc[0] == 20.0  # median of [10, 20, 30]

    def test_spread_round_trip(self):
        """Spread mode: prediction = looked-up spread + forecasted PMD."""
        hist = pd.DataFrame({
            "pmd": [50, 50, 50],
            "value": [60, 70, 80],  # spreads: 10, 20, 30
            "hour": [0, 0, 0],
        })
        cfg = BaselineConfig(
            band=BandConfig(band_width=10, band_min=45, band_max=55),
            quantiles=[0.5],
        )
        table = build_lookup_table(hist, config=cfg, use_spread=True)

        forecast = pd.DataFrame({"pmd": [55], "hour": [0]})
        pred = lookup_predict(table, forecast, config=cfg, use_spread=True)
        # Spread median = 20, PMD = 55 → predicted = 75
        assert pred.iloc[0] == 75.0
