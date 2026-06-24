"""Tests for PMD band discretisation."""

import numpy as np
import pandas as pd

from cogen_est_baseline.baseline.bands import assign_band, assign_band_label, compute_band_edges
from cogen_est_baseline.config import BandConfig


class TestComputeBandEdges:
    def test_default_edges_include_sentinels(self):
        edges = compute_band_edges()
        assert edges[0] == -np.inf
        assert edges[-1] == np.inf

    def test_default_inner_edges(self):
        edges = compute_band_edges()
        inner = edges[1:-1]
        assert inner[0] == -20.0
        assert inner[-1] == 250.0
        assert np.allclose(np.diff(inner), 5.0)

    def test_custom_config(self):
        cfg = BandConfig(band_width=10, band_min=0, band_max=100)
        edges = compute_band_edges(cfg)
        inner = edges[1:-1]
        assert inner[0] == 0.0
        assert inner[-1] == 100.0
        assert np.allclose(np.diff(inner), 10.0)


class TestAssignBand:
    def test_value_in_middle_of_range(self):
        s = pd.Series([52.0])
        result = assign_band(s)
        interval = result.iloc[0]
        assert 50.0 == interval.left
        assert 55.0 == interval.right

    def test_value_on_edge_goes_right(self):
        """pd.cut is right-closed by default: 50.0 goes into (45, 50]."""
        s = pd.Series([50.0])
        result = assign_band(s)
        interval = result.iloc[0]
        assert interval.left == 45.0
        assert interval.right == 50.0

    def test_extreme_low(self):
        s = pd.Series([-100.0])
        result = assign_band(s)
        interval = result.iloc[0]
        assert np.isneginf(interval.left)
        assert interval.right == -20.0

    def test_extreme_high(self):
        s = pd.Series([500.0])
        result = assign_band(s)
        interval = result.iloc[0]
        assert interval.left == 250.0
        assert np.isposinf(interval.right)


class TestAssignBandLabel:
    def test_label_low(self):
        s = pd.Series([-100.0])
        result = assign_band_label(s)
        assert result.iloc[0] == "< -20"

    def test_label_mid(self):
        s = pd.Series([52.0])
        result = assign_band_label(s)
        assert result.iloc[0] == "50 to 55"

    def test_label_high(self):
        s = pd.Series([300.0])
        result = assign_band_label(s)
        assert result.iloc[0] == ">= 250"
