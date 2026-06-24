"""Feature engineering for ML models.

Transforms the raw historical DataFrame (from ``prepare_historical``) into
a feature matrix suitable for gradient boosting or other ML models.

Key differences from the lookup baseline:
- PMD is continuous (not discretised into bands)
- Hour and month are encoded both as raw ordinals (for tree models) and
  cyclically (sin/cos, for future linear/neural models)
- Additional temporal features: day_of_week, is_weekend, quarter
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""

    # Whether to include cyclical (sin/cos) encodings
    cyclical_encoding: bool = True

    # Whether to include PMD-derived features
    pmd_rolling_windows: list[int] = field(default_factory=lambda: [24, 96])
    """Rolling window sizes (in 15-min steps) for PMD rolling mean/std.
    24 = 6 hours, 96 = 24 hours."""

    # Whether to include the PMD band as a categorical feature
    # (allows the model to learn band-like structure if useful)
    include_pmd_band: bool = False


CYCLICAL_COLS = {
    "hour": 24,
    "month": 12,
    "day_of_week": 7,
}


def build_features(
    df: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Build feature matrix from a historical or forecast DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a DatetimeIndex (timezone-aware) and a ``pmd`` column.
        Typically the output of ``prepare_historical`` or a forecast DataFrame.
    config : FeatureConfig | None
        Feature engineering options.

    Returns
    -------
    pd.DataFrame
        Feature matrix with the same index as ``df``. All columns are numeric.
    """
    config = config or FeatureConfig()
    features = pd.DataFrame(index=df.index)

    # ── PMD (continuous) ──────────────────────────────────────────────────
    features["pmd"] = df["pmd"]

    # ── Temporal features ─────────────────────────────────────────────────
    dt = df.index
    features["hour"] = dt.hour
    features["month"] = dt.month
    features["day_of_week"] = dt.dayofweek
    features["quarter"] = dt.quarter
    features["is_weekend"] = (dt.dayofweek >= 5).astype(int)

    # ── Cyclical encodings ────────────────────────────────────────────────
    if config.cyclical_encoding:
        for col, period in CYCLICAL_COLS.items():
            rad = 2 * np.pi * features[col] / period
            features[f"{col}_sin"] = np.sin(rad)
            features[f"{col}_cos"] = np.cos(rad)

    # ── PMD rolling statistics ────────────────────────────────────────────
    if config.pmd_rolling_windows:
        for window in config.pmd_rolling_windows:
            features[f"pmd_rmean_{window}"] = (
                df["pmd"].rolling(window, min_periods=1).mean()
            )
            features[f"pmd_rstd_{window}"] = (
                df["pmd"].rolling(window, min_periods=1).std().fillna(0)
            )

    # ── PMD band (optional, as ordinal) ───────────────────────────────────
    if config.include_pmd_band:
        from cogen_est_baseline.baseline.bands import assign_band

        bands = assign_band(df["pmd"])
        features["pmd_band_mid"] = bands.map(lambda iv: iv.mid if pd.notna(iv) else np.nan)

    return features


def get_feature_columns(config: FeatureConfig | None = None) -> list[str]:
    """Return the list of feature column names that ``build_features`` produces.

    Useful for ensuring consistent column ordering between train and predict.
    """
    config = config or FeatureConfig()
    cols = ["pmd", "hour", "month", "day_of_week", "quarter", "is_weekend"]

    if config.cyclical_encoding:
        for col in CYCLICAL_COLS:
            cols.extend([f"{col}_sin", f"{col}_cos"])

    if config.pmd_rolling_windows:
        for window in config.pmd_rolling_windows:
            cols.extend([f"pmd_rmean_{window}", f"pmd_rstd_{window}"])

    if config.include_pmd_band:
        cols.append("pmd_band_mid")

    return cols
