"""Stitching logic for indicator regime changes.

Handles two regime transitions:
- **634**: only valid from 20/11/2024 onwards. Data before that date is dropped.
- **2197**: replaced 676/677 on 10/12/2024. Before that date, 2197 is backfilled
  by selecting 677 (up) when 10250 > 0 and 676 (down) when 10250 < 0.
  Timesteps where 10250 == 0 are set to NaN.
"""

from __future__ import annotations

import pandas as pd

from cogen_est_baseline.data.indicators import REGIME_634_START, REGIME_2197_START


def apply_634_cutoff(df_634: pd.DataFrame) -> pd.DataFrame:
    """Drop 634 data before the regime split date (20/11/2024).

    Parameters
    ----------
    df_634 : pd.DataFrame
        Raw 634 series with a ``datetime`` column (timezone-aware).

    Returns
    -------
    pd.DataFrame
        Filtered to rows on or after ``REGIME_634_START``.
    """
    dt = pd.to_datetime(df_634["datetime"], utc=True).dt.tz_convert("Europe/Madrid")
    mask = dt >= pd.Timestamp(REGIME_634_START, tz="Europe/Madrid")
    return df_634.loc[mask].reset_index(drop=True)


def backfill_2197(
    df_676: pd.DataFrame,
    df_677: pd.DataFrame,
    df_2197: pd.DataFrame,
    df_10250: pd.DataFrame,
) -> pd.DataFrame:
    """Backfill 2197 before 10/12/2024 using 676/677 and sign of 10250.

    For timestamps before ``REGIME_2197_START``:
    - If 10250 > 0 (up activation): use 677 (up price)
    - If 10250 < 0 (down activation): use 676 (down price)
    - If 10250 == 0 or NaN: set to NaN

    For timestamps on or after ``REGIME_2197_START``, the native 2197 values
    are kept as-is.

    Parameters
    ----------
    df_676, df_677 : pd.DataFrame
        Legacy mFRR price series (down, up) with ``datetime`` and ``value`` columns.
    df_2197 : pd.DataFrame
        Native 2197 series (may be empty before the regime date).
    df_10250 : pd.DataFrame
        Net mFRR volume series with ``datetime`` and ``value`` columns.

    Returns
    -------
    pd.DataFrame
        Combined series with columns ``datetime``, ``value``, and ``id`` (= 2197).
    """
    regime_ts = pd.Timestamp(REGIME_2197_START, tz="Europe/Madrid")

    # Index everything by datetime for alignment
    def _to_series(df: pd.DataFrame) -> pd.Series:
        dt = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert("Europe/Madrid")
        return pd.Series(df["value"].values, index=dt, name="value")

    s_676 = _to_series(df_676)
    s_677 = _to_series(df_677)
    s_10250 = _to_series(df_10250)
    s_2197_native = _to_series(df_2197)

    # Build backfilled series for the pre-regime period
    pre_regime_index = s_10250.index[s_10250.index < regime_ts]

    # Align 676, 677, 10250 on the pre-regime index
    vol = s_10250.reindex(pre_regime_index)
    up_price = s_677.reindex(pre_regime_index)
    down_price = s_676.reindex(pre_regime_index)

    # Select based on sign of 10250
    backfilled = pd.Series(index=pre_regime_index, dtype=float)
    backfilled[vol > 0] = up_price[vol > 0]
    backfilled[vol < 0] = down_price[vol < 0]
    # vol == 0 or NaN → remains NaN (default)

    # Combine: backfilled pre-regime + native post-regime
    native_post = s_2197_native[s_2197_native.index >= regime_ts]
    combined = pd.concat([backfilled, native_post]).sort_index()

    # Convert back to DataFrame matching the standard format
    result = pd.DataFrame(
        {
            "datetime": combined.index,
            "value": combined.values,
            "id": 2197.0,
        }
    ).reset_index(drop=True)

    return result
