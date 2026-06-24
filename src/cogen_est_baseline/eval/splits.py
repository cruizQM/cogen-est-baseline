"""Temporal train/test splits."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from cogen_est_baseline.config import EvalConfig


def temporal_split(
    df: pd.DataFrame,
    config: EvalConfig | None = None,
    datetime_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train/test by date.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a DatetimeIndex or a column specified by ``datetime_col``.
    config : EvalConfig | None
        Contains the split date.
    datetime_col : str | None
        If provided, use this column for splitting; otherwise use the index.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train, test) DataFrames.
    """
    config = config or EvalConfig()
    split_ts = pd.Timestamp(config.split_date, tz="Europe/Madrid")

    if datetime_col is not None:
        dt = pd.to_datetime(df[datetime_col])
        if dt.dt.tz is None:
            dt = dt.dt.tz_localize("Europe/Madrid")
        mask = dt < split_ts
    else:
        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            raise TypeError("DataFrame must have a DatetimeIndex or specify datetime_col.")
        if idx.tz is None:
            idx = idx.tz_localize("Europe/Madrid")
        mask = idx < split_ts

    return df.loc[mask], df.loc[~mask]
