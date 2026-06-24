"""End-to-end prediction: given a PMD forecast, produce predicted indicators."""

from __future__ import annotations

import pandas as pd

from cogen_est_baseline.baseline.bands import assign_band
from cogen_est_baseline.baseline.lookup import build_lookup_table, lookup_predict
from cogen_est_baseline.config import BaselineConfig

# Indicators that are modelled as a spread vs. PMD (like the Excel's TC approach)
SPREAD_INDICATORS = {708}


def prepare_historical(
    indicators: dict[str, pd.DataFrame],
    pmd_df: pd.DataFrame,
) -> pd.DataFrame:
    """Align historical indicator data with PMD on a common 15-min datetime index.

    Parameters
    ----------
    indicators : dict[str, pd.DataFrame]
        Keyed by indicator id string (e.g. "634"), each with ``datetime``
        and ``value`` columns.
    pmd_df : pd.DataFrame
        PMD (indicator 600) with ``datetime`` and ``value`` columns.

    Returns
    -------
    pd.DataFrame
        With ``datetime`` index, ``pmd`` column, ``hour`` column, and one
        column per indicator (e.g. ``"634"``, ``"682"``, …).
    """
    # Use PMD datetime as the reference index
    result = pmd_df[["datetime", "value"]].copy()
    result = result.rename(columns={"value": "pmd"})
    result = result.set_index("datetime")

    for ind_id, df in indicators.items():
        s = df.set_index("datetime")["value"]
        s.name = ind_id
        result = result.join(s, how="left")

    result["hour"] = result.index.hour
    return result


def fit_all_lookup_tables(
    historical: pd.DataFrame,
    target_ids: list[str],
    config: BaselineConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Build one lookup table per target indicator.

    Parameters
    ----------
    historical : pd.DataFrame
        Output of ``prepare_historical``.
    target_ids : list[str]
        Indicator ID strings to build tables for.
    config : BaselineConfig | None
        Baseline configuration.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keyed by indicator id, values are lookup tables.
    """
    config = config or BaselineConfig()
    tables = {}

    for ind_id in target_ids:
        use_spread = int(ind_id) in SPREAD_INDICATORS
        # Drop rows where either PMD or target is NaN
        mask = historical[["pmd", ind_id]].notna().all(axis=1)
        tables[ind_id] = build_lookup_table(
            df=historical.loc[mask],
            pmd_col="pmd",
            target_col=ind_id,
            hour_col="hour",
            config=config,
            use_spread=use_spread,
        )

    return tables


def predict_all(
    tables: dict[str, pd.DataFrame],
    forecast_df: pd.DataFrame,
    quantile: float = 0.5,
    config: BaselineConfig | None = None,
) -> pd.DataFrame:
    """Predict all target indicators from a PMD forecast.

    Parameters
    ----------
    tables : dict[str, pd.DataFrame]
        Lookup tables from ``fit_all_lookup_tables``.
    forecast_df : pd.DataFrame
        Future PMD with ``pmd`` and ``hour`` columns.
    quantile : float
        Quantile to predict (0.5 = P50 baseline).
    config : BaselineConfig | None
        Baseline configuration.

    Returns
    -------
    pd.DataFrame
        With ``forecast_df``'s index and one column per indicator.
    """
    config = config or BaselineConfig()
    predictions = {}

    for ind_id, table in tables.items():
        use_spread = int(ind_id) in SPREAD_INDICATORS
        predictions[ind_id] = lookup_predict(
            lookup_table=table,
            forecast_df=forecast_df,
            quantile=quantile,
            use_spread=use_spread,
            config=config,
        )

    return pd.DataFrame(predictions, index=forecast_df.index)
