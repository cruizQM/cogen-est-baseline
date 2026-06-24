"""Build and query P50 (and other quantile) conditional lookup tables.

The core method from the client's Excel: for each target indicator, compute the
historical quantile of the value (or spread vs. PMD) conditional on
``(hour_of_day, PMD_band)``.
"""

from __future__ import annotations

import pandas as pd

from cogen_est_baseline.baseline.bands import assign_band
from cogen_est_baseline.config import BaselineConfig


def build_lookup_table(
    df: pd.DataFrame,
    pmd_col: str = "pmd",
    target_col: str = "value",
    hour_col: str = "hour",
    config: BaselineConfig | None = None,
    use_spread: bool = False,
) -> pd.DataFrame:
    """Build a quantile lookup table grouped by ``(hour, PMD_band)``.

    Parameters
    ----------
    df : pd.DataFrame
        Historical data with at least ``pmd_col``, ``target_col``, and
        ``hour_col`` columns.
    pmd_col : str
        Column with PMD (indicator 600) values.
    target_col : str
        Column with the target indicator values.
    hour_col : str
        Column with hour-of-day (0–23).
    config : BaselineConfig | None
        Configuration (band params, quantiles).
    use_spread : bool
        If ``True``, the lookup target is ``target - PMD`` (the spread)
        rather than the raw target value. Used for indicators like 708
        where the spread is more stable than the absolute level.

    Returns
    -------
    pd.DataFrame
        Indexed by ``(hour, pmd_band)`` with one column per quantile,
        named ``"q0.1"``, ``"q0.5"``, ``"q0.9"`` etc.
    """
    config = config or BaselineConfig()

    work = df[[pmd_col, target_col, hour_col]].copy()
    work["pmd_band"] = assign_band(work[pmd_col], config.band)

    if use_spread:
        work["target"] = work[target_col] - work[pmd_col]
    else:
        work["target"] = work[target_col]

    # Group by (hour, band) and compute requested quantiles
    grouped = work.groupby([hour_col, "pmd_band"], observed=True)["target"]

    quantile_dfs = []
    for q in config.quantiles:
        qs = grouped.quantile(q)
        qs.name = f"q{q}"
        quantile_dfs.append(qs)

    result = pd.concat(quantile_dfs, axis=1)
    result.index.names = ["hour", "pmd_band"]
    return result


def lookup_predict(
    lookup_table: pd.DataFrame,
    forecast_df: pd.DataFrame,
    pmd_col: str = "pmd",
    hour_col: str = "hour",
    quantile: float = 0.5,
    use_spread: bool = False,
    config: BaselineConfig | None = None,
) -> pd.Series:
    """Predict target values by joining a PMD forecast against a lookup table.

    Parameters
    ----------
    lookup_table : pd.DataFrame
        Output of ``build_lookup_table``.
    forecast_df : pd.DataFrame
        Future periods with at least ``pmd_col`` and ``hour_col``.
    pmd_col, hour_col : str
        Column names in ``forecast_df``.
    quantile : float
        Which quantile to return (must be one of the quantiles in the table).
    use_spread : bool
        If ``True``, the looked-up value is a spread and must be added back
        to the forecasted PMD to get the final prediction.
    config : BaselineConfig | None
        Band configuration.

    Returns
    -------
    pd.Series
        Predicted values aligned with ``forecast_df``'s index.
    """
    config = config or BaselineConfig()
    q_col = f"q{quantile}"

    work = forecast_df[[pmd_col, hour_col]].copy()
    work["pmd_band"] = assign_band(work[pmd_col], config.band)

    # Merge on (hour, band)
    work = work.set_index([hour_col, "pmd_band"])
    predicted = work.join(lookup_table[[q_col]], how="left")[q_col]

    # Restore original index
    predicted.index = forecast_df.index

    if use_spread:
        predicted = predicted + forecast_df[pmd_col]

    return predicted
