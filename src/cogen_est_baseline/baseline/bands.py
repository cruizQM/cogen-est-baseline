"""PMD (day-ahead price) band discretisation.

Reproduces the €5/MWh banding scheme from the client's Excel methodology:
bands start below -20 €/MWh and extend up to 250 €/MWh, with everything
above the last edge falling into a single overflow band.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cogen_est_baseline.config import BandConfig


def compute_band_edges(config: BandConfig | None = None) -> np.ndarray:
    """Return the array of band edges.

    For the default config (min=-20, max=250, width=5), this gives edges
    [-20, -15, -10, …, 245, 250] plus -inf and +inf sentinels.

    Returns
    -------
    np.ndarray
        Sorted edges including ``-inf`` and ``+inf``.
    """
    config = config or BandConfig()
    inner = np.arange(config.band_min, config.band_max + config.band_width, config.band_width)
    return np.concatenate([[-np.inf], inner, [np.inf]])


def assign_band(
    pmd_values: pd.Series,
    config: BandConfig | None = None,
) -> pd.Series:
    """Assign each PMD value to its €5/MWh band.

    Parameters
    ----------
    pmd_values : pd.Series
        Day-ahead price values (€/MWh).
    config : BandConfig | None
        Band parameters. Uses defaults if ``None``.

    Returns
    -------
    pd.Series
        Band labels as ``pd.Categorical`` with ordered interval categories.
        Each label is a ``pd.Interval`` like ``(-20.0, -15.0]``.
    """
    edges = compute_band_edges(config)
    return pd.cut(pmd_values, bins=edges, right=True)


def assign_band_label(
    pmd_values: pd.Series,
    config: BandConfig | None = None,
) -> pd.Series:
    """Assign each PMD value to a human-readable band label string.

    Produces labels like ``"< -20"``, ``"-20 to -15"``, ``"245 to 250"``,
    ``">= 250"``, matching the Excel's display convention.

    Parameters
    ----------
    pmd_values : pd.Series
        Day-ahead price values (€/MWh).
    config : BandConfig | None
        Band parameters.

    Returns
    -------
    pd.Series[str]
        String band labels.
    """
    config = config or BandConfig()
    edges = compute_band_edges(config)
    # pd.cut with labels
    labels = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if np.isneginf(lo):
            labels.append(f"< {hi:.0f}")
        elif np.isposinf(hi):
            labels.append(f">= {lo:.0f}")
        else:
            labels.append(f"{lo:.0f} to {hi:.0f}")
    return pd.cut(pmd_values, bins=edges, right=True, labels=labels)
