"""PMD degradation: simulate the information loss of a smooth price forecast.

When training on real (historical) PMD, the model learns to exploit fine-grained
intra-day volatility that the estimated 600 doesn't contain. PMD degradation
deliberately smooths the training PMD to match the information content available
at inference, closing the oracle-realistic gap.

Four modes:

- **none**: use real PMD as-is (baseline behavior).
- **rolling_weekly**: replace PMD with its 7-day rolling mean.
- **rolling_monthly**: replace PMD with its 30-day rolling mean.
- **rolling_weekly_noisy**: rolling weekly mean + calibrated Gaussian noise,
  preserving some realistic short-term variation without the exact structure
  the model could overfit to.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class PmdDegradationMode(str, Enum):
    """Available PMD degradation modes."""

    NONE = "none"
    ROLLING_WEEKLY = "rolling_weekly"
    ROLLING_MONTHLY = "rolling_monthly"
    ROLLING_WEEKLY_NOISY = "rolling_weekly_noisy"


# Window sizes in 15-min steps
_WEEKLY_WINDOW = 7 * 24 * 4   # 672
_MONTHLY_WINDOW = 30 * 24 * 4  # 2880


@dataclass
class PmdDegradationConfig:
    """Configuration for PMD degradation during training."""

    mode: PmdDegradationMode = PmdDegradationMode.NONE

    # For rolling_weekly_noisy: fraction of the historical residual std to add
    # back as noise. 1.0 = full historical noise level, 0.5 = half, etc.
    noise_fraction: float = 0.5

    # Random seed for reproducible noise generation
    noise_seed: int = 42


def degrade_pmd(
    df: pd.DataFrame,
    config: PmdDegradationConfig | None = None,
    pmd_col: str = "pmd",
) -> pd.DataFrame:
    """Apply PMD degradation to a DataFrame.

    Returns a **copy** of ``df`` with the ``pmd_col`` replaced by the
    degraded version. All other columns are untouched.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``pmd_col`` column. Typically the output of
        ``prepare_historical`` (training data only).
    config : PmdDegradationConfig | None
        Degradation parameters. ``None`` = no degradation.
    pmd_col : str
        Column name for the day-ahead price.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with degraded PMD.
    """
    config = config or PmdDegradationConfig()

    if config.mode == PmdDegradationMode.NONE:
        return df

    result = df.copy()
    pmd = result[pmd_col]

    if config.mode == PmdDegradationMode.ROLLING_WEEKLY:
        result[pmd_col] = pmd.rolling(_WEEKLY_WINDOW, min_periods=1, center=True).mean()

    elif config.mode == PmdDegradationMode.ROLLING_MONTHLY:
        result[pmd_col] = pmd.rolling(_MONTHLY_WINDOW, min_periods=1, center=True).mean()

    elif config.mode == PmdDegradationMode.ROLLING_WEEKLY_NOISY:
        smoothed = pmd.rolling(_WEEKLY_WINDOW, min_periods=1, center=True).mean()

        # Calibrate noise from the historical residual (real - smoothed)
        residual = pmd - smoothed
        residual_std = residual.std()

        rng = np.random.default_rng(config.noise_seed)
        noise = rng.normal(0, residual_std * config.noise_fraction, size=len(pmd))

        result[pmd_col] = smoothed + noise

    else:
        raise ValueError(f"Unknown degradation mode: {config.mode}")

    return result


def describe_degradation(config: PmdDegradationConfig) -> str:
    """Human-readable description of the degradation applied."""
    if config.mode == PmdDegradationMode.NONE:
        return "No degradation (real PMD)"
    elif config.mode == PmdDegradationMode.ROLLING_WEEKLY:
        return "Rolling 7-day mean"
    elif config.mode == PmdDegradationMode.ROLLING_MONTHLY:
        return "Rolling 30-day mean"
    elif config.mode == PmdDegradationMode.ROLLING_WEEKLY_NOISY:
        return f"Rolling 7-day mean + {config.noise_fraction:.0%} calibrated noise"
    return str(config.mode)
