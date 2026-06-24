"""Evaluation metrics for the baseline and future ML models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def r2_score(y_true: pd.Series, y_pred: pd.Series) -> float:
    """R² score, dropping NaN pairs."""
    mask = y_true.notna() & y_pred.notna()
    yt, yp = y_true[mask].values, y_pred[mask].values
    if len(yt) == 0:
        return float("nan")
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean absolute error, dropping NaN pairs."""
    mask = y_true.notna() & y_pred.notna()
    yt, yp = y_true[mask].values, y_pred[mask].values
    if len(yt) == 0:
        return float("nan")
    return float(np.mean(np.abs(yt - yp)))


def pinball_loss(
    y_true: pd.Series, y_pred: pd.Series, quantile: float = 0.5
) -> float:
    """Pinball (quantile) loss, dropping NaN pairs.

    Parameters
    ----------
    y_true, y_pred : pd.Series
        Actual and predicted values.
    quantile : float
        Target quantile (e.g. 0.1, 0.5, 0.9).

    Returns
    -------
    float
        Mean pinball loss.
    """
    mask = y_true.notna() & y_pred.notna()
    yt, yp = y_true[mask].values, y_pred[mask].values
    if len(yt) == 0:
        return float("nan")
    errors = yt - yp
    loss = np.where(errors >= 0, quantile * errors, (quantile - 1) * errors)
    return float(np.mean(loss))


def coverage(
    y_true: pd.Series,
    y_lower: pd.Series,
    y_upper: pd.Series,
) -> float:
    """Empirical coverage of a prediction interval.

    Parameters
    ----------
    y_true : pd.Series
        Actual values.
    y_lower, y_upper : pd.Series
        Lower and upper bounds of the prediction interval.

    Returns
    -------
    float
        Fraction of true values within [lower, upper].
    """
    mask = y_true.notna() & y_lower.notna() & y_upper.notna()
    yt = y_true[mask].values
    lo = y_lower[mask].values
    hi = y_upper[mask].values
    if len(yt) == 0:
        return float("nan")
    return float(np.mean((yt >= lo) & (yt <= hi)))
