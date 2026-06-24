"""Quantile gradient boosting model using LightGBM.

Replaces the lookup-table baseline with a continuous, regularised model
that naturally handles sparse regions and produces calibrated uncertainty bands.

One LightGBM model is trained per (indicator, quantile) pair.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from cogen_est_baseline.baseline.predict import SPREAD_INDICATORS
from cogen_est_baseline.models.features import FeatureConfig, build_features, get_feature_columns
from cogen_est_baseline.models.pmd_degradation import (
    PmdDegradationConfig,
    PmdDegradationMode,
    degrade_pmd,
    describe_degradation,
)


@dataclass
class QuantileGBMConfig:
    """Hyperparameters for the quantile GBM model."""

    # Quantiles to train models for
    quantiles: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])

    # LightGBM parameters
    n_estimators: int = 500
    learning_rate: float = 0.05
    max_depth: int = 6
    num_leaves: int = 31
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0

    # Early stopping
    early_stopping_rounds: int | None = 50
    validation_fraction: float = 0.1

    # Feature config
    features: FeatureConfig = field(default_factory=FeatureConfig)

    # PMD degradation (applied to training data only)
    pmd_degradation: PmdDegradationConfig = field(default_factory=PmdDegradationConfig)

    def to_lgb_params(self, quantile: float) -> dict[str, Any]:
        """Convert to LightGBM parameter dict for a specific quantile."""
        return {
            "objective": "quantile",
            "alpha": quantile,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "verbosity": -1,
            "n_jobs": -1,
        }


class QuantileGBMModel:
    """Collection of LightGBM quantile regressors for multiple indicators.

    Usage::

        model = QuantileGBMModel(config)
        model.fit(train_df)
        predictions = model.predict(test_df)
    """

    def __init__(self, config: QuantileGBMConfig | None = None):
        self.config = config or QuantileGBMConfig()
        self.feature_columns: list[str] = get_feature_columns(self.config.features)
        # Nested dict: models[indicator_id][quantile] = fitted LGBMRegressor
        self.models: dict[str, dict[float, lgb.LGBMRegressor]] = {}
        self.target_ids: list[str] = []

    def fit(
        self,
        df: pd.DataFrame,
        target_ids: list[str],
    ) -> QuantileGBMModel:
        """Fit quantile models for all indicators.

        Parameters
        ----------
        df : pd.DataFrame
            Historical data with DatetimeIndex, ``pmd`` column, and one column
            per indicator in ``target_ids``.
        target_ids : list[str]
            Indicator IDs to fit models for.

        Returns
        -------
        self
        """
        self.target_ids = target_ids

        # Apply PMD degradation to training data (not to prediction data)
        if self.config.pmd_degradation.mode != PmdDegradationMode.NONE:
            desc = describe_degradation(self.config.pmd_degradation)
            print(f"  PMD degradation: {desc}")
            df = degrade_pmd(df, self.config.pmd_degradation)

        features = build_features(df, self.config.features)

        for ind_id in target_ids:
            self.models[ind_id] = {}

            # Determine target: raw value or spread vs. PMD
            use_spread = int(ind_id) in SPREAD_INDICATORS
            if use_spread:
                y = df[ind_id] - df["pmd"]
            else:
                y = df[ind_id]

            # Drop NaN rows for this indicator
            mask = features.notna().all(axis=1) & y.notna()
            X_train = features.loc[mask, self.feature_columns]
            y_train = y[mask]

            if len(X_train) == 0:
                print(f"  {ind_id}: no valid training samples, skipping.")
                continue

            # Split off validation set for early stopping (temporal split)
            n_val = int(len(X_train) * self.config.validation_fraction)
            if n_val > 0 and self.config.early_stopping_rounds is not None:
                X_fit, X_val = X_train.iloc[:-n_val], X_train.iloc[-n_val:]
                y_fit, y_val = y_train.iloc[:-n_val], y_train.iloc[-n_val:]
                eval_set = [(X_val, y_val)]
                callbacks = [lgb.early_stopping(self.config.early_stopping_rounds, verbose=False)]
            else:
                X_fit, y_fit = X_train, y_train
                eval_set = None
                callbacks = None

            for q in self.config.quantiles:
                params = self.config.to_lgb_params(q)
                model = lgb.LGBMRegressor(**params)
                model.fit(
                    X_fit, y_fit,
                    eval_set=eval_set,
                    callbacks=callbacks,
                )
                self.models[ind_id][q] = model

            best_iters = {
                q: self.models[ind_id][q].best_iteration_
                for q in self.config.quantiles
                if hasattr(self.models[ind_id][q], "best_iteration_")
                and self.models[ind_id][q].best_iteration_ > 0
            }
            print(f"  {ind_id}: trained on {len(X_fit)} samples, "
                  f"val={n_val}, best_iters={best_iters or 'N/A'}")

        return self

    def predict(
        self,
        df: pd.DataFrame,
        quantile: float = 0.5,
    ) -> pd.DataFrame:
        """Predict for all fitted indicators at a given quantile.

        Parameters
        ----------
        df : pd.DataFrame
            Forecast data with DatetimeIndex and ``pmd`` column.
        quantile : float
            Which quantile to predict.

        Returns
        -------
        pd.DataFrame
            One column per indicator, aligned with ``df``'s index.
        """
        features = build_features(df, self.config.features)
        X = features[self.feature_columns]

        predictions = {}
        for ind_id in self.target_ids:
            if ind_id not in self.models or quantile not in self.models[ind_id]:
                predictions[ind_id] = pd.Series(np.nan, index=df.index)
                continue

            model = self.models[ind_id][quantile]
            y_pred = model.predict(X)

            # If spread model, add PMD back
            if int(ind_id) in SPREAD_INDICATORS:
                y_pred = y_pred + df["pmd"].values

            predictions[ind_id] = pd.Series(y_pred, index=df.index)

        return pd.DataFrame(predictions)

    def feature_importance(self, ind_id: str, quantile: float = 0.5) -> pd.Series:
        """Get feature importances for a specific (indicator, quantile) model.

        Returns
        -------
        pd.Series
            Feature importances indexed by feature name, sorted descending.
        """
        model = self.models[ind_id][quantile]
        imp = pd.Series(
            model.feature_importances_,
            index=self.feature_columns,
            name="importance",
        )
        return imp.sort_values(ascending=False)
