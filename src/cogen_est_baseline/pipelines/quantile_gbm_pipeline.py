"""Quantile GBM pipeline: train models → evaluate → compare against lookup baseline.

Usage::

    python -m cogen_est_baseline.pipelines.quantile_gbm_pipeline [--local-path /path/to/csvs]
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from cogen_est_baseline.config import BaselineConfig, EvalConfig
from cogen_est_baseline.baseline.predict import (
    fit_all_lookup_tables,
    predict_all,
    prepare_historical,
)
from cogen_est_baseline.data.indicators import TARGET_INDICATORS
from cogen_est_baseline.data.load import get_dataset_path, load_all_indicators
from cogen_est_baseline.eval.metrics import coverage, mae, pinball_loss, r2_score
from cogen_est_baseline.eval.splits import temporal_split
from cogen_est_baseline.models.features import FeatureConfig
from cogen_est_baseline.models.quantile_gbm import QuantileGBMConfig, QuantileGBMModel
from cogen_est_baseline.pipelines.baseline_pipeline import evaluate, expand_hourly_to_15min

try:
    from clearml import Task as _Task
except ImportError:
    _Task = None

TARGET_IDS = [str(ind.id) for ind in TARGET_INDICATORS]


def run_quantile_gbm(
    local_path: str | Path | None = None,
    dataset_id: str | None = None,
    gbm_config: QuantileGBMConfig | None = None,
    eval_config: EvalConfig | None = None,
    use_clearml: bool = True,
) -> dict:
    """Train quantile GBM models and evaluate against the lookup baseline.

    Returns
    -------
    dict
        Keys: ``"gbm_oracle"``, ``"gbm_realistic"``, ``"baseline_oracle"``,
        ``"baseline_realistic"`` — each a metrics DataFrame.
    """
    gbm_config = gbm_config or QuantileGBMConfig()
    eval_config = eval_config or EvalConfig()
    baseline_config = BaselineConfig(quantiles=gbm_config.quantiles)

    # ── ClearML task (optional) ───────────────────────────────────────────
    task = None

    class _NullLogger:
        def report_single_value(self, *a, **kw):
            pass
        def report_table(self, *a, **kw):
            pass

    logger = _NullLogger()

    if use_clearml and _Task is not None:
        try:
            task = _Task.init(
                project_name="cogen-est-baseline",
                task_name="quantile-gbm-eval",
                task_type=_Task.TaskTypes.training,
            )
            task.connect(gbm_config, name="gbm_config")
            task.connect(eval_config, name="eval_config")
            logger = task.get_logger()
        except Exception as e:
            print(f"ClearML init failed ({e}), continuing without logging.")
            task = None

    # ── 1. Load data ──────────────────────────────────────────────────────
    if local_path is not None:
        dataset_path = Path(local_path)
    else:
        dataset_path = get_dataset_path(dataset_id=dataset_id)

    all_data = load_all_indicators(dataset_path)
    print(f"Loaded indicators: {sorted(all_data.keys())}")

    # ── 2. Prepare historical ─────────────────────────────────────────────
    target_dfs = {k: v for k, v in all_data.items() if k in TARGET_IDS}
    pmd_real = all_data["600"]
    historical = prepare_historical(target_dfs, pmd_real)
    print(f"Historical: {historical.shape}")

    # ── 3. Split ──────────────────────────────────────────────────────────
    train, test = temporal_split(historical, eval_config)
    print(f"Train: {len(train)} | Test: {len(test)}")

    # ── 4. Train quantile GBM ─────────────────────────────────────────────
    print("\n--- Training quantile GBM ---")
    model = QuantileGBMModel(gbm_config)
    model.fit(train, TARGET_IDS)

    # Log feature importances
    print("\n--- Feature importances (P50) ---")
    for ind_id in TARGET_IDS:
        if ind_id in model.models and 0.5 in model.models[ind_id]:
            imp = model.feature_importance(ind_id, quantile=0.5)
            print(f"\n  {ind_id}:")
            for feat, val in imp.head(5).items():
                print(f"    {feat}: {val}")

    # ── 5. GBM oracle evaluation ──────────────────────────────────────────
    pred_gbm_oracle = model.predict(test, quantile=0.5)
    metrics_gbm_oracle = evaluate(test, pred_gbm_oracle, TARGET_IDS, gbm_config.quantiles)

    print("\n=== GBM ORACLE (real 600) ===")
    print(metrics_gbm_oracle.to_string())

    for ind_id in metrics_gbm_oracle.index:
        for col in metrics_gbm_oracle.columns:
            val = metrics_gbm_oracle.loc[ind_id, col]
            if isinstance(val, (int, float)):
                logger.report_single_value(f"gbm_oracle/{ind_id}/{col}", val)

    # ── 6. Lookup baseline oracle (for comparison) ────────────────────────
    tables = fit_all_lookup_tables(train, TARGET_IDS, baseline_config)
    test_forecast = test[["pmd", "hour"]].copy()
    pred_baseline_oracle = predict_all(tables, test_forecast, quantile=0.5, config=baseline_config)
    metrics_baseline_oracle = evaluate(test, pred_baseline_oracle, TARGET_IDS, baseline_config.quantiles)

    print("\n=== LOOKUP BASELINE ORACLE (real 600) ===")
    print(metrics_baseline_oracle.to_string())

    # ── 7. GBM realistic evaluation (estimated 600) ───────────────────────
    metrics_gbm_realistic = None
    metrics_baseline_realistic = None

    if "estimated_600" in all_data:
        est_600 = all_data["estimated_600"]
        est_600 = expand_hourly_to_15min(est_600)
        est_indexed = est_600.set_index(
            pd.to_datetime(est_600["datetime"], utc=True).dt.tz_convert("Europe/Madrid")
        )["value"]

        # Build test DataFrame with estimated PMD
        test_est = test.copy()
        test_est["pmd"] = est_indexed.reindex(test.index)
        mask = test_est["pmd"].notna()
        test_est_valid = test_est[mask]
        test_actual = test[mask]

        if len(test_est_valid) > 0:
            # GBM realistic
            pred_gbm_real = model.predict(test_est_valid, quantile=0.5)
            metrics_gbm_realistic = evaluate(
                test_actual, pred_gbm_real, TARGET_IDS, gbm_config.quantiles
            )
            print("\n=== GBM REALISTIC (estimated 600) ===")
            print(metrics_gbm_realistic.to_string())

            for ind_id in metrics_gbm_realistic.index:
                for col in metrics_gbm_realistic.columns:
                    val = metrics_gbm_realistic.loc[ind_id, col]
                    if isinstance(val, (int, float)):
                        logger.report_single_value(f"gbm_realistic/{ind_id}/{col}", val)

            # Baseline realistic
            test_est_forecast = test_est_valid[["pmd", "hour"]].copy()
            pred_bl_real = predict_all(
                tables, test_est_forecast, quantile=0.5, config=baseline_config
            )
            metrics_baseline_realistic = evaluate(
                test_actual, pred_bl_real, TARGET_IDS, baseline_config.quantiles
            )
            print("\n=== LOOKUP BASELINE REALISTIC (estimated 600) ===")
            print(metrics_baseline_realistic.to_string())

    # ── 8. Coverage comparison (P10–P90, oracle) ──────────────────────────
    if 0.1 in gbm_config.quantiles and 0.9 in gbm_config.quantiles:
        pred_gbm_p10 = model.predict(test, quantile=0.1)
        pred_gbm_p90 = model.predict(test, quantile=0.9)

        from cogen_est_baseline.baseline.lookup import lookup_predict
        from cogen_est_baseline.baseline.bands import assign_band

        print("\n=== COVERAGE P10–P90 (oracle) ===")
        print(f"  {'indicator':<12} {'GBM':>8} {'Lookup':>8}")
        for ind_id in TARGET_IDS:
            if ind_id not in test.columns:
                continue
            # GBM coverage
            cov_gbm = coverage(test[ind_id], pred_gbm_p10[ind_id], pred_gbm_p90[ind_id])

            # Baseline coverage
            use_spread = int(ind_id) in set()  # reuse from predict module
            from cogen_est_baseline.baseline.predict import SPREAD_INDICATORS as _SI
            pred_bl_p10 = predict_all(tables, test_forecast, quantile=0.1, config=baseline_config)
            pred_bl_p90 = predict_all(tables, test_forecast, quantile=0.9, config=baseline_config)
            cov_bl = coverage(test[ind_id], pred_bl_p10[ind_id], pred_bl_p90[ind_id])

            print(f"  {ind_id:<12} {cov_gbm:>8.3f} {cov_bl:>8.3f}")
            logger.report_single_value(f"coverage_80_gbm/{ind_id}", cov_gbm)
            logger.report_single_value(f"coverage_80_baseline/{ind_id}", cov_bl)

    # ── 9. Summary comparison ─────────────────────────────────────────────
    print("\n=== R² COMPARISON (oracle) ===")
    print(f"  {'indicator':<12} {'GBM':>8} {'Lookup':>8} {'Delta':>8}")
    for ind_id in TARGET_IDS:
        if ind_id in metrics_gbm_oracle.index and ind_id in metrics_baseline_oracle.index:
            r2_gbm = metrics_gbm_oracle.loc[ind_id, "r2"]
            r2_bl = metrics_baseline_oracle.loc[ind_id, "r2"]
            delta = r2_gbm - r2_bl
            print(f"  {ind_id:<12} {r2_gbm:>8.3f} {r2_bl:>8.3f} {delta:>+8.3f}")

    if metrics_gbm_realistic is not None and metrics_baseline_realistic is not None:
        print("\n=== R² COMPARISON (realistic) ===")
        print(f"  {'indicator':<12} {'GBM':>8} {'Lookup':>8} {'Delta':>8}")
        for ind_id in TARGET_IDS:
            if ind_id in metrics_gbm_realistic.index and ind_id in metrics_baseline_realistic.index:
                r2_gbm = metrics_gbm_realistic.loc[ind_id, "r2"]
                r2_bl = metrics_baseline_realistic.loc[ind_id, "r2"]
                delta = r2_gbm - r2_bl
                print(f"  {ind_id:<12} {r2_gbm:>8.3f} {r2_bl:>8.3f} {delta:>+8.3f}")

    if task is not None:
        task.close()

    return {
        "gbm_oracle": metrics_gbm_oracle,
        "gbm_realistic": metrics_gbm_realistic,
        "baseline_oracle": metrics_baseline_oracle,
        "baseline_realistic": metrics_baseline_realistic,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate quantile GBM")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--local-path", type=str, default=None)
    source.add_argument("--dataset-id", type=str, default=None)
    parser.add_argument("--split-date", type=str, default="2025-06-01")
    parser.add_argument("--no-clearml", action="store_true")

    # Key hyperparameters exposed as CLI args for quick experiments
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--num-leaves", type=int, default=31)

    # PMD degradation
    parser.add_argument(
        "--pmd-degradation",
        type=str,
        choices=["none", "rolling_weekly", "rolling_monthly", "rolling_weekly_noisy"],
        default="none",
        help="PMD degradation mode for training data",
    )
    parser.add_argument(
        "--no-rolling-features",
        action="store_true",
        help="Drop PMD rolling mean/std features (variant A4)",
    )

    args = parser.parse_args()

    eval_cfg = EvalConfig(split_date=datetime.fromisoformat(args.split_date))

    feature_cfg = FeatureConfig(
        pmd_rolling_windows=[] if args.no_rolling_features else [24, 96],
    )

    from cogen_est_baseline.models.pmd_degradation import PmdDegradationConfig, PmdDegradationMode

    gbm_cfg = QuantileGBMConfig(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        num_leaves=args.num_leaves,
        features=feature_cfg,
        pmd_degradation=PmdDegradationConfig(
            mode=PmdDegradationMode(args.pmd_degradation),
        ),
    )

    run_quantile_gbm(
        local_path=args.local_path,
        dataset_id=args.dataset_id,
        gbm_config=gbm_cfg,
        eval_config=eval_cfg,
        use_clearml=not args.no_clearml,
    )
