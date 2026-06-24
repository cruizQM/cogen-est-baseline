"""Compare all PMD degradation variants side-by-side.

Trains five GBM configurations (none, A1–A4) on the same train/test split
and produces a single comparison table of R² scores (oracle and realistic)
for each indicator.

Usage::

    python -m cogen_est_baseline.pipelines.compare_degradation [--local-path ...] [--dataset-id ...]
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
from cogen_est_baseline.eval.metrics import coverage, r2_score
from cogen_est_baseline.eval.splits import temporal_split
from cogen_est_baseline.models.features import FeatureConfig
from cogen_est_baseline.models.pmd_degradation import PmdDegradationConfig, PmdDegradationMode
from cogen_est_baseline.models.quantile_gbm import QuantileGBMConfig, QuantileGBMModel
from cogen_est_baseline.pipelines.baseline_pipeline import evaluate, expand_hourly_to_15min

TARGET_IDS = [str(ind.id) for ind in TARGET_INDICATORS]

# ── Variant definitions ───────────────────────────────────────────────────────

VARIANTS: dict[str, QuantileGBMConfig] = {
    "GBM (none)": QuantileGBMConfig(
        pmd_degradation=PmdDegradationConfig(mode=PmdDegradationMode.NONE),
    ),
    "A1 (weekly)": QuantileGBMConfig(
        pmd_degradation=PmdDegradationConfig(mode=PmdDegradationMode.ROLLING_WEEKLY),
    ),
    "A2 (monthly)": QuantileGBMConfig(
        pmd_degradation=PmdDegradationConfig(mode=PmdDegradationMode.ROLLING_MONTHLY),
    ),
    "A3 (weekly+noise)": QuantileGBMConfig(
        pmd_degradation=PmdDegradationConfig(
            mode=PmdDegradationMode.ROLLING_WEEKLY_NOISY,
            noise_fraction=0.5,
        ),
    ),
    "A4 (no rolling)": QuantileGBMConfig(
        features=FeatureConfig(pmd_rolling_windows=[]),
        pmd_degradation=PmdDegradationConfig(mode=PmdDegradationMode.NONE),
    ),
}


def run_comparison(
    local_path: str | Path | None = None,
    dataset_id: str | None = None,
    eval_config: EvalConfig | None = None,
) -> pd.DataFrame:
    """Run all degradation variants and return a comparison DataFrame.

    Returns
    -------
    pd.DataFrame
        MultiIndex columns: (variant, "oracle"/"realistic") × indicators as rows.
    """
    eval_config = eval_config or EvalConfig()

    # ── Load data ─────────────────────────────────────────────────────────
    if local_path is not None:
        dataset_path = Path(local_path)
    else:
        dataset_path = get_dataset_path(dataset_id=dataset_id)

    all_data = load_all_indicators(dataset_path)
    target_dfs = {k: v for k, v in all_data.items() if k in TARGET_IDS}
    pmd_real = all_data["600"]
    historical = prepare_historical(target_dfs, pmd_real)

    # ── Split ─────────────────────────────────────────────────────────────
    train, test = temporal_split(historical, eval_config)
    print(f"Train: {len(train)} | Test: {len(test)}")

    # ── Prepare estimated 600 for realistic eval ──────────────────────────
    est_600 = None
    test_est_valid = None
    test_actual = None

    if "estimated_600" in all_data:
        est_df = expand_hourly_to_15min(all_data["estimated_600"])
        est_indexed = est_df.set_index(
            pd.to_datetime(est_df["datetime"], utc=True).dt.tz_convert("Europe/Madrid")
        )["value"]

        test_est = test.copy()
        test_est["pmd"] = est_indexed.reindex(test.index)
        mask = test_est["pmd"].notna()
        test_est_valid = test_est[mask]
        test_actual = test[mask]

    # ── Lookup baseline ───────────────────────────────────────────────────
    baseline_config = BaselineConfig()
    tables = fit_all_lookup_tables(train, TARGET_IDS, baseline_config)
    test_forecast = test[["pmd", "hour"]].copy()

    pred_bl_oracle = predict_all(tables, test_forecast, quantile=0.5, config=baseline_config)
    metrics_bl_oracle = evaluate(test, pred_bl_oracle, TARGET_IDS, baseline_config.quantiles)

    results = {"Lookup": {"oracle": metrics_bl_oracle["r2"]}}

    if test_est_valid is not None:
        pred_bl_real = predict_all(
            tables, test_est_valid[["pmd", "hour"]], quantile=0.5, config=baseline_config
        )
        metrics_bl_real = evaluate(test_actual, pred_bl_real, TARGET_IDS, baseline_config.quantiles)
        results["Lookup"]["realistic"] = metrics_bl_real["r2"]

    # ── Run each GBM variant ──────────────────────────────────────────────
    for variant_name, config in VARIANTS.items():
        print(f"\n{'='*60}")
        print(f"Training: {variant_name}")
        print(f"{'='*60}")

        model = QuantileGBMModel(config)
        model.fit(train, TARGET_IDS)

        # Oracle
        pred_oracle = model.predict(test, quantile=0.5)
        metrics_oracle = evaluate(test, pred_oracle, TARGET_IDS, config.quantiles)
        results[variant_name] = {"oracle": metrics_oracle["r2"]}

        # Realistic
        if test_est_valid is not None:
            pred_real = model.predict(test_est_valid, quantile=0.5)
            metrics_real = evaluate(test_actual, pred_real, TARGET_IDS, config.quantiles)
            results[variant_name]["realistic"] = metrics_real["r2"]

        # Coverage (oracle)
        if 0.1 in config.quantiles and 0.9 in config.quantiles:
            pred_p10 = model.predict(test, quantile=0.1)
            pred_p90 = model.predict(test, quantile=0.9)
            cov = {}
            for ind_id in TARGET_IDS:
                if ind_id in test.columns:
                    cov[ind_id] = coverage(test[ind_id], pred_p10[ind_id], pred_p90[ind_id])
            results[variant_name]["coverage"] = pd.Series(cov, name="coverage")

    # ── Build comparison tables ───────────────────────────────────────────
    print("\n" + "=" * 80)
    print("COMPARISON: R² ORACLE (real 600)")
    print("=" * 80)
    oracle_df = pd.DataFrame({name: r["oracle"] for name, r in results.items()})
    print(oracle_df.to_string(float_format="{:.3f}".format))

    if test_est_valid is not None:
        print("\n" + "=" * 80)
        print("COMPARISON: R² REALISTIC (estimated 600)")
        print("=" * 80)
        realistic_df = pd.DataFrame(
            {name: r["realistic"] for name, r in results.items() if "realistic" in r}
        )
        print(realistic_df.to_string(float_format="{:.3f}".format))

        # Delta: realistic improvement over lookup baseline per variant
        print("\n" + "=" * 80)
        print("DELTA vs LOOKUP BASELINE: R² REALISTIC")
        print("=" * 80)
        baseline_r2 = results["Lookup"]["realistic"]
        delta_df = pd.DataFrame(
            {name: r["realistic"] - baseline_r2
             for name, r in results.items() if "realistic" in r and name != "Lookup"}
        )
        print(delta_df.to_string(float_format="{:+.3f}".format))

    # Coverage comparison
    cov_data = {name: r["coverage"] for name, r in results.items() if "coverage" in r}
    if cov_data:
        print("\n" + "=" * 80)
        print("COVERAGE P10-P90 (oracle)")
        print("=" * 80)
        cov_df = pd.DataFrame(cov_data)
        print(cov_df.to_string(float_format="{:.3f}".format))

    return oracle_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare PMD degradation variants")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--local-path", type=str, default=None)
    source.add_argument("--dataset-id", type=str, default=None)
    parser.add_argument("--split-date", type=str, default="2025-06-01")
    args = parser.parse_args()

    eval_cfg = EvalConfig(split_date=datetime.fromisoformat(args.split_date))
    run_comparison(
        local_path=args.local_path,
        dataset_id=args.dataset_id,
        eval_config=eval_cfg,
    )
