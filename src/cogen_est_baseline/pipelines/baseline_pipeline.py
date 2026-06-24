"""Baseline pipeline: load dataset → build lookup tables → evaluate.

Runs two evaluation modes:
1. **Oracle**: test-set predictions using the *real* indicator 600 — upper bound
   on what the lookup approach can achieve.
2. **Realistic**: test-set predictions using the *estimated* indicator 600 — what
   the client would actually get in production.

Usage::

    python -m cogen_est_baseline.pipelines.baseline_pipeline [--local-path /path/to/csvs]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cogen_est_baseline.config import BaselineConfig, EvalConfig
from cogen_est_baseline.baseline.predict import (
    SPREAD_INDICATORS,
    fit_all_lookup_tables,
    predict_all,
    prepare_historical,
)
from cogen_est_baseline.data.indicators import TARGET_INDICATORS
from cogen_est_baseline.data.load import get_dataset_path, load_all_indicators
from cogen_est_baseline.eval.metrics import coverage, mae, pinball_loss, r2_score
from cogen_est_baseline.eval.splits import temporal_split

try:
    from clearml import Task as _Task
except ImportError:
    _Task = None

TARGET_IDS = [str(ind.id) for ind in TARGET_INDICATORS]


# ── Data helpers ──────────────────────────────────────────────────────────────


def expand_hourly_to_15min(df: pd.DataFrame) -> pd.DataFrame:
    """Expand an hourly DataFrame to 15-min by forward-filling.

    The real indicator 600 from ESIOS is already at 15-min (same hourly value
    repeated 4×). The estimated 600 is hourly and needs the same treatment.

    Parameters
    ----------
    df : pd.DataFrame
        Hourly data with ``datetime`` and ``value`` columns.

    Returns
    -------
    pd.DataFrame
        15-min data with the hourly value assigned to all four quarter-hours.
    """
    dt = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert("Europe/Madrid")
    s = pd.Series(df["value"].values, index=dt).sort_index()

    # Build a 15-min index spanning the same range
    idx_15m = pd.date_range(s.index.min(), s.index.max(), freq="15min", tz="Europe/Madrid")

    # Reindex and forward-fill (each hourly value fills its 4 quarter-hours)
    s_15m = s.reindex(idx_15m, method="ffill")

    return pd.DataFrame({
        "datetime": s_15m.index,
        "value": s_15m.values,
        "id": 600.0,
    }).reset_index(drop=True)


# ── Evaluation ────────────────────────────────────────────────────────────────


def evaluate(
    actual: pd.DataFrame,
    predicted: pd.DataFrame,
    target_ids: list[str],
    quantiles: list[float],
) -> pd.DataFrame:
    """Compute metrics per indicator and quantile.

    Parameters
    ----------
    actual : pd.DataFrame
        Must contain columns for each indicator in ``target_ids``.
    predicted : pd.DataFrame
        Same columns, aligned index.
    target_ids : list[str]
        Indicator IDs to evaluate.
    quantiles : list[float]
        Quantiles that were predicted (for pinball loss).

    Returns
    -------
    pd.DataFrame
        Rows = indicators, columns = metric names.
    """
    rows = []
    for ind_id in target_ids:
        if ind_id not in actual.columns or ind_id not in predicted.columns:
            continue
        y_true = actual[ind_id]
        y_pred = predicted[ind_id]

        row = {
            "indicator": ind_id,
            "r2": r2_score(y_true, y_pred),
            "mae": mae(y_true, y_pred),
            "pinball_0.5": pinball_loss(y_true, y_pred, quantile=0.5),
            "n_samples": int(y_true.notna().sum()),
        }
        rows.append(row)

    return pd.DataFrame(rows).set_index("indicator")


# ── Main pipeline ─────────────────────────────────────────────────────────────


class _NullLogger:
    """Drop-in replacement when ClearML is disabled."""

    def report_single_value(self, *args, **kwargs):
        pass


def run_baseline(
    local_path: str | Path | None = None,
    dataset_id: str | None = None,
    baseline_config: BaselineConfig | None = None,
    eval_config: EvalConfig | None = None,
    use_clearml: bool = True,
) -> dict:
    """Run the full baseline fit-and-evaluate pipeline.

    Parameters
    ----------
    local_path : str | Path | None
        Path to a local directory with ``id_*.csv`` files.
    dataset_id : str | None
        ClearML Dataset ID. Ignored if ``local_path`` is set.
    baseline_config : BaselineConfig | None
        Lookup-table configuration.
    eval_config : EvalConfig | None
        Train/test split configuration.
    use_clearml : bool
        If ``True`` (default), log to ClearML. Set to ``False`` or use
        ``--no-clearml`` to run without a ClearML backend.

    Returns
    -------
    dict
        ``{"oracle": pd.DataFrame, "realistic": pd.DataFrame}`` with
        per-indicator metrics for each evaluation mode.
    """
    baseline_config = baseline_config or BaselineConfig()
    eval_config = eval_config or EvalConfig()

    # ── ClearML task (optional) ─────────────────────────────────────────────
    task = None
    logger = _NullLogger()

    if use_clearml and _Task is not None:
        try:
            task = _Task.init(
                project_name="cogen-est-baseline",
                task_name="baseline-eval",
                task_type=_Task.TaskTypes.testing,
            )
            task.connect(baseline_config, name="baseline_config")
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

    # ── 2. Prepare historical (real 600 + targets, aligned on 15-min) ────
    target_dfs = {k: v for k, v in all_data.items() if k in TARGET_IDS}
    pmd_real = all_data["600"]
    historical = prepare_historical(target_dfs, pmd_real)
    print(f"Historical shape: {historical.shape}")
    print(f"Date range: {historical.index.min()} to {historical.index.max()}")

    # ── 3. Train/test split ───────────────────────────────────────────────
    train, test = temporal_split(historical, eval_config)
    print(f"Train: {len(train)} rows ({train.index.min()} to {train.index.max()})")
    print(f"Test:  {len(test)} rows ({test.index.min()} to {test.index.max()})")

    # ── 4. Build lookup tables on training data ───────────────────────────
    tables = fit_all_lookup_tables(train, TARGET_IDS, baseline_config)
    for ind_id, table in tables.items():
        n_cells = len(table)
        print(f"Lookup table {ind_id}: {n_cells} (hour, band) cells")
        # Log table sizes to ClearML
        logger.report_single_value(f"lookup_cells/{ind_id}", n_cells)

    # ── 5. Oracle evaluation (real 600 on test set) ───────────────────────
    test_forecast_real = test[["pmd", "hour"]].copy()
    pred_oracle = predict_all(tables, test_forecast_real, quantile=0.5, config=baseline_config)
    metrics_oracle = evaluate(test, pred_oracle, TARGET_IDS, baseline_config.quantiles)

    print("\n=== ORACLE (real 600) ===")
    print(metrics_oracle.to_string())

    for ind_id in metrics_oracle.index:
        for col in metrics_oracle.columns:
            val = metrics_oracle.loc[ind_id, col]
            if isinstance(val, (int, float)):
                logger.report_single_value(f"oracle/{ind_id}/{col}", val)

    # ── 6. Realistic evaluation (estimated 600 on test set) ───────────────
    metrics_realistic = None
    if "estimated_600" in all_data:
        est_600 = all_data["estimated_600"]

        # Expand hourly → 15-min if needed
        dt = pd.to_datetime(est_600["datetime"])
        diffs = dt.diff().dropna()
        median_freq = diffs.median()
        if median_freq > pd.Timedelta(minutes=30):
            print("\nExpanding estimated_600 from hourly to 15-min...")
            est_600 = expand_hourly_to_15min(est_600)

        # Align estimated 600 with the test period
        est_indexed = est_600.set_index(
            pd.to_datetime(est_600["datetime"], utc=True).dt.tz_convert("Europe/Madrid")
        )["value"]

        # Build forecast DataFrame for the test period using estimated PMD
        test_realistic = test[[]].copy()  # empty df with test's datetime index
        test_realistic["pmd"] = est_indexed.reindex(test.index)
        test_realistic["hour"] = test.index.hour

        # Drop rows where estimated 600 is not available
        mask = test_realistic["pmd"].notna()
        test_realistic_valid = test_realistic[mask]
        test_valid = test[mask]

        if len(test_realistic_valid) > 0:
            pred_realistic = predict_all(
                tables, test_realistic_valid, quantile=0.5, config=baseline_config
            )
            metrics_realistic = evaluate(
                test_valid, pred_realistic, TARGET_IDS, baseline_config.quantiles
            )

            print("\n=== REALISTIC (estimated 600) ===")
            print(metrics_realistic.to_string())

            for ind_id in metrics_realistic.index:
                for col in metrics_realistic.columns:
                    val = metrics_realistic.loc[ind_id, col]
                    if isinstance(val, (int, float)):
                        logger.report_single_value(f"realistic/{ind_id}/{col}", val)
        else:
            print("\nNo overlap between estimated 600 and test period — skipping realistic eval.")
    else:
        print("\nNo estimated_600 found in dataset — skipping realistic eval.")

    # ── 7. Coverage evaluation (P10–P90) ──────────────────────────────────
    if 0.1 in baseline_config.quantiles and 0.9 in baseline_config.quantiles:
        print("\n=== COVERAGE (P10–P90, oracle) ===")
        pred_p10 = predict_all(tables, test_forecast_real, quantile=0.1, config=baseline_config)
        pred_p90 = predict_all(tables, test_forecast_real, quantile=0.9, config=baseline_config)

        for ind_id in TARGET_IDS:
            if ind_id in test.columns and ind_id in pred_p10.columns:
                cov = coverage(test[ind_id], pred_p10[ind_id], pred_p90[ind_id])
                print(f"  {ind_id}: {cov:.3f} (target: 0.800)")
                logger.report_single_value(f"coverage_80/{ind_id}", cov)

    if task is not None:
        task.close()
    return {"oracle": metrics_oracle, "realistic": metrics_realistic}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run P50 baseline evaluation")
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--local-path",
        type=str,
        default=None,
        help="Path to local directory with id_*.csv files",
    )
    source.add_argument(
        "--dataset-id",
        type=str,
        default=None,
        help="ClearML Dataset ID to download and use",
    )
    parser.add_argument(
        "--split-date",
        type=str,
        default="2025-06-01",
        help="Train/test split date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--no-clearml",
        action="store_true",
        help="Run without ClearML logging",
    )
    args = parser.parse_args()

    eval_cfg = EvalConfig()
    if args.split_date:
        from datetime import datetime
        eval_cfg.split_date = datetime.fromisoformat(args.split_date)

    run_baseline(
        local_path=args.local_path,
        dataset_id=args.dataset_id,
        eval_config=eval_cfg,
        use_clearml=not args.no_clearml,
    )
