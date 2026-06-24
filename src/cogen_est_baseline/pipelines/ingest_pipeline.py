"""Ingestion pipeline: fetch from ESIOS, apply stitching, publish ClearML Dataset.

Usage::

    python -m cogen_est_baseline.pipelines.ingest_pipeline
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
from clearml import Dataset, Task

from cogen_est_baseline.config import IngestConfig
from cogen_est_baseline.data.esios_client import fetch_indicator
from cogen_est_baseline.data.indicators import (
    ALL_INDICATORS,
    IND_600,
    IND_634,
    IND_676,
    IND_677,
    LEGACY_INDICATORS,
)
from cogen_est_baseline.data.stitching import apply_634_cutoff, backfill_2197


def run_ingest(config: IngestConfig | None = None) -> str:
    """Run the full ingestion pipeline.

    Parameters
    ----------
    config : IngestConfig | None
        Pipeline configuration. Uses defaults if ``None``.

    Returns
    -------
    str
        ClearML Dataset ID of the published dataset.
    """
    config = config or IngestConfig()
    end_date = config.end_date or datetime.now()

    task = Task.init(
        project_name=config.dataset_project,
        task_name=f"ingest-{datetime.now():%Y%m%d-%H%M}",
        task_type=Task.TaskTypes.data_processing,
    )
    task.connect(config, name="ingest_config")

    # ── 1. Fetch all indicators from ESIOS ───────────────────────────────
    raw: dict[int, pd.DataFrame] = {}
    all_to_fetch = ALL_INDICATORS + LEGACY_INDICATORS

    for ind in all_to_fetch:
        print(f"Fetching indicator {ind.id} ({ind.name})...")
        df = fetch_indicator(
            indicator_id=ind.id,
            start_date=config.start_date,
            end_date=end_date,
            # 600 uses geo_id=3 (España), the rest use 8741 (Península)
            geo_id=3 if ind.id == IND_600.id else 8741,
        )
        raw[ind.id] = df
        print(f"  → {len(df)} rows, {df['datetime'].min()} to {df['datetime'].max()}")

    # ── 2. Apply stitching ────────────────────────────────────────────────

    # 634: drop pre-regime data
    raw[634] = apply_634_cutoff(raw[634])
    print(f"634 after cutoff: {len(raw[634])} rows")

    # 2197: backfill using 676/677 + sign(10250)
    raw[2197] = backfill_2197(
        df_676=raw[IND_676.id],
        df_677=raw[IND_677.id],
        df_2197=raw[2197],
        df_10250=raw[10250],
    )
    print(f"2197 after backfill: {len(raw[2197])} rows")

    # ── 3. Write CSVs and publish as ClearML Dataset ─────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)

        for ind in ALL_INDICATORS:
            path = out_dir / f"id_{ind.id}.csv"
            raw[ind.id].to_csv(path, index=False)
            print(f"Wrote {path.name} ({len(raw[ind.id])} rows)")

        ds = Dataset.create(
            dataset_project=config.dataset_project,
            dataset_name=config.dataset_name,
        )
        ds.add_files(str(out_dir))
        ds.finalize(auto_upload=True)
        print(f"Published ClearML Dataset: {ds.id}")

    task.close()
    return ds.id


if __name__ == "__main__":
    run_ingest()
