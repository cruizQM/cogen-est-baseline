"""Load indicator DataFrames from a ClearML Dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from clearml import Dataset


def get_dataset_path(
    dataset_id: str | None = None,
    dataset_project: str = "cogen-est-baseline",
    dataset_name: str = "esios-balancing-indicators",
    dataset_version: str | None = None,
) -> Path:
    """Get local path to a ClearML Dataset, downloading if needed.

    Parameters
    ----------
    dataset_id : str | None
        ClearML dataset ID. If provided, takes precedence over
        project/name/version.
    dataset_project, dataset_name : str
        ClearML dataset identifiers. Ignored if ``dataset_id`` is set.
    dataset_version : str | None
        Specific version. ``None`` → latest. Ignored if ``dataset_id`` is set.

    Returns
    -------
    Path
        Local directory containing the dataset files.
    """
    if dataset_id is not None:
        ds = Dataset.get(dataset_id=dataset_id)
    else:
        ds = Dataset.get(
            dataset_project=dataset_project,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
        )
    return Path(ds.get_local_copy())


def load_indicator(
    dataset_path: Path,
    indicator_id: int | str,
) -> pd.DataFrame:
    """Load a single indicator CSV from a dataset directory.

    Parameters
    ----------
    dataset_path : Path
        Local dataset root (from ``get_dataset_path``).
    indicator_id : int | str
        Indicator ID or filename stem (e.g. 634, "estimated_600").

    Returns
    -------
    pd.DataFrame
        With ``datetime`` as timezone-aware DatetimeIndex and a ``value`` column.
    """
    csv_path = dataset_path / f"id_{indicator_id}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Indicator file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    # Parse datetime handling mixed UTC offsets (summer +02:00 / winter +01:00)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert("Europe/Madrid")
    return df


def load_all_indicators(
    dataset_path: Path,
) -> dict[str, pd.DataFrame]:
    """Load all indicator CSVs found in a dataset directory.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keyed by filename stem (e.g. "600", "634", "estimated_600").
    """
    result = {}
    for csv_path in sorted(dataset_path.glob("id_*.csv")):
        stem = csv_path.stem.removeprefix("id_")
        result[stem] = load_indicator(dataset_path, stem)
    return result
