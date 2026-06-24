"""Thin wrapper around the ESIOS REST API."""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

ESIOS_BASE_URL = "https://api.esios.ree.es"
DEFAULT_GEO_ID = 8741  # Península


def _get_token() -> str:
    token = os.environ.get("ESIOS_API_TOKEN")
    if not token:
        raise EnvironmentError(
            "ESIOS_API_TOKEN not set. Copy .env.example to .env and fill it in."
        )
    return token


def fetch_indicator(
    indicator_id: int,
    start_date: datetime,
    end_date: datetime,
    geo_id: int = DEFAULT_GEO_ID,
) -> pd.DataFrame:
    """Fetch a single ESIOS indicator as a DataFrame.

    Parameters
    ----------
    indicator_id : int
        ESIOS indicator ID (e.g. 600, 634, …).
    start_date, end_date : datetime
        Date range (inclusive).
    geo_id : int
        Geographic zone. 8741 = Península, 3 = España.

    Returns
    -------
    pd.DataFrame
        Columns: ``datetime``, ``value``, ``datetime_utc``, ``tz_time``,
        ``geo_id``, ``geo_name``, ``id``.
        ``datetime`` is timezone-aware (Europe/Madrid).
    """
    token = _get_token()
    headers = {"x-api-key": token}

    # ESIOS expects ISO-8601 date strings
    params = {
        "start_date": start_date.strftime("%Y-%m-%dT%H:%M"),
        "end_date": end_date.strftime("%Y-%m-%dT%H:%M"),
        "geo_ids[]": geo_id,
    }

    url = f"{ESIOS_BASE_URL}/indicators/{indicator_id}"
    resp = requests.get(url, headers=headers, params=params, timeout=120)
    resp.raise_for_status()

    data = resp.json()
    values = data.get("indicator", {}).get("values", [])
    if not values:
        return pd.DataFrame(
            columns=["datetime", "value", "datetime_utc", "tz_time", "geo_id", "geo_name", "id"]
        )

    df = pd.DataFrame(values)
    # Normalise column names to match the CSV format from the existing ClearML dataset
    rename_map = {
        "datetime": "datetime",
        "value": "value",
        "datetime_utc": "datetime_utc",
        "tz_time": "tz_time",
        "geo_id": "geo_id",
        "geo_name": "geo_name",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Add indicator id column
    df["id"] = float(indicator_id)

    # Parse datetime to timezone-aware
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert("Europe/Madrid")

    return df
