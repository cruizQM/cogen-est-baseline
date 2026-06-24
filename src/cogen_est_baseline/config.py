"""Plain-dataclass configuration — no Hydra, ClearML-native."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class IngestConfig:
    """Configuration for ESIOS data ingestion."""

    # Date range for historical data fetch
    start_date: datetime = field(default_factory=lambda: datetime(2022, 5, 1))
    end_date: datetime | None = None  # None → fetch up to now

    # ClearML dataset metadata
    dataset_project: str = "cogen-est-baseline"
    dataset_name: str = "esios-balancing-indicators"


@dataclass
class BandConfig:
    """Configuration for PMD band discretisation."""

    band_width: float = 5.0  # €/MWh per band
    band_min: float = -20.0  # lower edge of first band
    band_max: float = 250.0  # upper edge of last band (everything above → last band)


@dataclass
class BaselineConfig:
    """Configuration for the P50 conditional-lookup baseline."""

    band: BandConfig = field(default_factory=BandConfig)

    # Quantiles to compute in the lookup tables (P50 = baseline, P10/P90 = bands)
    quantiles: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])


@dataclass
class EvalConfig:
    """Configuration for evaluation splits and metrics."""

    # Train/test split date: everything before is used to build lookup tables,
    # everything after is held out for evaluation.
    split_date: datetime = field(default_factory=lambda: datetime(2025, 6, 1))
