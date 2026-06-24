"""ESIOS indicator metadata, regime-change dates, and family groupings."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class IndicatorFamily(str, Enum):
    """Mechanism-based grouping of balancing market indicators."""

    PMD = "pmd"
    AFRR = "afrr"  # secondary regulation
    MFRR = "mfrr"  # manual tertiary regulation
    TC = "tc"  # technical constraints


@dataclass(frozen=True)
class IndicatorMeta:
    """Metadata for a single ESIOS indicator."""

    id: int
    name: str
    family: IndicatorFamily
    unit: str
    frequency_minutes: int = 15
    description: str = ""


# ── Target indicators ────────────────────────────────────────────────────────

# PMD — day-ahead market price (driver, not a target)
IND_600 = IndicatorMeta(
    id=600,
    name="Precio mercado SPOT España",
    family=IndicatorFamily.PMD,
    unit="€/MWh",
    frequency_minutes=60,
    description="Day-ahead wholesale electricity price (hourly, expanded to 15-min).",
)

# aFRR — secondary regulation
IND_634 = IndicatorMeta(
    id=634,
    name="Precio reserva regulación secundaria a bajar",
    family=IndicatorFamily.AFRR,
    unit="€/MW",
    description=(
        "Secondary reserve capacity price (down). "
        "Used from 20/11/2024 onwards (previously covered both directions)."
    ),
)
IND_682 = IndicatorMeta(
    id=682,
    name="Precio energía regulación secundaria a subir",
    family=IndicatorFamily.AFRR,
    unit="€/MWh",
    description="Secondary regulation energy price (up).",
)
IND_683 = IndicatorMeta(
    id=683,
    name="Precio energía regulación secundaria a bajar",
    family=IndicatorFamily.AFRR,
    unit="€/MWh",
    description="Secondary regulation energy price (down).",
)

# mFRR — manual tertiary regulation
IND_2197 = IndicatorMeta(
    id=2197,
    name="Precio energías balance mFRR activación programada",
    family=IndicatorFamily.MFRR,
    unit="€/MWh",
    description=(
        "mFRR scheduled activation price. "
        "Replaced 676/677 on 10/12/2024; backfilled using 676/677 + sign(10250)."
    ),
)
IND_10250 = IndicatorMeta(
    id=10250,
    name="Volumen neto asignación energías mFRR",
    family=IndicatorFamily.MFRR,
    unit="MWh",
    description=(
        "Net mFRR energy assignment volume. "
        "Positive = up activation, negative = down activation."
    ),
)

# TC — technical constraints
IND_708 = IndicatorMeta(
    id=708,
    name="Precio medio restricciones técnicas diario Fase II bajar",
    family=IndicatorFamily.TC,
    unit="€/MWh",
    description="Average technical constraints price (Phase II, down).",
)

# ── Legacy indicators (used only for backfilling 2197) ───────────────────────

IND_676 = IndicatorMeta(
    id=676,
    name="Precio marginal regulación terciaria a bajar de AP",
    family=IndicatorFamily.MFRR,
    unit="€/MWh",
    description="Legacy mFRR down price. Discontinued 10/12/2024.",
)
IND_677 = IndicatorMeta(
    id=677,
    name="Precio marginal regulación terciaria a subir de AP",
    family=IndicatorFamily.MFRR,
    unit="€/MWh",
    description="Legacy mFRR up price. Discontinued 10/12/2024.",
)

# ── Convenience collections ──────────────────────────────────────────────────

TARGET_INDICATORS = [IND_634, IND_682, IND_683, IND_708, IND_2197, IND_10250]
ALL_INDICATORS = [IND_600] + TARGET_INDICATORS
LEGACY_INDICATORS = [IND_676, IND_677]

# Indicators to fetch from ESIOS for ingestion (includes legacy for backfill)
INGEST_INDICATORS = ALL_INDICATORS + LEGACY_INDICATORS

# ── Regime-change dates ──────────────────────────────────────────────────────

# 634 only valid from this date (previously covered both up and down)
REGIME_634_START = datetime(2024, 11, 20)

# 2197 started on this date; before it, use 676/677 + sign(10250) to backfill
REGIME_2197_START = datetime(2024, 12, 10)

# ── Indicator family groupings (for joint-model experiments) ─────────────────

FAMILY_GROUPS: dict[IndicatorFamily, list[IndicatorMeta]] = {
    IndicatorFamily.AFRR: [IND_634, IND_682, IND_683],
    IndicatorFamily.MFRR: [IND_2197, IND_10250],
    IndicatorFamily.TC: [IND_708],
}
