"""mFRR energy direction sampling with block persistence.

Reproduces the Excel's approach: for each (month, solar-group), compute the
historical probability of Up vs. Down direction, then sample quarter-hourly
directions using persistence-aware block sampling (continuous blocks of the
predominant direction, positioned randomly within each time group).

TODO: implement after the basic (hour, PMD-band) lookup baseline is validated.
"""

from __future__ import annotations

raise_stub = True


def sample_mfrr_direction() -> None:
    """Sample mFRR Up/Down direction for a forecast horizon."""
    raise NotImplementedError(
        "mFRR direction sampling not yet implemented. "
        "The basic P50 lookup baseline treats 2197 and 10250 with the same "
        "(hour, PMD-band) grouping as other indicators."
    )
