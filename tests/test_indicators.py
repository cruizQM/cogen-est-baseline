"""Sanity checks on indicator metadata."""

from cogen_est_baseline.data.indicators import (
    ALL_INDICATORS,
    FAMILY_GROUPS,
    LEGACY_INDICATORS,
    TARGET_INDICATORS,
)


def test_no_duplicate_ids():
    all_ids = [ind.id for ind in ALL_INDICATORS + LEGACY_INDICATORS]
    assert len(all_ids) == len(set(all_ids))


def test_target_count():
    assert len(TARGET_INDICATORS) == 6


def test_family_groups_cover_all_targets():
    grouped_ids = {ind.id for inds in FAMILY_GROUPS.values() for ind in inds}
    target_ids = {ind.id for ind in TARGET_INDICATORS}
    assert grouped_ids == target_ids


def test_legacy_indicators_are_676_677():
    legacy_ids = {ind.id for ind in LEGACY_INDICATORS}
    assert legacy_ids == {676, 677}
