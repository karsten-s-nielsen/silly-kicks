"""The restart gate contract (`silly_kicks.tracking._restart_validation`, TF-46 Arm 1 Part 2, spec sections 5.1, 15):
the data profile, the gate entry, and the fail-closed lookup over the shipped (empty) table.

Every bar is tested from both sides (spec section 3)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest

from silly_kicks.tracking import _restart_validation as rv
from silly_kicks.tracking._restart_validation import DataProfile, GateEntry

PROFILE = DataProfile(provider="skillcorner", tracking_release="r1", gi_release="g1", competition_season="2022")


def test_gate_is_fail_closed_on_the_empty_shipped_table():
    before = rv.withheld_lookups
    assert rv.gate(PROFILE, "corner", "kick_time", "both") is None
    assert rv.gate(PROFILE, "*", "footprint", "-") is None
    assert rv.withheld_lookups == before + 2
    assert dict(rv.RESTART_GATE_TABLE) == {}


def test_gate_returns_an_injected_entry_without_counting_a_withhold():
    entry = GateEntry("reference", "idsse", 0.95, 0.91, 0.91, 40, 5, "cell")
    before = rv.withheld_lookups
    assert rv.gate(PROFILE, "*", "footprint", "-", table={(PROFILE, "*", "footprint", "-"): entry}) is entry
    assert rv.withheld_lookups == before  # a hit is not a withhold
    other = DataProfile(provider="idsse", tracking_release="r1", gi_release=None, competition_season="2022")
    assert rv.gate(other, "*", "footprint", "-", table={(PROFILE, "*", "footprint", "-"): entry}) is None


def test_data_profile_has_no_pairing_field():
    names = {f.name for f in fields(DataProfile)}
    assert names == {"provider", "tracking_release", "gi_release", "competition_season"}
    assert "pairing" not in names


def test_gate_entry_is_frozen():
    e = GateEntry("reference", "idsse", 0.95, 0.91, 0.91, 40, 5, "cell")
    with pytest.raises(FrozenInstanceError):
        e.metric = 0.5  # type: ignore[misc]
