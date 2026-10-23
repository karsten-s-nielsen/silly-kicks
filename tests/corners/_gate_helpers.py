"""Shared first-contact gate injection for the corner-window tests (TF-46 Arm 1 Part 2, spec section 9.2).

The commit-2 ``RESTART_GATE_TABLE`` is empty, so ``add_corner_windows`` withholds every winner by default.
These helpers inject a table whose lookup passes for every ``first_contact`` key, so a test can exercise the
LIVE (gated-through) path as well as the withheld default.
"""

from __future__ import annotations

from silly_kicks.tracking._restart_validation import DataProfile, GateEntry
from tests.tracking._restart_fixtures import GateTableStub

PROFILE = DataProfile(provider="skillcorner", tracking_release="r1", gi_release="g1", competition_season="cs1")


def _entry() -> GateEntry:
    return GateEntry(
        tier="transfer", base="idsse", metric=0.9, wilson_lb=0.82, bootstrap_lb=0.80, n=50, m=5, cell_resolution="cell"
    )


class PassAllFirstContact(GateTableStub):
    """A gate table whose ``.get`` returns a passing entry for every ``first_contact`` key."""

    def get(self, key, default=None):
        return _entry() if isinstance(key, tuple) and len(key) == 4 and key[2] == "first_contact" else default


def passing() -> dict:
    """Kwargs that pass the first-contact gate: ``add_corner_windows(..., **passing())``."""
    return {"data_profile": PROFILE, "gate_table": PassAllFirstContact()}
