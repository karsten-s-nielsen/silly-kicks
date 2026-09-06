"""TerritoryReport -- counterfactual census fields (SPEC-04 Task 8).

n_target_modeled / n_target_unresolved are additive, defaulted fields so every v1 construction
(positional or keyword) still works unchanged. Documented cf identity:
n_target_modeled + n_target_unresolved == (failed passes considered into the aimed set).
"""

from __future__ import annotations

from silly_kicks.territory import TerritoryParams
from silly_kicks.territory._report import TerritoryReport


def test_new_fields_default_to_zero():
    r = TerritoryReport(
        TerritoryParams(),
        n_players_in=5,
        n_scored=4,
        n_degenerate_hull=1,
        n_no_actions=0,
        n_passes_considered=200,
        n_passes_into_hull=37,
    )
    assert r.n_target_modeled == 0
    assert r.n_target_unresolved == 0


def test_v1_construction_still_works_positionally():
    # Pre-existing positional construction (7 args) must remain valid with the new fields defaulted.
    r = TerritoryReport(TerritoryParams(), 5, 4, 1, 0, 200, 37)
    assert r.n_scored + r.n_degenerate_hull + r.n_no_actions == r.n_players_in
    assert r.n_target_modeled == 0
    assert r.n_target_unresolved == 0


def test_cf_identity_on_hand_built_report():
    # 12 failed passes considered into the aimed set: 9 resolved to a modeled target, 3 unresolved.
    failed_passes_considered_into_aimed_set = 12
    r = TerritoryReport(
        TerritoryParams(),
        n_players_in=3,
        n_scored=3,
        n_degenerate_hull=0,
        n_no_actions=0,
        n_passes_considered=50,
        n_passes_into_hull=20,
        n_target_modeled=9,
        n_target_unresolved=3,
    )
    assert r.n_target_modeled + r.n_target_unresolved == failed_passes_considered_into_aimed_set
