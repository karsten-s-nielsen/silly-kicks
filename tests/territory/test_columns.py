"""TERRITORY_COLUMNS -- pinned schema (names + dtypes) for the TF-54 output table."""

from __future__ import annotations

from silly_kicks.territory._columns import (
    TERRITORY_COLUMNS,
    TERRITORY_HULL_SOURCE_VALUES,
    TERRITORY_METHODS,
    TERRITORY_METRIC_COLUMNS,
    TR_HULL_SOURCE,
)


def test_column_set_and_order():
    cols = list(TERRITORY_COLUMNS)
    assert cols[:2] == ["game_id", "player_id"]
    assert cols[-1] == TR_HULL_SOURCE
    # every metric column is present, exactly once.
    assert set(TERRITORY_METRIC_COLUMNS) <= set(cols)
    assert len(cols) == len(set(cols))


def test_dtypes():
    d = TERRITORY_COLUMNS
    assert d["game_id"] == "object" and d["player_id"] == "object"
    assert d[TR_HULL_SOURCE] == "object"
    assert d["territory_passes_into_hull"] == "Int64"
    assert d["territory_defensive_actions_in_hull"] == "Int64"
    for c in (
        "territory_xt_conceded",
        "territory_xt_prevented",
        "territory_xt_net",
        "territory_hull_area_m2",
        "territory_xt_conceded_rate",
    ):
        assert d[c] == "float64"


def test_method_family_and_source_vocab():
    assert TERRITORY_METHODS == frozenset({"completed_failed", "counterfactual"})
    assert TERRITORY_HULL_SOURCE_VALUES == frozenset({"resolved", "degenerate", "no_actions"})
