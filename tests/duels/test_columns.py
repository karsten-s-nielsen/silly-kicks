"""DUEL_COLUMNS -- pinned schema (names + dtypes) for the TF-55 output table."""

from __future__ import annotations

from silly_kicks.duels._columns import (
    DU_WINNER_SOURCE,
    DUEL_COLUMNS,
    DUEL_METRIC_COLUMNS,
    DUEL_WINNER_SOURCE_VALUES,
)


def test_column_set_and_order():
    cols = list(DUEL_COLUMNS)
    assert cols[:2] == ["game_id", "player_id"]
    assert cols[-1] == DU_WINNER_SOURCE
    assert set(DUEL_METRIC_COLUMNS) <= set(cols)
    assert len(cols) == len(set(cols))


def test_dtypes():
    d = DUEL_COLUMNS
    assert d["game_id"] == "object" and d["player_id"] == "object"
    assert d[DU_WINNER_SOURCE] == "object"
    for c in ("duel_rating", "duel_rating_deviation", "duel_volatility"):
        assert d[c] == "float64"
    for c in ("duels_contested", "duels_won", "duels_lost"):
        assert d[c] == "Int64"


def test_winner_source_vocab():
    assert DUEL_WINNER_SOURCE_VALUES == frozenset({"native", "derived"})
