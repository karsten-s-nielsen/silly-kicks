"""TF-58 Task 14: coordination vocabularies, signal catalogue and output schemas."""

from __future__ import annotations

import silly_kicks.coordination._columns as col
from silly_kicks.coordination._columns import (
    COORD_AXES,
    COORD_LEVELS,
    COORD_METHOD_FAMILIES,
    COORD_SIGNALS,
    COORD_SOURCE_VALUES,
    COORD_SURROGATE_SOURCE_VALUES,
    COORD_WINDOW_KINDS,
    RP_HIST,
    TEAM_SIGNALS,
    SignalSpec,
)

_FAMILIES = [
    ("pair", col.COORDINATION_PAIR_KEYS, col.COORDINATION_PAIR_METRIC_COLUMNS, col.COORDINATION_PAIR_COLUMNS),
    (
        "pair_phase",
        col.COORDINATION_PAIR_PHASE_KEYS,
        col.COORDINATION_PAIR_PHASE_METRIC_COLUMNS,
        col.COORDINATION_PAIR_PHASE_COLUMNS,
    ),
    (
        "spectral",
        col.COORDINATION_SPECTRAL_KEYS,
        col.COORDINATION_SPECTRAL_METRIC_COLUMNS,
        col.COORDINATION_SPECTRAL_COLUMNS,
    ),
    (
        "cluster_team",
        col.COORDINATION_CLUSTER_TEAM_KEYS,
        col.COORDINATION_CLUSTER_TEAM_METRIC_COLUMNS,
        col.COORDINATION_CLUSTER_TEAM_COLUMNS,
    ),
    (
        "cluster_player",
        col.COORDINATION_CLUSTER_PLAYER_KEYS,
        col.COORDINATION_CLUSTER_PLAYER_METRIC_COLUMNS,
        col.COORDINATION_CLUSTER_PLAYER_COLUMNS,
    ),
    (
        "team_sync",
        col.COORDINATION_TEAM_SYNC_KEYS,
        col.COORDINATION_TEAM_SYNC_METRIC_COLUMNS,
        col.COORDINATION_TEAM_SYNC_COLUMNS,
    ),
    ("rsi", col.COORDINATION_RSI_KEYS, col.COORDINATION_RSI_METRIC_COLUMNS, col.COORDINATION_RSI_COLUMNS),
]


def test_columns_start_with_keys_in_order():
    for name, keys, _metric, columns in _FAMILIES:
        assert list(columns)[: len(keys)] == list(keys), name


def test_metric_subset_of_columns_and_all_coord_prefixed():
    for name, _keys, metric, columns in _FAMILIES:
        assert set(metric) <= set(columns), name
        assert all(m.startswith("coord_") for m in metric), name


def test_metric_and_provenance_disjoint_and_provenance_shape():
    for name, keys, metric, columns in _FAMILIES:
        provenance = set(columns) - set(keys) - set(metric)
        assert set(metric).isdisjoint(provenance), name
        for c in provenance:
            assert c.endswith("_source") or c in {"team_a_id", "team_b_id"}, (name, c)


def test_vocabularies_are_dedup_tuples():
    for vocab in (
        COORD_LEVELS,
        COORD_WINDOW_KINDS,
        COORD_AXES,
        COORD_METHOD_FAMILIES,
        COORD_SOURCE_VALUES,
        COORD_SURROGATE_SOURCE_VALUES,
        TEAM_SIGNALS,
    ):
        assert isinstance(vocab, tuple)
        assert len(vocab) == len(set(vocab))


def test_coord_signals_has_15_entries_with_specs():
    assert len(COORD_SIGNALS) == 15
    assert all(isinstance(spec, SignalSpec) for spec in COORD_SIGNALS.values())
    assert len(TEAM_SIGNALS) == 12
    assert COORD_SIGNALS["convex_hull_area"].unit == "m^2"
    assert COORD_SIGNALS["possession"].scope == "match"
    assert COORD_SIGNALS["player_x"].scope == "player"


def test_rp_hist_is_twelve():
    assert len(RP_HIST) == 12


def test_counts_are_nullable_int():
    for name, _keys, _metric, columns in _FAMILIES:
        for count_col in ("coord_n_samples", "coord_n_segments", "coord_vc_n_stationary", "coord_coh_n_segments"):
            if count_col in columns:
                assert columns[count_col] == "Int64", (name, count_col)


def test_window_id_and_period_id_dtypes():
    assert col.COORD_WINDOW_COLUMNS["window_id"] == "Int64"
    assert col.COORD_WINDOW_COLUMNS["period_id"] == "int64"
    assert col.COORD_WINDOW_COLUMNS["game_id"] == "object"


def test_pair_ids_are_object():
    for c in ("team_a_id", "team_b_id", "player_a_id", "player_b_id"):
        assert col.COORDINATION_PAIR_COLUMNS[c] == "object"
