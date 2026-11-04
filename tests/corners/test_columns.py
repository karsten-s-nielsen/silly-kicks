"""TF-46 corners column-schema shape gates."""

from __future__ import annotations

from silly_kicks.corners import (
    CORNER_CHANCE_COLUMNS,
    CORNER_CHANCE_METRIC_COLUMNS,
    CORNER_IDENTIFY_COLUMNS,
    CORNER_KEYS,
    CORNER_WINDOW_COLUMNS,
)
from silly_kicks.corners._columns import (
    CORNER_KIND_VALUES,
    CORNER_SIDE_VALUES,
    CORNER_SOURCE_VALUES,
    TARGET_SNAPSHOT_SOURCE_VALUES,
)


def test_keys_are_plain_list():
    assert CORNER_KEYS == ["game_id", "corner_id"]


def test_chance_metric_columns_is_plain_list_of_value_cols():
    assert isinstance(CORNER_CHANCE_METRIC_COLUMNS, list)
    assert CORNER_CHANCE_METRIC_COLUMNS == ["n_shots", "n_goals", "sum_xg"]


def test_chance_columns_superset_of_metric_plus_keys():
    assert set(CORNER_CHANCE_METRIC_COLUMNS) <= set(CORNER_CHANCE_COLUMNS)
    assert set(CORNER_KEYS) <= set(CORNER_CHANCE_COLUMNS)


def test_every_schema_is_a_dtype_string_dict():
    for schema in (CORNER_IDENTIFY_COLUMNS, CORNER_WINDOW_COLUMNS, CORNER_CHANCE_COLUMNS):
        assert isinstance(schema, dict)
        assert all(isinstance(c, str) and isinstance(t, str) for c, t in schema.items())


def test_keys_lead_the_identify_and_chance_schemas():
    assert list(CORNER_IDENTIFY_COLUMNS)[:2] == CORNER_KEYS
    assert list(CORNER_CHANCE_COLUMNS)[:2] == CORNER_KEYS


def test_window_columns_are_additive_not_overlapping_identify():
    assert set(CORNER_WINDOW_COLUMNS).isdisjoint(set(CORNER_IDENTIFY_COLUMNS))


def test_closed_vocabularies_are_frozensets():
    assert CORNER_SIDE_VALUES == frozenset({"left", "right"})
    assert CORNER_KIND_VALUES == frozenset({"direct", "short", "unclear"})
    assert CORNER_SOURCE_VALUES == frozenset({"skillcorner_marker", "spadl_corner", "sb360_event"})
    assert TARGET_SNAPSHOT_SOURCE_VALUES == frozenset({"first_contact", "kick_plus_2s", "sb360_event"})
