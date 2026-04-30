"""Tests for SPADL and Atomic-SPADL schema constants and ConversionReport."""

import pytest

from silly_kicks.atomic.spadl.schema import ATOMIC_SPADL_COLUMNS
from silly_kicks.spadl.schema import (
    KLOPPY_SPADL_COLUMNS,
    SPADL_COLUMNS,
    SPADL_CONSTRAINTS,
    SPADL_NAME_COLUMNS,
    ConversionReport,
)


def test_spadl_columns_count():
    assert len(SPADL_COLUMNS) == 14


def test_atomic_spadl_columns_count():
    assert len(ATOMIC_SPADL_COLUMNS) == 13


def test_spadl_columns_are_strings():
    for col, dtype in SPADL_COLUMNS.items():
        assert isinstance(col, str)
        assert isinstance(dtype, str)


def test_spadl_name_columns():
    assert set(SPADL_NAME_COLUMNS.keys()) == {"type_name", "result_name", "bodypart_name"}


def test_kloppy_overrides_id_columns():
    assert KLOPPY_SPADL_COLUMNS["game_id"] == "object"
    assert KLOPPY_SPADL_COLUMNS["team_id"] == "object"
    assert KLOPPY_SPADL_COLUMNS["player_id"] == "object"
    assert KLOPPY_SPADL_COLUMNS["start_x"] == SPADL_COLUMNS["start_x"]


def test_spadl_constraints_cover_coordinate_columns():
    for col in ["start_x", "start_y", "end_x", "end_y"]:
        assert col in SPADL_CONSTRAINTS


def test_conversion_report_creation():
    report = ConversionReport(
        provider="StatsBomb",
        total_events=100,
        total_actions=80,
        mapped_counts={"Pass": 50, "Shot": 10, "Duel": 20},
        excluded_counts={"Pressure": 15, "Substitution": 5},
        unrecognized_counts={},
    )
    assert report.provider == "StatsBomb"
    assert report.total_events == 100
    assert report.total_actions == 80
    assert not report.has_unrecognized


def test_conversion_report_has_unrecognized():
    report = ConversionReport(
        provider="StatsBomb",
        total_events=10,
        total_actions=5,
        mapped_counts={"Pass": 5},
        excluded_counts={"Pressure": 3},
        unrecognized_counts={"NewEventType": 2},
    )
    assert report.has_unrecognized


def test_conversion_report_is_frozen():
    report = ConversionReport(
        provider="test",
        total_events=0,
        total_actions=0,
        mapped_counts={},
        excluded_counts={},
        unrecognized_counts={},
    )
    with pytest.raises(AttributeError):
        report.provider = "other"


def test_config_df_caching():
    """O-15: Config DataFrame factories should return cached instances."""
    import silly_kicks.spadl.config as spadlcfg

    assert spadlcfg.actiontypes_df() is spadlcfg.actiontypes_df()
    assert spadlcfg.results_df() is spadlcfg.results_df()
    assert spadlcfg.bodyparts_df() is spadlcfg.bodyparts_df()


def test_atomic_config_df_caching():
    """O-15: Atomic config DataFrame factories should return cached instances."""
    import silly_kicks.atomic.spadl.config as atomicconfig

    assert atomicconfig.actiontypes_df() is atomicconfig.actiontypes_df()
    assert atomicconfig.bodyparts_df() is atomicconfig.bodyparts_df()


def test_pff_spadl_columns_extends_spadl_columns():
    """PFF_SPADL_COLUMNS is SPADL_COLUMNS + 4 tackle-passthrough columns."""
    from silly_kicks.spadl.schema import PFF_SPADL_COLUMNS, SPADL_COLUMNS

    expected_extras = {
        "tackle_winner_player_id": "Int64",
        "tackle_winner_team_id": "Int64",
        "tackle_loser_player_id": "Int64",
        "tackle_loser_team_id": "Int64",
    }
    # SPADL_COLUMNS appears verbatim, in order, at the front
    assert list(PFF_SPADL_COLUMNS.keys())[: len(SPADL_COLUMNS)] == list(SPADL_COLUMNS.keys())
    # 4 extras follow, in declared order
    extras_part = {k: PFF_SPADL_COLUMNS[k] for k in list(PFF_SPADL_COLUMNS.keys())[len(SPADL_COLUMNS) :]}
    assert extras_part == expected_extras


def test_pff_spadl_columns_uses_int64_extension_dtype():
    """The 4 tackle-passthrough columns use pandas nullable Int64 (capital I)."""
    from silly_kicks.spadl.schema import PFF_SPADL_COLUMNS

    for col in (
        "tackle_winner_player_id",
        "tackle_winner_team_id",
        "tackle_loser_player_id",
        "tackle_loser_team_id",
    ):
        assert PFF_SPADL_COLUMNS[col] == "Int64"


def test_pff_spadl_columns_exported_from_top_level():
    """PFF_SPADL_COLUMNS is reachable from `silly_kicks.spadl`."""
    from silly_kicks.spadl import PFF_SPADL_COLUMNS

    assert "tackle_winner_player_id" in PFF_SPADL_COLUMNS
