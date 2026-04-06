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
