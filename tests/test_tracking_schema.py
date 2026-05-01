"""Schema tests for silly_kicks.tracking — column set, dtype variants, dataclasses."""

import dataclasses

from silly_kicks.tracking.schema import (
    KLOPPY_TRACKING_FRAMES_COLUMNS,
    PFF_TRACKING_FRAMES_COLUMNS,
    SPORTEC_TRACKING_FRAMES_COLUMNS,
    TRACKING_CATEGORICAL_DOMAINS,
    TRACKING_CONSTRAINTS,
    TRACKING_FRAMES_COLUMNS,
    LinkReport,
    TrackingConversionReport,
)


def test_tracking_frames_columns_is_19_columns():
    assert len(TRACKING_FRAMES_COLUMNS) == 19


def test_tracking_frames_columns_required_keys():
    expected = {
        "game_id",
        "period_id",
        "frame_id",
        "time_seconds",
        "frame_rate",
        "player_id",
        "team_id",
        "is_ball",
        "is_goalkeeper",
        "x",
        "y",
        "z",
        "speed",
        "speed_source",
        "ball_state",
        "team_attacking_direction",
        "confidence",
        "visibility",
        "source_provider",
    }
    assert set(TRACKING_FRAMES_COLUMNS) == expected


def test_kloppy_variant_overrides_identifiers_to_object():
    assert KLOPPY_TRACKING_FRAMES_COLUMNS["game_id"] == "object"
    assert KLOPPY_TRACKING_FRAMES_COLUMNS["player_id"] == "object"
    assert KLOPPY_TRACKING_FRAMES_COLUMNS["team_id"] == "object"
    for k, v in TRACKING_FRAMES_COLUMNS.items():
        if k not in {"game_id", "player_id", "team_id"}:
            assert KLOPPY_TRACKING_FRAMES_COLUMNS[k] == v


def test_sportec_variant_matches_kloppy_variant():
    assert SPORTEC_TRACKING_FRAMES_COLUMNS == KLOPPY_TRACKING_FRAMES_COLUMNS


def test_pff_variant_uses_nullable_int64_identifiers():
    assert PFF_TRACKING_FRAMES_COLUMNS["player_id"] == "Int64"
    assert PFF_TRACKING_FRAMES_COLUMNS["team_id"] == "Int64"
    assert PFF_TRACKING_FRAMES_COLUMNS["game_id"] == "int64"


def test_tracking_constraints_keys_subset_of_columns():
    assert set(TRACKING_CONSTRAINTS) <= set(TRACKING_FRAMES_COLUMNS)


def test_tracking_constraints_x_y_match_spadl_field_dimensions():
    assert TRACKING_CONSTRAINTS["x"] == (0, 105.0)
    assert TRACKING_CONSTRAINTS["y"] == (0, 68.0)


def test_tracking_categorical_domains_keys_subset_of_columns():
    assert set(TRACKING_CATEGORICAL_DOMAINS) <= set(TRACKING_FRAMES_COLUMNS)


def test_tracking_categorical_domains_values():
    assert TRACKING_CATEGORICAL_DOMAINS["ball_state"] == frozenset({"alive", "dead"})
    assert TRACKING_CATEGORICAL_DOMAINS["team_attacking_direction"] == frozenset({"ltr", "rtl"})
    assert TRACKING_CATEGORICAL_DOMAINS["speed_source"] == frozenset({"native", "derived"})
    assert TRACKING_CATEGORICAL_DOMAINS["source_provider"] == frozenset({"pff", "sportec", "metrica", "skillcorner"})


def test_tracking_conversion_report_is_frozen_dataclass():
    r = TrackingConversionReport(
        provider="pff",
        total_input_frames=100,
        total_output_rows=2200,
        n_periods=2,
        frame_coverage_per_period={1: 1.0, 2: 0.99},
        ball_out_seconds_per_period={1: 12.4, 2: 8.7},
        nan_rate_per_column={"z": 0.95, "speed": 0.0},
        derived_speed_rows=0,
        unrecognized_player_ids=set(),
    )
    assert dataclasses.is_dataclass(r) and r.__dataclass_params__.frozen


def test_tracking_conversion_report_has_unrecognized():
    r1 = TrackingConversionReport("pff", 0, 0, 0, {}, {}, {}, 0, set())
    assert r1.has_unrecognized is False
    r2 = TrackingConversionReport("pff", 0, 0, 0, {}, {}, {}, 0, {123})
    assert r2.has_unrecognized is True


def test_link_report_link_rate_zero_when_empty():
    r = LinkReport(
        n_actions_in=0,
        n_actions_linked=0,
        n_actions_unlinked=0,
        n_actions_multi_candidate=0,
        per_provider_link_rate={},
        max_time_offset_seconds=0.0,
        tolerance_seconds=0.2,
    )
    assert r.link_rate == 0.0


def test_link_report_link_rate_nonzero():
    r = LinkReport(
        n_actions_in=100,
        n_actions_linked=95,
        n_actions_unlinked=5,
        n_actions_multi_candidate=10,
        per_provider_link_rate={"pff": 0.95},
        max_time_offset_seconds=0.18,
        tolerance_seconds=0.2,
    )
    assert r.link_rate == 0.95
