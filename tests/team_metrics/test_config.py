"""TeamKpiParams + CounterpressWindow (TF-52 Task 1). Mirrors tests/shot_stopping/test_config.py."""

from __future__ import annotations

import dataclasses

import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.team_metrics import COUNTERPRESS_PRESETS, CounterpressWindow, TeamKpiParams


def test_default_is_flagged():
    assert TeamKpiParams.default().is_default() is True
    assert TeamKpiParams().is_default() is False


def test_default_force_universal_is_not_flagged():
    assert TeamKpiParams.default(force_universal=True).is_default() is False


def test_for_provider_empty_map_returns_base():
    assert TeamKpiParams.for_provider("statsbomb") == TeamKpiParams()
    assert TeamKpiParams.for_provider("wyscout") == TeamKpiParams()
    assert TeamKpiParams.for_provider("nonexistent") == TeamKpiParams()


def test_frozen_and_field_defaults():
    p = TeamKpiParams()
    assert p.ppda_zone_fraction == 0.6
    assert p.long_ball_distance_m == 32.0
    assert p.retained_after_seconds == 5.0
    assert p.high_opportunity_xg == 0.15
    assert p.possession_max_gap_seconds == 7.0
    assert p.possession_retain_on_set_pieces is True
    assert p.counterpress_seconds == 5.0
    assert p.post_recovery_window_seconds == 10.0
    assert p.switch_min_lateral_m == 30.0
    assert p.defensive_action_types == ("tackle", "interception", "foul")
    assert p.counterpress_window == CounterpressWindow(seconds=5.0)
    assert p.channel_boundaries == (spadlconfig.field_width / 3.0, 2.0 * spadlconfig.field_width / 3.0)
    assert p.build_up_zone_max_x == spadlconfig.field_length / 3.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.ppda_zone_fraction = 0.5  # type: ignore[reportAttributeAccessIssue]  # frozen: runtime raise is the point


def test_counterpress_window_xor():
    assert CounterpressWindow(seconds=6.0).seconds == 6.0
    assert CounterpressWindow(passes=3).passes == 3
    with pytest.raises(ValueError):
        CounterpressWindow(seconds=5.0, passes=3)  # both set
    with pytest.raises(ValueError):
        CounterpressWindow()  # neither set


def test_presets_are_windows():
    assert COUNTERPRESS_PRESETS["tigres_hunt"] == CounterpressWindow(passes=3)
    assert COUNTERPRESS_PRESETS["barcelona"] == CounterpressWindow(seconds=6.0)
    assert COUNTERPRESS_PRESETS["leipzig"] == CounterpressWindow(seconds=10.0)
    # every preset is a valid (XOR-satisfying) window
    for window in COUNTERPRESS_PRESETS.values():
        assert isinstance(window, CounterpressWindow)
        assert (window.seconds is None) != (window.passes is None)


def test_for_provider_is_universal_default_false():
    # A per-provider config is a hand-built one, not the universal default (mirrors shot_stopping).
    assert TeamKpiParams.for_provider("statsbomb").is_default() is False
