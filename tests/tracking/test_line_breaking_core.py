"""Characterization + direct-core tests for the extracted _straddle_core (TF-51 v2, N3).

Pins detect_line_breaking's output after the extraction (byte-identical) AND exercises the
home_team_id-free core directly on action-LTR coordinates, so the core and detect_line_breaking
are held to ONE implementation.
"""

from __future__ import annotations

import numpy as np

from tests.tracking.test_defensive_line import _make_frame_rows
from tests.tracking.test_off_ball_runs import _make_action_at


def _three_line_fixture():
    return _make_frame_rows(
        home_outfield_xs=[20.0, 25.0, 30.0, 35.0, 40.0],
        home_outfield_ys=[10.0, 20.0, 34.0, 48.0, 58.0],
        away_outfield_xs=[50.0, 50.0, 50.0, 70.0, 70.0, 70.0, 90.0, 90.0, 90.0, 90.0],
        away_outfield_ys=[15.0, 34.0, 53.0, 15.0, 34.0, 53.0, 10.0, 24.0, 44.0, 58.0],
    )


def test_detect_line_breaking_characterization_unchanged():
    """Pin detect_line_breaking's output after the _straddle_core extraction (byte-identical)."""
    from silly_kicks.tracking._line_breaking import detect_line_breaking

    frames = _three_line_fixture()
    actions = _make_action_at(
        time_seconds=1.0, player_id=2, team_id=1, start_x=10.0, start_y=34.0, end_x=100.0, end_y=34.0
    )
    row = detect_line_breaking(actions, frames).iloc[0]
    assert bool(row["line_break__ward"]) is True
    assert int(row["lines_broken__ward"]) == 3
    assert row["line_breaking_type__ward"] == "between_lines"


def test_straddle_core_direct_action_ltr():
    """_straddle_core on already-action-LTR inputs: pass through 3 lines -> between_lines, n=3."""
    from silly_kicks.tracking._line_breaking import LineBreakingParams, _straddle_core

    opp_x = np.array([50.0, 50.0, 50.0, 70.0, 70.0, 70.0, 90.0, 90.0, 90.0, 90.0])
    opp_y = np.array([15.0, 34.0, 53.0, 15.0, 34.0, 53.0, 10.0, 24.0, 44.0, 58.0])
    is_break, break_type, n = _straddle_core(10.0, 34.0, 100.0, 34.0, opp_x, opp_y, LineBreakingParams())
    assert is_break is True
    assert break_type == "between_lines"
    assert n == 3


def test_straddle_core_short_circuits_return_false():
    """min_pass_length / min_opponents / min_x_spread short-circuits -> (False, None, 0)."""
    from silly_kicks.tracking._line_breaking import LineBreakingParams, _straddle_core

    opp_x = np.array([50.0, 50.0, 50.0, 70.0, 70.0, 70.0, 90.0, 90.0, 90.0, 90.0])
    opp_y = np.array([15.0, 34.0, 53.0, 15.0, 34.0, 53.0, 10.0, 24.0, 44.0, 58.0])
    p = LineBreakingParams()
    # too-short pass
    assert _straddle_core(50.0, 34.0, 51.0, 34.0, opp_x, opp_y, p) == (False, None, 0)
    # too few opponents
    assert _straddle_core(10.0, 34.0, 100.0, 34.0, opp_x[:2], opp_y[:2], p) == (False, None, 0)
    # no x-spread
    flat = np.full(5, 70.0)
    assert _straddle_core(10.0, 34.0, 100.0, 34.0, flat, np.array([10.0, 20.0, 34.0, 48.0, 58.0]), p) == (
        False,
        None,
        0,
    )


def test_straddle_core_and_detect_agree_on_the_same_geometry():
    """The core (fed the same tracking coords detect uses internally) matches detect's output."""
    from silly_kicks.tracking._line_breaking import (
        LineBreakingParams,
        _straddle_core,
        detect_line_breaking,
    )

    frames = _three_line_fixture()
    actions = _make_action_at(
        time_seconds=1.0, player_id=2, team_id=1, start_x=55.0, start_y=34.0, end_x=75.0, end_y=34.0
    )
    detected = detect_line_breaking(actions, frames).iloc[0]

    away = frames[(~frames["is_ball"]) & (~frames["is_goalkeeper"]) & (frames["team_id"] == 2)]
    opp_x = away["x"].to_numpy(dtype="float64")
    opp_y = away["y"].to_numpy(dtype="float64")
    is_break, _break_type, n = _straddle_core(55.0, 34.0, 75.0, 34.0, opp_x, opp_y, LineBreakingParams())
    assert bool(detected["line_break__ward"]) == is_break
    assert int(detected["lines_broken__ward"]) == n
