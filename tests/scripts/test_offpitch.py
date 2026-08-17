"""Unit tests for the shared off-pitch mask (keeper-box detection-quality cycle, A1 Step 3a)."""

from __future__ import annotations

import numpy as np

from scripts._offpitch import OFF_PITCH_MARGIN_M, off_pitch_mask


def test_default_margin_is_two_metres():
    assert OFF_PITCH_MARGIN_M == 2.0


def test_point_past_touchline_is_offpitch():
    # 3 m past the y=0 sideline, at the default 2.0 m margin -> off-pitch
    assert bool(off_pitch_mask(np.array([50.0]), np.array([-3.0]))[0])


def test_keeper_just_behind_goal_line_is_not_offpitch():
    # 1 m behind the x=0 goal line is within the 2.0 m tolerance -> NOT off-pitch
    assert not bool(off_pitch_mask(np.array([-1.0]), np.array([34.0]))[0])


def test_on_pitch_point_is_not_offpitch():
    assert not bool(off_pitch_mask(np.array([52.5]), np.array([34.0]))[0])


def test_vectorized_over_mixed_points():
    x = np.array([52.5, -3.0, 108.0, 50.0])
    y = np.array([34.0, 34.0, 34.0, 71.0])
    got = off_pitch_mask(x, y)
    assert got.tolist() == [False, True, True, True]


def test_margin_is_overridable():
    # at a 5 m margin a point 3 m past the line is within tolerance
    assert not bool(off_pitch_mask(np.array([50.0]), np.array([-3.0]), margin_m=5.0)[0])
