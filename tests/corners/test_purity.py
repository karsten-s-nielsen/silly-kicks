"""TF-46 corners: local purity checks (the central registry is tests/test_add_star_purity.py)."""

from __future__ import annotations

import pandas as pd

from silly_kicks.corners import add_corner_windows, identify_corners

_CORNER_CROSSED = 5


def _actions():
    return pd.DataFrame(
        {
            "game_id": ["g1"],
            "period_id": [1],
            "time_seconds": [100.0],
            "team_id": ["tA"],
            "player_id": ["p1"],
            "start_x": [104.5],
            "start_y": [1.0],
            "type_id": [_CORNER_CROSSED],
            "result_id": [1],
            "bodypart_id": [0],
        }
    )


def test_identify_corners_is_pure():
    acts = _actions()
    before = acts.copy(deep=True)
    out = identify_corners(acts, provider="sportec")
    pd.testing.assert_frame_equal(acts, before)
    assert out is not acts


def test_add_corner_windows_is_pure():
    corners = identify_corners(_actions(), provider="sportec")
    frames = pd.DataFrame(
        {"frame_id": [5], "period_id": [1], "time_seconds": [100.5], "player_id": ["pW"], "team_id": ["tW"]}
    )
    contacts = pd.DataFrame(
        {
            "period_id": [1],
            "frame_id": [5],
            "time_seconds": [100.5],
            "player_id": ["pW"],
            "contact_source": pd.Series(["velocity_break"], dtype="object"),
        }
    )
    cb, fb, kb = corners.copy(deep=True), frames.copy(deep=True), contacts.copy(deep=True)
    out = add_corner_windows(corners, frames, contacts)
    pd.testing.assert_frame_equal(corners, cb)
    pd.testing.assert_frame_equal(frames, fb)
    pd.testing.assert_frame_equal(contacts, kb)
    assert out is not corners
