"""Unit tests for silly_kicks.tracking.utils.slice_around_event."""

import pandas as pd

from silly_kicks.tracking.utils import link_actions_to_frames, slice_around_event


def _frames(period_id: int, n: int, hz: float = 25.0, t0: float = 0.0) -> pd.DataFrame:
    rows = []
    for i in range(n):
        for player_id in (7, 8):
            rows.append(
                {
                    "game_id": 1,
                    "period_id": period_id,
                    "frame_id": i,
                    "time_seconds": t0 + i / hz,
                    "frame_rate": hz,
                    "player_id": player_id,
                    "team_id": 100,
                    "is_ball": False,
                    "is_goalkeeper": False,
                    "x": 50.0,
                    "y": 34.0,
                    "z": float("nan"),
                    "speed": 5.0,
                    "speed_source": "native",
                    "ball_state": "alive",
                    "team_attacking_direction": "ltr",
                    "confidence": None,
                    "visibility": None,
                    "source_provider": "pff",
                }
            )
        rows.append(
            {
                "game_id": 1,
                "period_id": period_id,
                "frame_id": i,
                "time_seconds": t0 + i / hz,
                "frame_rate": hz,
                "player_id": pd.NA,
                "team_id": pd.NA,
                "is_ball": True,
                "is_goalkeeper": False,
                "x": 50.0,
                "y": 34.0,
                "z": 0.5,
                "speed": 8.0,
                "speed_source": "native",
                "ball_state": "alive",
                "team_attacking_direction": None,
                "confidence": None,
                "visibility": None,
                "source_provider": "pff",
            }
        )
    return pd.DataFrame(rows)


def _action(action_id, period_id, t):
    return {
        "game_id": 1,
        "action_id": action_id,
        "period_id": period_id,
        "time_seconds": t,
        "team_id": 100,
        "player_id": 7,
        "type_id": 0,
        "result_id": 1,
        "bodypart_id": 0,
        "start_x": 50.0,
        "start_y": 34.0,
        "end_x": 60.0,
        "end_y": 34.0,
    }


def test_zero_window_returns_one_frame_per_action():
    frames = _frames(1, 5)
    actions = pd.DataFrame([_action(0, 1, 0.04)])
    out = slice_around_event(actions, frames, pre_seconds=0.0, post_seconds=0.0)
    assert len(out) == 3
    assert (out["action_id"] == 0).all()
    assert (out["frame_id"] == 1).all()


def test_half_second_window_returns_full_neighbourhood():
    frames = _frames(1, 50, hz=25.0)
    actions = pd.DataFrame([_action(0, 1, 1.0)])
    out = slice_around_event(actions, frames, pre_seconds=0.5, post_seconds=0.5)
    # ~26 frames * 3 rows = ~78 rows expected (boundary inclusivity 0.5/0.04 = 12.5 frames each side + center)
    assert 70 <= len(out) <= 85


def test_window_does_not_cross_periods():
    p1 = _frames(1, 25, hz=25.0)
    p2 = _frames(2, 25, hz=25.0, t0=0.0)
    frames = pd.concat([p1, p2], ignore_index=True)
    actions = pd.DataFrame([_action(0, 1, 0.96)])
    out = slice_around_event(actions, frames, pre_seconds=1.0, post_seconds=1.0)
    assert (out["period_id"] == 1).all()


def test_zero_window_consistent_with_link_actions_to_frames():
    """slice_around_event(pre=0, post=0) should yield same frame_id set as link."""
    frames = _frames(1, 25, hz=25.0)
    actions = pd.DataFrame(
        [
            _action(0, 1, 0.04),
            _action(1, 1, 0.40),
        ]
    )
    pointers, _ = link_actions_to_frames(actions, frames, tolerance_seconds=0.05)
    sliced = slice_around_event(actions, frames, pre_seconds=0.0, post_seconds=0.0)
    linked_frame_ids = pointers.dropna(subset=["frame_id"])["frame_id"].astype(int).tolist()
    sliced_frame_ids = sliced["frame_id"].drop_duplicates().tolist()
    assert set(linked_frame_ids) == set(sliced_frame_ids)


def test_empty_intersection_returns_empty():
    frames = _frames(1, 5)
    actions = pd.DataFrame([_action(0, 2, 0.0)])
    out = slice_around_event(actions, frames)
    assert len(out) == 0
