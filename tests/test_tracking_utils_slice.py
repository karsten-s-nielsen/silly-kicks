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
                    "source_provider": "gradientsports",
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
                "source_provider": "gradientsports",
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


# ---------------------------------------------------------------------------
# PR-S52: OOM fix — searchsorted replaces cartesian merge
# ---------------------------------------------------------------------------


def test_high_framerate_no_cartesian_blowup():
    """5000 frames x 200 actions per period must not create cartesian intermediate.

    Cartesian merge would produce 5000*3*200 = 3M intermediate rows;
    searchsorted produces only ~200*26*3 = ~15,600 rows (1s window at 25fps).
    """
    frames = _frames(1, 5000, hz=25.0)  # 5000 frames, 3 entities = 15,000 rows
    actions = pd.DataFrame([_action(i, 1, 1.0 + i * 0.5) for i in range(200)])
    out = slice_around_event(actions, frames, pre_seconds=0.5, post_seconds=0.5)
    assert len(out) > 0
    # Each action gets ~26 frames, 3 entities = ~78 rows (within 1s window at 25fps)
    # 200 actions * 78 = ~15,600 total, but some share frames near boundaries
    assert len(out) < 200 * 30 * 3 + 1000  # well under cartesian 3M
    assert set(out.columns) == {*frames.columns, "action_id", "time_offset_seconds"}
    # Every row's time_offset_seconds must be within [-0.5, +0.5]
    assert out["time_offset_seconds"].min() >= -0.5 - 1e-9
    assert out["time_offset_seconds"].max() <= 0.5 + 1e-9


def test_multi_action_per_period_correct_windows():
    """Each action in the same period gets its own independent window."""
    frames = _frames(1, 100, hz=25.0)  # 0.00 .. 3.96s
    actions = pd.DataFrame(
        [
            _action(0, 1, 0.5),  # window [0.0, 1.0]
            _action(1, 1, 2.0),  # window [1.5, 2.5]
        ]
    )
    out = slice_around_event(actions, frames, pre_seconds=0.5, post_seconds=0.5)

    a0 = out[out["action_id"] == 0]
    a1 = out[out["action_id"] == 1]

    # Action 0: frames with time in [0.0, 1.0] → frame_ids 0..25
    a0_frame_ids = a0["frame_id"].unique()
    a0_times = a0.drop_duplicates("frame_id")["time_seconds"]
    assert (a0_times >= 0.0 - 1e-9).all()
    assert (a0_times <= 1.0 + 1e-9).all()

    # Action 1: frames with time in [1.5, 2.5] → frame_ids 37..62
    a1_frame_ids = a1["frame_id"].unique()
    a1_times = a1.drop_duplicates("frame_id")["time_seconds"]
    assert (a1_times >= 1.5 - 1e-9).all()
    assert (a1_times <= 2.5 + 1e-9).all()

    # The two windows should NOT overlap (gap at [1.0, 1.5])
    assert len(set(a0_frame_ids) & set(a1_frame_ids)) == 0


def test_boundary_inclusivity():
    """Verify that frame times exactly at window edges are included (>= / <=)."""
    # 5 frames at 0.0, 0.5, 1.0, 1.5, 2.0
    frames = _frames(1, 5, hz=2.0)
    actions = pd.DataFrame([_action(0, 1, 1.0)])
    out = slice_around_event(actions, frames, pre_seconds=1.0, post_seconds=1.0)
    # Window is [0.0, 2.0] — all 5 frames should be included
    assert set(out["frame_id"].unique()) == {0, 1, 2, 3, 4}


def test_output_columns_match_frames_plus_action_id():
    """Output must contain all frames columns plus action_id and time_offset_seconds."""
    frames = _frames(1, 10)
    actions = pd.DataFrame([_action(0, 1, 0.1)])
    out = slice_around_event(actions, frames, pre_seconds=0.5, post_seconds=0.5)
    expected_cols = set(frames.columns) | {"action_id", "time_offset_seconds"}
    assert set(out.columns) == expected_cols


def test_multi_period_scale():
    """Two periods x 2000 frames x 100 actions each -- verifies period isolation at scale."""
    p1 = _frames(1, 2000, hz=25.0)
    p2 = _frames(2, 2000, hz=25.0, t0=0.0)
    frames = pd.concat([p1, p2], ignore_index=True)
    actions = pd.DataFrame(
        [_action(i, 1, 1.0 + i * 0.3) for i in range(100)] + [_action(100 + i, 2, 1.0 + i * 0.3) for i in range(100)]
    )
    out = slice_around_event(actions, frames, pre_seconds=0.5, post_seconds=0.5)
    # Each action's frames must be in its own period
    for _, row in actions.iterrows():
        action_rows = out[out["action_id"] == row["action_id"]]
        if len(action_rows) > 0:
            assert (action_rows["period_id"] == row["period_id"]).all()
