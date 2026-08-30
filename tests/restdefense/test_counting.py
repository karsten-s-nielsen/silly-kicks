"""count_goalside (TF-60, ADR-080) -- the behind-the-ball counting primitive."""

import pandas as pd

from silly_kicks.restdefense._counting import count_goalside


def _rows(team_id, xs, gk_flags=None):
    gk_flags = gk_flags if gk_flags is not None else [False] * len(xs)
    return pd.DataFrame(
        {
            "team_id": team_id,
            "player_id": range(len(xs)),
            "x": xs,
            "y": 34.0,
            "is_ball": False,
            "is_goalkeeper": gk_flags,
        }
    )


def test_counts_goalside_reference_goal_low_x():
    # reference goal G at 0, ball at 40; goal-side = players with x in [0, 40]
    rows = _rows(3, [5.0, 20.0, 45.0])  # two in [0, 40], one beyond the ball
    assert count_goalside(rows, team_id=3, ball_x=40.0, goal_x=0.0) == 2


def test_counts_goalside_reference_goal_high_x():
    rows = _rows(3, [100.0, 85.0, 60.0])  # G=105, ball 65 -> x in [65, 105] => 100, 85
    assert count_goalside(rows, team_id=3, ball_x=65.0, goal_x=105.0) == 2


def test_include_gk_flag():
    rows = _rows(3, [5.0, 2.0], gk_flags=[False, True])  # both goal-side; GK is one of them
    assert count_goalside(rows, team_id=3, ball_x=40.0, goal_x=0.0, include_gk=True) == 2
    assert count_goalside(rows, team_id=3, ball_x=40.0, goal_x=0.0, include_gk=False) == 1


def test_only_counts_the_named_team():
    a = _rows(3, [5.0, 20.0])
    b = _rows(4, [10.0, 15.0])
    both = pd.concat([a, b], ignore_index=True)
    assert count_goalside(both, team_id=3, ball_x=40.0, goal_x=0.0) == 2
    assert count_goalside(both, team_id=4, ball_x=40.0, goal_x=0.0) == 2


def test_string_team_id_matches_numeric_column():  # ADR-019
    rows = _rows(3, [5.0])
    rows["team_id"] = rows["team_id"].astype("Int64")
    assert count_goalside(rows, team_id="3", ball_x=40.0, goal_x=0.0) == 1


def test_ball_row_with_na_team_is_excluded():  # ADR-058: ball rows carry NA team_id
    rows = _rows(3, [5.0, 20.0])
    ball = pd.DataFrame(
        {
            "team_id": pd.array([pd.NA], dtype="Int64"),
            "player_id": pd.array([pd.NA], dtype="Int64"),
            "x": [10.0],
            "y": [34.0],
            "is_ball": [True],
            "is_goalkeeper": pd.array([pd.NA], dtype="boolean"),
        }
    )
    rows["team_id"] = rows["team_id"].astype("Int64")
    with_ball = pd.concat([rows, ball], ignore_index=True)
    assert count_goalside(with_ball, team_id=3, ball_x=40.0, goal_x=0.0) == 2


def test_nonfinite_x_is_ignored():
    rows = _rows(3, [5.0, float("nan"), 20.0])
    assert count_goalside(rows, team_id=3, ball_x=40.0, goal_x=0.0) == 2


def test_empty_frame_is_zero():
    rows = _rows(3, [])
    assert count_goalside(rows, team_id=3, ball_x=40.0, goal_x=0.0) == 0


def test_count_goalside_by_sample_matches_scalar_per_row():
    from silly_kicks.restdefense._counting import count_goalside_by_sample

    frames = pd.DataFrame(
        {
            "game_id": [1, 1, 1, 1, 1, 1],
            "period_id": [1, 1, 1, 1, 1, 1],
            "frame_id": [10, 10, 10, 11, 11, 11],
            "team_id": pd.array([1, 1, 2, 1, 2, 2], dtype="Int64"),
            "player_id": pd.array([1, 2, 3, 4, 5, 6], dtype="Int64"),
            "x": [5.0, 20.0, 30.0, 45.0, 10.0, 15.0],
            "y": 34.0,
            "is_ball": False,
            "is_goalkeeper": pd.array([False] * 6, dtype="boolean"),
        }
    )
    samples = pd.DataFrame(
        {
            "game_id": [1, 1],
            "period_id": [1, 1],
            "frame_id": [10, 11],
            "team_id": pd.array([1, 2], dtype="Int64"),
            "ball_x": [40.0, 40.0],
            "own_goal_x": [0.0, 0.0],
        }
    )
    out = count_goalside_by_sample(samples, frames)
    # frame 10, team 1: x in [0,40] -> {5,20} = 2 ; frame 11, team 2: x in [0,40] -> {10,15} = 2
    assert out.tolist() == [2, 2]
    assert str(out.dtype) == "Int64"


def test_count_goalside_by_sample_threaded_groups_match_self_built():
    from silly_kicks._frame_index import group_rows
    from silly_kicks.restdefense._columns import RD_FRAME_KEYS
    from silly_kicks.restdefense._counting import count_goalside_by_sample

    frames, samples = _scaling_like()
    self_built = count_goalside_by_sample(samples, frames)
    threaded = count_goalside_by_sample(samples, frames, groups=group_rows(frames, tuple(RD_FRAME_KEYS)))
    assert self_built.tolist() == threaded.tolist()


def _scaling_like():
    from tests.restdefense._fixtures import make_scaling_fixture

    return make_scaling_fixture(6)
