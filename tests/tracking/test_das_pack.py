"""Task 3 gates: the DAS input port -- contract, keys, ordering, direction, reason codes, sentinel."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._das_pack import Reason, pack_frames
from silly_kicks.tracking._das_taxonomy import DasUnscoreableError
from tests.tracking._das_golden import load_golden
from tests.tracking._das_helpers import single_frame


def _pack(df, **kw):
    kw.setdefault("attacking_direction_col", "dir")
    return pack_frames(df, **kw)


# --- fail-loud contract (spec 6.7) ---------------------------------------------------------------


def test_duplicate_player_rows_raise():
    df = single_frame()
    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        _pack(dup)


def test_goal_map_and_direction_col_mutually_exclusive():
    df = single_frame()
    with pytest.raises(ValueError, match="not both"):
        pack_frames(df, goal_map=object(), attacking_direction_col="dir")  # type: ignore[arg-type]  # wrong type is the point


def test_direction_values_must_be_plus_minus_one():
    df = single_frame(direction=0.5)
    with pytest.raises(ValueError, match=r"\+1/-1"):
        _pack(df)


def test_possession_varying_within_frame_raises():
    df = single_frame()
    df.loc[df.index[:5], "team_in_possession"] = 2
    with pytest.raises(ValueError, match="team_in_possession varies"):
        _pack(df)


def test_carrier_varying_within_frame_raises():
    df = single_frame()
    df.loc[df.index[:5], "ball_carrier_player_id"] = 999
    with pytest.raises(ValueError, match="carrier column varies"):
        _pack(df)


def test_two_ball_rows_raise():
    df = single_frame()
    ball = df[df["is_ball"]].iloc[[0]].copy()
    with pytest.raises(ValueError, match="more than one ball"):
        _pack(pd.concat([df, ball], ignore_index=True))


def test_missing_velocity_without_marker_raises_valueerror():
    df = single_frame().drop(columns=["vx", "vy"])
    with pytest.raises(ValueError, match="velocity columns"):
        _pack(df)


def test_velocity_unavailable_marker_degrades_before_missing_team_in_possession():
    """A velocity-less-BY-DESIGN source (SB360 freeze-frame) degrades even when
    team_in_possession is ALSO absent -- the honest 'no velocity here' wins over a generic
    missing-column error, so add_das NaN-degrades (ADR-063) rather than raising. Regression
    for the validation-order bug that raised plain ValueError('team_in_possession') and made
    run_tracking_features drop the whole add_das family on SB360."""
    from silly_kicks.tracking._das_taxonomy import DAS_SOURCE_UNSCOREABLE_FRAME
    from silly_kicks.tracking.schema import SPEED_SOURCE_UNAVAILABLE

    df = single_frame().drop(columns=["vx", "vy", "team_in_possession"])
    df["speed_source"] = SPEED_SOURCE_UNAVAILABLE
    with pytest.raises(DasUnscoreableError) as excinfo:
        _pack(df)
    assert excinfo.value.das_source == DAS_SOURCE_UNSCOREABLE_FRAME


def test_all_nan_possession_raises_unscoreable():
    df = single_frame()
    df["team_in_possession"] = pd.NA
    df["team_in_possession"] = df["team_in_possession"].astype("Int64")
    with pytest.raises(DasUnscoreableError):
        _pack(df)


# --- keys, ordering, dtypes, purity, sentinel ----------------------------------------------------


def test_keys_include_game_and_period_no_conflation():
    g = load_golden()
    p = pack_frames(g.frames_for("S03"), attacking_direction_col="dir")
    # S03 has two periods with disjoint frame ids -> 6 distinct frames, none conflated.
    assert p.n_frames == 6
    assert p.keys["period_id"].nunique() == 2


def test_two_games_not_conflated():
    g = load_golden()
    p = pack_frames(g.frames_for("S04"), attacking_direction_col="dir")
    assert p.keys["game_id"].nunique() == 2


def test_packed_kinematics_are_float64_from_float32_storage():
    df = single_frame()
    assert str(df["x"].dtype) == "float32"
    p = _pack(df)
    assert p.px.dtype == p.py.dtype == p.pvx.dtype == p.pvy.dtype == np.float64


def test_coords_are_centred():
    df = single_frame(ball_xy=(52.5, 34.0))
    p = _pack(df)
    assert np.allclose(p.ball_xy[0], (0.0, 0.0), atol=1e-6)


def test_input_unmodified_and_player_id_never_written():
    df = single_frame()
    before = df.copy(deep=True)
    _pack(df)
    pd.testing.assert_frame_equal(df, before)


def test_category_player_id_packs():
    df = single_frame()
    df["player_id"] = df["player_id"].astype("category")
    p = _pack(df)  # no raise
    assert p.n_frames == 1


def test_players_ordered_by_canonical_id_within_frame():
    df = single_frame(n1=3, n2=2)
    p = _pack(df)
    ids = df.loc[p.p_input_pos, "player_id"].to_numpy()
    assert list(ids) == [100, 101, 102, 200, 201]


def test_attacking_mask_matches_possession():
    df = single_frame(poss=1)
    p = _pack(df)
    teams = df.loc[p.p_input_pos, "team_id"].to_numpy()
    assert np.array_equal(p.p_attacking, teams == 1)


def test_passer_mask_marks_the_carrier():
    df = single_frame(carrier=100)
    p = _pack(df)
    ids = df.loc[p.p_input_pos, "player_id"].to_numpy()
    assert np.array_equal(p.p_is_passer, ids == 100)


# --- direction ------------------------------------------------------------------------------------


def test_direction_from_column_sign():
    assert _pack(single_frame(direction=1.0)).direction[0] == 1.0
    assert _pack(single_frame(direction=-1.0)).direction[0] == -1.0


def test_direction_from_goal_map():
    # team 1 GK deep at x=4 -> team 1 defends x=0, attacks x=105 -> +1 when team 1 in possession.
    df = single_frame(poss=1)
    df["is_goalkeeper"] = False
    df.loc[df["player_id"] == 100, ["x", "is_goalkeeper"]] = [4.0, True]
    df.loc[df["player_id"] == 200, ["x", "is_goalkeeper"]] = [101.0, True]
    p = pack_frames(df, goal_map=None)  # builds GoalMap from frames
    assert p.direction[0] == 1.0


# --- reason codes ---------------------------------------------------------------------------------


def test_reason_ok_on_normal_frame():
    assert _pack(single_frame()).reason[0] == Reason.OK


def test_reason_no_possession():
    df = single_frame()
    # a second frame with NaN possession -> that frame is NO_POSSESSION, the live one stays OK
    dead = single_frame(frame=1)
    dead["team_in_possession"] = pd.NA
    dead["team_in_possession"] = dead["team_in_possession"].astype("Int64")
    dead["dir"] = np.nan
    p = _pack(pd.concat([df, dead], ignore_index=True))
    order = {tuple(k): i for i, k in enumerate(p.keys[["game_id", "period_id", "frame_id"]].to_numpy())}
    assert p.reason[order[(1, 1, 1)]] == Reason.NO_POSSESSION
    assert p.reason[order[(1, 1, 0)]] == Reason.OK


def test_reason_no_ball():
    df = single_frame()
    df = df[~df["is_ball"]].reset_index(drop=True)
    assert _pack(df).reason[0] == Reason.NO_BALL


def test_reason_ball_nan():
    df = single_frame()
    df.loc[df["is_ball"], ["x", "y"]] = np.nan
    assert _pack(df).reason[0] == Reason.BALL_NAN


def test_reason_possession_team_absent():
    df = single_frame(poss=1)
    df["team_in_possession"] = 9
    df["team_in_possession"] = df["team_in_possession"].astype("Int64")
    df["dir"] = 1.0
    assert _pack(df).reason[0] == Reason.POSSESSION_TEAM_ABSENT


def test_reason_direction_unresolved():
    df = single_frame()
    df["dir"] = np.nan
    assert _pack(df).reason[0] == Reason.DIRECTION_UNRESOLVED
