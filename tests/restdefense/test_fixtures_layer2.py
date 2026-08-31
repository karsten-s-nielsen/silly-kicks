"""Self-checks for the Layer-2 fixtures (TF-60 PR2)."""

from tests.restdefense._fixtures import (
    make_fitted_xt,
    make_keeper_sensitive_fixture,
    make_rest_defense_fixture,
)


def test_frames_carry_velocity():
    _actions, frames = make_rest_defense_fixture()
    assert {"vx", "vy"} <= set(frames.columns)


def test_fitted_xt_is_fitted():
    from silly_kicks.xthreat import require_fitted_xt

    require_fitted_xt(make_fitted_xt(), caller="test")  # does not raise


def test_keeper_sensitive_fixture_has_receiver_behind_A_rearguard():
    _actions, frames, _xt = make_keeper_sensitive_fixture()
    f = frames[frames["frame_id"] == frames["frame_id"].iloc[0]]
    outfield = f[(f["team_id"] == 1) & ~f["is_goalkeeper"].astype(bool) & ~f["is_ball"].astype(bool)]
    a_line = outfield["x"].nsmallest(4).max()
    a_gk = f[(f["team_id"] == 1) & f["is_goalkeeper"].astype(bool)]["x"].iloc[0]
    b_deep = f[(f["team_id"] == 2) & ~f["is_ball"].astype(bool)]["x"].min()
    assert a_gk < b_deep < a_line  # B receiver between A's keeper and A's back line
    assert {"vx", "vy"} <= set(frames.columns)
