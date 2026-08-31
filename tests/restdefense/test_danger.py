"""Layer-2 danger metrics (TF-60 PR2, ADR-081)."""

import math

import numpy as np

from silly_kicks.restdefense import RD_LAYER2_COLUMNS, RestDefenseParams
from silly_kicks.restdefense._danger import layer2_metrics
from silly_kicks.restdefense._structure import SampleContext
from silly_kicks.tracking import resolve_defended_goals
from tests.restdefense._fixtures import make_keeper_sensitive_fixture


def _ctx_and_frame():
    _actions, frames, xt = make_keeper_sensitive_fixture()
    gm = resolve_defended_goals(frames)
    fid = frames["frame_id"].iloc[0]
    frame_rows = frames[frames["frame_id"] == fid]
    ctx = SampleContext(
        team_id=1,
        opponent_id=2,
        ball_x=float(frame_rows[frame_rows["is_ball"]]["x"].iloc[0]),
        own_goal_x=0.0,
        attacked_goal_x=105.0,
        defensive_line_x=28.0,
        compactness_x=math.nan,
        lateral_width=math.nan,
        team_length=math.nan,
    )
    return frame_rows, ctx, gm, xt


def test_all_five_columns_finite_with_fitted_xt():
    frame_rows, ctx, gm, xt = _ctx_and_frame()
    m = layer2_metrics(frame_rows, ctx, xt=xt, goal_map=gm, params=RestDefenseParams())
    assert set(m) == set(RD_LAYER2_COLUMNS)
    for c in RD_LAYER2_COLUMNS:
        assert np.isfinite(m[c]), f"{c} is not finite"


def test_space_control_and_coverage_are_fractions():
    frame_rows, ctx, gm, xt = _ctx_and_frame()
    m = layer2_metrics(frame_rows, ctx, xt=xt, goal_map=gm, params=RestDefenseParams())
    assert 0.0 <= m["rd_attacker_space_control"] <= 1.0
    assert 0.0 <= m["rd_gk_coverage_behind_line"] <= 1.0


def test_all_columns_nan_without_xt():
    """P2-02: Layer 2 is gated ENTIRELY on xt -- no xt -> all five NaN, before any pitch-control call."""
    frame_rows, ctx, gm, _xt = _ctx_and_frame()
    m = layer2_metrics(frame_rows, ctx, xt=None, goal_map=gm, params=RestDefenseParams())
    for c in RD_LAYER2_COLUMNS:
        assert math.isnan(m[c]), f"{c} should be NaN without xt (Layer-2 gate)"


def test_gk_dependent_columns_nan_when_A_keeper_absent():
    frame_rows, ctx, gm, xt = _ctx_and_frame()
    no_gk = frame_rows[~((frame_rows["team_id"] == 1) & frame_rows["is_goalkeeper"].astype(bool))]
    m = layer2_metrics(no_gk, ctx, xt=xt, goal_map=gm, params=RestDefenseParams())
    assert math.isnan(m["rd_danger_behind_line_gk"])
    assert math.isnan(m["rd_gk_coverage_behind_line"])
    assert math.isnan(m["rd_gk_reachable_coverage_m2"])
    assert np.isfinite(m["rd_danger_behind_line"])  # GK-blind base still computes
    assert np.isfinite(m["rd_attacker_space_control"])  # xt-free (but gated) still computes


def test_unresolvable_zone_yields_all_nan():
    frame_rows, ctx, gm, xt = _ctx_and_frame()
    bad_ctx = SampleContext(
        team_id=1,
        opponent_id=2,
        ball_x=ctx.ball_x,
        own_goal_x=float("nan"),
        attacked_goal_x=105.0,
        defensive_line_x=float("nan"),
        compactness_x=float("nan"),
        lateral_width=float("nan"),
        team_length=float("nan"),
    )
    m = layer2_metrics(frame_rows, bad_ctx, xt=xt, goal_map=gm, params=RestDefenseParams())
    for c in RD_LAYER2_COLUMNS:
        assert math.isnan(m[c])
