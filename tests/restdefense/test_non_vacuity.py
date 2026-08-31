"""Non-vacuity: the GK-inclusion and w_field re-weighting measurably move the danger (CLAUDE.md)."""

import numpy as np

from silly_kicks.restdefense import RestDefenseParams
from silly_kicks.restdefense._danger import layer2_metrics
from silly_kicks.restdefense._structure import SampleContext
from silly_kicks.tracking import resolve_defended_goals
from tests.restdefense._fixtures import make_keeper_sensitive_fixture


def _ctx(frame_rows):
    return SampleContext(
        team_id=1,
        opponent_id=2,
        ball_x=float(frame_rows[frame_rows["is_ball"]]["x"].iloc[0]),
        own_goal_x=0.0,
        attacked_goal_x=105.0,
        defensive_line_x=28.0,
        compactness_x=float("nan"),
        lateral_width=float("nan"),
        team_length=float("nan"),
    )


def test_gk_inclusion_measurably_reduces_deep_danger():
    """On a keeper-sensitive frame (a B receiver in behind A's rearguard), including A's keeper as a
    control agent LOWERS B's deep danger: base (GK-blind) > gk (GK-included), by a real margin."""
    _actions, frames, xt = make_keeper_sensitive_fixture()
    fr = frames[frames["frame_id"] == frames["frame_id"].iloc[0]]
    m = layer2_metrics(fr, _ctx(fr), xt=xt, goal_map=resolve_defended_goals(frames), params=RestDefenseParams())
    base, gk = m["rd_danger_behind_line"], m["rd_danger_behind_line_gk"]
    assert np.isfinite(base) and np.isfinite(gk)
    assert base > gk, f"keeper did not deter: base={base} gk={gk}"
    assert (base - gk) > 1e-6, "GK contribution is vacuously ~0 -- keeper is screened; fix the fixture"


def test_w_field_measurably_changes_the_danger():
    _actions, frames, xt = make_keeper_sensitive_fixture()
    fr = frames[frames["frame_id"] == frames["frame_id"].iloc[0]]
    ctx, gm = _ctx(fr), resolve_defended_goals(frames)
    off = layer2_metrics(fr, ctx, xt=xt, goal_map=gm, params=RestDefenseParams(danger_field_weight=False))
    on = layer2_metrics(fr, ctx, xt=xt, goal_map=gm, params=RestDefenseParams(danger_field_weight=True))
    assert off["rd_danger_behind_line"] != on["rd_danger_behind_line"]
    # w_field weights in (0,1] -> it can only down-weight; the deep zone stays a real fraction of it.
    assert 0.0 < on["rd_danger_behind_line"] <= off["rd_danger_behind_line"] + 1e-9
