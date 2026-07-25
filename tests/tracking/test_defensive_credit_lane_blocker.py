"""Item 2 lane-geometry shot_block blocker (TF-51 v2, spec section 4).

Three-fixture discriminator: credit the defender in the shot->goal corridor, NOT the one nearest the
origin; exclude the goalkeeper (by flag AND distance-along-lane); drop the origin-proximity threshold
so a far-but-on-lane blocker is credited.
"""

from __future__ import annotations

from typing import get_args

import numpy as np
import pandas as pd

from silly_kicks.tracking.defensive_credit import DefensiveCreditParams, compute_defensive_credits
from silly_kicks.tracking.defensive_credit._resolution import Mode
from tests.tracking._defensive_credit_fixtures import frame_with_defender, one_action

_SHOT_BLOCK = frozenset({"shot_block"})


def test_mode_literal_is_a_closed_set():
    assert set(get_args(Mode)) == {"nearest", "all_within", "all_within_beyond_nearest", "lane_blocker"}


def _shot_frame(defenders, *, acting_team_id=10, defender_team_id=20, home_team_id=10):
    """One frame: (x, y, is_gk) opponent defenders + a team-10 acting player + the ball (home=10)."""
    common = dict(
        game_id="g1", period_id=1, frame_id=500, time_seconds=50.0, vx=0.0, vy=0.0,
        home_team_id=home_team_id, source_provider="test",
    )  # fmt: skip
    def_dir = "ltr" if defender_team_id == home_team_id else "rtl"
    act_dir = "ltr" if acting_team_id == home_team_id else "rtl"
    rows = [
        {
            **common, "team_id": defender_team_id, "player_id": 900 + i, "x": float(x), "y": float(y),
            "is_ball": False, "is_goalkeeper": bool(is_gk), "team_attacking_direction": def_dir,
        }
        for i, (x, y, is_gk) in enumerate(defenders)
    ]  # fmt: skip
    rows.append(
        {
            **common, "team_id": acting_team_id, "player_id": 800, "x": 52.0, "y": 34.0,
            "is_ball": False, "is_goalkeeper": False, "team_attacking_direction": act_dir,
        }
    )  # fmt: skip
    rows.append(
        {
            **common, "team_id": np.nan, "player_id": np.nan, "x": 90.0, "y": 34.0,
            "is_ball": True, "is_goalkeeper": False, "team_attacking_direction": act_dir,
        }
    )  # fmt: skip
    return pd.DataFrame(rows)


def _shot_action(*, start_x=85.0, start_y=34.0):
    a = one_action(
        type_name="shot", result_name="fail", start_x=start_x, start_y=start_y,
        team_id=10, player_id=5, time_seconds=50.0,
    )  # fmt: skip
    a["shot_blocked"] = pd.array([True], dtype="boolean")
    a["cross_blocked"] = pd.array([pd.NA], dtype="boolean")
    a["shot_on_target_derived"] = pd.array([pd.NA], dtype="boolean")
    a["xg"] = [0.2]
    return a


def _run(a, f, fitted_xt):
    return compute_defensive_credits(
        a, f, xg_column="xg", xt=fitted_xt, params=DefensiveCreditParams(rules=_SHOT_BLOCK)
    )


def _shot_block_row(out):
    b = out[out["rule"] == "shot_block"]
    return b.iloc[0] if not b.empty else None


def test_lane_blocker_credits_on_lane_not_nearest_origin(fitted_xt):
    # A (900): near origin (4.1 m) but OFF the lane (y=38); B (901): dead on the lane but 10 m away.
    a = _shot_action(start_x=85.0, start_y=34.0)
    f = _shot_frame([(86.0, 38.0, False), (95.0, 34.0, False)])
    block = _shot_block_row(_run(a, f, fitted_xt))
    assert block is not None
    assert block["player_id"] == 901  # v1 credits the near-origin A (900); v2 credits the on-lane B


def test_lane_blocker_excludes_goalkeeper(fitted_xt):
    # GK (900) on the lane, nearest the origin; an outfielder (901) also on the lane, farther.
    a = _shot_action(start_x=85.0, start_y=34.0)
    f = _shot_frame([(86.0, 34.0, True), (95.0, 34.0, False)])
    assert bool(f[f["player_id"] == 900]["is_goalkeeper"].iloc[0]) is True  # N5: flag genuinely set
    block = _shot_block_row(_run(a, f, fitted_xt))
    assert block is not None
    assert block["player_id"] == 901  # NOT the GK (v1 would credit the nearest-origin GK)


def test_lane_blocker_credits_far_but_on_lane_defender(fitted_xt):
    # A single defender 10 m from the origin, dead on the lane -> credited (origin threshold dropped).
    a = _shot_action(start_x=85.0, start_y=34.0)
    f = _shot_frame([(95.0, 34.0, False)])
    block = _shot_block_row(_run(a, f, fitted_xt))
    assert block is not None  # v1 finds nobody within the 4.5 m origin threshold -> no row
    assert block["player_id"] == 900
    assert block["resolution"] == "lane"


def test_resolution_records_nearest_fallback(fitted_xt):
    # A defender near the origin but OFF the lane -> no corridor blocker -> nearest-to-origin fallback.
    a = _shot_action(start_x=85.0, start_y=34.0)
    f = _shot_frame([(86.0, 38.0, False)])
    block = _shot_block_row(_run(a, f, fitted_xt))
    assert block is not None
    assert block["player_id"] == 900
    assert block["resolution"] == "nearest_fallback"


def test_anchor_actor_recorded_on_passer_debit(fitted_xt):
    # The pressure_pass_fail -passer credits the acting-team actor -> resolution == "anchor_actor";
    # the +presser is proximity-resolved -> "nearest".
    a = one_action(type_name="pass", result_name="fail", start_x=95.0, start_y=34.0, team_id=10, player_id=5)
    a["shot_blocked"] = pd.array([pd.NA], dtype="boolean")
    a["cross_blocked"] = pd.array([pd.NA], dtype="boolean")
    a["shot_on_target_derived"] = pd.array([pd.NA], dtype="boolean")
    a["xg"] = [np.nan]
    f = frame_with_defender(defender_x=96.0, defender_y=34.0)
    out = compute_defensive_credits(
        a, f, xg_column="xg", xt=fitted_xt, params=DefensiveCreditParams(rules=frozenset({"pressure_pass_fail"}))
    )
    assert (out[out["player_id"] == 5]["resolution"] == "anchor_actor").all()  # -passer
    assert (out[out["player_id"] == 900]["resolution"] == "nearest").all()  # +presser
