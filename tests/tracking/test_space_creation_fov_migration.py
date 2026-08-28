"""M4 (ADR-077, Task 7): space_creation opponent-perspective softening on the REAL FOV signal.

Before this cycle, ``compute_space_created`` softened a one-team opponent-perspective frame when the
frame declared velocity structurally unavailable (``velocity_unavailable_by_design``) -- a PRAGMATIC
PROXY for "this is an SB360 FOV crop" (ADR-054 amendment). The proxy is retired here: softening is now
driven by the action's REAL per-action FOV signal -- soften iff the ``visible_area`` polygon is present
and FOV-cropped (observed pitch fraction below the full-coverage floor); otherwise raise.

These fixtures DECOUPLE velocity from FOV (today's providers conflate them), which is the whole point:
a velocity-less frame with FULL coverage must now RAISE (it is not a crop), and a velocity-BEARING
frame with a CROPPED polygon must now SOFTEN. Under the OLD proxy both would behave oppositely.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._velocity_availability import velocity_unavailable_by_design
from silly_kicks.tracking.features import add_space_creation
from silly_kicks.tracking.schema import SPEED_SOURCE_UNAVAILABLE
from silly_kicks.xthreat import ExpectedThreat


def _xt() -> ExpectedThreat:
    # A tiny fitted xT avoids the escalated SyntheticEPVWarning (same precedent as the NaN-safety
    # gate's obso/pausa branch); it does not affect the opponent-resolution path under test.
    xt = ExpectedThreat(l=16, w=12)
    xt.xT = np.tile(np.linspace(0.0, 1.0, 16), (12, 1))
    return xt


def _stamp(df: pd.DataFrame, *, velocity_less: bool) -> pd.DataFrame:
    df["game_id"] = 1
    df["period_id"] = 1
    df["frame_id"] = 10
    df["time_seconds"] = 5.0
    df["ball_state"] = "alive"
    df["team_attacking_direction"] = "ltr"
    df["source_provider"] = "synthetic"
    if velocity_less:
        # SB360 declared-velocity-less shape: no vx/vy, marker on every row -> zero-velocity model.
        df["speed"] = np.nan
        df["speed_source"] = SPEED_SOURCE_UNAVAILABLE
    else:
        # Velocity-BEARING: vx/vy present (any value); the marker is NOT set.
        df["speed"] = 0.0
        df["speed_source"] = "derived"
        df["vx"] = 0.0
        df["vy"] = 0.0
    return df


def _one_team_frame(*, velocity_less: bool) -> pd.DataFrame:
    rows = []
    x = 30.0
    for j in range(2):  # >=2 attacking players -> a non-trivial leave-one-out
        rows.append(
            dict(player_id=100 + j, team_id=1, is_ball=False, is_goalkeeper=(j == 0), x=x, y=30.0 + 8 * j, z=0.0)
        )
        x += 6.0
    rows.append(dict(player_id=-1, team_id=-1, is_ball=True, is_goalkeeper=False, x=52.5, y=34.0, z=0.0))
    return _stamp(pd.DataFrame(rows), velocity_less=velocity_less)


def _two_team_frame(*, velocity_less: bool) -> pd.DataFrame:
    rows = []
    x = 30.0
    for t in (1, 2):
        for j in range(2):
            rows.append(
                dict(
                    player_id=100 * t + j,
                    team_id=t,
                    is_ball=False,
                    is_goalkeeper=(j == 0),
                    x=x,
                    y=30.0 + 8 * j,
                    z=0.0,
                )
            )
            x += 6.0
    rows.append(dict(player_id=-1, team_id=-1, is_ball=True, is_goalkeeper=False, x=52.5, y=34.0, z=0.0))
    return _stamp(pd.DataFrame(rows), velocity_less=velocity_less)


def _actions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [1],
            "period_id": [1],
            "action_id": [1],
            "time_seconds": [5.0],
            "team_id": [1],
            "player_id": [100],  # a team-1 player present in every frame above
            "start_x": [30.0],
            "start_y": [30.0],
        }
    )


def _full_pitch_va(actions: pd.DataFrame) -> pd.DataFrame:
    poly = np.array([[0.0, 0.0], [105.0, 0.0], [105.0, 68.0], [0.0, 68.0]])
    return pd.DataFrame({"action_id": list(actions["action_id"]), "polygon": [poly] * len(actions)})


def _left_half_va(actions: pd.DataFrame) -> pd.DataFrame:
    poly = np.array([[0.0, 0.0], [52.5, 0.0], [52.5, 68.0], [0.0, 68.0]])
    return pd.DataFrame({"action_id": list(actions["action_id"]), "polygon": [poly] * len(actions)})


def test_velocity_less_but_full_coverage_now_raises():
    # velocity-unavailable one-team frame BUT full-pitch visible_area -> NOT an FOV crop -> must RAISE.
    # Under the OLD velocity proxy this SOFTENED (velocity_unavailable_by_design == True).
    a = _actions()
    f = _one_team_frame(velocity_less=True)
    with pytest.raises(ValueError):
        add_space_creation(a, f, home_team_id=1, xt=_xt(), visible_area=_full_pitch_va(a))


def test_velocity_bearing_but_cropped_now_softens():
    # velocity-BEARING one-team frame BUT half-pitch visible_area -> a real FOV crop -> SOFTEN, not raise.
    # Under the OLD velocity proxy this RAISED (velocity_unavailable_by_design == False).
    a = _actions()
    f = _one_team_frame(velocity_less=False)
    out = add_space_creation(a, f, home_team_id=1, xt=_xt(), visible_area=_left_half_va(a))
    assert (out["space_opponent_source"] == "unresolved_one_team").all()
    assert out["space_denied_m2_opponent"].isna().all()  # opponent unresolvable -> NaN, no raise
    assert out["space_created_m2"].notna().any()  # team side still computes


def test_behaviour_flips_versus_old_velocity_proxy():
    # The same two inputs under the OLD proxy would behave OPPOSITELY. Assert the NEW result differs
    # from the OLD proxy's decision on BOTH -- a fixture where old==new would prove nothing.
    a = _actions()
    f_velless_full = _one_team_frame(velocity_less=True)
    f_velbearing_cropped = _one_team_frame(velocity_less=False)

    # What velocity_unavailable_by_design ALONE (the old proxy) would have decided:
    assert velocity_unavailable_by_design(f_velless_full) is True  # OLD: soften
    assert velocity_unavailable_by_design(f_velbearing_cropped) is False  # OLD: raise

    # NEW behaviour is the OPPOSITE on both axes:
    with pytest.raises(ValueError):  # full coverage -> raise (OLD would have softened)
        add_space_creation(a, f_velless_full, home_team_id=1, xt=_xt(), visible_area=_full_pitch_va(a))
    out_cropped = add_space_creation(
        a, f_velbearing_cropped, home_team_id=1, xt=_xt(), visible_area=_left_half_va(a)
    )  # cropped -> soften (OLD would have raised)
    # Self-non-vacuity: the `== "unresolved_one_team"` .all() is vacuously true on an all-NA column,
    # so pin that the team side actually computed (a soften, not a silently-empty result).
    assert out_cropped["space_created_m2"].notna().any()
    assert (out_cropped["space_opponent_source"] == "unresolved_one_team").all()

    # Non-vacuity: a factual two-team twin measurably differs from the cropped one-team result --
    # the factual resolves the opponent (a real space_denied_m2_opponent), the crop NaNs it.
    factual = add_space_creation(a, _two_team_frame(velocity_less=False), home_team_id=1, xt=_xt())
    assert (factual["space_opponent_source"] == "resolved").all()
    assert factual["space_denied_m2_opponent"].notna().any()
    assert out_cropped["space_denied_m2_opponent"].isna().all()
