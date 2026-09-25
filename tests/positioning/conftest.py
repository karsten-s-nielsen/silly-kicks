"""Shared fixtures for the TF-56 positioning suite (spec section 9; plan Task 0 Step 5b).

Frames are built by REUSING the tracking test frame builder
(``tests.tracking._gk_test_helpers._make_two_team_frame``) -- never hand-rolled column dicts --
then decorated with the columns the domain gate / DAS / pressure seams read
(``ball_state``, ``team_in_possession``, ``speed_source``). ``fitted_xt`` is FREE from the root
``tests/conftest.py`` -- do NOT redefine it here.

Geometry convention across the suite: home (team 1) is the DEFENDING team, defending the x=0
goal (GK at x~3); away (team 2) is the in-possession ATTACKING team, attacking toward x=0; the
ball sits near the x=0 goal so the domain gate admits the frame. Movable = home outfielders
(player_ids 10..13); home GK (player_id 1) is EXCLUDED from movable but kept as a fixed agent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.tracking._gk_test_helpers import _make_two_team_frame

# Home (defending, team 1). Player_ids 10 & 11 are the DEEP flank defenders -- the last
# outfielders between the two advanced away receivers and the x=0 goal, so they are the nearest
# defenders to the high-xT cells and moving them measurably changes the conceded threat (the
# compute_threat_pc "screened defender" property: only the nearest defender to a dangerous cell
# registers). They carry velocity toward goal, so a reachable (<=0.7 s) nudge deeper LOWERS the
# conceded threat -- a real, non-degenerate gap exists (measured: dx=-1 -> -0.46, dx=-3 -> -1.44).
_HOME_OUTFIELD = [(12.0, 30.0), (12.0, 38.0), (40.0, 34.0), (46.0, 30.0)]
_HOME_VELS = [(-3.0, 0.0), (-3.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
# Away (attacking, team 2): two advanced receivers near goal + a ball-carrier + a wide man.
_AWAY_OUTFIELD = [(16.0, 30.0), (16.0, 38.0), (24.0, 34.0), (34.0, 44.0)]
_AWAY_VELS = [(-2.0, 0.0), (-2.0, 0.0), (-2.0, 0.0), (-2.0, 0.0)]

_HOME_GK = (3.0, 34.0)
_AWAY_GK = (100.0, 34.0)

#: Ball at the away ball-carrier (24, 34) -- 26 m from the x=0 (home-defended) goal, inside the
#: 35 m danger domain, and within infer_ball_carrier's 3 m tolerance of an away player so away is
#: the in-possession (attacking) team.
_BALL_XY = (26.0, 34.0)

DEFENDING_TEAM_ID = 1
ATTACKING_TEAM_ID = 2
MOVABLE_PLAYER_IDS = [10, 11, 12, 13]  # home outfielders (GK player_id=1 excluded)


def _decorate(
    frame: pd.DataFrame, *, ball_xy: tuple[float, float] = _BALL_XY, possession: int = ATTACKING_TEAM_ID
) -> pd.DataFrame:
    """Add domain-gate / DAS / pressure columns and place the ball."""
    out = frame.copy()
    ball = out["is_ball"].astype(bool)
    out.loc[ball, "x"] = ball_xy[0]
    out.loc[ball, "y"] = ball_xy[1]
    out["ball_state"] = "alive"
    out["team_in_possession"] = possession
    out["speed_source"] = "derived"  # velocity-informed (not the SB360 "unavailable" marker)
    out["speed"] = np.hypot(out["vx"].astype(float), out["vy"].astype(float))  # derive_velocities emits this
    return out


def _base_frame() -> pd.DataFrame:
    return _make_two_team_frame(
        home_positions=_HOME_OUTFIELD,
        away_positions=_AWAY_OUTFIELD,
        home_gk_pos=_HOME_GK,
        away_gk_pos=_AWAY_GK,
        home_velocities=_HOME_VELS,
        away_velocities=_AWAY_VELS,
    )


@pytest.fixture
def one_frame() -> pd.DataFrame:
    """One tracking frame: defending (home) + attacking (away) team + ball; vx/vy + is_goalkeeper
    present; team_in_possession = attacking; the ball is inside the danger domain and a defender
    reposition measurably lowers ``compute_threat_pc``."""
    frame = _decorate(_base_frame())
    # Structural guarantees the whole suite leans on:
    assert frame["team_id"].dropna().nunique() == 2
    assert frame["is_ball"].astype(bool).sum() == 1
    players = frame[~frame["is_ball"].astype(bool)]
    assert players[["vx", "vy"]].notna().all().all()
    assert players["is_goalkeeper"].astype(bool).sum() == 2
    return frame


@pytest.fixture
def two_team_velocity_frame(one_frame) -> pd.DataFrame:
    return one_frame


@pytest.fixture
def attacking_team_id() -> int:
    return ATTACKING_TEAM_ID


@pytest.fixture
def movable_ids() -> list[int]:
    """The defending (home) outfielders' player_ids in one_frame (GK excluded)."""
    return list(MOVABLE_PLAYER_IDS)


@pytest.fixture
def goal_map(one_frame):
    from silly_kicks.tracking import resolve_defended_goals

    return resolve_defended_goals(one_frame)


@pytest.fixture
def threat_objective(one_frame, fitted_xt, goal_map):
    """A ThreatObjective oriented for the away (team 2) attack (built lazily -- Task 1+)."""
    from silly_kicks.positioning import ThreatObjective

    return ThreatObjective(xt=fitted_xt, goal_map=goal_map, attacking_team_id=ATTACKING_TEAM_ID)


@pytest.fixture
def reachability_constraint():
    from silly_kicks.positioning import ReachabilityConstraint, ReachabilityParams

    return ReachabilityConstraint(ReachabilityParams.default())


@pytest.fixture
def mixed_domain_frames() -> pd.DataFrame:
    """Concat spanning: in-domain, ball-far (out_of_domain), one-team, velocity-less (declared),
    unresolved-goal, no-movable -- for the Task 8 conservation census. Each slice is a distinct
    (game_id, period_id, frame_id) so the domain gate scores per frame."""
    slices: list[pd.DataFrame] = []

    def _set_keys(df: pd.DataFrame, *, game: int, period: int, frame: int) -> pd.DataFrame:
        df = df.copy()
        df["game_id"] = game
        df["period_id"] = period
        df["frame_id"] = frame
        # Distinct integer time per frame within a game so 1 fps down-sampling keeps them all.
        df["time_seconds"] = float(frame)
        return df

    # 1) in-domain, scoreable (ball near x=0, away in possession)
    slices.append(_set_keys(_decorate(_base_frame()), game=1, period=1, frame=1))

    # 2) out-of-domain: ball far from the defended (x=0) goal
    slices.append(_set_keys(_decorate(_base_frame(), ball_xy=(80.0, 34.0)), game=1, period=1, frame=2))

    # 3) one-team frame: drop the away team + away GK -> not two teams
    one_team = _base_frame()
    one_team = one_team[(one_team["team_id"] == 1) | one_team["is_ball"].astype(bool)]
    slices.append(_set_keys(_decorate(one_team), game=1, period=1, frame=3))

    # 4) velocity-less (DECLARED via speed_source marker): still two teams, in-domain geometry
    velless = _decorate(_base_frame())
    velless["speed_source"] = "unavailable"  # SPEED_SOURCE_UNAVAILABLE
    slices.append(_set_keys(velless, game=1, period=1, frame=4))

    # 5) unresolved goal geometry: a SEPARATE game (resolve_defended_goals aggregates per
    #    (game, period, team)), where the DEFENDING team (home) has every x NaN so neither its
    #    GK-x nor its outfield-x fallback resolves -> goal_map.get(home) is None. The attacking
    #    team (away) keeps real positions so possession + the two-team check still pass, isolating
    #    the unresolved-goal reason (not folded into no-possession / not-two-team).
    nogeom = _decorate(_base_frame())
    home_rows = (nogeom["team_id"] == DEFENDING_TEAM_ID).to_numpy()
    nogeom.loc[home_rows, "x"] = float("nan")
    slices.append(_set_keys(nogeom, game=2, period=1, frame=1))

    # 6) no-movable: drop all home outfielders (keep home GK) -> defending team has 0 movable
    nomov = _base_frame()
    nomov = nomov[~nomov["player_id"].isin(MOVABLE_PLAYER_IDS)]
    slices.append(_set_keys(_decorate(nomov), game=1, period=1, frame=6))

    return pd.concat(slices, ignore_index=True)


@pytest.fixture
def actions_frames_fixture(one_frame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(actions, frames) for any pressure regression -- a single synthesized on-ball action for
    a home player at the frame's key, paired with one_frame."""
    actions = pd.DataFrame(
        {
            "game_id": [1],
            "period_id": [1],
            "action_id": [0],
            "time_seconds": [1.0],
            "team_id": [DEFENDING_TEAM_ID],
            "player_id": [10],
            "start_x": [_HOME_OUTFIELD[0][0]],
            "start_y": [_HOME_OUTFIELD[0][1]],
        }
    )
    return actions, one_frame
