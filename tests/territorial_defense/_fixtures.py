"""Shared fixtures for the territorial_defense suite (TF-54b).

Velocity-ZERO LTR frames (team 1 defends x=0, team 2 defends x=105). The load-bearing geometry
(mirrors ``tests/restdefense/_fixtures.make_keeper_sensitive_fixture`` and
``tests/tracking/test_compute_threat_pc``): a defender D registers in the opponent's threat integral
only when it is the NEAREST defender to cells inside a dangerous opponent receiver's Voronoi region,
so **removing D measurably raises the attacking team's threat** -- the non-vacuity anchor for the
removal counterfactual. D is an OUTFIELD actor-defender; the keeper is laterally separated (opposite
touchline) so it does NOT cover D's deep zone, i.e. D is genuinely the sole coverer there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_GAME = 1
_PERIOD = 1


def _player(team_id, player_id, x, y, *, is_actor=False, gk=False):
    return {
        "game_id": _GAME,
        "period_id": _PERIOD,
        "frame_id": 0,
        "time_seconds": 10.0,
        "team_id": team_id,
        "player_id": player_id,
        "is_ball": False,
        "is_goalkeeper": gk,
        "is_actor": is_actor,
        "x": float(x),
        "y": float(y),
        "vx": 0.0,
        "vy": 0.0,
        "ball_state": "alive",
        "source_provider": "fixture",
    }


def _ball(x, y=34.0):
    return {
        "game_id": _GAME,
        "period_id": _PERIOD,
        "frame_id": 0,
        "time_seconds": 10.0,
        "team_id": pd.NA,
        "player_id": pd.NA,
        "is_ball": True,
        "is_goalkeeper": False,
        "is_actor": False,
        "x": float(x),
        "y": float(y),
        "vx": 0.0,
        "vy": 0.0,
        "ball_state": "alive",
        "source_provider": "fixture",
    }


def make_fitted_xt():
    """A fitted ExpectedThreat (x-increasing grid), matching the tracking ``fitted_xt`` fixture."""
    from silly_kicks.xthreat import ExpectedThreat

    xt = ExpectedThreat(l=16, w=12)
    xt.xT = np.tile(np.linspace(0.0, 1.0, 16), (12, 1))
    return xt


#: D = team-1 outfield defender #102, the actor. Attacking team = 2 (attacks toward x=0). Team-2
#: striker #201 sits in D's deep bottom-corner zone; the team-1 keeper is at the TOP touchline so it
#: does not cover that zone. Removing D raises team-2's threat there.
ATTACKING_TEAM_ID = 2
DEFENDING_TEAM_ID = 1
ACTOR_PLAYER_ID = 102


def _rich_rows():
    # Mirrors tests/tracking/test_compute_threat_pc's keeper-sensitive geometry, but the deep
    # sole-defender is OUTFIELD D (#102, the actor) not the keeper. The team-1 keeper sits in the
    # TOP corner so it does NOT cover D's deep-CENTRE zone; the ball is FORWARD (x=60), so team-2's
    # counter-threat receiver (#201, broken in behind at deep centre) is "ahead of the ball" and
    # registers -- removing D opens that zone and raises team-2 threat (non-vacuity).
    return [
        _player(1, 101, 2.0, 60.0, gk=True),  # team-1 keeper, TOP corner (not covering deep centre)
        _player(1, 102, 4.0, 34.0, is_actor=True),  # D: deep CENTRE, sole coverer, THE ACTOR
        _player(1, 103, 26.0, 20.0),  # back line, upfield of the striker
        _player(1, 104, 28.0, 48.0),
        _player(1, 105, 30.0, 34.0),
        _player(1, 106, 58.0, 30.0),  # attacker near the ball
        _player(2, 201, 14.0, 34.0),  # team-2 striker broken in behind, in D's deep-centre zone
        _player(2, 202, 48.0, 22.0),  # team-2 rest, upfield
        _player(2, 203, 52.0, 46.0),
        _player(2, 204, 68.0, 30.0),
        _player(2, 205, 72.0, 20.0),
        _player(2, 206, 100.0, 34.0, gk=True),  # team-2 keeper
        _ball(60.0, 34.0),  # ball FORWARD (opponent counter-threat in progress)
    ]


def _typed(frames: pd.DataFrame) -> pd.DataFrame:
    frames = frames.copy()
    frames["team_id"] = frames["team_id"].astype("Int64")
    frames["player_id"] = frames["player_id"].astype("Int64")
    frames["is_ball"] = frames["is_ball"].astype(bool)
    frames["is_goalkeeper"] = frames["is_goalkeeper"].astype(bool)
    frames["is_actor"] = frames["is_actor"].astype("boolean")
    return frames


def make_rich_frame() -> pd.DataFrame:
    """A single scoreable Arm-A frame: D (#102) is the actor, the keeper + 4 others remain after
    removal, and removing D raises team-2 threat (non-vacuity). Velocity-zero, LTR."""
    return _typed(pd.DataFrame(_rich_rows()))


def make_thin_one_defender_frame() -> pd.DataFrame:
    """D is the SOLE team-1 defender: removing D -> 0 defenders (PLAN-01 depletion). Same deep-centre
    geometry as the rich frame (D goal-side of the team-2 striker, ball forward), so removing D
    inflates team-2 threat -- the evidence the depletion drop is not cosmetic. team-1 goal is
    resolved from a richer frame's goal_map (this frame is keeper-less by construction)."""
    rows = [
        _player(1, 102, 4.0, 34.0, is_actor=True),  # D, the only team-1 player, deep-centre coverer
        _player(2, 201, 14.0, 34.0),  # team-2 striker in D's zone
        _player(2, 202, 48.0, 22.0),  # team-2 support, upfield
        _ball(60.0, 34.0),  # ball FORWARD (counter-threat in progress)
    ]
    return _typed(pd.DataFrame(rows))


def make_e2e_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(actions, frames)`` exercising BOTH arms end-to-end for D (#102, team 1).

    Arm A: 3 interceptions by D (frames 0-2, rich geometry -> scored). Arm B: D's hull from D's 3
    own-half defensive-action locations, and a team-2 pass (action 3) whose end (90, 34) reflects to
    (15, 34) INSIDE D's hull -- at frame 3 the nearest team-1 defender to the frame target is D, and
    removing it raises team-2 threat. All four frames reuse the rich geometry (frame positions are
    decoupled from the event locations, as in real freeze-frames)."""
    frame_parts = []
    for fid in range(4):
        fr = pd.DataFrame(_rich_rows())
        fr["frame_id"] = fid
        frame_parts.append(_typed(fr))
    frames = pd.concat(frame_parts, ignore_index=True)

    actions = pd.DataFrame(
        {
            "game_id": [_GAME] * 4,
            "period_id": [_PERIOD] * 4,
            "action_id": [0, 1, 2, 3],
            "team_id": pd.array([1, 1, 1, 2], dtype="Int64"),
            "player_id": pd.array([102, 102, 102, 201], dtype="Int64"),
            "type_id": [10, 10, 10, 0],  # 3 interceptions by D + 1 team-2 pass
            "type_name": ["interception", "interception", "interception", "pass"],
            "result_id": [1, 1, 1, 1],
            "start_x": [10.0, 14.0, 20.0, 60.0],
            "start_y": [20.0, 44.0, 30.0, 34.0],
            "end_x": [10.0, 14.0, 20.0, 90.0],  # team-2 pass ends at 90 -> reflects to (15, 34) in D's hull
            "end_y": [20.0, 44.0, 30.0, 34.0],
            "time_seconds": [10.0, 11.0, 12.0, 13.0],
        }
    )
    return actions, frames


def make_arm_a_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(actions, frames)`` -- the single action is D's interception (type_id=10), so
    ``select_arm_a_domain`` picks it and the frame carries ``is_actor`` on D. ``frame_id ==
    action_id == 0`` (the snapshot-port convention)."""
    frames = make_rich_frame()
    actions = pd.DataFrame(
        {
            "game_id": [_GAME],
            "period_id": [_PERIOD],
            "action_id": [0],
            "team_id": pd.array([DEFENDING_TEAM_ID], dtype="Int64"),
            "player_id": pd.array([ACTOR_PLAYER_ID], dtype="Int64"),
            "type_id": [10],  # interception (a defensive action in the Arm-A domain)
            "type_name": ["interception"],
            "result_id": [1],
            "result_name": ["success"],
            "start_x": [6.0],
            "start_y": [13.0],
            "end_x": [6.0],
            "end_y": [13.0],
            "time_seconds": [10.0],
        }
    )
    return actions, frames


def _reflect_rows(rows: list[dict]) -> list[dict]:
    """Point-reflect a per-action-LTR row set into the OTHER team's LTR (x->105-x, y->68-y).

    A per-action freeze-frame is aligned to its action's acting-team LTR (ADR-028): the acting team
    attacks x=105. Reflecting the rich (team-1 acting) geometry yields the SAME physical scene as it
    would appear in team-2's action frame -- team-2's keeper drops to low x, team-1's rises to high x.
    """
    out = []
    for r in rows:
        r = dict(r)
        r["x"] = 105.0 - float(r["x"])
        r["y"] = 68.0 - float(r["y"])
        out.append(r)
    return out


def make_per_action_ltr_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(actions, frames)`` in the REAL SB360 per-action-LTR convention -- each frame is in ITS acting
    team's LTR (ADR-028), the convention ``compute_territorial_defense`` defaults to (``per_action_ltr``).

    BALANCED acting teams by design (3 team-1 D-interception frames + 3 team-2 pass frames): each team's
    keeper is therefore bimodal across the match (low x in its own actions, high x in the opponent's),
    the per-``(game, period, team)`` mean lands at midfield, and the per-MATCH ``resolve_defended_goals``
    (``match_ltr``) collapses BOTH teams to the same end -> ``attacked_goal`` None -> 0 scored. Under the
    ``per_action_ltr`` default each frame resolves from its action's acting team, so BOTH arms score. This
    single fixture is the SB360 scoring fixture AND the two-sided-regression fixture.

    Arm A: D (#102, team 1) interceptions in frames 0-2 (team-1's LTR). Arm B: 3 team-2 passes (frames
    3-5, team-2's LTR = reflected rich geometry) whose ends reflect into D's own-half hull; at each the
    nearest team-1 defender to the frame target is D, and removing D raises team-2 threat.
    """
    rich = _rich_rows()
    refl = _reflect_rows(rich)
    parts = []
    for fid in range(3):  # team-1 (D) interception frames -> team-1's LTR
        fr = pd.DataFrame(rich)
        fr["frame_id"] = fid
        parts.append(_typed(fr))
    for fid in range(3, 6):  # team-2 pass frames -> team-2's LTR (reflected)
        fr = pd.DataFrame(refl)
        fr["frame_id"] = fid
        parts.append(_typed(fr))
    frames = pd.concat(parts, ignore_index=True)

    actions = pd.DataFrame(
        {
            "game_id": [_GAME] * 6,
            "period_id": [_PERIOD] * 6,
            "action_id": [0, 1, 2, 3, 4, 5],
            "team_id": pd.array([1, 1, 1, 2, 2, 2], dtype="Int64"),
            "player_id": pd.array([102, 102, 102, 201, 201, 201], dtype="Int64"),
            "type_id": [10, 10, 10, 0, 0, 0],  # 3 interceptions by D + 3 team-2 passes
            "type_name": ["interception"] * 3 + ["pass"] * 3,
            "result_id": [1] * 6,
            # interceptions in D's own half (team-1 LTR) -> the hull; passes in team-2's LTR whose ends
            # reflect to (~15, 34) inside D's hull.
            "start_x": [10.0, 14.0, 20.0, 60.0, 60.0, 60.0],
            "start_y": [20.0, 44.0, 30.0, 34.0, 34.0, 34.0],
            "end_x": [10.0, 14.0, 20.0, 88.0, 90.0, 92.0],
            "end_y": [20.0, 44.0, 30.0, 34.0, 34.0, 34.0],
            "time_seconds": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        }
    )
    return actions, frames


def make_td_scaling_fixture(n_games: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``n_games`` copies of the e2e fixture with distinct ``game_id`` -- the ADR-073 growth fixture.

    Scales the GROUP (loop-iteration) dimension: each game contributes 3 D-interceptions (Arm-A
    domain / classified rows) + 1 opponent pass, i.e. one distinct ``(game, D)`` unit for Arm B.
    **Scaling GAMES (not within-game defenders) is the discrimination proof for ``_score_arm_b``**:
    the ADR-068 fix groups opponent passes by game ONCE, so each defender touches only its own game's
    single pass -> O(games); the regression it guards (a per-defender ``passes[mask]`` over the WHOLE
    batch) rescans every game's passes for every defender -> O(games^2). A single-game fixture would
    leave that rescan linear (a guard that cannot guard)."""
    base_a, base_f = make_e2e_fixture()
    actions = pd.concat([base_a.assign(game_id=g) for g in range(n_games)], ignore_index=True)
    frames = pd.concat([base_f.assign(game_id=g) for g in range(n_games)], ignore_index=True)
    return actions, frames


def single_match_scored(n_frames: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ONE game with ``n_frames`` SCORED Arm-A interception frames (all by D #102) -- the ADR-073
    growth fixture for the validation DRIVER's per-scored-frame dose-battery loop.

    Scales the scored-frame (loop) dimension WITHIN a single match: the driver builds ``group_rows``
    ONCE and does one O(1) ``.get`` per scored frame, so its rows-scanned grows linearly. ``frame_id ==
    action_id`` (snapshot convention); each frame is the rich geometry, so every candidate classifies
    ``scored``. Arm B is degenerate here (all interceptions share one location -> no hull), which is
    irrelevant to the Arm-A dose battery this fixture drives."""
    frame_parts = []
    for fid in range(n_frames):
        fr = pd.DataFrame(_rich_rows())
        fr["frame_id"] = fid
        frame_parts.append(_typed(fr))
    frames = pd.concat(frame_parts, ignore_index=True)
    actions = pd.DataFrame(
        {
            "game_id": [_GAME] * n_frames,
            "period_id": [_PERIOD] * n_frames,
            "action_id": list(range(n_frames)),
            "team_id": pd.array([DEFENDING_TEAM_ID] * n_frames, dtype="Int64"),
            "player_id": pd.array([ACTOR_PLAYER_ID] * n_frames, dtype="Int64"),
            "type_id": [10] * n_frames,  # interception (Arm-A domain)
            "type_name": ["interception"] * n_frames,
            "result_id": [1] * n_frames,
            "start_x": [10.0] * n_frames,
            "start_y": [20.0] * n_frames,
            "end_x": [10.0] * n_frames,
            "end_y": [20.0] * n_frames,
            "time_seconds": [10.0 + i for i in range(n_frames)],
        }
    )
    return actions, frames
