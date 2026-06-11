"""Repo-wide aggregator column-LIVENESS gate (lakehouse mandate, 4.23.0).

Every tracking enrichment in the public ``add_*`` surface must produce LIVE
values: every column an aggregator ADDS to its input must be non-null for at
least one row on a fixture that genuinely exercises that family's domain
(shots for the pre-shot-GK family, a GK goalkick for the xT-GK/completion
family, attacking-third ball for xS, wide-area ball for xCross, ...).

A documented column that is 100%-null on the fixture FAILS CI. This is the
structural guarantee that a hard-coded-NaN contract column — the TF-41
``*_opponent`` defect that shipped 3.21.0-4.22.1 — can never ship again, for
ANY aggregator. There is deliberately NO exception set: a conditional column
gets a fixture that exercises its condition, never an exclusion.

Meta-assertion: the wired surface == the registered ``tracking.__all__``
``add_*`` exports (mirrors the ADR-019 dtype gate's B3 pattern), so a new
aggregator cannot land without a liveness entry.
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd
import pytest

import silly_kicks.tracking as tracking
from silly_kicks.tracking import features as F
from silly_kicks.tracking import link_actions_to_frames

# ---------------------------------------------------------------------------
# Fixture: 5 action windows, each exercising a distinct feature domain.
#   t=10 pass (midfield)           -> context/pressure/structural/elastic/...
#   t=20 shot (edge of box)        -> pre-shot-GK family, ghost-GK
#   t=30 goalkick by the team-5 GK -> gk_completion + xt_gk domain
#   t=40 shot, ball x=80 (att 3rd) -> xshot_occurrence domain
#   t=50 cross from the wide right -> xcross_attempt domain
# Geometry mirrors the ADR-019 dtype-gate fixture (5 outfield + GK per team +
# ball, sampled across a 1.4 s pre-window) so off-ball/defensive-line/run
# features produce real values; ball/player x tracks each window's domain.
# ---------------------------------------------------------------------------

_WINDOWS = (
    # (t_action, ball_x, ball_y, actor_pid) — the actor carries the ball into the
    # action (carrier inference + ELASTIC player-ball proximity need them together).
    (10.0, 58.0, 32.0, 10),
    (20.0, 62.0, 30.0, 11),
    (30.0, 5.0, 34.0, 1),
    (40.0, 80.0, 34.0, 10),
    (50.0, 78.0, 60.0, 11),
)


def make_actions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [1] * 5,
            "action_id": [0, 1, 2, 3, 4],
            "period_id": [1] * 5,
            "time_seconds": [10.0, 20.0, 30.0, 40.0, 50.0],
            "team_id": pd.Series([5, 5, 5, 5, 5], dtype="int64"),
            # action 2 is the goalkick by the team-5 GK (player 1)
            "player_id": pd.Series([10, 11, 1, 10, 11], dtype="int64"),
            "defending_gk_player_id": pd.Series([2] * 5, dtype="int64"),
            "start_x": [50.0, 60.0, 8.0, 80.0, 78.0],
            "start_y": [34.0, 30.0, 34.0, 34.0, 60.0],
            "end_x": [60.0, 90.0, 45.0, 95.0, 95.0],
            "end_y": [30.0, 34.0, 30.0, 34.0, 34.0],
            "type_id": [0, 11, 22, 11, 1],  # pass, shot, goalkick, shot, cross
            "type_name": ["pass", "shot", "goalkick", "shot", "cross"],
            "result_id": [1, 1, 1, 1, 1],
            "result_name": ["success"] * 5,
            "bodypart_id": [0] * 5,
            "bodypart_name": ["foot"] * 5,
        }
    )


def _frow(pid, team, gk, x, y, t, *, is_ball=False):
    return dict(
        game_id=1,
        period_id=1,
        frame_id=round(t * 25),
        time_seconds=t,
        frame_rate=25.0,
        player_id=pid,
        team_id=team,
        is_ball=is_ball,
        is_goalkeeper=gk,
        x=float(min(max(x, 0.5), 104.5)),
        y=float(min(max(y, 0.5), 67.5)),
        z=0.0,
        speed=4.3 if not is_ball else 6.0,
        vx=4.3,
        vy=0.0,
        speed_source="native",
        ball_state="alive",
        team_attacking_direction="ltr",
        confidence=None,
        visibility=None,
        source_provider="gradientsports",
        is_goalkeeper_source="native",
    )


def make_frames() -> pd.DataFrame:
    team5 = {10: -8.0, 11: -13.0, 12: -28.0, 13: -33.0, 14: -38.0}  # offset from ball_x
    team6 = {20: 12.0, 21: 14.0, 22: 22.0, 23: 27.0, 24: 30.0}
    gks = {1: (5, 4.0), 2: (6, 101.0)}
    offsets = (-1.4, -1.05, -0.7, -0.35, 0.0)

    rows = []
    for t_a, ball_x, ball_y, actor_pid in _WINDOWS:
        for off in offsets:
            t = round(t_a + off, 3)
            frac = (off + 1.4) / 1.4  # 0 at window start -> 1 at the action
            # Ball: near-still while carried, kicked at the action (acceleration
            # signature for ELASTIC; finite-difference accel spikes at frac>0.75).
            kick = max(0.0, frac - 0.75) * 16.0
            bx, by = ball_x + kick, ball_y
            for pid, dx in team5.items():
                if pid == actor_pid:
                    rows.append(_frow(pid, 5, False, bx - 0.6, ball_y, t))
                else:
                    rows.append(_frow(pid, 5, False, ball_x + dx + 6.0 * frac, 34.0 + pid % 3, t))
            for pid, dx in team6.items():
                rows.append(_frow(pid, 6, False, ball_x + dx - 2.0 * frac, 30.0 + pid % 3, t))
            for pid, (team, gx) in gks.items():
                if pid == actor_pid:
                    rows.append(_frow(pid, team, True, bx - 0.6, ball_y, t))
                else:
                    rows.append(_frow(pid, team, True, gx, 34.0, t))
            rows.append(_frow(pd.NA, pd.NA, False, bx, by, t, is_ball=True))
    f = pd.DataFrame(rows)
    f["player_id"] = f["player_id"].astype("Int64")
    f["team_id"] = f["team_id"].astype("Int64")
    return f


@functools.cache
def _actions() -> pd.DataFrame:
    return make_actions()


@functools.cache
def _frames() -> pd.DataFrame:
    return make_frames()


@functools.cache
def _xt():
    from silly_kicks.xthreat import ExpectedThreat

    xt = ExpectedThreat(l=16, w=12)
    xt.xT = np.tile(np.linspace(0.0, 1.0, 16), (12, 1))
    return xt


@functools.cache
def _frames_with_possession() -> pd.DataFrame:
    """Frames + team_in_possession (the DAS prerequisite)."""
    from silly_kicks.tracking import derive_team_in_possession, infer_ball_carrier

    frames = _frames()
    carrier = infer_ball_carrier(frames)
    return derive_team_in_possession(frames, carrier)


def _gs_jersey_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Minimal GS jersey_frames + roster for add_gradientsports_player_ids."""
    jersey_frames = pd.DataFrame(
        {
            "game_id": [1, 1, 1],
            "period_id": [1, 1, 1],
            "frame_id": [250, 250, 250],
            "time_seconds": [10.0, 10.0, 10.0],
            "jersey_number": [7, 9, pd.NA],
            "team_side": ["home", "away", None],
            "is_ball": [False, False, True],
            "x": [50.0, 60.0, 58.0],
            "y": [34.0, 30.0, 32.0],
        }
    )
    roster = pd.DataFrame(
        {
            "team_id": [5, 6],
            "player_id": [10, 20],
            "shirt_number": [7, 9],
            "position_group_type": ["DF", "FW"],
        }
    )
    return jersey_frames, roster


# ---------------------------------------------------------------------------
# The wired surface. Each entry: (name, runner) where runner() returns
# (input_df, output_df); liveness = every column ADDED by the aggregator is
# non-null somewhere.
# ---------------------------------------------------------------------------


def _std(fn, **kw):
    return lambda: (_actions(), fn(_actions(), _frames(), **kw))


def _xtf(fn, **kw):
    return lambda: (_actions(), fn(_actions(), _frames(), _xt(), home_team_id=5, **kw))


def _run_sync_score():
    links, _report = link_actions_to_frames(_actions(), _frames())
    from silly_kicks.tracking import add_sync_score

    return links, add_sync_score(_actions(), links)


def _run_das():
    return _actions(), F.add_das(_actions(), _frames_with_possession())


def _run_gradientsports_player_ids():
    from silly_kicks.tracking import add_gradientsports_player_ids

    jersey_frames, roster = _gs_jersey_inputs()
    out, _report = add_gradientsports_player_ids(jersey_frames, roster, home_team_id=5, away_team_id=6)
    return jersey_frames, out


ENTRIES: dict[str, object] = {
    "add_action_context": _std(F.add_action_context),
    "add_actor_pre_window": _std(F.add_actor_pre_window),
    "add_cover_shadows": _xtf(F.add_cover_shadows),
    "add_das": _run_das,
    "add_defensive_line": _std(F.add_defensive_line, home_team_id=5, n=4),
    "add_elastic_sync": _std(F.add_elastic_sync),
    "add_ghost_gk": _std(F.add_ghost_gk, home_team_id=5),
    "add_gk_completion": _std(F.add_gk_completion),
    "add_gk_influence": _xtf(F.add_gk_influence),
    "add_gradientsports_player_ids": _run_gradientsports_player_ids,
    "add_line_break": _std(F.add_line_break, home_team_id=5),
    "add_obso": _std(F.add_obso),
    "add_off_ball_context": _std(F.add_off_ball_context, home_team_id=5),
    "add_off_ball_runs": _std(F.add_off_ball_runs, home_team_id=5),
    "add_pausa": _std(F.add_pausa),
    "add_pitch_control": _std(F.add_pitch_control),
    "add_player_influence": _xtf(F.add_player_influence),
    "add_pre_shot_gk_angle": (lambda: (_actions(), F.add_pre_shot_gk_angle(_actions(), frames=_frames()))),
    "add_pre_shot_gk_position": _std(F.add_pre_shot_gk_position),
    "add_pressure_on_actor": _std(F.add_pressure_on_actor),
    "add_shape_graph": _std(F.add_shape_graph, home_team_id=5),
    "add_space_creation": _std(F.add_space_creation, home_team_id=5),
    "add_structural_pass": _std(F.add_structural_pass, home_team_id=5),
    "add_sync_score": _run_sync_score,
    "add_team_shape": _std(F.add_team_shape, home_team_id=5),
    "add_xcross_attempt": _std(tracking.add_xcross_attempt, home_team_id=5),
    "add_xshot_occurrence": _std(tracking.add_xshot_occurrence, home_team_id=5),
    "add_xt_gk": _xtf(F.add_xt_gk),
}


def test_meta_surface_complete():
    """Every registered add_* export is wired into the liveness gate (B3 pattern)."""
    registered = {n for n in tracking.__all__ if n.startswith("add_")}
    assert set(ENTRIES) == registered, (
        f"liveness gate surface drift: missing={registered - set(ENTRIES)}, stale={set(ENTRIES) - registered}"
    )


@pytest.mark.parametrize("name", sorted(ENTRIES))
def test_aggregator_columns_live(name):
    runner = ENTRIES[name]
    input_df, out = runner()  # type: ignore[operator]
    added = [c for c in out.columns if c not in set(input_df.columns)]
    assert added, f"{name} added no columns on the liveness fixture"
    dead = [c for c in added if not out[c].notna().any()]
    assert not dead, (
        f"{name}: dead contract column(s) {dead} — 100%-null on a fixture that "
        f"exercises the family's domain (added columns: {added})"
    )
