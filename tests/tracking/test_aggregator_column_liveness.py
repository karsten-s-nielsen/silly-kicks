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
            # Event clocks jittered off the 25 fps frame grid (real providers never
            # align perfectly): sync/elastic quality varies instead of saturating.
            "time_seconds": [10.0, 20.06, 30.0, 40.11, 50.02],
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


def _frow(pid, team, gk, x, y, t, *, is_ball=False, vx=4.3, vy=0.0):
    speed = float(np.hypot(vx, vy))
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
        speed=speed if not is_ball else 6.0,
        vx=float(vx),
        vy=float(vy),
        speed_source="native",
        ball_state="alive",
        team_attacking_direction="ltr",
        confidence=None,
        visibility=None,
        source_provider="gradientsports",
        is_goalkeeper_source="native",
    )


def make_frames() -> pd.DataFrame:
    """Per-window geometry VARIES (velocities, y-layout, GK positions, kick power):
    a fixture that repeats one geometric pattern makes genuinely-live metrics
    constant across actions (actor_speed, defensive-line widths, GK reachable
    area, ...) and false-fails the non-constant check."""
    # Receivers 12/13 run AHEAD of the ball with the defensive block between them
    # and the passer (cover-shadow lanes can actually be blocked); 10/11/14 offer
    # behind. Offsets scale down near the pitch edges (no clamp pile-ups).
    team5 = {10: -8.0, 11: -13.0, 12: 9.0, 13: 16.0, 14: -24.0}  # offset from ball_x
    team6 = {20: 12.0, 21: 14.0, 22: 22.0, 23: 27.0, 24: 30.0}
    offsets = (-1.4, -1.05, -0.7, -0.35, 0.0)
    actor_gaps = (0.05, 0.65, 0.45, 0.9, 1.25)  # one ~at-ball window: PC-at-ball varies

    rows = []
    for w_idx, (t_a, ball_x, ball_y, actor_pid) in enumerate(_WINDOWS):
        gk_jitter = 0.9 * w_idx
        gks = {1: (5, 4.0 + gk_jitter, 31.0 + 1.5 * w_idx), 2: (6, 101.0 - gk_jitter, 36.5 - 1.2 * w_idx)}
        kick_power = 12.0 + 3.0 * w_idx
        # Kick fires on the FINAL inter-frame step only: the ball stays at the action's
        # start point until the event (PC/lane sampling at start_x is live), while the
        # last-step velocity jump still gives ELASTIC its acceleration signature.
        kick_threshold = 0.93 + 0.01 * w_idx
        # Teammates advance in the pre-window — except the cross window, where the
        # runs retreat (OBSO declines toward the event: PAUSA's temporal term varies).
        advance = 4.0 + 0.9 * w_idx if w_idx != 4 else -3.0
        actor_gap = actor_gaps[w_idx]
        # Nearest defender presses the carrier — except the cross window, where the
        # carrier is ISOLATED (nobody within reach: uniquely-reachable area > 0).
        presser_gap = 2.2 + 0.9 * w_idx if w_idx != 4 else 14.0
        scale_back = min(1.0, max(0.15, (ball_x - 2.0) / 42.0))  # room behind the ball
        scale_fwd = min(1.0, max(0.15, (103.0 - ball_x) / 32.0))  # room ahead of the ball
        for off in offsets:
            t = round(t_a + off, 3)
            frac = (off + 1.4) / 1.4  # 0 at window start -> 1 at the action
            # Ball: near-still while carried, kicked at the action (acceleration
            # signature for ELASTIC; finite-difference accel spikes past the threshold).
            kick = max(0.0, frac - kick_threshold) * kick_power
            bx, by = ball_x + kick, ball_y
            for pid, dx in team5.items():
                if pid == 14 and w_idx == 3:
                    continue  # attacking side a man down once: squad-count metrics vary
                v = 2.2 + 0.45 * (pid % 4) + 0.35 * w_idx
                scale = scale_fwd if dx > 0 else scale_back
                if pid == actor_pid:
                    # Reach must EXCEED the ~2.1 m grid spacing for uniquely-reachable
                    # cells to exist at all (TTI = reaction time + kinematics leaves
                    # ~0.3 s of motion within tau=1 s); the cross window's carrier
                    # sprints (8.5 m/s) into open space -> unique cells > 0 there.
                    actor_vx = 8.5 if w_idx == 4 else 3.2 + 0.45 * w_idx
                    rows.append(_frow(pid, 5, False, bx - actor_gap, ball_y, t, vx=actor_vx))
                else:
                    y = 33.0 + (pid % 3) * (2.0 + 0.6 * w_idx)
                    x = ball_x + dx * scale + advance * frac
                    rows.append(_frow(pid, 5, False, x, y, t, vx=v, vy=0.3 * (pid % 2)))
            for pid, dx in team6.items():
                if pid == 24 and w_idx == 4:
                    continue  # one window plays a man down: squad-count metrics vary
                v = 1.8 + 0.4 * (pid % 3) + 0.3 * w_idx
                y = 29.0 + (pid % 3) * (2.2 + 0.5 * w_idx)
                # pid 20 presses the carrier (goal-side of the ball: cover-shadow lanes,
                # pressure, PC-at-ball variation); the rest hold a varying block.
                x = bx + presser_gap if pid == 20 else ball_x + dx * scale_fwd - 2.0 * frac
                rows.append(_frow(pid, 6, False, x, y if pid != 20 else ball_y + 0.8, t, vx=-v, vy=-0.25 * (pid % 2)))
            for pid, (team, gx, gy) in gks.items():
                if pid == actor_pid:
                    rows.append(_frow(pid, team, True, bx - actor_gap, ball_y, t, vx=1.0))
                else:
                    rows.append(_frow(pid, team, True, gx, gy, t, vx=0.6 + 0.15 * w_idx))
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


def _run_shot_goalmouth():
    """TF-48 domain: the goalmouth-crossing family needs a POST-contact ball flight
    that straddles the goal plane (the shared 5-sample pre-window fixture ends at the
    action) plus GKs at both ends for the goal map and a ballistic z for the z-profile
    columns -- an OBSERVED on-target crossing makes every contract column live."""
    t_a = 60.0
    rows = []
    for i in range(-10, 34):  # 25 fps, t in [-0.4, +1.32] around the shot
        t = round(t_a + i / 25.0, 3)
        bt = max(t - t_a, 0.0)  # ball waits at the spot, then flies at 25 m/s
        bx, by = 85.0 + 25.0 * bt, 30.0 + 2.0 * bt
        rows.append(_frow(1, 5, True, 4.0, 34.0, t))
        rows.append(_frow(2, 6, True, 101.0, 34.0, t))
        ball = _frow(pd.NA, pd.NA, False, bx, by, t, is_ball=True)
        ball["x"] = bx  # un-clamped: the crossing needs samples BEYOND x=105
        ball["z"] = max(4.0 * bt - 4.905 * bt * bt, 0.0)
        rows.append(ball)
    frames = pd.DataFrame(rows)
    frames["player_id"] = frames["player_id"].astype("Int64")
    frames["team_id"] = frames["team_id"].astype("Int64")
    actions = pd.DataFrame(
        {
            "game_id": [1],
            "action_id": [0],
            "period_id": [1],
            "time_seconds": [t_a],
            "team_id": pd.Series([5], dtype="int64"),
            "player_id": pd.Series([10], dtype="int64"),
            "start_x": [85.0],
            "start_y": [30.0],
            "end_x": [105.0],
            "end_y": [34.0],
            "type_id": [11],
            "type_name": ["shot"],
            "result_id": [1],
            "result_name": ["success"],
            "bodypart_id": [0],
            "bodypart_name": ["foot"],
        }
    )
    return actions, F.add_shot_goalmouth(actions, frames)


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
    "add_shot_goalmouth": _run_shot_goalmouth,
    "add_space_creation": _std(F.add_space_creation, home_team_id=5),
    "add_structural_pass": _std(F.add_structural_pass, home_team_id=5),
    "add_sync_score": _run_sync_score,
    "add_team_shape": _std(F.add_team_shape, home_team_id=5),
    "add_xcross_attempt": _std(tracking.add_xcross_attempt, home_team_id=5),
    "add_xshot_occurrence": _std(tracking.add_xshot_occurrence, home_team_id=5),
    "add_xt_gk": _xtf(F.add_xt_gk),
}


# Documented STRUCTURAL CONSTANTS (lakehouse round-2 amendment: non-NaN alone is not
# enough — `space_destroyed_m2` was 0-everywhere since TF-41). These are columns that
# are constant BY DESIGN, each backed by a dedicated invariant test proving the
# constant and a docstring/CHANGELOG entry. They are NOT exclusions: the liveness
# (non-NaN) check still applies; only the non-constant check defers to the invariant
# test named in the justification.
STRUCTURAL_CONSTANTS: dict[str, dict[str, str]] = {
    # add_space_creation declares none: its structurally-zero columns were RETIRED
    # from the contract entirely (4.24.0 lean contract; resurrection is blocked by
    # test_space_creation.py::TestComputeSpaceCreated::test_retired_columns_never_emitted).
    #
    # add_pitch_control declared none as of ADR-032 (4.31.0): the dead near-ball
    # `pitch_control_at_ball__spearman` (~0.5 everywhere) was RETIRED and replaced by the LIVE
    # `pitch_control_at_target__spearman` (sampled at the action destination, ball-travel-time
    # positive), which passes the standard non-constant liveness check. The off-ball-destination
    # precondition the gate's teeth rest on is asserted by
    # test_pitch_control_at_target_fixture_has_offball_destinations below.
}


def test_pitch_control_at_target_fixture_has_offball_destinations():
    """Hard precondition for the add_pitch_control liveness teeth (ADR-032): the fixture MUST have
    >=2 actions whose destination is off-ball (>R from the frame ball), else at_target is ~0.5
    everywhere and the non-constant check would pass weakly. Fail loudly on a fixture refactor that
    accidentally makes every action in-place, rather than silently neutering the gate."""
    import numpy as np

    actions, frames = _actions(), _frames()
    offball = 0
    for _, a in actions.iterrows():
        # actions link to frames by time within a period; use the period's ball as the off-ball reference.
        ball = frames[(frames["period_id"] == a["period_id"]) & frames["is_ball"]]
        if ball.empty or np.isnan(a["end_x"]):
            continue
        bx, by = float(ball["x"].iloc[0]), float(ball["y"].iloc[0])
        if np.hypot(a["end_x"] - bx, a["end_y"] - by) > 10.0:
            offball += 1
    assert offball >= 2, f"liveness fixture has only {offball} off-ball-destination actions; gate would lose teeth"


def test_meta_surface_complete():
    """Every registered add_* export is wired into the liveness gate (B3 pattern)."""
    registered = {n for n in tracking.__all__ if n.startswith("add_")}
    assert set(ENTRIES) == registered, (
        f"liveness gate surface drift: missing={registered - set(ENTRIES)}, stale={set(ENTRIES) - registered}"
    )


def test_meta_structural_constants_are_wired():
    """Every declared structural constant belongs to a wired aggregator."""
    assert set(STRUCTURAL_CONSTANTS) <= set(ENTRIES)


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
    # Non-constant (lakehouse round-2): a float METRIC column with >= 2 observed
    # values carrying a single distinct value is informationally dead even when
    # non-NaN (the `space_destroyed_m2 ≡ 0` failure mode). Categorical/provenance
    # columns are exempt: object/bool/int by dtype, plus the four documented
    # linkage-provenance floats (legitimately constant when every action links at
    # offset 0 / quality 1; their merge semantics are gated by the provenance-skip
    # guard, not here). The check targets metrics.
    provenance = {"frame_id", "time_offset_seconds", "link_quality_score", "n_candidate_frames"}
    declared = STRUCTURAL_CONSTANTS.get(name, {})
    flat = [
        c
        for c in added
        if c not in declared
        and c not in provenance
        and pd.api.types.is_float_dtype(out[c])
        and out[c].notna().sum() >= 2
        and out[c].dropna().nunique() == 1
    ]
    assert not flat, (
        f"{name}: constant metric column(s) {flat} — single distinct value across the "
        f"fixture; either the fixture does not exercise the metric or the column is "
        f"structurally dead (declare + invariant-test it in STRUCTURAL_CONSTANTS if BY DESIGN)"
    )
