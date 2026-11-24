"""Shared production-shape fixtures (plan 2026-11-08 tracking-batch, Task 1).

- :func:`make_production_frames`: a seeded multi-period match in the ``TRACKING_FRAMES_COLUMNS`` schema (float32
  coordinates, ``team_id`` category) with 22 players and the ball, detection gaps, NaN detections, and goalkeeper
  flags. The bit-identity gates (spec §5.1) run on it.
- :func:`make_two_game_inputs`: two games that share ``period_id``, ``frame_id``, and ``action_id``, with game 8's
  clock 2 s later (the spec §4.9 defect shape). The geometry is the aggregator-liveness fixture's, so every registered
  ``add_*`` aggregator produces real values on it.
- :func:`spec_4_9_reproduction_inputs`: the exact spec §4.9 probe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.tracking.schema import TRACKING_FRAMES_COLUMNS
from silly_kicks.vaep.feature_framework import gamestates
from tests.tracking.test_aggregator_column_liveness import make_actions, make_frames

FIELD_LENGTH = 105.0
FIELD_WIDTH = 68.0
GAME_A, GAME_B = 7, 8
CLOCK_OFFSET_S = 2.0
#: Game 8's actions run this much earlier than game 7's on the shifted clock: one frame step of the liveness
#: frames, so action k of game 8 links to the frame BEFORE the one that action k of game 7 links to. Then one
#: ``action_id`` names a different ``frame_id`` in each game, and a pointer re-attached by ``action_id`` reads
#: the other game's frame (the defensive-line defect that the real-data W9.6 run found, 2026-11-12).
ACTION_LEAD_S = 0.35
_HOME, _AWAY = 5, 6
#: The home team of both games of make_two_game_inputs (and of each game alone).
HOME_TEAM_ID = _HOME
#: Game 8's player ids are game 7's plus this offset (the teams are the same two clubs).
PLAYER_ID_OFFSET = 100

# 4-4-2 base shape for the team that attacks +x (index 0 is the goalkeeper).
_SHAPE_442 = np.array(
    [
        [5.0, 34.0],
        [25.0, 8.0],
        [22.0, 26.0],
        [22.0, 42.0],
        [25.0, 60.0],
        [45.0, 10.0],
        [42.0, 28.0],
        [42.0, 40.0],
        [45.0, 58.0],
        [62.0, 28.0],
        [62.0, 40.0],
    ]
)


def make_production_frames(
    seed: int = 20261108,
    n_frames: int = 2000,
    n_periods: int = 2,
    *,
    mixed_ids: bool = False,
) -> pd.DataFrame:
    """A seeded match: ``n_frames`` frames per period at 25 fps, 22 players and the ball.

    Each player follows a smoothed random walk around a 4-4-2 shape. About 3 % of player rows are missing (detection
    gaps), and about 1 % carry NaN ``x``/``y``. The home team (5) attacks +x in odd periods, and the away team (6)
    the other way. With ``mixed_ids=True``, the first half of each period's frames carries ``int`` player and team
    ids and the second half ``str`` ids (``player_id`` and ``team_id`` are then ``object``).
    """
    rng = np.random.default_rng(seed)
    rows: list[pd.DataFrame] = []
    frame_ids = np.arange(n_frames)
    for period in range(1, n_periods + 1):
        home_ltr = period % 2 == 1
        drift = np.cumsum(rng.normal(0.0, 0.08, size=(n_frames, 2)), axis=0)  # the block moves together
        for team, base_ltr in ((_HOME, home_ltr), (_AWAY, not home_ltr)):
            shape = _SHAPE_442 if base_ltr else np.column_stack([FIELD_LENGTH - _SHAPE_442[:, 0], _SHAPE_442[:, 1]])
            for k in range(11):
                walk = np.cumsum(rng.normal(0.0, 0.12, size=(n_frames, 2)), axis=0)
                walk -= np.linspace(0.0, 1.0, n_frames)[:, None] * walk[-1]  # stay near the base shape
                xy = shape[k] + drift + walk
                keep = rng.random(n_frames) >= 0.03
                xs = np.clip(xy[keep, 0], 0.0, FIELD_LENGTH)
                ys = np.clip(xy[keep, 1], 0.0, FIELD_WIDTH)
                nan = rng.random(len(xs)) < 0.01
                xs[nan] = np.nan
                ys[nan] = np.nan
                rows.append(
                    pd.DataFrame(
                        {
                            "period_id": period,
                            "frame_id": frame_ids[keep],
                            "player_id": team * 100 + k,
                            "team_id": team,
                            "is_ball": False,
                            "is_goalkeeper": k == 0,
                            "x": xs,
                            "y": ys,
                            "team_attacking_direction": "ltr" if base_ltr else "rtl",
                        }
                    )
                )
        ball = np.column_stack([52.5 + 30.0 * np.sin(frame_ids / 90.0), 34.0 + 20.0 * np.cos(frame_ids / 70.0)])
        ball += drift
        rows.append(
            pd.DataFrame(
                {
                    "period_id": period,
                    "frame_id": frame_ids,
                    "player_id": pd.NA,
                    "team_id": pd.NA,
                    "is_ball": True,
                    "is_goalkeeper": False,
                    "x": np.clip(ball[:, 0], 0.0, FIELD_LENGTH),
                    "y": np.clip(ball[:, 1], 0.0, FIELD_WIDTH),
                    "team_attacking_direction": None,
                }
            )
        )
    f = pd.concat(rows, ignore_index=True).sort_values(["period_id", "frame_id", "is_ball"], kind="stable")
    f = f.reset_index(drop=True)
    f["game_id"] = GAME_A
    f["time_seconds"] = f["frame_id"] / 25.0
    f["frame_rate"] = 25.0
    f["z"] = np.where(f["is_ball"], 0.0, np.nan)
    f["speed"] = np.nan
    f["speed_source"] = "unavailable"
    f["ball_state"] = "alive"
    f["visibility"] = None
    f["source_provider"] = "synthetic"
    f["is_goalkeeper_source"] = "native"
    f["z_source"] = "measured"
    if mixed_ids:
        str_half = f["frame_id"].to_numpy() >= n_frames // 2
        for col in ("player_id", "team_id"):
            vals = f[col].astype(object).to_numpy()
            out = np.array(
                [v if (pd.isna(v) or not s) else str(v) for v, s in zip(vals, str_half, strict=True)], object
            )
            f[col] = pd.Series(out, index=f.index, dtype=object)
        dtypes = {k: v for k, v in TRACKING_FRAMES_COLUMNS.items() if k not in ("player_id", "team_id")}
    else:
        dtypes = dict(TRACKING_FRAMES_COLUMNS)
    return f[list(TRACKING_FRAMES_COLUMNS)].astype(dtypes)


def make_two_game_inputs(seed: int = 20261108) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(actions, frames)`` for games 7 and 8.

    Both games use the aggregator-liveness geometry, the same ``period_id``, ``frame_id``, and ``action_id`` values,
    and the same two teams with the same home team (5). Game 8's clock runs ``CLOCK_OFFSET_S`` later, its actions
    run a further ``ACTION_LEAD_S`` earlier (so one ``action_id`` links to a different ``frame_id`` in each game),
    its player positions and velocities carry a seeded jitter, and its player ids are offset by
    ``PLAYER_ID_OFFSET``, so a cross-game link changes values and ids. A scalar ``home_team_id`` is therefore valid
    for the whole call and for each game (A3-PLAN-68).
    """
    rng = np.random.default_rng(seed)
    actions_a, frames_a = _with_interception(make_actions()), make_frames()
    actions_a["game_id"] = GAME_A
    frames_a["game_id"] = GAME_A
    actions_b, frames_b = actions_a.copy(), frames_a.copy()
    actions_b["game_id"] = GAME_B
    frames_b["game_id"] = GAME_B
    actions_b["time_seconds"] = actions_b["time_seconds"] + CLOCK_OFFSET_S - ACTION_LEAD_S
    frames_b["time_seconds"] = frames_b["time_seconds"] + CLOCK_OFFSET_S
    players = ~frames_b["is_ball"].to_numpy()
    for col, hi in (("x", FIELD_LENGTH - 0.5), ("y", FIELD_WIDTH - 0.5)):
        jitter = rng.normal(0.0, 0.6, size=int(players.sum()))
        frames_b.loc[players, col] = np.clip(frames_b.loc[players, col].to_numpy() + jitter, 0.5, hi)
    for col in ("vx", "vy"):  # the kinematics differ too, so a cross-game speed or velocity read changes values
        frames_b.loc[players, col] = frames_b.loc[players, col].to_numpy() + rng.normal(0.0, 0.4, int(players.sum()))
    frames_b.loc[players, "speed"] = np.hypot(frames_b.loc[players, "vx"], frames_b.loc[players, "vy"])
    # game 8 fields other players: an id read from the other game's frames is a different id
    frames_b["player_id"] = frames_b["player_id"] + PLAYER_ID_OFFSET
    for col in ("player_id", "defending_gk_player_id"):
        actions_b[col] = actions_b[col] + PLAYER_ID_OFFSET
    actions = pd.concat([_complete_spadl(actions_a), _complete_spadl(actions_b)], ignore_index=True)
    frames = pd.concat([_with_possession(frames_a), _with_possession(frames_b)], ignore_index=True)
    return actions, frames


def _with_interception(actions: pd.DataFrame) -> pd.DataFrame:
    """The liveness actions plus one away-team interception (SPADL type 10) at t=49.67 s, inside the cross window
    (its pressing defender 20 is near the ball there), so the defensive members have a defensive action to score.
    ``action_id`` is renumbered chronologically (the converter contract, ADR-065)."""
    row = actions[actions["type_name"] == "cross"].copy()
    row = row.assign(
        time_seconds=49.67,
        team_id=pd.Series([_AWAY], index=row.index, dtype="int64"),
        player_id=pd.Series([20], index=row.index, dtype="int64"),
        defending_gk_player_id=pd.Series([1], index=row.index, dtype="int64"),
        start_x=80.0,
        start_y=60.5,
        end_x=80.0,
        end_y=60.5,
        type_id=10,
        type_name="interception",
    )
    out = pd.concat([actions, row], ignore_index=True).sort_values("time_seconds", kind="stable")
    return out.reset_index(drop=True).assign(action_id=np.arange(len(out)))


def _complete_spadl(actions: pd.DataFrame) -> pd.DataFrame:
    """Add the SPADL columns that the liveness fixture leaves out (``original_event_id``, the 3-valued block flags)."""
    out = actions.copy()
    out["original_event_id"] = [f"{g}-{a}" for g, a in zip(out["game_id"], out["action_id"], strict=True)]
    for col in ("shot_blocked", "cross_blocked"):
        out[col] = pd.array([pd.NA] * len(out), dtype="boolean")
    return out


def _with_possession(frames: pd.DataFrame) -> pd.DataFrame:
    """One game's frames plus ``team_in_possession`` (the DAS prerequisite), derived per game as the liveness
    fixture does."""
    from silly_kicks.tracking import derive_team_in_possession, infer_ball_carrier

    return derive_team_in_possession(frames, infer_ball_carrier(frames))


def make_single_game_inputs(seed: int = 20261108) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Game 7 of :func:`make_two_game_inputs`."""
    actions, frames = make_two_game_inputs(seed)
    keep_a = actions["game_id"].to_numpy() == GAME_A
    keep_f = frames["game_id"].to_numpy() == GAME_A
    return actions[keep_a].reset_index(drop=True), frames[keep_f].reset_index(drop=True)


def make_gamestate_slots(actions: pd.DataFrame, nb_prev_actions: int = 3) -> list[pd.DataFrame]:
    """The VAEP gamestate slots of ``actions`` (duplicate ``action_id``s by construction), from the public
    ``silly_kicks.vaep.feature_framework.gamestates``."""
    return gamestates(actions, nb_prev_actions)


def spec_4_9_reproduction_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The spec §4.9 probe: games 7 and 8, 100 frames each (``frame_id`` 0..99, period 1, 25 fps), game 8's clock
    2 s later, a ball row per frame, and actions 0, 1 in each game at ``[0.40, 1.20]`` and ``[2.40, 3.20]`` s."""
    frames = []
    for game, offset in ((GAME_A, 0.0), (GAME_B, CLOCK_OFFSET_S)):
        fid = np.arange(100)
        frames.append(
            pd.DataFrame(
                {
                    "game_id": game,
                    "period_id": 1,
                    "frame_id": fid,
                    "time_seconds": fid / 25.0 + offset,
                    "frame_rate": 25.0,
                    "player_id": pd.array([pd.NA] * 100, dtype="Int64"),
                    "team_id": pd.array([pd.NA] * 100, dtype="Int64"),
                    "is_ball": True,
                    "is_goalkeeper": False,
                    "x": 52.5,
                    "y": 34.0,
                    "team_attacking_direction": None,
                    "source_provider": "synthetic",  # a TRACKING_FRAMES_COLUMNS column the linker reads
                }
            )
        )
    actions = pd.DataFrame(
        {
            "game_id": [GAME_A, GAME_A, GAME_B, GAME_B],
            "period_id": 1,
            "action_id": [0, 1, 0, 1],
            "time_seconds": [0.40, 1.20, 2.40, 3.20],
            "team_id": _HOME,
            "player_id": 10,
            "start_x": 50.0,
            "start_y": 34.0,
            "end_x": 60.0,
            "end_y": 34.0,
            "type_id": 0,
            "result_id": 1,
            "bodypart_id": 0,
        }
    )
    return actions, pd.concat(frames, ignore_index=True)
