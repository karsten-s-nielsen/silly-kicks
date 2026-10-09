"""Deterministic synthetic match for the TF-58 coordination suite (Tasks 15-18).

Team centroids oscillate longitudinally at ``oscillation_cpm``; team B lags team A by ``phase_offset_deg``.
Players jitter around symmetric formation slots (so each team's centroid IS the planted oscillation). Team 1
attacks LTR in period 1 (GK near x=4), team 2 RTL (GK near x=101). Style follows ``tests/restdefense/_fixtures``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

import silly_kicks.spadl.config as spadlconfig

_AMP_M = 15.0  # centroid longitudinal oscillation amplitude
_CENTRE_X = 52.5
_PITCH_Y = 68.0
_DETECTION_AWARE = {"skillcorner"}


def make_coordination_match(
    *,
    seconds: float = 900.0,
    hz: float = 10.0,
    provider: str = "sportec",
    n_outfield: int = 10,
    with_gk: bool = True,
    oscillation_cpm: float = 0.5,
    phase_offset_deg: float = 40.0,
    noise_m: float = 0.3,
    dead_intervals: Sequence[tuple[float, float]] = (),
    visibility_drop: float = 0.0,
    substitution: tuple[float, int] | None = None,
    red_card: tuple[float, int] | None = None,
    team_ids: tuple[object, object] = (1, 2),
    seed: int = 58,
    periods: int = 1,
) -> pd.DataFrame:
    """A TRACKING_FRAMES_COLUMNS-conformant synthetic match (see module docstring)."""
    rng = np.random.default_rng(seed)
    f_hz = oscillation_cpm / 60.0
    # symmetric longitudinal depth + lateral slots so the team mean equals the centroid oscillation.
    depth = np.linspace(-12.0, 12.0, n_outfield)
    depth = depth - depth.mean()
    lateral = np.linspace(6.0, _PITCH_Y - 6.0, n_outfield)

    cols: dict[str, list[np.ndarray]] = {c: [] for c in _COLS}
    frame_base = 0
    for period in range(1, periods + 1):
        n = round(seconds * hz)
        t = np.arange(n) / hz
        frame_id = frame_base + np.arange(n)
        dead = _dead_mask(t, dead_intervals)
        dir_a = "ltr" if period == 1 else "rtl"
        dir_b = "rtl" if period == 1 else "ltr"
        centroid_a = _CENTRE_X + _AMP_M * np.sin(2 * np.pi * f_hz * t)
        centroid_b = _CENTRE_X + _AMP_M * np.sin(2 * np.pi * f_hz * t - np.radians(phase_offset_deg))

        _append_ball(cols, period, frame_id, t, hz, provider, dead, centroid_a)
        for team, centroid, direction in (
            (team_ids[0], centroid_a, dir_a),
            (team_ids[1], centroid_b, dir_b),
        ):
            gk_x = 4.0 if direction == "ltr" else 101.0  # GK sits by the team's own (defended) goal
            if with_gk:
                _append_player(
                    cols,
                    period,
                    frame_id,
                    t,
                    hz,
                    provider,
                    dead,
                    direction,
                    rng,
                    noise_m,
                    visibility_drop,
                    team=team,
                    player_id=_pid(team, 0),
                    is_gk=True,
                    x=np.full(n, gk_x),
                    y=np.full(n, _PITCH_Y / 2),
                )
            for k in range(n_outfield):
                pid = _pid(team, k + 1)
                x = centroid + depth[k]
                y = np.full(n, lateral[k])
                mask = _entity_mask(t, k, substitution, red_card)
                if substitution is not None and substitution[1] == k and t.max() >= substitution[0]:
                    pid_arr = np.where(t >= substitution[0], _pid(team, 900 + k), pid)
                else:
                    pid_arr = np.full(n, pid)
                _append_player(
                    cols,
                    period,
                    frame_id,
                    t,
                    hz,
                    provider,
                    dead,
                    direction,
                    rng,
                    noise_m,
                    visibility_drop,
                    team=team,
                    player_id=pid_arr,
                    is_gk=False,
                    x=x,
                    y=y,
                    keep=mask,
                )
        frame_base += n

    data = {c: np.concatenate(cols[c]) if cols[c] else np.array([]) for c in _COLS}
    return _finalize(pd.DataFrame(data))


def make_coordination_actions(
    frames: pd.DataFrame,
    *,
    restarts: Sequence[tuple[float, str]] = (),
    goals: Sequence[float] = (),
    possession_every_s: float = 12.0,
) -> pd.DataFrame:
    """Minimal SPADL actions with alternating possessions plus the given restarts and goals."""
    period = int(frames["period_id"].iloc[0])
    game = int(frames["game_id"].iloc[0])
    teams = [int(x) for x in pd.unique(frames.loc[~frames["is_ball"], "team_id"].dropna())][:2]
    end_t = float(frames["time_seconds"].max())
    rows: list[dict[str, object]] = []
    goal_set = set(goals)
    restart_map = dict((round(t, 3), name) for t, name in restarts)
    t = 0.0
    while t < end_t:
        team = teams[int(t // possession_every_s) % 2]
        name = restart_map.get(round(t, 3), "pass")
        result = "success"
        if t in goal_set:
            name, result = "shot", "success"
        rows.append(_action_row(game, period, team, t, name, result))
        t += possession_every_s
    for gt in goals:
        if gt not in {r["time_seconds"] for r in rows}:
            rows.append(_action_row(game, period, teams[0], float(gt), "shot", "success"))
    actions = pd.DataFrame(rows).sort_values("time_seconds").reset_index(drop=True)
    actions["action_id"] = np.arange(len(actions))
    return actions


# --------------------------------------------------------------------------- internals
_COLS = (
    "game_id",
    "period_id",
    "frame_id",
    "time_seconds",
    "frame_rate",
    "player_id",
    "team_id",
    "is_ball",
    "is_goalkeeper",
    "x",
    "y",
    "z",
    "speed",
    "speed_source",
    "ball_state",
    "team_attacking_direction",
    "visibility",
    "source_provider",
    "is_goalkeeper_source",
)


def _pid(team, k: int) -> int:
    return int(team) * 100 + k


def _dead_mask(t: np.ndarray, intervals: Sequence[tuple[float, float]]) -> np.ndarray:
    mask = np.zeros(t.shape, dtype=bool)
    for lo, hi in intervals:
        mask |= (t >= lo) & (t < hi)
    return mask


def _entity_mask(t: np.ndarray, k: int, sub, red) -> np.ndarray:
    keep = np.ones(t.shape, dtype=bool)
    if red is not None and red[1] == k:
        keep &= t < red[0]
    return keep


def _append_ball(cols, period, frame_id, t, hz, provider, dead, ball_x) -> None:
    n = t.shape[0]
    _push(
        cols,
        n,
        period=period,
        frame_id=frame_id,
        t=t,
        hz=hz,
        provider=provider,
        player_id=np.full(n, np.nan),
        team_id=np.full(n, np.nan),
        is_ball=True,
        is_gk=False,
        x=ball_x,
        y=np.full(n, _PITCH_Y / 2),
        dead=dead,
        direction="",
        visibility=np.full(n, None),
    )


def _append_player(
    cols,
    period,
    frame_id,
    t,
    hz,
    provider,
    dead,
    direction,
    rng,
    noise_m,
    visibility_drop,
    *,
    team,
    player_id,
    is_gk,
    x,
    y,
    keep=None,
) -> None:
    n = t.shape[0]
    xj = x + rng.normal(0, noise_m, n)
    yj = y + rng.normal(0, noise_m, n)
    if provider in _DETECTION_AWARE:
        vis = rng.random(n) >= visibility_drop
        visibility = vis.astype(object)
    else:
        visibility = np.full(n, None)
    pid = player_id if isinstance(player_id, np.ndarray) else np.full(n, player_id)
    team_arr = np.full(n, int(team))
    _push(
        cols,
        n,
        period=period,
        frame_id=frame_id,
        t=t,
        hz=hz,
        provider=provider,
        player_id=pid,
        team_id=team_arr,
        is_ball=False,
        is_gk=is_gk,
        x=xj,
        y=yj,
        dead=dead,
        direction=direction,
        visibility=visibility,
        keep=keep,
    )


def _push(
    cols,
    n,
    *,
    period,
    frame_id,
    t,
    hz,
    provider,
    player_id,
    team_id,
    is_ball,
    is_gk,
    x,
    y,
    dead,
    direction,
    visibility,
    keep=None,
) -> None:
    if keep is None:
        keep = np.ones(n, dtype=bool)
    arrays = {
        "game_id": np.ones(n, dtype=np.int64),
        "period_id": np.full(n, period, dtype=np.int64),
        "frame_id": np.asarray(frame_id, dtype=np.int64),
        "time_seconds": np.asarray(t, dtype=np.float64),
        "frame_rate": np.full(n, hz, dtype=np.float64),
        "player_id": np.asarray(player_id, dtype=object),
        "team_id": np.asarray(team_id, dtype=object),
        "is_ball": np.full(n, is_ball, dtype=bool),
        "is_goalkeeper": np.full(n, is_gk, dtype=bool),
        "x": np.asarray(x, dtype=np.float64),
        "y": np.asarray(y, dtype=np.float64),
        "z": np.zeros(n, dtype=np.float64),
        "speed": np.zeros(n, dtype=np.float64),
        "speed_source": np.full(n, "unavailable", dtype=object),
        "ball_state": np.where(dead, "dead", "alive").astype(object),
        "team_attacking_direction": np.full(n, direction, dtype=object),
        "visibility": np.asarray(visibility, dtype=object),
        "source_provider": np.full(n, provider, dtype=object),
        "is_goalkeeper_source": np.full(n, "provided", dtype=object),
    }
    for c in _COLS:
        cols[c].append(arrays[c][keep])


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce").astype("Int64")
    df["ball_state"] = df["ball_state"].astype("category")
    df["source_provider"] = df["source_provider"].astype("category")
    df["is_goalkeeper_source"] = df["is_goalkeeper_source"].astype("category")
    return df.reset_index(drop=True)


def _action_row(game: int, period: int, team: int, t: float, name: str, result: str) -> dict[str, object]:
    return {
        "game_id": game,
        "period_id": period,
        "time_seconds": float(t),
        "team_id": team,
        "player_id": team * 100 + 1,
        "type_id": spadlconfig.actiontypes.index(name),
        "result_id": spadlconfig.results.index(result),
        "start_x": _CENTRE_X,
        "start_y": _PITCH_Y / 2,
        "end_x": _CENTRE_X + 10.0,
        "end_y": _PITCH_Y / 2,
        "bodypart_id": spadlconfig.bodyparts.index("foot"),
    }
