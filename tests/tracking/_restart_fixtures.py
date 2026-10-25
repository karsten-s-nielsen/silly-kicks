"""Synthetic ball / frame / action builders for the kick-resolver tests (TF-46 Arm 1 Part 2, Task 2.5)."""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import numpy as np
import pandas as pd

from silly_kicks.spadl import config as spadlconfig


def rest_then_kick(
    *,
    rest_s: float = 1.0,
    kick_v: float = 6.0,
    fps: float = 25.0,
    rest_xy: tuple[float, float] = (50.0, 34.0),
    t0: float = 9.0,
    post_s: float = 1.0,
    seed: int = 0,
) -> pd.DataFrame:
    """A ball trace: rests at ``rest_xy`` (sub-cm jitter) for ``rest_s`` s, then launches +x at ``kick_v`` m/s.
    Columns: ``time_seconds``, ``x``, ``y``."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / fps
    n_rest = round(rest_s * fps)
    n_post = round(post_s * fps)
    t = t0 + dt * np.arange(n_rest + n_post)
    x = np.empty(len(t))
    y = np.empty(len(t))
    x[:n_rest] = rest_xy[0] + rng.normal(0, 0.02, n_rest)
    y[:n_rest] = rest_xy[1] + rng.normal(0, 0.02, n_rest)
    for k in range(n_rest, len(t)):
        x[k] = rest_xy[0] + kick_v * dt * (k - n_rest + 1)
        y[k] = rest_xy[1]
    return pd.DataFrame({"time_seconds": t, "x": x, "y": y})


def frames_from_ball(
    ball: pd.DataFrame,
    *,
    game_id: str = "g",
    period_id: int = 1,
    players: list[dict] | None = None,
) -> pd.DataFrame:
    """Build a frames table from a ball trace plus optional per-frame player rows. ``players`` is a list of
    dicts ``{player_id, team_id, x, y, is_goalkeeper?}`` repeated on every frame."""
    fps_dt = float(np.median(np.diff(ball["time_seconds"]))) if len(ball) > 1 else 0.04
    frame_ids = np.arange(len(ball)) + 1000
    rows = []
    for fi, (_, b) in zip(frame_ids, ball.iterrows(), strict=True):
        rows.append(
            {
                "game_id": game_id,
                "period_id": period_id,
                "frame_id": int(fi),
                "time_seconds": float(b["time_seconds"]),
                "player_id": None,
                "team_id": None,
                "is_ball": True,
                "is_goalkeeper": False,
                "x": float(b["x"]),
                "y": float(b["y"]),
                "visibility": True,
            }
        )
        for p in players or []:
            rows.append(
                {
                    "game_id": game_id,
                    "period_id": period_id,
                    "frame_id": int(fi),
                    "time_seconds": float(b["time_seconds"]),
                    "player_id": p["player_id"],
                    "team_id": p["team_id"],
                    "is_ball": False,
                    "is_goalkeeper": bool(p.get("is_goalkeeper", False)),
                    "x": float(p["x"]),
                    "y": float(p["y"]),
                    "visibility": True,
                }
            )
    df = pd.DataFrame(rows)
    df.attrs["frame_period"] = fps_dt
    return df


def restart_action(
    restart_type: str,
    *,
    game_id: str = "g",
    period_id: int = 1,
    t_event: float = 10.0,
    team_id: str = "H",
    player_id: str | None = "h7",
    provenance: str = "native",
    gi_marker_end: object = pd.NA,
    gi_next_start: object = pd.NA,
) -> pd.DataFrame:
    """One restart action row carrying the columns resolve_kicks reads."""
    return pd.DataFrame(
        [
            {
                "game_id": game_id,
                "period_id": period_id,
                "time_seconds": t_event,
                "team_id": team_id,
                "player_id": player_id,
                "type_id": spadlconfig.actiontype_id[restart_type],
                "action_provenance": provenance,
                "gi_marker_possession_frame_end": gi_marker_end,
                "gi_next_possession_frame_start": gi_next_start,
                "gi_attacking_side": pd.NA,
            }
        ]
    )


class GateTableStub(Mapping):
    """Base for a test gate table: a subclass defines ``get(key, default=None)``; ``Mapping`` is completed from it,
    so the stub satisfies the ports' ``gate_table: Mapping[GateKey, GateEntry]`` annotation. It enumerates no
    keys (``gate`` only looks keys up)."""

    def get(self, key, default=None):
        return default

    def __getitem__(self, key):
        entry = self.get(key)
        if entry is None:
            raise KeyError(key)
        return entry

    def __iter__(self) -> Iterator:
        return iter(())

    def __len__(self) -> int:
        return 0


def piecewise_trace(segments, *, fps: float = 25.0, t0: float = 0.0, start_xy=(0.0, 0.0)) -> pd.DataFrame:
    """A ball trace built from segments, each ``("rest", seconds)`` (stay put, no jitter) or
    ``("move", (x, y), speed_ms)`` (straight line to ``(x, y)`` at ``speed_ms``). Columns ``time_seconds``,
    ``x``, ``y``. Returns the trace; the time each segment STARTS is in ``trace.attrs["segment_starts"]``."""
    dt = 1.0 / fps
    xs, ys = [float(start_xy[0])], [float(start_xy[1])]
    starts = []
    for seg in segments:
        starts.append(t0 + dt * (len(xs) - 1))
        if seg[0] == "rest":
            for _ in range(round(seg[1] * fps)):
                xs.append(xs[-1])
                ys.append(ys[-1])
        else:
            (tx, ty), v = seg[1], seg[2]
            d = float(np.hypot(tx - xs[-1], ty - ys[-1]))
            steps = max(1, int(np.ceil(d / (v * dt))))
            x0, y0 = xs[-1], ys[-1]
            for k in range(1, steps + 1):
                xs.append(x0 + (tx - x0) * k / steps)
                ys.append(y0 + (ty - y0) * k / steps)
    t = t0 + dt * np.arange(len(xs))
    out = pd.DataFrame({"time_seconds": t, "x": xs, "y": ys})
    out.attrs["segment_starts"] = starts
    return out
