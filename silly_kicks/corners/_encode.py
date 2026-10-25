"""compute_corner_run_encoding -- per-corner probabilistic run vector (TF-46 Arm 1, Shaw §3.1).

For each DIRECT corner, slice the attacking team's player positions at ``t_init`` and ``t_target`` from
the tracking frames, reflect them to the canonical side (Y-flip for right-flag corners), align the two
snapshots by player, and encode via ``RunZoneModel`` into the ``n_initial x n_target`` run vector. Short
and unclear corners are skipped (kept as NaN rows, ADR-042). The columns are the NMF term-matrix columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.id_compat import ids_match
from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking._gk_resolve import GoalMap

from ._columns import CORNER_KEYS
from ._config import RunEncodingParams
from ._zones import RunZoneModel

_DEFAULT_ENCODING_PARAMS = RunEncodingParams()
_FIELD_WIDTH = float(spadlconfig.field_width)


def _run_column_names(model: RunZoneModel) -> list[str]:
    """The ``run_<i><t>`` column names (initial 1..n, target a..g), Shaw's alphanumeric run code."""
    return [f"run_{i + 1}{chr(97 + t)}" for i in range(model.n_initial_zones) for t in range(model.n_target_zones)]


def _nearest_frame_xy(frames: pd.DataFrame, period: int, t: float, team, *, goal_map: GoalMap) -> pd.DataFrame | None:
    """The attacking team's player (player_id, x, y) at the frame nearest ``t`` in ``period``, in the ACTION
    frame of that team (it attacks +x)."""
    # NA-safe id compare (ADR-019): the ball row carries a <NA> team_id, so a raw ``==`` yields an
    # NA-bearing mask that crashes under boolean indexing. Both masks are positional numpy arrays --
    # ids_match returns a fresh-RangeIndex Series, so never ``&`` it with a label-indexed Series.
    period_mask = (frames["period_id"] == period).to_numpy()
    team_mask = ids_match(frames["team_id"], team).to_numpy()
    sub = frames[period_mask & team_mask]
    sub = sub[~sub.get("is_ball", pd.Series(False, index=sub.index)).astype(bool)] if "is_ball" in sub.columns else sub
    if sub.empty:
        return None
    fid = sub.iloc[(sub["time_seconds"] - t).abs().to_numpy().argmin()]["frame_id"]
    snap = sub[sub["frame_id"] == fid].reset_index(drop=True)
    if snap.empty:
        return None
    # F0 (TF-46 Arm 1 Part 2, spec section 13.3): orient the snapshot to the ACTION frame (the attacking
    # team attacks +x) so the zone geometry (_ATTACKED_GOAL_X = 105) holds for either play direction. The
    # direction comes from the GoalMap ONLY (ADR-055: one defended-goal implementation; never a frame label
    # or team identity, ADR-051). An unresolved direction (<NA>) gives NaN coords (the encoding withholds).
    from silly_kicks.tracking._restart_frame import team_attacks_rtl, to_action_frame

    gid = snap["game_id"].iloc[0] if "game_id" in snap.columns else None
    key = pd.DataFrame({"game_id": [gid] * len(snap), "period_id": [period] * len(snap), "team_id": [team] * len(snap)})
    rtl = team_attacks_rtl(key, goal_map)
    ax, ay = to_action_frame(snap["x"].to_numpy(dtype=np.float64), snap["y"].to_numpy(dtype=np.float64), rtl)
    snap = snap.assign(x=ax, y=ay)
    snap = snap[np.isfinite(ax) & np.isfinite(ay)]
    return snap[["player_id", "x", "y"]].reset_index(drop=True) if not snap.empty else None


def _reflect_y(snap: pd.DataFrame, side: str) -> pd.DataFrame:
    """Canonical-side Y-reflection of a snapshot (x invariant; right-flag corners flip y)."""
    if side == "left":
        return snap
    out = snap.copy()
    out["y"] = _FIELD_WIDTH - out["y"].to_numpy(dtype=np.float64)
    return out


def _encode_one(
    corner: pd.Series, frames: pd.DataFrame, model: RunZoneModel, params: RunEncodingParams, goal_map: GoalMap
):
    period = int(corner["period_id"])
    team = corner["attacking_team_id"]
    # SB360 (freeze-frame only): the initial snapshot is the KICK freeze frame (t_kick), not the -2s
    # continuous snapshot -- consistent with the fit driver (spec §7; divergence measured in §12.7).
    sb360 = "corner_source" in corner and str(corner["corner_source"]) == "sb360_event"
    init_t = float(corner["t_kick"]) if sb360 else float(corner["t_init"])
    init = _nearest_frame_xy(frames, period, init_t, team, goal_map=goal_map)
    target = _nearest_frame_xy(frames, period, float(corner["t_target"]), team, goal_map=goal_map)
    if init is None or target is None:
        return None
    side = str(corner["corner_side"])
    init, target = _reflect_y(init, side), _reflect_y(target, side)
    # align the two snapshots by player (a run is one player's init -> target)
    merged = init.merge(target, on="player_id", suffixes=("_i", "_t"))
    if merged.empty:
        return None
    init_xy = merged[["x_i", "y_i"]].to_numpy(dtype=np.float64)
    target_xy = merged[["x_t", "y_t"]].to_numpy(dtype=np.float64)
    # Active players follow the model's OCCUPANCY rule (target in the danger+approach region); no
    # displacement threshold, so a stationary second-ball/edge holder is counted (spec §6, 2026-10-18).
    return model.encode_runs(init_xy, target_xy)


def compute_corner_run_encoding(
    corners: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    model: RunZoneModel,
    params: RunEncodingParams = _DEFAULT_ENCODING_PARAMS,
    goal_map: GoalMap | None = None,
) -> pd.DataFrame:
    """Per-corner run vector at grain ``(game_id, corner_id)``. Pure (new frame).

    ``goal_map`` (ADR-055) orients each corner's snapshots to the attacking team's action frame (F0); when
    omitted it is resolved ONCE from ``frames`` via ``resolve_defended_goals``.

    Examples
    --------
    Encode corner routines once a RunZoneModel is fitted::

        from silly_kicks.corners import compute_corner_run_encoding
        encoding = compute_corner_run_encoding(windowed_corners, frames, model=run_zone_model)
    """
    run_cols = _run_column_names(model)
    if goal_map is None and len(corners):
        from silly_kicks.tracking._gk_resolve import resolve_defended_goals

        goal_map = resolve_defended_goals(frames)
    rows: list[dict] = []
    for _, c in corners.iterrows():
        row: dict = {"game_id": c["game_id"], "corner_id": int(c["corner_id"])}
        skip = str(c["corner_kind"]) in ("short", "unclear")
        vec = None if skip or goal_map is None else _encode_one(c, frames, model, params, goal_map)
        if vec is None:
            row.update(dict.fromkeys(run_cols, np.nan))
        else:
            row.update(dict(zip(run_cols, vec, strict=True)))
        rows.append(row)
    cols = [*CORNER_KEYS, *run_cols]
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype="float64" if c in run_cols else "object") for c in cols})
    return pd.DataFrame(rows, columns=cols)
