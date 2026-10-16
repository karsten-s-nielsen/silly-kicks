"""compute_corner_run_encoding -- per-corner probabilistic run vector (TF-46 Arm 1, Shaw §3.1).

For each DIRECT corner, slice the attacking team's player positions at ``t_init`` and ``t_target`` from
the tracking frames, reflect them to the canonical side (Y-flip for right-flag corners), align the two
snapshots by player, and encode via ``RunZoneModel`` into the ``n_initial x n_target`` run vector. Short
and unclear corners are skipped (kept as NaN rows, ADR-042). The columns are the NMF term-matrix columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.id_compat import canonical_id, canonical_id_series
from silly_kicks.spadl import config as spadlconfig

from ._columns import CORNER_KEYS
from ._config import RunEncodingParams
from ._zones import RunZoneModel

_DEFAULT_ENCODING_PARAMS = RunEncodingParams()
_FIELD_WIDTH = float(spadlconfig.field_width)


def _run_column_names(model: RunZoneModel) -> list[str]:
    """The ``run_<i><t>`` column names (initial 1..n, target a..g), Shaw's alphanumeric run code."""
    return [f"run_{i + 1}{chr(97 + t)}" for i in range(model.n_initial_zones) for t in range(model.n_target_zones)]


def _nearest_frame_xy(frames: pd.DataFrame, period: int, t: float, team) -> pd.DataFrame | None:
    """The attacking team's player (player_id, x, y) at the frame nearest ``t`` in ``period``."""
    sub = frames[(frames["period_id"] == period) & (canonical_id_series(frames["team_id"]) == canonical_id(team))]
    sub = sub[~sub.get("is_ball", pd.Series(False, index=sub.index)).astype(bool)] if "is_ball" in sub.columns else sub
    if sub.empty:
        return None
    fid = sub.iloc[(sub["time_seconds"] - t).abs().to_numpy().argmin()]["frame_id"]
    snap = sub[sub["frame_id"] == fid][["player_id", "x", "y"]].reset_index(drop=True)
    return snap if not snap.empty else None


def _reflect_y(snap: pd.DataFrame, side: str) -> pd.DataFrame:
    """Canonical-side Y-reflection of a snapshot (x invariant; right-flag corners flip y)."""
    if side == "left":
        return snap
    out = snap.copy()
    out["y"] = _FIELD_WIDTH - out["y"].to_numpy(dtype=np.float64)
    return out


def _encode_one(corner: pd.Series, frames: pd.DataFrame, model: RunZoneModel, params: RunEncodingParams):
    period = int(corner["period_id"])
    team = corner["attacking_team_id"]
    # SB360 (freeze-frame only): the initial snapshot is the KICK freeze frame (t_kick), not the -2s
    # continuous snapshot -- consistent with the fit driver (spec §7; divergence measured in §12.7).
    sb360 = "corner_source" in corner and str(corner["corner_source"]) == "sb360_event"
    init_t = float(corner["t_kick"]) if sb360 else float(corner["t_init"])
    init = _nearest_frame_xy(frames, period, init_t, team)
    target = _nearest_frame_xy(frames, period, float(corner["t_target"]), team)
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
    moved = np.hypot(target_xy[:, 0] - init_xy[:, 0], target_xy[:, 1] - init_xy[:, 1]) >= params.min_run_distance_m
    return model.encode_runs(init_xy, target_xy, active_mask=moved if moved.any() else None)


def compute_corner_run_encoding(
    corners: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    model: RunZoneModel,
    params: RunEncodingParams = _DEFAULT_ENCODING_PARAMS,
) -> pd.DataFrame:
    """Per-corner run vector at grain ``(game_id, corner_id)``. Pure (new frame).

    Examples
    --------
    Encode corner routines once a RunZoneModel is fitted::

        from silly_kicks.corners import compute_corner_run_encoding
        encoding = compute_corner_run_encoding(windowed_corners, frames, model=run_zone_model)
    """
    run_cols = _run_column_names(model)
    rows: list[dict] = []
    for _, c in corners.iterrows():
        row: dict = {"game_id": c["game_id"], "corner_id": int(c["corner_id"])}
        vec = None if str(c["corner_kind"]) in ("short", "unclear") else _encode_one(c, frames, model, params)
        if vec is None:
            row.update(dict.fromkeys(run_cols, np.nan))
        else:
            row.update(dict(zip(run_cols, vec, strict=True)))
        rows.append(row)
    cols = [*CORNER_KEYS, *run_cols]
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype="float64" if c in run_cols else "object") for c in cols})
    return pd.DataFrame(rows, columns=cols)
