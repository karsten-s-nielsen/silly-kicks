"""Nearest-opponent(s) resolution within a box-aware threshold, in action-LTR coords."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from silly_kicks.id_compat import ids_match
from silly_kicks.tracking._action_orientation import acting_team_attacks_rtl

from ._params import DefensiveCreditParams

Mode = Literal["nearest", "all_within", "all_within_beyond_nearest"]

_FIELD_LENGTH = 105.0
_FIELD_WIDTH = 68.0


def resolve_responsible_defenders(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    anchor_x: float,
    anchor_y: float,
    acting_team_id,
    mode: Mode,
    params: DefensiveCreditParams,
    frame_id: int | None = None,
    flip: bool | None = None,
) -> pd.DataFrame:
    """Return opponents within the box-aware threshold of the (anchor_x, anchor_y) action-LTR point.

    Columns: player_id, team_id, distance_m (ascending). Empty when none are within threshold.
    ``frame_id``: the linked frame for the triggering action; if None, uses the single frame present.
    ``flip``: the precomputed action-LTR reprojection decision; if None, computed here (unit-test path).
    """
    if frame_id is not None:
        fr = frames[frames["frame_id"] == frame_id]
    else:
        fr = frames
    # opponents only (team_id != acting team) -- dtype-safe (ADR-019); exclude the ball + NaN teams
    is_opponent = ~ids_match(fr["team_id"], acting_team_id) & fr["team_id"].notna() & ~fr["is_ball"].astype(bool)
    opp = fr[is_opponent.to_numpy()].copy()
    if opp.empty:
        return _empty()

    # reproject opponent positions to action-LTR for THIS action (ADR-028). The orchestrator passes
    # the precomputed `flip`; the single-action unit path computes it here.
    if flip is None:
        flip = bool(acting_team_attacks_rtl(actions, frames).iloc[0])
    px = _FIELD_LENGTH - opp["x"].to_numpy() if flip else opp["x"].to_numpy()
    py = _FIELD_WIDTH - opp["y"].to_numpy() if flip else opp["y"].to_numpy()

    dist = np.hypot(px - anchor_x, py - anchor_y)
    thr = params._proximity_threshold(anchor_x, anchor_y)
    within = dist <= thr
    if not within.any():
        return _empty()

    out = (
        pd.DataFrame(
            {
                "player_id": opp["player_id"].to_numpy()[within],
                "team_id": opp["team_id"].to_numpy()[within],
                "distance_m": dist[within],
            }
        )
        .sort_values("distance_m", kind="stable")
        .reset_index(drop=True)
    )

    if mode == "nearest":
        return out.iloc[[0]].reset_index(drop=True)
    if mode == "all_within_beyond_nearest":
        return out.iloc[1:].reset_index(drop=True)
    return out  # all_within


def _empty() -> pd.DataFrame:
    return pd.DataFrame({"player_id": [], "team_id": [], "distance_m": pd.Series([], dtype="float64")})
