"""Shared nearest-opponent resolution primitive (TF-51 v2 lift, N6).

The ONE producer of "who is the nearest opponent(s) within a distance", in action-LTR coordinates.
``defensive_credit._resolution`` (box-aware threshold) and ``_press_commitment`` (flat press
distance) both consume it, so the opponent mask + scalar-flip reprojection live in exactly one place.
Takes a ``threshold_m: float`` (dependency inverted -- NOT a ``DefensiveCreditParams``), so a generic
tracking feature never depends on the defensive-credit sub-package's params.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.id_compat import ids_match

_FIELD_LENGTH = 105.0
_FIELD_WIDTH = 68.0


def opponents_action_ltr(
    frame_slice: pd.DataFrame, acting_team_id, flip: bool, *, exclude_goalkeeper: bool = False
) -> pd.DataFrame:
    """Opponent rows (team != acting, non-ball, [non-GK]) with action-LTR positions in ``_px``/``_py``.

    ADR-019 dtype-safe opponent mask + ADR-027 NaN-team exclusion + the family's scalar-flip
    reprojection. Returns a COPY with ``_px``/``_py`` added (empty frame if no opponent). GK exclusion
    is the CALLER's choice -- a keeper can press (Item 5) but is never an outfield shot-blocker (Item 2).
    """
    fr = frame_slice
    is_opp = ~ids_match(fr["team_id"], acting_team_id) & fr["team_id"].notna() & ~fr["is_ball"].astype(bool)
    if exclude_goalkeeper:
        is_opp = is_opp & ~fr["is_goalkeeper"].astype(bool)
    opp = fr[is_opp.to_numpy()].copy()
    if opp.empty:
        return opp
    x = opp["x"].to_numpy()
    y = opp["y"].to_numpy()
    opp["_px"] = _FIELD_LENGTH - x if flip else x
    opp["_py"] = _FIELD_WIDTH - y if flip else y
    return opp


def opponents_within(
    frame_slice: pd.DataFrame,
    *,
    anchor_x: float,
    anchor_y: float,
    acting_team_id,
    threshold_m: float,
    flip: bool,
    exclude_goalkeeper: bool = False,
) -> pd.DataFrame:
    """Opponents within ``threshold_m`` of the action-LTR ``(anchor_x, anchor_y)``, sorted by distance.

    Columns: ``player_id``, ``team_id``, ``distance_m`` (ascending). Empty when none within threshold.
    """
    opp = opponents_action_ltr(frame_slice, acting_team_id, flip, exclude_goalkeeper=exclude_goalkeeper)
    if opp.empty:
        return _empty()
    dist = np.hypot(opp["_px"].to_numpy() - anchor_x, opp["_py"].to_numpy() - anchor_y)
    within = dist <= threshold_m
    if not within.any():
        return _empty()
    return (
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


def _empty() -> pd.DataFrame:
    return pd.DataFrame({"player_id": [], "team_id": [], "distance_m": pd.Series([], dtype="float64")})
