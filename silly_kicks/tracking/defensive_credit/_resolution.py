"""Opponent resolution in action-LTR coords: box-aware nearest(s) + the Item-2 shot->goal lane.

A thin adapter over the shared ``tracking._opponent_resolution`` core (N6): the proximity modes call
``opponents_within`` with the box-aware threshold; ``lane_blocker`` BYPASSES the threshold entirely
(spec section 4 / Q2) and uses the shot->goal corridor. Every returned frame carries a ``resolution``
column recording HOW the credited player was determined.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from silly_kicks._frame_index import RowGroups

from silly_kicks.tracking._action_orientation import acting_team_attacks_rtl
from silly_kicks.tracking._opponent_resolution import opponents_action_ltr, opponents_within

from ._params import (
    RESOLUTION_LANE,
    RESOLUTION_NEAREST_FALLBACK,
    DefensiveCreditParams,
)

Mode = Literal["nearest", "all_within", "all_within_beyond_nearest", "lane_blocker"]

_GOAL_XY = (105.0, 34.0)  # attacked goal centre in action-LTR


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
    frame_groups: RowGroups | None = None,
) -> pd.DataFrame:
    """Return opponents responsible for an action-LTR ``(anchor_x, anchor_y)`` anchor.

    Columns: player_id, team_id, distance_m (ascending), resolution. Empty when none qualify.
    ``mode`` -- ``nearest`` / ``all_within`` / ``all_within_beyond_nearest`` use the box-aware origin
    threshold; ``lane_blocker`` (Item 2) uses the shot->goal corridor, excludes the GK, and does NOT
    origin-threshold (with a nearest-to-origin fallback).
    ``frame_id``: the linked frame for the triggering action; if None, uses the single frame present.
    ``flip``: the precomputed action-LTR reprojection decision; if None, computed here (unit-test path).
    ``frame_groups``: an optional ADR-068 ``RowGroups`` prebuilt over ``frames`` keyed on ``frame_id``
    (passed by ``compute_defensive_credits`` to avoid re-scanning ``frames`` per action x rule). When
    ``None`` (unit-test path) the per-call boolean filter is used -- byte-identical, since the lookup
    keys on ``frame_id`` ALONE, exactly matching the filter.
    """
    if frame_groups is not None and frame_id is not None:
        fr = frame_groups.get(frame_id)
    elif frame_id is not None:
        fr = frames[frames["frame_id"] == frame_id]
    else:
        fr = frames
    if flip is None:
        _resolved = acting_team_attacks_rtl(actions, frames).iloc[0]
        if pd.isna(_resolved):
            # Documented unit-test convenience path. The production caller (_orchestration)
            # decides this policy for itself and never reaches here, so refusing is safe and
            # says the useful thing: these frames cannot orient this action, pass `flip=`
            # explicitly if the test means to fix a direction.
            raise ValueError(
                "defensive_credit: the acting team's attacking direction does not resolve from "
                "these frames, so `flip` cannot be derived. Pass `flip=` explicitly, or orient "
                "the frames (tracking.orient_frames_to_ltr)."
            )
        flip = bool(_resolved)

    if mode == "lane_blocker":
        return _lane_blocker(fr, acting_team_id, flip, anchor_x, anchor_y, params)

    threshold = params._proximity_threshold(anchor_x, anchor_y)
    out = opponents_within(
        fr,
        anchor_x=anchor_x,
        anchor_y=anchor_y,
        acting_team_id=acting_team_id,
        threshold_m=threshold,
        flip=flip,
    )
    if out.empty:
        return _empty()
    out = out.copy()
    out["resolution"] = mode  # "nearest" / "all_within" / "all_within_beyond_nearest"
    # Task 6 (ADR-077): carry the resolution ANCHOR + box-aware search radius so the FOV companion
    # can rebuild the proximity DISK this mode searched. Purely ADDITIVE -- the net/plus/minus/n
    # aggregate reads only signed_value/team_id, so these columns leave it byte-identical.
    out["origin_x"] = float(anchor_x)
    out["origin_y"] = float(anchor_y)
    out["region_radius"] = float(threshold)

    if mode == "nearest":
        return out.iloc[[0]].reset_index(drop=True)
    if mode == "all_within_beyond_nearest":
        return out.iloc[1:].reset_index(drop=True)
    return out  # all_within


def _lane_blocker(
    fr: pd.DataFrame, acting_team_id, flip: bool, origin_x: float, origin_y: float, params: DefensiveCreditParams
) -> pd.DataFrame:
    """Blocker = the in-corridor, in-front, non-GK defender with minimum perpendicular offset to the
    shot->goal lane (spec section 4). No origin-proximity threshold (Q2). GK excluded by BOTH the flag
    AND the distance-along-lane cap (``shot_lane_max_t``) -- the GS flag can be all-False (N5). Falls
    back to the nearest-to-origin non-GK defender within the box threshold (``nearest_fallback``)."""
    opp = opponents_action_ltr(fr, acting_team_id, flip, exclude_goalkeeper=False)
    if opp.empty:
        return _empty()
    px = opp["_px"].to_numpy()
    py = opp["_py"].to_numpy()
    non_gk = ~opp["is_goalkeeper"].astype(bool).to_numpy()

    gx, gy = _GOAL_XY
    lane = np.array([gx - origin_x, gy - origin_y], dtype="float64")
    shot_dist = float(np.hypot(lane[0], lane[1]))
    if shot_dist < 1e-6:
        return _empty()  # degenerate: shot from the goal line
    u = lane / shot_dist
    dx = px - origin_x
    dy = py - origin_y
    t = np.clip((dx * u[0] + dy * u[1]) / shot_dist, 0.0, 1.0)  # fraction along the lane; in-front constraint
    # corridor half-width scales with lane distance (matches _cover_shadows), FLOORED so it does not
    # pinch to 0 at the shooter -- a close-range blocker sits at small t (Q4).
    half_width_at_t = np.maximum(
        params.shot_lane_cone_width_factor * shot_dist / 2.0 * t, params.shot_lane_min_half_width_m
    )
    perp = np.abs(u[0] * dy - u[1] * dx)  # scalar 2-D cross (np.cross on 2-D is deprecated in numpy>=2, Q5)
    origin_dist = np.hypot(dx, dy)
    in_corridor = non_gk & (perp <= half_width_at_t) & (t <= params.shot_lane_max_t)
    if in_corridor.any():
        cand = np.where(in_corridor)[0]
        best = cand[np.argmin(perp[cand])]
        # Task 6: the lane region is the shot->goal CORRIDOR (rebuilt from the origin + lane params),
        # not a disk -- so region_radius is NaN.
        return _single_row(opp, int(best), float(origin_dist[best]), RESOLUTION_LANE, origin_x, origin_y, float("nan"))
    # fallback: nearest non-GK within the box-aware origin threshold (a real xG-sizable block still
    # deserves an approximate attributee over NaN/no-row, B8) -- recorded as nearest_fallback.
    thr = params._proximity_threshold(origin_x, origin_y)
    within = non_gk & (origin_dist <= thr)
    if within.any():
        cand = np.where(within)[0]
        best = cand[np.argmin(origin_dist[cand])]
        # Task 6: the fallback searched a proximity DISK of the box-aware origin threshold.
        return _single_row(
            opp, int(best), float(origin_dist[best]), RESOLUTION_NEAREST_FALLBACK, origin_x, origin_y, float(thr)
        )
    return _empty()


def _single_row(
    opp: pd.DataFrame,
    pos: int,
    distance_m: float,
    resolution: str,
    origin_x: float,
    origin_y: float,
    region_radius: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [opp["player_id"].to_numpy()[pos]],
            "team_id": [opp["team_id"].to_numpy()[pos]],
            "distance_m": [distance_m],
            "resolution": [resolution],
            "origin_x": [float(origin_x)],
            "origin_y": [float(origin_y)],
            "region_radius": [float(region_radius)],
        }
    )


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [],
            "team_id": [],
            "distance_m": pd.Series([], dtype="float64"),
            "resolution": pd.Series([], dtype="object"),
            "origin_x": pd.Series([], dtype="float64"),
            "origin_y": pd.Series([], dtype="float64"),
            "region_radius": pd.Series([], dtype="float64"),
        }
    )
