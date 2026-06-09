"""Goal-kick geometry resolution for xT-GK (scoped; promotable -- see TODO general-enrichment).

Derives a goal-kick's origin/destination when the SPADL event omits them (real GS data:
~67% NaN origin), WITHOUT mutating the shared ``actions`` frame. Conditional origin in
confidence order (native -> in-area tracking-GK -> goal-area rule point); destination
native -> next-event -> unresolved. Emits per-row source + continuous confidence. All tiers
measured on owner data (spec 2026-06-08-xt-gk-goalkick-coverage-design.md). See NOTICE.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.spadl import config as spadlconfig

from ._id_compat import ids_match

_GOALKICK = spadlconfig.actiontype_id["goalkick"]
_GOAL_AREA_DEPTH = 16.5  # m from own goal line; tracking-GK beyond this is "off position" (measured 48%)
_RULE_POINT = (5.5, 34.0)  # 6-yard-box centre, LTR
# Effective tiers are 3 (native / in-area-tracking-GK / rule-point): the "empirical median" and
# "rule point" are not data-distinguishable, so they collapse to one positional prior.
_CONF = {"native": 1.0, "tracking_gk": 0.7, "goalkick_prior": 0.2, "unresolved": 0.0}


def resolve_gk_geometry(
    actions: pd.DataFrame, *, frames: pd.DataFrame | None, links: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Return a frame indexed like ``actions`` with origin_x/origin_y/origin_source/
    origin_confidence + dest_x/dest_y/dest_source. Only goal-kicks (type 22) get origin
    imputation; other rows pass native coords through. ``actions`` is never mutated."""
    out = pd.DataFrame(index=actions.index)
    sx = actions["start_x"].to_numpy(float)
    sy = actions["start_y"].to_numpy(float)
    ex = actions["end_x"].to_numpy(float)
    ey = actions["end_y"].to_numpy(float)
    is_goalkick = actions["type_id"].to_numpy() == _GOALKICK

    origin_x = sx.copy()
    origin_y = sy.copy()
    source = np.where(np.isfinite(sx) & np.isfinite(sy), "native", "unresolved").astype(object)

    # tier 2: in-area tracking-GK (goal-kicks with NaN native origin only)
    need = is_goalkick & (source == "unresolved")
    if need.any() and frames is not None:
        gk_xy = _tracking_gk_xy(actions, frames, links)  # (n,2) float, NaN where unavailable/off-area
        use = need & np.isfinite(gk_xy[:, 0])
        origin_x[use] = gk_xy[use, 0]
        origin_y[use] = gk_xy[use, 1]
        source[use] = "tracking_gk"

    # tier 3: goal-area rule point (goal-kicks still unresolved)
    still = is_goalkick & (source == "unresolved")
    origin_x[still] = _RULE_POINT[0]
    origin_y[still] = _RULE_POINT[1]
    source[still] = "goalkick_prior"

    out["origin_x"] = origin_x
    out["origin_y"] = origin_y
    out["origin_source"] = source
    out["origin_confidence"] = np.array([_CONF[s] for s in source], dtype=float)

    # destination: native -> next-event -> unresolved
    dest_x = ex.copy()
    dest_y = ey.copy()
    dsource = np.where(np.isfinite(ex) & np.isfinite(ey), "native", "unresolved").astype(object)
    nan_dest = is_goalkick & (dsource == "unresolved")
    if nan_dest.any():
        nx, ny = _next_event_start(actions)
        use = nan_dest & np.isfinite(nx) & np.isfinite(ny)
        dest_x[use] = nx[use]
        dest_y[use] = ny[use]
        dsource[use] = "next_event"
    out["dest_x"] = dest_x
    out["dest_y"] = dest_y
    out["dest_source"] = dsource
    return out


def _next_event_start(actions: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Next row's start coords (the receiver location), positionally. NaN'd across a
    (game_id, period_id) boundary (a period-/match-final goalkick must NOT take the next
    period's first action as its destination -> falls to dest_source == 'unresolved')."""
    nx = actions["start_x"].shift(-1).to_numpy(float)
    ny = actions["start_y"].shift(-1).to_numpy(float)
    same = np.ones(len(actions), dtype=bool)
    for col in ("game_id", "period_id"):
        if col in actions.columns:
            same &= actions[col].to_numpy() == actions[col].shift(-1).to_numpy()
    nx = np.where(same, nx, np.nan)
    ny = np.where(same, ny, np.nan)
    return nx, ny


def _tracking_gk_xy(actions: pd.DataFrame, frames: pd.DataFrame, links: pd.DataFrame | None) -> np.ndarray:
    """Acting-team GK position at each goal-kick's linked frame, CLAMPED to the goal area
    (x <= _GOAL_AREA_DEPTH in LTR own-half coords); NaN where unavailable or off-position."""
    from ._kernels import resolve_frame_ids_by_position

    n = len(actions)
    res = np.full((n, 2), np.nan, dtype=float)
    fid = resolve_frame_ids_by_position(actions, frames, links=links)
    fg = frames.groupby("frame_id")
    team_ids = actions["team_id"].to_numpy()
    for i in range(n):
        if not np.isfinite(fid[i]):
            continue
        try:
            fr = fg.get_group(int(fid[i]))
        except KeyError:
            continue
        gk = fr[
            fr["is_goalkeeper"].astype(bool) & (~fr["is_ball"].astype(bool)) & ids_match(fr["team_id"], team_ids[i])
        ]
        if gk.empty:
            continue
        gx, gy = float(gk.iloc[0]["x"]), float(gk.iloc[0]["y"])
        if gx <= _GOAL_AREA_DEPTH:  # clamp: off-position GK falls through to the rule point
            res[i] = (gx, gy)
    return res
