"""Schema-agnostic compute kernels for tracking-aware action_context features.

Private module. Public per-schema wrappers live in
silly_kicks.tracking.features (standard SPADL) and
silly_kicks.atomic.tracking.features (atomic SPADL).

All kernels accept anchor_x / anchor_y as pd.Series (caller-supplied, allowing
per-schema column choice: standard's start_x/y, atomic's x/y, or end-anchors)
and an ActionFrameContext built once via _resolve_action_frame_context.

See spec docs/superpowers/specs/2026-04-30-action-context-pr1-design.md s 4.3
for the kernel pattern.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from ._action_orientation import acting_team_attacks_rtl, reproject_to_action_ltr
from ._id_compat import align_join_keys, ids_differ, ids_equal
from .feature_framework import ActionFrameContext

if TYPE_CHECKING:
    from .pressure import AndrienkoParams, BekkersParams, LinkParams

# Goal-mouth coordinates per spadl.config (105 x 68 m pitch, goal post-to-post 7.32 m centered on y=34)
_GOAL_X = 105.0
_GOAL_Y_CENTER = 34.0
_GOAL_LEFT_POST_Y = 30.34
_GOAL_RIGHT_POST_Y = 37.66


def _nearest_defender_distance(
    anchor_x: pd.Series,
    anchor_y: pd.Series,
    ctx: ActionFrameContext,
) -> pd.Series:
    """Per action: distance from (anchor_x, anchor_y) to nearest opposite-team frame row.

    Returns NaN for actions with no opposite rows in ctx (unlinked or no defenders).
    """
    actions_id = ctx.actions["action_id"].to_numpy()
    n = len(actions_id)
    out = pd.Series(np.full(n, np.nan), index=ctx.actions.index, dtype="float64")

    if len(ctx.opposite_rows_per_action) == 0:
        return out

    a_xy = pd.DataFrame(
        {
            "action_id": actions_id,
            "anchor_x": anchor_x.to_numpy(),
            "anchor_y": anchor_y.to_numpy(),
        }
    )
    merged = ctx.opposite_rows_per_action.merge(a_xy, on="action_id", how="left")
    dx = merged["x"] - merged["anchor_x"]
    dy = merged["y"] - merged["anchor_y"]
    dist = np.sqrt(dx * dx + dy * dy)
    merged["_dist"] = dist
    min_per_action = merged.groupby("action_id")["_dist"].min()

    action_to_idx = pd.Series(ctx.actions.index, index=actions_id)
    for aid, d in min_per_action.items():
        if aid in action_to_idx.index:
            out.loc[action_to_idx.loc[aid]] = d  # type: ignore[arg-type]  # .loc[scalar] returns Hashable; pandas-stubs limitation
    return out


def _actor_speed_from_ctx(ctx: ActionFrameContext) -> pd.Series:
    """Per action: actor's speed from the linked frame row.

    NaN where action couldn't link, actor's player_id missing, or speed itself is NaN.
    """
    out = pd.Series(np.full(len(ctx.actions), np.nan), index=ctx.actions.index, dtype="float64")
    if len(ctx.actor_rows) == 0 or "speed" not in ctx.actor_rows.columns:
        return out
    actions_id = ctx.actions["action_id"].to_numpy()
    action_to_idx = pd.Series(ctx.actions.index, index=actions_id)
    for _, row in ctx.actor_rows.iterrows():
        aid = row["action_id"]
        speed = row["speed"]
        if aid in action_to_idx.index and pd.notna(speed):
            out.loc[action_to_idx.loc[aid]] = float(speed)
    return out


def _receiver_zone_density(
    anchor_x: pd.Series,
    anchor_y: pd.Series,
    ctx: ActionFrameContext,
    *,
    radius: float,
) -> pd.Series:
    """Per action: count of opposite-team frame rows within radius of (anchor_x, anchor_y).

    Returns NaN for unlinked actions (no pointer); 0 for linked actions with no defenders
    in radius (genuine count-zero distinction).
    """
    out = pd.Series(np.full(len(ctx.actions), np.nan), index=ctx.actions.index, dtype="float64")

    actions_id = ctx.actions["action_id"].to_numpy()
    action_to_idx = pd.Series(ctx.actions.index, index=actions_id)
    linked_aids = set(ctx.pointers.loc[ctx.pointers["frame_id"].notna(), "action_id"].to_numpy())
    for aid in linked_aids:
        if aid in action_to_idx.index:
            out.loc[action_to_idx.loc[aid]] = 0.0

    if len(ctx.opposite_rows_per_action) == 0:
        return out

    a_xy = pd.DataFrame(
        {
            "action_id": actions_id,
            "anchor_x": anchor_x.to_numpy(),
            "anchor_y": anchor_y.to_numpy(),
        }
    )
    merged = ctx.opposite_rows_per_action.merge(a_xy, on="action_id", how="left")
    dx = merged["x"] - merged["anchor_x"]
    dy = merged["y"] - merged["anchor_y"]
    dist = np.sqrt(dx * dx + dy * dy)
    in_radius = dist <= radius
    merged["_in"] = in_radius.astype("int64")
    counts = merged.groupby("action_id")["_in"].sum()
    for aid, c in counts.items():
        if aid in action_to_idx.index:
            out.loc[action_to_idx.loc[aid]] = float(c)  # type: ignore[arg-type]  # .loc[scalar] returns Hashable; pandas-stubs limitation
    return out


def _defenders_in_triangle_to_goal(
    anchor_x: pd.Series,
    anchor_y: pd.Series,
    ctx: ActionFrameContext,
) -> pd.Series:
    """Per action: count of opposite-team frame rows inside the triangle
    (anchor, left_goalpost, right_goalpost).

    NaN for unlinked actions; 0 for linked-but-no-defenders-in-triangle.
    Triangle vertices: (anchor_x, anchor_y), (105, 30.34), (105, 37.66).
    Vectorized point-in-triangle via sign-of-cross-product test.
    """
    out = pd.Series(np.full(len(ctx.actions), np.nan), index=ctx.actions.index, dtype="float64")

    actions_id = ctx.actions["action_id"].to_numpy()
    action_to_idx = pd.Series(ctx.actions.index, index=actions_id)
    linked_aids = set(ctx.pointers.loc[ctx.pointers["frame_id"].notna(), "action_id"].to_numpy())
    for aid in linked_aids:
        if aid in action_to_idx.index:
            out.loc[action_to_idx.loc[aid]] = 0.0

    if len(ctx.opposite_rows_per_action) == 0:
        return out

    a_xy = pd.DataFrame(
        {
            "action_id": actions_id,
            "ax": anchor_x.to_numpy(),
            "ay": anchor_y.to_numpy(),
        }
    )
    merged = ctx.opposite_rows_per_action.merge(a_xy, on="action_id", how="left")

    ax = merged["ax"].to_numpy()
    ay = merged["ay"].to_numpy()
    bx = np.full_like(ax, _GOAL_X)
    by = np.full_like(ay, _GOAL_LEFT_POST_Y)
    cx = np.full_like(ax, _GOAL_X)
    cy = np.full_like(ay, _GOAL_RIGHT_POST_Y)
    px = merged["x"].to_numpy()
    py = merged["y"].to_numpy()

    def _sign(x1, y1, x2, y2, x3, y3):
        return (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3)

    d1 = _sign(px, py, ax, ay, bx, by)
    d2 = _sign(px, py, bx, by, cx, cy)
    d3 = _sign(px, py, cx, cy, ax, ay)
    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    in_triangle = ~(has_neg & has_pos)

    merged["_in"] = in_triangle.astype("int64")
    counts = merged.groupby("action_id")["_in"].sum()
    for aid, c in counts.items():
        if aid in action_to_idx.index:
            out.loc[action_to_idx.loc[aid]] = float(c)  # type: ignore[arg-type]  # .loc[scalar] returns Hashable; pandas-stubs limitation
    return out


def _pre_shot_gk_position(
    anchor_x: pd.Series,
    anchor_y: pd.Series,
    ctx: ActionFrameContext,
    *,
    shot_type_ids: frozenset[int],
) -> pd.DataFrame:
    """Per shot action: defending GK's x/y at the linked frame + 2 derived distances.

    Returns
    -------
    pd.DataFrame
        Indexed identically to ctx.actions with 4 columns:
        - pre_shot_gk_x (float64)
        - pre_shot_gk_y (float64)
        - pre_shot_gk_distance_to_goal (float64) — Euclidean to goal-mouth center (105, 34)
        - pre_shot_gk_distance_to_shot (float64) — Euclidean to (anchor_x, anchor_y)

    All NaN for: non-shot rows; rows with no defending_gk_row in ctx
    (covers unlinked / pre-engagement / GK-absent-from-frame cases).

    Schema-agnostic per ADR-005 s 3 — caller supplies anchor columns and shot_type_ids.

    See NOTICE for full bibliographic citations (Anzer & Bauer 2021).
    """
    actions = ctx.actions
    out = pd.DataFrame(
        {
            "pre_shot_gk_x": np.full(len(actions), np.nan, dtype="float64"),
            "pre_shot_gk_y": np.full(len(actions), np.nan, dtype="float64"),
            "pre_shot_gk_distance_to_goal": np.full(len(actions), np.nan, dtype="float64"),
            "pre_shot_gk_distance_to_shot": np.full(len(actions), np.nan, dtype="float64"),
        },
        index=actions.index,
    )
    if "type_id" not in actions.columns:
        return out
    is_shot = actions["type_id"].isin(shot_type_ids).to_numpy()
    if not is_shot.any():
        return out

    # Left-join GK x/y on action_id; non-shot/pre-engagement/GK-absent rows have NaN gk x/y.
    if len(ctx.defending_gk_rows) > 0:
        gk = ctx.defending_gk_rows[["action_id", "x", "y"]].rename(columns={"x": "_gk_x", "y": "_gk_y"})
        gk = gk.drop_duplicates("action_id", keep="first")
        per_action = actions[["action_id"]].merge(gk, on="action_id", how="left")
    else:
        per_action = actions[["action_id"]].assign(_gk_x=np.nan, _gk_y=np.nan)
    per_action.index = actions.index

    shot_mask = pd.Series(is_shot, index=actions.index)
    gk_present_mask = per_action["_gk_x"].notna()
    valid = shot_mask & gk_present_mask

    if valid.any():
        gx = per_action.loc[valid, "_gk_x"].astype("float64")
        gy = per_action.loc[valid, "_gk_y"].astype("float64")
        ax = anchor_x.loc[valid].astype("float64")
        ay = anchor_y.loc[valid].astype("float64")

        out.loc[valid, "pre_shot_gk_x"] = gx
        out.loc[valid, "pre_shot_gk_y"] = gy
        out.loc[valid, "pre_shot_gk_distance_to_goal"] = np.sqrt((_GOAL_X - gx) ** 2 + (_GOAL_Y_CENTER - gy) ** 2)
        out.loc[valid, "pre_shot_gk_distance_to_shot"] = np.sqrt((ax - gx) ** 2 + (ay - gy) ** 2)
    return out


def _pre_shot_gk_angle(
    anchor_x: pd.Series,
    anchor_y: pd.Series,
    ctx: ActionFrameContext,
    *,
    shot_type_ids: frozenset[int],
) -> pd.DataFrame:
    """Per shot action: GK angle vs shot trajectory and goal-line normal.

    Returns
    -------
    pd.DataFrame
        Indexed identically to ctx.actions with 2 columns:
        - pre_shot_gk_angle_to_shot_trajectory (float64, radians, signed)
        - pre_shot_gk_angle_off_goal_line (float64, radians, signed)

    All NaN for non-shot rows or rows with no defending_gk_row in ctx.

    See NOTICE for full bibliographic citations (Anzer & Bauer 2021).
    """
    actions = ctx.actions
    out = pd.DataFrame(
        {
            "pre_shot_gk_angle_to_shot_trajectory": np.full(len(actions), np.nan, dtype="float64"),
            "pre_shot_gk_angle_off_goal_line": np.full(len(actions), np.nan, dtype="float64"),
        },
        index=actions.index,
    )
    if "type_id" not in actions.columns:
        return out
    is_shot = actions["type_id"].isin(shot_type_ids).to_numpy()
    if not is_shot.any():
        return out

    if len(ctx.defending_gk_rows) > 0:
        gk = ctx.defending_gk_rows[["action_id", "x", "y"]].rename(columns={"x": "_gk_x", "y": "_gk_y"})
        gk = gk.drop_duplicates("action_id", keep="first")
        per_action = actions[["action_id"]].merge(gk, on="action_id", how="left")
    else:
        per_action = actions[["action_id"]].assign(_gk_x=np.nan, _gk_y=np.nan)
    per_action.index = actions.index

    shot_mask = pd.Series(is_shot, index=actions.index)
    gk_present_mask = per_action["_gk_x"].notna()
    valid = shot_mask & gk_present_mask
    if not valid.any():
        return out

    gx = per_action.loc[valid, "_gk_x"].astype("float64").to_numpy()
    gy = per_action.loc[valid, "_gk_y"].astype("float64").to_numpy()
    ax = anchor_x.loc[valid].astype("float64").to_numpy()
    ay = anchor_y.loc[valid].astype("float64").to_numpy()

    v1x = _GOAL_X - ax
    v1y = _GOAL_Y_CENTER - ay
    v2x = gx - ax
    v2y = gy - ay
    cross = v1x * v2y - v1y * v2x
    dot = v1x * v2x + v1y * v2y
    out.loc[valid, "pre_shot_gk_angle_to_shot_trajectory"] = np.arctan2(cross, dot)

    out.loc[valid, "pre_shot_gk_angle_off_goal_line"] = np.arctan2(gy - _GOAL_Y_CENTER, _GOAL_X - gx)
    return out


# ---------------------------------------------------------------------------
# PR-S25 -- TF-2: pressure_on_actor multi-flavor kernels
# ---------------------------------------------------------------------------


def _pressure_andrienko(
    anchor_x: pd.Series,
    anchor_y: pd.Series,
    ctx: ActionFrameContext,
    *,
    params: AndrienkoParams,
    goal_x: float = _GOAL_X,
    goal_y: float = _GOAL_Y_CENTER,
) -> pd.Series:
    """Andrienko et al. 2017 directional-oval pressure (sum-of-pressers).

    Per defender:
        vec_to_threat = (goal - anchor) normalized
        vec_presser   = (defender - anchor) normalized
        cos_theta     = dot(vec_to_threat, vec_presser)
        z             = (1 + cos_theta) / 2
        L             = d_back + (d_front - d_back) * (z^3 + 0.3*z) / 1.3
        d             = ||defender - anchor||
        pr_i          = (1 - d/L)^q * 100   if d < L else 0.0

    Aggregation: sum across all opp-team defenders. NaN unlinked actions;
    0.0 for linked-but-no-defenders or all-defenders-outside-zone.

    See spec section 4.4, NOTICE (Andrienko et al. 2017).
    """
    out = pd.Series(np.full(len(ctx.actions), np.nan), index=ctx.actions.index, dtype="float64")
    actions_id = ctx.actions["action_id"].to_numpy()
    action_to_idx = pd.Series(ctx.actions.index, index=actions_id)

    # Track which actions have a finite anchor; NaN-anchor actions stay NaN
    # in output (per ADR-003 NaN-safe contract -- pressure is undefined when
    # anchor position is unknown, distinct from "linked but no defenders" 0.0).
    anchor_finite_per_aid = pd.Series(
        np.isfinite(anchor_x.to_numpy()) & np.isfinite(anchor_y.to_numpy()),
        index=actions_id,
    )

    linked_aids = set(ctx.pointers.loc[ctx.pointers["frame_id"].notna(), "action_id"].to_numpy())
    for aid in linked_aids:
        if aid in action_to_idx.index and bool(anchor_finite_per_aid.get(aid, False)):
            out.loc[action_to_idx.loc[aid]] = 0.0  # type: ignore[arg-type]

    if len(ctx.opposite_rows_per_action) == 0:
        return out

    a_xy = pd.DataFrame(
        {
            "action_id": actions_id,
            "ax": anchor_x.to_numpy(),
            "ay": anchor_y.to_numpy(),
        }
    )
    merged = ctx.opposite_rows_per_action.merge(a_xy, on="action_id", how="left")

    ax = merged["ax"].to_numpy()
    ay = merged["ay"].to_numpy()
    px = merged["x"].to_numpy()
    py = merged["y"].to_numpy()

    threat_dx = goal_x - ax
    threat_dy = goal_y - ay
    threat_mag = np.sqrt(threat_dx**2 + threat_dy**2) + 1e-12
    threat_unit_x = threat_dx / threat_mag
    threat_unit_y = threat_dy / threat_mag

    presser_dx = px - ax
    presser_dy = py - ay
    d = np.sqrt(presser_dx**2 + presser_dy**2)
    presser_mag = d + 1e-12
    presser_unit_x = presser_dx / presser_mag
    presser_unit_y = presser_dy / presser_mag

    cos_theta = threat_unit_x * presser_unit_x + threat_unit_y * presser_unit_y
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    z = (1.0 + cos_theta) / 2.0
    # Andrienko Eq 1: L is the directional zone half-extent (paper's symbol).
    L = params.d_back + (params.d_front - params.d_back) * (z**3 + 0.3 * z) / 1.3  # noqa: N806

    in_zone = d < L
    pr_per_defender = np.where(in_zone, np.power(np.maximum(0.0, 1.0 - d / L), params.q) * 100.0, 0.0)

    merged["_pr"] = pr_per_defender
    sums = merged.groupby("action_id")["_pr"].sum()
    for aid, pr_total in sums.items():
        if aid in action_to_idx.index and bool(anchor_finite_per_aid.get(aid, False)):
            out.loc[action_to_idx.loc[aid]] = float(pr_total)  # type: ignore[arg-type]
    return out


def _pressure_link(
    anchor_x: pd.Series,
    anchor_y: pd.Series,
    ctx: ActionFrameContext,
    *,
    params: LinkParams,
    goal_x: float = _GOAL_X,
    goal_y: float = _GOAL_Y_CENTER,
) -> pd.Series:
    """Link, Lang & Seidenschwarz 2016 piecewise-zone saturating-aggregation.

    Per defender:
        d            = ||defender - anchor||
        cos_theta    = dot(unit(goal - anchor), unit(defender - anchor))
        alpha_deg    = degrees(arccos(clip(cos_theta, -1, 1)))   # in [0, 180]
        r_zo         = r_hoz if alpha_deg < angle_hoz_lz_deg
                      else r_lz if alpha_deg < angle_lz_hz_deg
                      else r_hz
        pr_i         = max(0, 1 - d/r_zo)   if d < r_zo else 0.0

    Aggregation:
        PR(x) = 1 - exp(-k3 * x)   where x = sum(pr_i)

    See spec section 4.4, NOTICE (Link et al. 2016).
    """
    out = pd.Series(np.full(len(ctx.actions), np.nan), index=ctx.actions.index, dtype="float64")
    actions_id = ctx.actions["action_id"].to_numpy()
    action_to_idx = pd.Series(ctx.actions.index, index=actions_id)

    anchor_finite_per_aid = pd.Series(
        np.isfinite(anchor_x.to_numpy()) & np.isfinite(anchor_y.to_numpy()),
        index=actions_id,
    )

    linked_aids = set(ctx.pointers.loc[ctx.pointers["frame_id"].notna(), "action_id"].to_numpy())
    for aid in linked_aids:
        if aid in action_to_idx.index and bool(anchor_finite_per_aid.get(aid, False)):
            out.loc[action_to_idx.loc[aid]] = 0.0  # type: ignore[arg-type]

    if len(ctx.opposite_rows_per_action) == 0:
        return out

    a_xy = pd.DataFrame(
        {
            "action_id": actions_id,
            "ax": anchor_x.to_numpy(),
            "ay": anchor_y.to_numpy(),
        }
    )
    merged = ctx.opposite_rows_per_action.merge(a_xy, on="action_id", how="left")

    ax = merged["ax"].to_numpy()
    ay = merged["ay"].to_numpy()
    px = merged["x"].to_numpy()
    py = merged["y"].to_numpy()

    presser_dx = px - ax
    presser_dy = py - ay
    d = np.sqrt(presser_dx**2 + presser_dy**2)

    threat_dx = goal_x - ax
    threat_dy = goal_y - ay
    threat_mag = np.sqrt(threat_dx**2 + threat_dy**2) + 1e-12
    presser_mag = d + 1e-12
    cos_theta = (threat_dx * presser_dx + threat_dy * presser_dy) / (threat_mag * presser_mag)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    alpha_deg = np.degrees(np.arccos(cos_theta))

    r_zo = np.where(
        alpha_deg < params.angle_hoz_lz_deg,
        params.r_hoz,
        np.where(alpha_deg < params.angle_lz_hz_deg, params.r_lz, params.r_hz),
    )

    in_zone = d < r_zo
    pr_per_defender = np.where(in_zone, np.maximum(0.0, 1.0 - d / r_zo), 0.0)

    merged["_pr"] = pr_per_defender
    sums = merged.groupby("action_id")["_pr"].sum()
    for aid, x_total in sums.items():
        if aid in action_to_idx.index and bool(anchor_finite_per_aid.get(aid, False)):
            agg = 1.0 - math.exp(-params.k3 * float(x_total))
            out.loc[action_to_idx.loc[aid]] = agg  # type: ignore[arg-type]
    return out


def _bekkers_tti(
    *,
    p1: np.ndarray,
    p2: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    reaction_time: float,
    max_object_speed: float,
) -> np.ndarray:
    """Bekkers/Spearman/Shaw/Pleuler time-to-intercept matrix.

    Re-implementation of UnravelSports/unravelsports
    unravel/soccer/models/utils.py:time_to_intercept (BSD-3-Clause).
    See NOTICE for full attribution.

    Parameters
    ----------
    p1 : (n, 2) ndarray
        Pressing-player positions.
    p2 : (m, 2) ndarray
        Target positions (attackers / ball).
    v1 : (n, 2) ndarray
        Pressing-player velocities.
    v2 : (m, 2) ndarray
        Target velocities.
    reaction_time : float
        Pressing-player reaction time before accelerating (s).
    max_object_speed : float
        Pressing-player maximum running speed (m/s).

    Returns
    -------
    (m, n) ndarray
        TTI matrix; element [i, j] = time for pressing-player j to intercept
        target i.
    """
    u = v1
    d2 = p2 + v2
    v = d2[:, None, :] - p1[None, :, :]

    u_mag = np.linalg.norm(u, axis=-1)
    v_mag = np.linalg.norm(v, axis=-1)
    dot = np.sum(u * v, axis=-1)

    eps = 1e-10
    angle = np.arccos(np.clip(dot / (u_mag * v_mag + eps), -1.0, 1.0))

    r_reaction = p1 + v1 * reaction_time
    d = d2[:, None, :] - r_reaction[None, :, :]

    t = u_mag * angle / np.pi + reaction_time + np.linalg.norm(d, axis=-1) / max_object_speed
    return t


def _bekkers_p_intercept(
    *,
    tti: np.ndarray,
    sigma: float,
    time_threshold: float,
) -> np.ndarray:
    """Logistic transform of TTI to per-pair intercept probability.

    Re-implementation of unravel/soccer/models/utils.py:probability_to_intercept
    (BSD-3-Clause). See NOTICE.
    """
    exponent = -math.pi / math.sqrt(3.0) / sigma * (time_threshold - tti)
    exponent = np.clip(exponent, -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(exponent))


def _pressure_bekkers(
    anchor_x: pd.Series,
    anchor_y: pd.Series,
    ctx: ActionFrameContext,
    *,
    params: BekkersParams,
    ball_xy_v_per_action: pd.DataFrame,
) -> pd.Series:
    """Bekkers 2024 Pressing Intensity probabilistic model.

    Per defender: TTI -> p via logistic. Optional ball-carrier-max
    (max of p_to_player and p_to_ball per defender). Aggregation:
    1 - prod(1 - p_i_final).

    See spec section 4.4, NOTICE (Bekkers 2025; BSD-3-Clause UnravelSports).
    """
    out = pd.Series(np.full(len(ctx.actions), np.nan), index=ctx.actions.index, dtype="float64")
    actions_id = ctx.actions["action_id"].to_numpy()
    action_to_idx = pd.Series(ctx.actions.index, index=actions_id)

    anchor_finite_per_aid = pd.Series(
        np.isfinite(anchor_x.to_numpy()) & np.isfinite(anchor_y.to_numpy()),
        index=actions_id,
    )

    linked_aids = set(ctx.pointers.loc[ctx.pointers["frame_id"].notna(), "action_id"].to_numpy())
    for aid in linked_aids:
        if aid in action_to_idx.index and bool(anchor_finite_per_aid.get(aid, False)):
            out.loc[action_to_idx.loc[aid]] = 0.0  # type: ignore[arg-type]

    if len(ctx.opposite_rows_per_action) == 0:
        return out

    # Build per-action arrays: actor pos+v, ball pos+v (if used), defenders pos+v+speed.
    # ctx.actor_rows can have duplicate action_id rows in real-data slim slices when
    # the linked frame happens to contain multiple rows tagged with the actor's
    # player_id (rare but observed). Collapse to first per action.
    actor_per_action = ctx.actor_rows.drop_duplicates(subset=["action_id"], keep="first").set_index("action_id")[
        ["x", "y", "vx", "vy"]
    ]

    if params.use_ball_carrier_max:
        # Mirror the actor_per_action dedup above: a duplicated frame record (e.g. Gradient Sports
        # ships some (period, frame) records up to 16x) yields >1 ball row for one action_id, so
        # ball_per_action_indexed.loc[aid] would return a DataFrame and ball_pos would become 3-D,
        # crashing _bekkers_tti with a broadcast error. Collapse to first per action.
        ball_per_action_indexed = (
            ball_xy_v_per_action.drop_duplicates(subset=["action_id"], keep="first").set_index("action_id")
            if len(ball_xy_v_per_action)
            else pd.DataFrame()
        )

    grouped = ctx.opposite_rows_per_action.groupby("action_id")
    for aid, defender_group in grouped:
        if aid not in action_to_idx.index:
            continue
        if not bool(anchor_finite_per_aid.get(aid, False)):
            # NaN anchor -> output stays NaN (set at init); don't compute.
            continue
        if aid not in actor_per_action.index:
            continue
        actor_row = actor_per_action.loc[aid]
        actor_pos = np.array([[actor_row["x"], actor_row["y"]]], dtype="float64")
        actor_vel = np.array([[actor_row["vx"], actor_row["vy"]]], dtype="float64")
        if pd.isna(actor_pos).any() or pd.isna(actor_vel).any():
            out.loc[action_to_idx.loc[aid]] = float("nan")  # type: ignore[arg-type]
            continue

        defender_pos = defender_group[["x", "y"]].to_numpy(dtype="float64")
        defender_vel = defender_group[["vx", "vy"]].to_numpy(dtype="float64")
        defender_speed = defender_group["speed"].to_numpy(dtype="float64")

        if np.isnan(defender_pos).any() or np.isnan(defender_vel).any():
            out.loc[action_to_idx.loc[aid]] = float("nan")  # type: ignore[arg-type]
            continue

        tti_to_actor = _bekkers_tti(
            p1=defender_pos,
            p2=actor_pos,
            v1=defender_vel,
            v2=actor_vel,
            reaction_time=params.reaction_time,
            max_object_speed=params.max_player_speed,
        )
        p_to_actor = _bekkers_p_intercept(
            tti=tti_to_actor,
            sigma=params.sigma,
            time_threshold=params.time_threshold,
        )[0, :]  # shape (n_defenders,)

        p_per_defender = p_to_actor.copy()

        # ball-carrier-max (Bekkers 2024 section 2.4): per defender, take
        # max(p_to_actor, p_to_ball) when this action's linked frame has a finite ball
        # position. When the ball is missing or NaN at the linked frame (e.g. Metrica
        # windows where kloppy returned no ball_coordinates), fall back PER ACTION to
        # the Bekkers base model (pressure-on-player only) -- a documented variant, not
        # NaN and not a raise. ball-carrier-max is an improvement, not a requirement. (3.30.0)
        if params.use_ball_carrier_max and aid in ball_per_action_indexed.index:
            ball_row = ball_per_action_indexed.loc[aid]
            ball_pos = np.array([[ball_row["x"], ball_row["y"]]], dtype="float64")
            ball_vel = np.array([[ball_row["vx"], ball_row["vy"]]], dtype="float64")
            if not (np.isnan(ball_pos).any() or np.isnan(ball_vel).any()):
                tti_to_ball = _bekkers_tti(
                    p1=defender_pos,
                    p2=ball_pos,
                    v1=defender_vel,
                    v2=ball_vel,
                    reaction_time=params.reaction_time,
                    max_object_speed=params.max_player_speed,
                )
                p_to_ball = _bekkers_p_intercept(
                    tti=tti_to_ball,
                    sigma=params.sigma,
                    time_threshold=params.time_threshold,
                )[0, :]
                p_per_defender = np.maximum(p_to_actor, p_to_ball)
            # else: NaN ball at this frame -> keep p_per_defender = p_to_actor (base model).
        # else: no ball row for this action (or use_ball_carrier_max=False) ->
        # keep p_per_defender = p_to_actor (base model).

        # Active-pressing filter: defenders below speed_threshold contribute 0
        below_thresh = defender_speed < params.speed_threshold
        p_per_defender = np.where(below_thresh, 0.0, p_per_defender)

        # Aggregation: 1 - prod(1 - p)
        agg = 1.0 - float(np.prod(1.0 - p_per_defender))
        out.loc[action_to_idx.loc[aid]] = agg  # type: ignore[arg-type]

    return out


# ---------------------------------------------------------------------------
# PR-S25 -- TF-3: actor_*_pre_window kernel
# ---------------------------------------------------------------------------


def _actor_pre_window_kernel(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    pre_seconds: float,
) -> pd.DataFrame:
    """TF-3 shared kernel: emits both arc-length and displacement.

    REQUIREMENT per spec section 3.3: sort frames by (action_id, time_seconds ASC)
    inside each (action, actor_player_id) group BEFORE computing segments or
    selecting first/last. Drops NaN-position rows entirely (bridge rule).

    Returns DataFrame with columns:
        - actor_arc_length_pre_window (float64, m)
        - actor_displacement_pre_window (float64, m)
    indexed identically to ``actions``.
    """
    from .utils import slice_around_event

    out = pd.DataFrame(
        {
            "actor_arc_length_pre_window": np.full(len(actions), np.nan, dtype="float64"),
            "actor_displacement_pre_window": np.full(len(actions), np.nan, dtype="float64"),
        },
        index=actions.index,
    )

    if len(actions) == 0 or len(frames) == 0:
        return out

    sliced = slice_around_event(actions, frames, pre_seconds=pre_seconds, post_seconds=0.0)
    if len(sliced) == 0:
        return out

    # Cast is_ball to bool explicitly: slim-parquet providers store it as object
    # dtype where Python `~` yields -1 / -2 ints rather than logical negation.
    sliced = sliced[~sliced["is_ball"].astype(bool)].copy()
    actor_id_per_action = actions[["action_id", "player_id"]].rename(columns={"player_id": "actor_player_id"})
    sliced = sliced.merge(actor_id_per_action, on="action_id", how="left")
    sliced = sliced[ids_equal(sliced["player_id"], sliced["actor_player_id"]).to_numpy()].copy()
    if len(sliced) == 0:
        return out

    # Drop NaN-position rows (bridge rule per spec section 3.2)
    valid = sliced.dropna(subset=["x", "y"]).copy()
    if len(valid) == 0:
        return out

    # Sort by (action_id, time_seconds ASC) per kernel requirement
    valid = valid.sort_values(["action_id", "time_seconds"], kind="mergesort")

    action_to_idx = pd.Series(actions.index.values, index=actions["action_id"].values)
    grouped = valid.groupby("action_id", sort=False)
    for aid, group in grouped:
        if len(group) < 2:
            continue
        xs = group["x"].to_numpy()
        ys = group["y"].to_numpy()
        dx = np.diff(xs)
        dy = np.diff(ys)
        arc = float(np.sqrt(dx * dx + dy * dy).sum())
        disp = float(math.hypot(xs[-1] - xs[0], ys[-1] - ys[0]))
        if aid in action_to_idx.index:
            row_idx = action_to_idx.loc[aid]
            out.loc[row_idx, "actor_arc_length_pre_window"] = arc
            out.loc[row_idx, "actor_displacement_pre_window"] = disp
    return out


def _defensive_line_at_actions(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    home_team_id: int | str,
    n: int | Literal["adaptive"] = 4,
    links: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """All 6 defensive-line columns for the defending team at each action's linked frame.

    Calls compute_defensive_line ONCE on the full frames DataFrame, then
    joins on (period_id, frame_id, opposing_team_id) per action.

    Returns DataFrame aligned with actions.index.
    """
    from ._defensive_line import compute_defensive_line
    from .utils import link_actions_to_frames

    feature_cols = [
        "defensive_line_x",
        "back_line_high_x",
        "compactness_x",
        "lateral_width",
        "max_lateral_gap",
        "back_n_count",
    ]
    n_actions = len(actions)
    empty = pd.DataFrame(
        {col: np.full(n_actions, np.nan) for col in feature_cols},
        index=actions.index,
    )
    empty["back_n_count"] = pd.array([pd.NA] * n_actions, dtype="Int64")

    if n_actions == 0 or len(frames) == 0:
        return empty

    # Compute defensive line for all frames (ONCE)
    dl = compute_defensive_line(frames, home_team_id=home_team_id, n=n)
    if dl.empty:
        return empty

    # Link actions to frames — use _row_idx as stable positional key
    # (action_id may not be unique in gamestates shifted slots)
    actions_with_idx = actions.copy()
    actions_with_idx["_row_idx"] = np.arange(n_actions)
    if links is not None:
        pointers = links
    else:
        pointers, _report = link_actions_to_frames(actions_with_idx, frames)
    linked = pointers[pointers["frame_id"].notna()].copy()
    if linked.empty:
        return empty

    # Re-attach _row_idx + team_id + period_id from the indexed actions
    linked = linked.merge(
        actions_with_idx[["action_id", "_row_idx", "team_id", "period_id", "game_id"]],
        on="action_id",
        how="left",
    )
    # For duplicate action_ids (gamestates), keep all rows — _row_idx disambiguates
    linked["frame_id_int"] = linked["frame_id"].astype("int64")

    # Align id-valued join keys (incl. the differently-named frame_id_int<->frame_id pair) so a
    # string-id caller does not raise on the merge (ADR-019; replaces the ad-hoc astype(str)).
    linked, dl = align_join_keys(linked, dl, ["game_id", "period_id", ("frame_id_int", "frame_id")])

    # Join with defensive-line data: match on (period_id, frame_id) then filter to opposing team
    merged = linked.merge(
        dl,
        left_on=["game_id", "period_id", "frame_id_int"],
        right_on=["game_id", "period_id", "frame_id"],
        how="left",
        suffixes=("_action", "_dl"),
    )
    # Keep only rows where dl team != action team (opposing team's line)
    opposing = merged[ids_differ(merged["team_id_dl"], merged["team_id_action"])]

    # One result per _row_idx (positional, always unique)
    opposing = opposing.drop_duplicates("_row_idx", keep="first")

    # Map back to actions index by position
    out = empty.copy()
    for _, row in opposing.iterrows():
        pos = int(row["_row_idx"])
        idx = actions.index[pos]
        for col in feature_cols:
            out.at[idx, col] = row[col]

    out["back_n_count"] = out["back_n_count"].astype("Int64")

    # ADR-028: re-project the two x-positions into each action's LTR frame.
    # compactness_x / lateral_width / max_lateral_gap / back_n_count are spans/counts
    # (flip-invariant) and are left unchanged.
    flip = acting_team_attacks_rtl(actions, frames)
    out = reproject_to_action_ltr(out, flip, x_cols=["defensive_line_x", "back_line_high_x"], y_cols=[])
    return out


def resolve_frame_ids_by_position(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    links: pd.DataFrame | None = None,
) -> np.ndarray:
    """Float64 array (len == len(actions)): linked frame_id per action ROW, NaN if
    unlinked. Robust to NON-UNIQUE action_id (VAEP shifted gamestate slots repeat the
    boundary action). NEVER uses .at on a possibly-non-unique index.

    add_*-supplied ``links`` are keyed by the unique action_id; the internal-link path
    re-keys to a unique positional surrogate before linking.
    """
    from .utils import link_actions_to_frames

    n = len(actions)
    if n == 0:
        return np.full(0, np.nan)
    if links is not None:
        # links action_id and actions action_id are both the post-link int64 SPADL id
        # (link_actions_to_frames casts to int64); reindex aligns by exact id. The
        # internal-link path below uses an int64 surrogate, so both paths are
        # dtype-consistent -- locked by test_resolve_frame_ids_by_position.
        fid = links.drop_duplicates("action_id").set_index("action_id")["frame_id"]
        return fid.reindex(actions["action_id"].to_numpy()).to_numpy(dtype="float64")
    link_input = actions.copy()
    link_input["action_id"] = np.arange(n)  # unique positional surrogate
    pointers, _ = link_actions_to_frames(link_input, frames)
    return pointers.set_index("action_id")["frame_id"].reindex(np.arange(n)).to_numpy(dtype="float64")


def _structural_pass_at_actions(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    home_team_id: int | str,
    params=None,
    links: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """structural_lbs / structural_sgm / structural_sdi for each pass/cross action at
    its linked frame. Returns a DataFrame aligned with actions.index (3 columns).

    Shared by add_structural_pass and structural_pass_xfns (DRY + 3x-not-9x budget).
    Non-pass/cross rows and unlinked/degenerate rows -> NaN. Robust to non-unique
    action_id in shifted VAEP gamestate slots (frame_id resolved by position).
    """
    from ._structural_pass import StructuralPassParams, compute_structural_pass_metrics

    if params is None:
        params = StructuralPassParams()

    n = len(actions)
    out = pd.DataFrame(
        {
            "structural_lbs": np.full(n, np.nan),
            "structural_sgm": np.full(n, np.nan),
            "structural_sdi": np.full(n, np.nan),
        },
        index=actions.index,
    )
    if n == 0 or len(frames) == 0:
        return out

    fid_by_pos = resolve_frame_ids_by_position(actions, frames, links=links)

    col_lbs = np.full(n, np.nan)
    col_sgm = np.full(n, np.nan)
    col_sdi = np.full(n, np.nan)
    frame_groups = frames.groupby(["period_id", "frame_id"])

    for j, (_idx, row) in enumerate(actions.iterrows()):
        if row.get("type_id") not in (0, 1):  # 0=pass, 1=cross (SPADL type ids)
            continue
        tid = row["team_id"]
        if pd.isna(tid) or np.isnan(fid_by_pos[j]):
            continue
        pid = int(row["period_id"])  # shift may promote period_id to float
        fid = int(fid_by_pos[j])
        try:
            frame_data = frame_groups.get_group((pid, fid))
        except KeyError:
            continue

        m = compute_structural_pass_metrics(
            frame_data,
            attacking_team_id=tid,
            home_team_id=home_team_id,
            passer_xy=(float(row["start_x"]), float(row["start_y"])),
            receiver_xy=(float(row["end_x"]), float(row["end_y"])),
            params=params,
        )
        col_lbs[j] = m["structural_lbs"]
        col_sgm[j] = m["structural_sgm"]
        col_sdi[j] = m["structural_sdi"]

    out["structural_lbs"] = col_lbs
    out["structural_sgm"] = col_sgm
    out["structural_sdi"] = col_sdi
    return out
