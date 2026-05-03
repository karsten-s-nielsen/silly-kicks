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

import numpy as np
import pandas as pd

from .feature_framework import ActionFrameContext

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
