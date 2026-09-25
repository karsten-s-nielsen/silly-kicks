"""Layer-2 danger-behind-line valuation (TF-60 PR2, ADR-081, spec §7.2).

The five Layer-2 metrics for one sample, keyed by the ``RD_LAYER2_COLUMNS`` names. Reuses TF-7 pitch
control (``compute_pitch_control`` / ``PitchControlSurface.control_in_region`` / ``compute_threat_pc``)
and TF-15 GK reachable area (``compute_gk_influence``), oriented via the ``GoalMap`` scalars threaded
in on the :class:`SampleContext`. Additive; nothing here mutates ``frame_rows``.

The whole family is gated on a fitted ``xt`` (P2-02): without one, all five are NaN before any
pitch-control call, so a Layer-1-only caller pays no cost and hits no velocity precondition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.id_compat import ids_match
from silly_kicks.tracking import (
    GoalEndUnresolvedError,
    compute_gk_influence,
    compute_pitch_control,
    compute_threat_pc,
    compute_threat_pc_batch,
    zero_velocity_if_unavailable,
)

from ._columns import (
    RD_ATTACKER_SPACE_CONTROL,
    RD_DANGER_BEHIND_LINE,
    RD_DANGER_BEHIND_LINE_GK,
    RD_GK_COVERAGE_BEHIND_LINE,
    RD_GK_REACHABLE_COVERAGE_M2,
)
from ._counting import bool_flag
from ._geometry import danger_zone_bounds
from ._wfield import build_w_field

_NAN = float("nan")
_SPEARMAN = "spearman"
_PITCH_HEIGHT = 68.0


def _zone(ctx, params):
    """``(x_min, x_max)`` of the danger zone Z, or None if the geometry is unresolved."""
    if not (np.isfinite(ctx.defensive_line_x) and np.isfinite(ctx.own_goal_x)):
        return None
    return danger_zone_bounds(ctx.defensive_line_x, ctx.own_goal_x, zone_depth_m=params.zone_depth_m)


def _resolve_a_keeper_id(frame_rows, team_id):
    """The player_id of team A's keeper in this frame, or None if not exactly one is observed."""
    if "is_goalkeeper" not in frame_rows.columns:
        return None
    mask = ids_match(frame_rows["team_id"], team_id).to_numpy() & bool_flag(frame_rows["is_goalkeeper"])
    ids = frame_rows.loc[mask, "player_id"].dropna().unique()
    return ids[0] if len(ids) == 1 else None


def _gk_share_in_zone(surface, gk_id, team_id, lo, hi):
    """Keeper's mean share of team A's per-cell influence over the Z x-band -- MIRRORS the TF-15
    ``compute_gk_influence`` share_grid (``_gk_influence.py:396``), restricted to Z. NaN if the
    decompose fields are missing. DRIFT NOTE: this duplicates that share formula; if TF-15's
    per-cell share changes, update here too (the reuse alternative -- a region-share field on
    ``compute_gk_influence`` -- was declined to keep the tracking blast radius to the ``region`` param)."""
    if surface.per_player_influence is None or surface.player_ids is None or surface.player_team_ids is None:
        return _NAN
    gk_surface = surface.player_surface(gk_id)  # (ny, nx)
    # ADR-019: team_id is ACTION-sourced (ctx.team_id) while player_team_ids is FRAME-sourced -> a
    # CROSS-source compare (the numeric-actions x string-frames case the id-dtype gate tests). Route
    # via ids_match (unlike _gk_influence.py:396, whose scalar is drawn from the frame's own gk_row).
    team_mask = ids_match(surface.player_team_ids, team_id).to_numpy()
    team_surface = surface.per_player_influence[np.flatnonzero(team_mask)].sum(axis=0)
    safe = np.where(team_surface < 1e-8, np.inf, team_surface)
    share = np.where(team_surface < 1e-8, 0.0, gk_surface / safe)  # (ny, nx)
    xmask = (surface.grid_x >= lo) & (surface.grid_x <= hi)
    region = share[:, xmask]
    return float(region.mean()) if region.size else _NAN


def layer2_metrics(frame_rows, ctx, *, xt, goal_map, params, pitch_control_cache=None, threat_values=None) -> dict:
    """All five Layer-2 metrics for one sample, keyed by the ``RD_LAYER2_COLUMNS`` names.

    ``xt=None`` -> all five NaN (the Layer-2 gate, P2-02). A keeper absent/unresolved -> the three
    keeper-dependent metrics NaN; the GK-blind ``rd_danger_behind_line`` still computes. An
    unresolvable zone -> all five NaN. ``compute_threat_pc`` on an unfitted ``xt`` propagates
    (fail-closed).

    ``threat_values`` (ADR-105 Task 2): a pre-batched ``(gk_leg, blind_leg)`` pair. When given, the two
    ``compute_threat_pc`` legs are NOT recomputed -- the caller (``_score_samples``) has already scored
    them across a chunk via ``compute_threat_pc_batch`` (byte-identical). ``None`` = compute directly (the
    per-frame path, for direct callers). Independently, ``pitch_control_cache`` is threaded into
    ``compute_gk_influence`` so its @opponent_id surface HITS a warmed cache.
    """
    out = dict.fromkeys(
        (
            RD_ATTACKER_SPACE_CONTROL,
            RD_DANGER_BEHIND_LINE,
            RD_DANGER_BEHIND_LINE_GK,
            RD_GK_COVERAGE_BEHIND_LINE,
            RD_GK_REACHABLE_COVERAGE_M2,
        ),
        _NAN,
    )
    # P2-02: Layer 2 is the danger valuation -- gated ENTIRELY on a fitted xt. Everything below is
    # reached ONLY with an xt, so a Layer-1-only caller pays nothing and hits no velocity precondition.
    if xt is None:
        return out
    zone = _zone(ctx, params)
    if zone is None or pd.isna(ctx.opponent_id) or pd.isna(ctx.team_id):
        return out
    lo, hi = zone
    frame = zero_velocity_if_unavailable(frame_rows, method=_SPEARMAN)
    a_keeper = _resolve_a_keeper_id(frame, ctx.team_id)

    # #1 space control + #4 gk coverage share (ONE canonical decompose surface for team A).
    # Cacheable: the frame is UNMODIFIED and attacking=A is canonical, so cache.surface (keyed on
    # frame_id) is correct here -- unlike the keeper-removed #2 leg, which must never touch the cache.
    try:
        surf_a = (
            pitch_control_cache.surface(frame, ctx.team_id, method=_SPEARMAN, decompose=True)
            if pitch_control_cache is not None
            else compute_pitch_control(frame, ctx.team_id, method=_SPEARMAN, decompose=True)
        )
        out[RD_ATTACKER_SPACE_CONTROL] = 1.0 - surf_a.control_in_region(lo, hi, 0.0, _PITCH_HEIGHT)
        if a_keeper is not None:
            out[RD_GK_COVERAGE_BEHIND_LINE] = _gk_share_in_zone(surf_a, a_keeper, ctx.team_id, lo, hi)
    except (GoalEndUnresolvedError, ValueError):
        pass

    # #2 GK-blind + #3 GK-included danger
    if threat_values is not None:
        # Pre-batched across a chunk (ADR-105 Task 2). `(gk_leg, blind_leg)` mirror the direct branch:
        # gk_leg is NaN when a_keeper is None; blind_leg is scored on frame_no_gk (or frame). A chunk
        # whose batch raised GoalEndUnresolvedError falls back to threat_values=None (the branch below).
        out[RD_DANGER_BEHIND_LINE_GK], out[RD_DANGER_BEHIND_LINE] = threat_values
    else:
        w = build_w_field(ctx.own_goal_x, params.w_field_params) if params.danger_field_weight else None
        try:
            out[RD_DANGER_BEHIND_LINE_GK] = (
                compute_threat_pc(frame, attacking_team_id=ctx.opponent_id, xt=xt, goal_map=goal_map, field_weight=w)
                if a_keeper is not None
                else _NAN
            )
            frame_no_gk = frame[~ids_match(frame["player_id"], a_keeper).to_numpy()] if a_keeper is not None else frame
            out[RD_DANGER_BEHIND_LINE] = compute_threat_pc(
                frame_no_gk, attacking_team_id=ctx.opponent_id, xt=xt, goal_map=goal_map, field_weight=w
            )
        except GoalEndUnresolvedError:
            pass

    # #5 reachable ∩ Z (needs xt for the compute_gk_influence seam; the reachable value itself ignores xt)
    if a_keeper is not None:
        try:
            out[RD_GK_REACHABLE_COVERAGE_M2] = compute_gk_influence(
                frame,
                attacking_team_id=ctx.opponent_id,
                gk_player_id=a_keeper,
                xt=xt,
                goal_map=goal_map,
                region=(lo, hi, 0.0, _PITCH_HEIGHT),
                pitch_control_cache=pitch_control_cache,
            ).reachable_area_m2
        except (GoalEndUnresolvedError, ValueError):
            pass
    return out


def batch_threat_values(prepared, *, xt, goal_map, params):
    """Per-sample ``(gk_leg, blind_leg)`` for a CHUNK of scored samples via ONE
    :func:`compute_threat_pc_batch` (ADR-105 Task 2) -- byte-identical to the per-sample
    :func:`layer2_metrics` threat block.

    ``prepared`` is a list of ``(frame_rows, ctx)``. Returns a list aligned to ``prepared``; each entry
    is a ``(gk_val, blind_val)`` tuple. A sample outside the Layer-2 threat domain (no ``xt`` / no zone /
    NA opp-or-team) gets ``(NaN, NaN)`` -- harmless, since :func:`layer2_metrics` returns early there. A
    ``GoalEndUnresolvedError`` from the batch returns ``None`` for the WHOLE chunk, so the caller falls
    back to the direct per-sample path (byte-identical values, just not batched).
    """
    items: list = []
    plan: list = []
    for frame_rows, ctx in prepared:
        zone = _zone(ctx, params)
        if xt is None or zone is None or pd.isna(ctx.opponent_id) or pd.isna(ctx.team_id):
            plan.append(None)
            continue
        frame = zero_velocity_if_unavailable(frame_rows, method=_SPEARMAN)
        a_keeper = _resolve_a_keeper_id(frame, ctx.team_id)
        w = build_w_field(ctx.own_goal_x, params.w_field_params) if params.danger_field_weight else None
        gk_idx = None
        if a_keeper is not None:
            gk_idx = len(items)
            items.append((frame, ctx.opponent_id, goal_map, w))
            frame_no_gk = frame[~ids_match(frame["player_id"], a_keeper).to_numpy()]
        else:
            frame_no_gk = frame
        blind_idx = len(items)
        items.append((frame_no_gk, ctx.opponent_id, goal_map, w))
        plan.append((gk_idx, blind_idx))

    if not items or xt is None:
        return None  # nothing to batch (or no xt) -> caller uses the direct per-sample path
    try:
        vals = compute_threat_pc_batch(items, xt=xt)
    except GoalEndUnresolvedError:
        return None  # rare (scored samples have resolved geometry) -> direct fallback, byte-identical

    out: list = []
    for p in plan:
        if p is None:
            out.append((_NAN, _NAN))
        else:
            gk_idx, blind_idx = p
            out.append((vals[gk_idx] if gk_idx is not None else _NAN, vals[blind_idx]))
    return out
