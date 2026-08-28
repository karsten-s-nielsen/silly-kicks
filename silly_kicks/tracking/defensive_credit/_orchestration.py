"""Batch orchestration: actions -> long-form defensive-credit rows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks._frame_index import group_rows
from silly_kicks.id_compat import ids_differ
from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking._action_orientation import acting_team_attacks_rtl
from silly_kicks.tracking.utils import link_actions_to_frames

from ._chaining import with_possessions
from ._line_break_signal import precompute_line_break_between_lines
from ._params import DEFENSIVE_CREDIT_RULES, RULE_FAILED_MARKING_THROUGH_BALL, DefensiveCreditParams
from ._rules import RULE_REGISTRY, CreditRow, RuleContext

_LONG_COLS = [
    "game_id",
    "period_id",
    "action_id",
    "player_id",
    "team_id",
    "rule",
    "signed_value",
    "anchor_type",
    "frame_id",
    "sizing",
    "resolution",
    # Task 6 (ADR-077), appended ADDITIVELY at the END: the per-credit resolution ANCHOR (action-LTR)
    # + box-aware search radius the FOV companion rebuilds each credit's region from. NaN on the
    # event-resolved anchor_actor rows; region_radius NaN on the lane corridor rows.
    "origin_x",
    "origin_y",
    "region_radius",
]

_SHOT_TYPE = spadlconfig.actiontype_id["shot"]
_GOAL_RESULT = spadlconfig.result_id["success"]


def _ensure_on_target(
    act: pd.DataFrame, frames: pd.DataFrame, pointers: pd.DataFrame, on_target_column: str
) -> pd.DataFrame:
    """Attach a nullable-boolean ``_on_target`` per action (P-1). Shots only; others NA.

    Goal (result success) -> True. Else the injected ``on_target_column`` if present, else the
    frame-based TF-48 ``shot_on_target_derived`` fallback (reuses the ONE link via ``links=pointers``).
    ``pd.NA`` stays NA (unknown) -> the pressure rules do not fire, so a saved shot is never mis-signed.
    """
    act = act.copy()
    if on_target_column in act.columns:
        base = pd.array(act[on_target_column], dtype="boolean")  # type: ignore[reportCallIssue, reportArgumentType]
    else:
        from silly_kicks.tracking.features import add_shot_goalmouth

        gm = add_shot_goalmouth(act, frames, links=pointers)
        col = (
            gm["shot_on_target_derived"]
            if "shot_on_target_derived" in gm.columns
            else pd.Series(pd.NA, index=act.index)
        )
        base = pd.array(col, dtype="boolean")  # type: ignore[reportCallIssue, reportArgumentType]
    is_shot = (act["type_id"] == _SHOT_TYPE).to_numpy()
    is_goal = ((act["type_id"] == _SHOT_TYPE) & (act["result_id"] == _GOAL_RESULT)).to_numpy()
    base[~is_shot] = pd.NA  # only meaningful for shots
    base[is_goal] = True  # a scored shot is on-target
    act["_on_target"] = base
    return act


def compute_defensive_credits(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    xg_column: str,
    xt,
    blocked_column: str = "shot_blocked",
    on_target_column: str = "shot_on_target_derived",
    links: pd.DataFrame | None = None,
    params: DefensiveCreditParams | None = None,
) -> pd.DataFrame:
    """Long-form: one row per (triggering action, credited player, rule). Pure.

    Examples
    --------
    Per-defender signed credit rows (needs a fitted ``ExpectedThreat`` and an injected per-shot xG)::

        credits = compute_defensive_credits(actions, frames, xg_column="xg", xt=xt)
        credits.groupby("player_id")["signed_value"].sum()  # per-player rollup
    """
    params = params or DefensiveCreditParams()
    act = with_possessions(actions).reset_index(drop=True)

    pointers = links if links is not None else link_actions_to_frames(act, frames)[0]
    act = _ensure_on_target(act, frames, pointers, on_target_column)  # P-1: attach _on_target (tri-state)
    fid_by_pos = (
        pointers.drop_duplicates("action_id")
        .set_index("action_id")["frame_id"]
        .reindex(act["action_id"].to_numpy())
        .to_numpy()
    )
    flip_series = acting_team_attacks_rtl(act, frames)  # ONE reprojection decision per action

    enabled = [r for r in DEFENSIVE_CREDIT_RULES if r in params.rules]
    # Item 3: precompute the between_lines line-break signal ONCE (threading the single link via
    # fid_by_pos); skip the Ward-clustering cost entirely when the rule is disabled (Q8).
    if RULE_FAILED_MARKING_THROUGH_BALL in enabled:
        line_break_between_lines = precompute_line_break_between_lines(
            # fillna(False) is safe HERE and only here: the actions whose direction is
            # unresolved are skipped wholesale in the loop below, so their precomputed entry is
            # never read. Passing the nullable array instead would push the NA into a numpy
            # boolean context inside the Ward clustering.
            act,
            frames,
            fid_by_pos=fid_by_pos,
            flip_by_pos=flip_series.fillna(False).to_numpy(dtype=bool),
        )
    else:
        line_break_between_lines = pd.array([pd.NA] * len(act), dtype="boolean")
    # ADR-068: build the per-frame lookup ONCE (keyed on frame_id, matching resolve_responsible_
    # defenders' filter exactly) instead of re-scanning the whole `frames` table per action x per rule.
    frame_groups = group_rows(frames, "frame_id")
    rows: list[CreditRow] = []
    for idx in range(len(act)):
        if pd.isna(flip_series.iloc[idx]):
            # Unresolved direction -> this action assigns NO credit. Every geometric rule below
            # consumes `flip` (lane blocking, proximity, between-lines), so there is no subset
            # that could still be evaluated honestly. Deliberately NOT the unlinked-frame route
            # just below, which keeps the action and lets each rule decide: an unlinked action is
            # missing its frame, whereas this one has a frame it cannot orient.
            continue
        fid = fid_by_pos[idx]
        fid = None if pd.isna(fid) else int(fid)
        ctx = RuleContext(
            actions=act,
            frames=frames,
            idx=idx,
            xg_column=xg_column,
            xt=xt,
            blocked_column=blocked_column,
            params=params,
            frame_id=fid,
            acting_team_id=act.iloc[idx]["team_id"],
            flip=bool(flip_series.iloc[idx]),
            line_break_between_lines=line_break_between_lines,
            frame_groups=frame_groups,
        )
        for rule_name in enabled:
            rows.extend(RULE_REGISTRY[rule_name](ctx))

    return _to_long_form(rows, act)


def _to_long_form(rows: list[CreditRow], act: pd.DataFrame) -> pd.DataFrame:
    if not rows:
        return _empty_long_form(act)
    df = pd.DataFrame([r.__dict__ for r in rows])  # 9 fields; period_id is NOT per-credit -> merge it
    pid = act[["action_id", "period_id"]].drop_duplicates("action_id")
    df = df.merge(pid, on="action_id", how="left")
    df = df[_LONG_COLS]  # reorder to the canonical schema (period_id now present via the merge)
    df["signed_value"] = df["signed_value"].astype("float64")
    df["frame_id"] = df["frame_id"].astype("Int64")
    df["period_id"] = df["period_id"].astype("int64")
    for c in ("origin_x", "origin_y", "region_radius"):  # Task 6 region-anchor columns
        df[c] = df[c].astype("float64")
    return df.reset_index(drop=True)


def _empty_long_form(act: pd.DataFrame) -> pd.DataFrame:
    empty = {c: pd.Series([], dtype="object") for c in _LONG_COLS}  # resolution stays object
    empty["signed_value"] = pd.Series([], dtype="float64")
    empty["frame_id"] = pd.Series([], dtype="Int64")
    empty["period_id"] = pd.Series([], dtype="int64")
    for c in ("origin_x", "origin_y", "region_radius"):
        empty[c] = pd.Series([], dtype="float64")
    return pd.DataFrame(empty)


def _aggregate_defensive_credit(
    actions,
    frames,
    *,
    xg_column,
    xt,
    blocked_column="shot_blocked",
    on_target_column="shot_on_target_derived",
    links=None,
    params=None,
    visible_area=None,
) -> pd.DataFrame:
    """actions + per-action aggregate columns (defending-team-scoped). Pure -- returns a NEW frame.

    No ``home_team_id`` (P-2): the defending/attacking split derives from ``team_id != acting-team``
    and reprojection uses ``acting_team_attacks_rtl``, so a home_team_id would be a dead required param.

    When ``visible_area`` (an ``action_id`` -> polygon table) is supplied, TWO additional
    companion columns are appended -- ``defensive_credit_observed_fraction`` /
    ``defensive_credit_observed_source`` -- the mode-aware FOV-observability rollup over the SAME
    defending credit set (ADR-077, Task 6). OPT-IN and additive: net/plus/minus/n are byte-identical.
    """
    params = params or DefensiveCreditParams()  # resolve ONCE (byte-identical) so the rollup below
    # and compute_defensive_credits share the exact lane geometry the FOV corridor is rebuilt from.
    long = compute_defensive_credits(
        actions,
        frames,
        xg_column=xg_column,
        xt=xt,
        blocked_column=blocked_column,
        on_target_column=on_target_column,
        links=links,
        params=params,
    )
    out = actions.copy()
    act_team = actions.set_index("action_id")["team_id"]
    if long.empty:
        defending = long
    else:
        long = long.copy()
        long["_acting_team"] = long["action_id"].map(act_team)
        keep = ids_differ(long["team_id"], long["_acting_team"])  # credited team != acting team -> defender
        defending = long[keep.to_numpy()]

    if defending.empty:
        net = plus = minus = pd.Series(dtype="float64")
        cnt = pd.Series(dtype="int64")
    else:
        net = defending.groupby("action_id")["signed_value"].sum(min_count=0)
        plus = defending[defending["signed_value"] > 0].groupby("action_id")["signed_value"].sum()
        minus = defending[defending["signed_value"] < 0].groupby("action_id")["signed_value"].sum()
        cnt = defending.groupby("action_id").size()

    aid = out["action_id"]
    out["defensive_credit_net"] = aid.map(net).fillna(0.0).astype("float64")
    out["defensive_credit_plus"] = aid.map(plus).fillna(0.0).astype("float64")
    out["defensive_credit_minus"] = aid.map(minus).fillna(0.0).astype("float64")
    out["n_defensive_credits"] = aid.map(cnt).fillna(0).astype("Int64")

    if visible_area is not None:  # ADR-077/Task 6: the opt-in mode-aware credit-region rollup.
        fractions, sources = _defensive_credit_observation_rollup(
            actions, defending, params=params, visible_area=visible_area, links=links
        )
        out["defensive_credit_observed_fraction"] = fractions
        out["defensive_credit_observed_source"] = sources
    return out


def _defensive_credit_observation_rollup(actions, defending, *, params, visible_area, links):
    """Per-action FOV-observability rollup for the defensive-credit family (ADR-077, Task 6).

    ONE companion ``(fraction, source)`` per action, over the SAME defending credit set the
    net/plus/minus/n aggregate uses. Each defending credit's region is built BY MODE -- a proximity
    DISK (nearest / all_within / all_within_beyond_nearest / nearest_fallback), the shot->goal
    CORRIDOR (lane), or NO region (anchor_actor, event-resolved) -- and scored against the action's
    ``visible_area`` polygon; the per-action fraction is the credit-MAGNITUDE-weighted mean over the
    region-bearing OBSERVED credits (P5). Returns ``(fractions ndarray, sources list)`` aligned to
    ``actions``.
    """
    # Function-local imports keep this seam neutral (the _fov_registry engine imports NEITHER
    # _orchestration nor features, so it may serve this custom path without a cycle).
    from silly_kicks.id_compat import canonical_id
    from silly_kicks.tracking._fov_registry import (
        _NO_REGION,
        _rollup_credit_observed_fraction,
        defensive_credit_region_for_mode,
    )
    from silly_kicks.tracking._visibility import (
        REGION_OBSERVATION_DEGENERATE_REGION,
        VISIBLE_AREA_UNLINKED,
        _polygons_by_action,
        classify_region_observation,
    )

    polygons = _polygons_by_action(visible_area)
    linked_ids: set | None = None
    if links is not None and len(links) > 0:
        ok = links[links["frame_id"].notna()] if "frame_id" in links.columns else links
        linked_ids = {canonical_id(a) for a in ok["action_id"]}

    lane_kw = {
        "lane_cone_width_factor": params.shot_lane_cone_width_factor,
        "lane_max_t": params.shot_lane_max_t,
        "lane_min_half_width_m": params.shot_lane_min_half_width_m,
    }
    # ONE pass over the defending credits, grouped by action -- never a per-action rescan.
    per_action: dict = {}
    if defending is not None and not defending.empty:
        cols = ["resolution", "origin_x", "origin_y", "region_radius", "signed_value"]
        for aid, group in defending.groupby("action_id", sort=False):
            ckey = canonical_id(aid)
            polygon = polygons.get(ckey)
            observations = []
            for c in group[cols].itertuples(index=False):
                region = defensive_credit_region_for_mode(
                    c.resolution,
                    origin_x=c.origin_x,
                    origin_y=c.origin_y,
                    region_radius=c.region_radius,
                    **lane_kw,
                )
                if region is _NO_REGION:  # anchor_actor / degenerate -> excluded from BOTH sums (P5)
                    observations.append((c.signed_value, float("nan"), REGION_OBSERVATION_DEGENERATE_REGION, False))
                    continue
                frac, s = classify_region_observation(polygon, region)
                observations.append((c.signed_value, frac, s, True))
            per_action[ckey] = _rollup_credit_observed_fraction(observations)

    n = len(actions)
    fractions = np.full(n, np.nan)
    sources: list[str] = []
    for i, aid in enumerate(actions["action_id"]):
        ckey = canonical_id(aid)
        if linked_ids is not None and ckey not in linked_ids:
            sources.append(VISIBLE_AREA_UNLINKED)  # fraction stays NaN
            continue
        if ckey in per_action:
            frac, s = per_action[ckey]
            fractions[i] = frac
            sources.append(s)
        else:  # no defending credit at all -> no region to observe
            sources.append(REGION_OBSERVATION_DEGENERATE_REGION)
    return fractions, sources
