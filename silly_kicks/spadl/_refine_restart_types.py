"""The single row-mutating restart stage (TF-46 Arm 1 Part 2, spec section 11).

``refine_restart_types`` is PURE and IDEMPOTENT. It is the ONLY stage that rewrites a restart row's type,
adds derived penalty / keeper-save rows, or merges a SkillCorner derived free-kick row with its shot -- and
every change is REVERSIBLE: the prior values live in ``*_events`` columns, a dropped merged row's content in
``merged_derived_*``, and every added row carries ``refine_created = True`` (CL-9) + ``result_source =
"inferred"`` (so ``training_exclusion_mask`` drops it). A native row (``type_source`` native) is NEVER
rewritten (spec section 3); refine only flags ``type_tracking_disagrees``. The gate table is EMPTY in commit 2
(CL-3), so no subtype rewrite, merge, or penalty row ships; an injected passing table exercises them.

All additions / removals happen here, followed by ONE atomic ``action_id`` renumber (ADR-065, section 11.5).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from silly_kicks._frame_index import column_position
from silly_kicks.id_compat import same_id
from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl.base import sort_actions_chronologically
from silly_kicks.tracking._gk_resolve import GoalMap, resolve_defended_goals
from silly_kicks.tracking._restart_frame import team_attacks_rtl, to_action_frame
from silly_kicks.tracking._restart_validation import (
    GATE_TABLE_VERSION,
    RATIFIED,
    RESOLVE_KICKS_COLUMNS,
    DataProfile,
    gate,
)

_CORNER_SHORT = spadlconfig.actiontype_id.get("corner_short")
_CORNER_CROSSED = spadlconfig.actiontype_id.get("corner_crossed")
_FK_SHORT = spadlconfig.actiontype_id.get("freekick_short")
_FK_CROSSED = spadlconfig.actiontype_id.get("freekick_crossed")
_SHOT = spadlconfig.actiontype_id.get("shot")
_SHOT_PENALTY = spadlconfig.actiontype_id.get("shot_penalty")
_KEEPER_SAVE = spadlconfig.actiontype_id.get("keeper_save")
_FAIL = spadlconfig.result_id.get("fail")
_SUCCESS = spadlconfig.result_id.get("success")
_SHOT_LIKE = frozenset({t for t in (_SHOT, _SHOT_PENALTY) if t is not None})
# The Part-1 constant penalty-spot start (spec section 11.4 events-first merge form); the merge radius tolerates
# the constant's <= 0.55 m offset from the pitch-scaled spot.
_MERGE_SPOT = (94.0, 34.0)
_SUBTYPE_PAIRS = {
    _CORNER_SHORT: _CORNER_CROSSED,
    _CORNER_CROSSED: _CORNER_SHORT,
    _FK_SHORT: _FK_CROSSED,
    _FK_CROSSED: _FK_SHORT,
}
_PA_X = spadlconfig.field_length - 16.5
_PA_HALF = spadlconfig.penalty_area_half_width

_STR_COLS = (
    "type_source_events",
    "subtype_evidence",
    "subtype_gate_version",
    "subtype_cell_resolution",
    "merged_derived_result_source",
    "merged_derived_restart_evidence",
    "merged_derived_type_source",
    "merged_derived_original_event_id",
    "merged_derived_repair_kind",
    "merged_derived_gi_attacking_side",
    "merge_evidence",
    "merge_gate_version",
    "merge_cell_resolution",
    "penalty_evidence",
    "penalty_gate_version",
    "penalty_cell_resolution",
    "penalty_crosscheck",
    "penalty_outcome",
)
_INT_COLS = (
    "type_id_events",
    "merged_from_derived_action_id",
    "merged_from_shot_action_id",
    "merged_derived_type_id",
    "merged_derived_result_id",
    "merged_derived_gi_marker_possession_frame_end",
    "merged_derived_gi_next_possession_frame_start",
)
_FLOAT_COLS = (
    "start_x_events",
    "start_y_events",
    "merged_derived_time_seconds",
    "merged_derived_end_x",
    "merged_derived_end_y",
)


def _init_columns(out: pd.DataFrame) -> pd.DataFrame:
    for c in _STR_COLS:
        out[c] = pd.NA
    for c in _INT_COLS:
        out[c] = pd.array([pd.NA] * len(out), dtype="Int64")
    for c in _FLOAT_COLS:
        out[c] = np.nan
    out["type_tracking_disagrees"] = pd.array([pd.NA] * len(out), dtype="boolean")
    out["refine_created"] = pd.array([False] * len(out), dtype="boolean")
    return out


def _as_int(v) -> int:
    """A pandas group key -> int; the untyped arg sidesteps the Hashable stub friction."""
    return int(v)


def _ball_by_period(frames: pd.DataFrame) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per period, the ball's time-sorted ``(t, x, y)``, built ONCE (ADR-068: the per-restart window is a
    ``searchsorted`` slice, never a re-filter of the whole frame table)."""
    if frames.empty:
        return {}
    ball = frames[frames["is_ball"].to_numpy(dtype=bool)] if "is_ball" in frames else frames
    out: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for period, pos in ball.groupby("period_id", sort=True).indices.items():
        g = ball.iloc[pos].sort_values("time_seconds", kind="mergesort")
        out[_as_int(period)] = (
            g["time_seconds"].to_numpy(dtype=np.float64),
            g["x"].to_numpy(dtype=np.float64),
            g["y"].to_numpy(dtype=np.float64),
        )
    return out


def _ball_enters_pa_after(ball_by_period, period: int, t0: float, rtl, window: float = 6.0) -> bool | None:
    """The section 11.2 rule: does the ball enter the ATTACKED penalty area within ``window`` s after ``t0``?

    Evaluated in the restarting team's action frame (it attacks +x; direction from GoalMap, ADR-055 -- never a
    frame direction label), so the PA is ``x >= L-16.5`` within the half-width band. Fails closed with
    ``None`` (no tracking verdict) when the direction is unresolved or no ball frame is observed in the window."""
    if rtl is None or pd.isna(rtl):
        return None
    hit = ball_by_period.get(period)
    if hit is None:
        return None
    t, x, y = hit
    lo = int(np.searchsorted(t, t0, side="right"))
    hi = int(np.searchsorted(t, t0 + window, side="right"))
    if hi <= lo:
        return None
    ax, ay = to_action_frame(x[lo:hi], y[lo:hi], pd.Series([bool(rtl)] * (hi - lo), dtype="boolean"))
    return bool(np.any((ax >= _PA_X) & (np.abs(ay - spadlconfig.field_width / 2.0) <= _PA_HALF)))


def refine_restart_types(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    provider: str,
    penalties: pd.DataFrame | None = None,
    data_profile: DataProfile | None = None,
    gate_table: Mapping | None = None,
    goal_map: GoalMap | None = None,
) -> pd.DataFrame:
    """Refine restart rows (section 11). Returns a COPY of ``actions`` with the section 11 columns, the gated
    subtype rewrites / merges / penalty rows applied, and one atomic ``action_id`` renumber.

    ``gate_table`` overrides the shipped restart gate table (default ``None`` = shipped); the pre-flight drivers
    inject one to score each rule regardless of the shipped gate. ``goal_map`` is the match's
    ``resolve_defended_goals(frames)`` (built here when ``None``); the subtype rule reads direction from it.

    Examples
    --------
    Refine after the kicks are resolved and the penalties detected (one renumber at the end)::

        goal_map = resolve_defended_goals(frames)
        kicks = resolve_kicks(actions, frames, provider="skillcorner", goal_map=goal_map, data_profile=profile)
        pens = detect_penalties(kicks, frames, provider="skillcorner", goal_map=goal_map, data_profile=profile)
        refined = refine_restart_types(
            kicks, frames, provider="skillcorner", penalties=pens, data_profile=profile, goal_map=goal_map
        )
    """
    out = _init_columns(actions.copy())
    if goal_map is None:
        goal_map = resolve_defended_goals(frames)

    # --- 11.2 subtype rewrite (gated per data profile x direction) ---
    tsrc = out["type_source"] if "type_source" in out else pd.Series([pd.NA] * len(out))
    type_ids = out["type_id"].to_numpy(dtype=np.int64)
    sub_pos = np.flatnonzero(np.isin(type_ids, list(_SUBTYPE_PAIRS)))
    balls = _ball_by_period(frames) if len(sub_pos) else {}
    rtl_sub = team_attacks_rtl(out.iloc[sub_pos], goal_map).to_numpy(dtype=object) if len(sub_pos) else []
    rtl_at = dict(zip(sub_pos.tolist(), rtl_sub, strict=True))
    for idx in sub_pos.tolist():
        tid = int(type_ids[idx])
        base = "corner" if tid in (_CORNER_SHORT, _CORNER_CROSSED) else "free_kick"
        src = None if pd.isna(tsrc.iloc[idx]) else str(tsrc.iloc[idx])  # Part-1 repair rows carry <NA>
        crossed = _ball_enters_pa_after(
            balls, int(out.iloc[idx]["period_id"]), float(out.iloc[idx]["time_seconds"]), rtl_at[idx]
        )
        if crossed is None:
            continue  # no tracking verdict (unresolved direction / unobserved ball): type_tracking_disagrees stays <NA>
        tracking_tid = (
            (_CORNER_CROSSED if crossed else _CORNER_SHORT)
            if base == "corner"
            else (_FK_CROSSED if crossed else _FK_SHORT)
        )
        disagrees = tracking_tid != tid
        if src not in ("default", "inferred"):
            # native (or Part-1 <NA>): never rewrite; only flag disagreement.
            out.iat[int(idx), column_position(out, "type_tracking_disagrees")] = bool(disagrees)
            continue
        direction = "crossed" if crossed else "short"
        entry = gate(data_profile, base, "subtype", direction, table=gate_table) if data_profile is not None else None
        out.iat[int(idx), column_position(out, "type_tracking_disagrees")] = bool(disagrees)
        if entry is not None and disagrees:
            out.iat[int(idx), column_position(out, "type_id_events")] = int(tid)
            out.iat[int(idx), column_position(out, "type_source_events")] = src
            out.iat[int(idx), column_position(out, "type_id")] = tracking_tid
            out.iat[int(idx), column_position(out, "type_source")] = "tracking"
            out.iat[int(idx), column_position(out, "subtype_evidence")] = entry.tier
            out.iat[int(idx), column_position(out, "subtype_gate_version")] = GATE_TABLE_VERSION
            out.iat[int(idx), column_position(out, "subtype_cell_resolution")] = entry.cell_resolution

    # --- 11.4 B8 merge (SkillCorner derived free-kick + following restarting-team shot) ---
    # The dropped derived row's merged_derived_* content is the PRE-refine value (B-42), so it is read from
    # the original `actions`, not from `out` (which the subtype rewrite above may already have changed).
    if provider == "skillcorner":
        out = _merge_b8(out, actions, data_profile, gate_table, frames)

    # --- 11.3 penalty rows (from detect_penalties). SkillCorner adds / annotates / retypes; a native provider
    # codes penalties natively (Sportec since section 25), so there the detector only ANNOTATES the native
    # penalty shot within the tau_event ceiling of the release and never adds a row.
    if penalties is not None and len(penalties):
        out = _apply_penalties(out, penalties, data_profile, gate_table, native=provider != "skillcorner")

    # --- 11.5 one atomic renumber ---
    out = sort_actions_chronologically(out, by=("period_id", "time_seconds")).reset_index(drop=True)
    out["action_id"] = np.arange(len(out), dtype="int64")
    return out


def _merge_form(a: pd.Series, b: pd.Series, origin: pd.Series | None = None) -> str | None:
    """Which section 11.4 merge form (if any) a derived free-kick row ``a`` and its restarting-team shot ``b``
    satisfy.

    - ``events_first``: the shot starts within ``merge_r_m`` of the constant spot AND within ``merge_dt_s`` of the
      derived restart (the Part-1 section 13 mis-tagged penalty).
    - ``tracking``: otherwise, the derived row's GATED taker (``taker_source == "tracking"``) is the shooter AND
      its GATED origin lies within ``merge_r_m`` of the shot start (a genuine direct free-kick shot). The origin is
      the row's own ``origin_evidence`` / ``enriched_start_x/_y`` when present, else ``origin`` -- the
      section 10.1 gated tiers computed by ``gated_origin`` (refine runs before ``add_restart_coordinates``). If
      either input is withheld, there is no merge.
    """

    r = RATIFIED["merge_r_m"]
    sx, sy = float(b.get("start_x", np.nan)), float(b.get("start_y", np.nan))
    if (
        np.hypot(sx - _MERGE_SPOT[0], sy - _MERGE_SPOT[1]) <= r
        and abs(float(b["time_seconds"]) - float(a["time_seconds"])) <= RATIFIED["merge_dt_s"]
    ):
        return "events_first"
    taker = a.get("taker_player_id")
    if a.get("taker_source") != "tracking" or not _known(taker) or not _known(b.get("player_id")):
        return None
    if not same_id(taker, b.get("player_id")):
        return None
    src = a if _known(a.get("origin_evidence")) else origin
    if src is None or not _known(src.get("origin_evidence")):
        return None
    ox, oy = float(src.get("enriched_start_x", np.nan)), float(src.get("enriched_start_y", np.nan))
    if not (np.isfinite(ox) and np.isfinite(oy)) or np.hypot(ox - sx, oy - sy) > r:
        return None
    return "tracking"


def _merge_b8(
    out: pd.DataFrame,
    orig: pd.DataFrame,
    data_profile: DataProfile | None,
    gate_table: Mapping | None = None,
    frames: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge a derived free-kick row immediately followed by a restarting-team shot into one row (section 11.4),
    keeping the dropped derived row's PRE-refine content in ``merged_derived_*`` (reversible). Each form has its
    own gate cell (``events_first`` / ``tracking``); the merged row is ``shot_penalty`` for the events-first form,
    else it keeps the shot's type (C-34). No form, or a withheld cell -> no merge."""
    if data_profile is None:
        return out
    from silly_kicks.tracking._gk_geometry import gated_origin

    orig_by_id = {int(r["action_id"]): r for _, r in orig.iterrows()}
    out = out.sort_values(["period_id", "time_seconds"]).reset_index(drop=True)
    # The section 10.1 gated origin, computed ONCE and only when a merge candidate exists (a derived free kick).
    has_candidate = bool(
        (out.get("action_provenance", pd.Series(dtype=object)) == "derived").any()
        and out["type_id"].isin([_FK_SHORT, _FK_CROSSED]).any()
    )
    origin = (
        gated_origin(out, frames, data_profile, gate_table)
        if has_candidate
        else pd.DataFrame({"origin_evidence": [pd.NA] * len(out)}, index=out.index)
    )
    drop_idx: list[int] = []
    for i in range(len(out) - 1):
        a = out.iloc[i]
        b = out.iloc[i + 1]
        if a.get("action_provenance") != "derived" or a["type_id"] not in (_FK_SHORT, _FK_CROSSED):
            continue
        if b["type_id"] != _SHOT or a.get("period_id") != b.get("period_id"):
            continue
        if not _same_team(a.get("team_id"), b.get("team_id")):
            continue  # the shot must be the RESTARTING team's (spec section 11.4 population)
        form = _merge_form(a, b, origin.iloc[i])
        if form is None:
            continue
        entry = gate(data_profile, "free_kick", "merge", form, table=gate_table)
        if entry is None:
            continue  # withheld in commit 2 -> no merge
        shot_i = i + 1
        od = orig_by_id[int(a["action_id"])]  # the derived row's PRE-refine values (B-42)
        out.iat[int(shot_i), column_position(out, "type_id_events")] = int(b["type_id"])
        out.iat[int(shot_i), column_position(out, "type_source_events")] = b.get("type_source")
        if form == "events_first":
            out.iat[int(shot_i), column_position(out, "type_id")] = _SHOT_PENALTY
            out.iat[int(shot_i), column_position(out, "type_source")] = "inferred"
        else:  # the tracking rule decided the merge (spec section 11.4, C-34)
            out.iat[int(shot_i), column_position(out, "type_source")] = "tracking"
        # Section 11.4: the merged row's restart_evidence is the derived row's; section 11.5: it keeps the derived
        # row's kick / taker / pairing columns (a shot row carries none of its own; unrefine restores them).
        if "restart_evidence" in out.columns:
            out.iat[int(shot_i), column_position(out, "restart_evidence")] = a.get("restart_evidence")
        for col in RESOLVE_KICKS_COLUMNS:
            if col in out.columns:
                out.iat[int(shot_i), column_position(out, col)] = a.get(col)
        out.iat[int(shot_i), column_position(out, "merged_from_derived_action_id")] = int(a["action_id"])
        out.iat[int(shot_i), column_position(out, "merged_from_shot_action_id")] = int(b["action_id"])
        for col, val in (
            ("merged_derived_type_id", int(od["type_id"])),
            ("merged_derived_result_id", int(od["result_id"])),
            ("merged_derived_time_seconds", float(od["time_seconds"])),
            ("merged_derived_end_x", float(od.get("end_x", np.nan))),
            ("merged_derived_end_y", float(od.get("end_y", np.nan))),
            ("merged_derived_result_source", od.get("result_source")),
            ("merged_derived_restart_evidence", od.get("restart_evidence")),
            ("merged_derived_type_source", od.get("type_source")),
            ("merged_derived_original_event_id", od.get("original_event_id")),
            ("merged_derived_repair_kind", od.get("repair_kind")),
            ("merged_derived_gi_marker_possession_frame_end", od.get("gi_marker_possession_frame_end")),
            ("merged_derived_gi_next_possession_frame_start", od.get("gi_next_possession_frame_start")),
            ("merged_derived_gi_attacking_side", od.get("gi_attacking_side")),
            ("merge_evidence", entry.tier),
            ("merge_gate_version", GATE_TABLE_VERSION),
            ("merge_cell_resolution", entry.cell_resolution),
        ):
            out.iat[int(shot_i), column_position(out, col)] = val if _known(val) else pd.NA
        drop_idx.append(i)
    if drop_idx:
        out = out.drop(index=drop_idx).reset_index(drop=True)
    return out


def _same_team(a, b) -> bool:
    return same_id(a, b)


def _known(v) -> bool:
    return v is not None and not (isinstance(v, float) and np.isnan(v)) and v is not pd.NA


def _rows_near(out: pd.DataFrame, period: int, team, lo: float, hi: float, types) -> list[int]:
    """Positional indices of rows of ``types`` for ``team`` in ``period`` with time in ``[lo, hi]``."""
    hits = []
    for i in range(len(out)):
        r = out.iloc[i]
        if r["type_id"] not in types or int(r["period_id"]) != period or not _same_team(r.get("team_id"), team):
            continue
        if lo <= float(r["time_seconds"]) <= hi:
            hits.append(i)
    return hits


def _set(out: pd.DataFrame, i: int, col: str, val) -> None:
    out.iat[int(i), column_position(out, col)] = val


def _annotate(out: pd.DataFrame, i: int, det, outcome, crosscheck) -> None:
    """Annotate an existing row (section 11.3, B-52): provenance + outcome columns ONLY; never result_id/type_id."""
    _set(out, i, "penalty_evidence", det.tier)
    _set(out, i, "penalty_gate_version", GATE_TABLE_VERSION)
    _set(out, i, "penalty_cell_resolution", det.cell_resolution)
    _set(out, i, "penalty_outcome", outcome)
    if _known(crosscheck):
        _set(out, i, "penalty_crosscheck", crosscheck)


def _reconciliation_row(out: pd.DataFrame, period: int, team, release: float) -> int | None:
    """The Part-1 reconciliation goal row (``shot_penalty``, success) of the attacking team that pairs with this
    detection (the nearest in the period; section 12 consistency guard, penalty context only)."""
    best, best_d = None, np.inf
    for i in range(len(out)):
        r = out.iloc[i]
        if r["type_id"] != _SHOT_PENALTY or int(r["period_id"]) != period or not _same_team(r.get("team_id"), team):
            continue
        if r.get("result_id") != _SUCCESS or bool(r.get("refine_created")) is True:
            continue
        d = abs(float(r["time_seconds"]) - release)
        if d < best_d:
            best, best_d = i, d
    return best


def _retype_to_shot(out: pd.DataFrame, i: int) -> None:
    """The section 11.1 rebound extension (B-48/B-54): retype the reconciliation goal row ``shot_penalty -> shot``;
    canonical start -> NaN (the spot constant would be fabricated), prior start / type kept in ``*_events``."""
    _set(out, i, "type_id_events", int(out.iloc[i]["type_id"]))
    _set(out, i, "type_source_events", out.iloc[i].get("type_source"))
    _set(out, i, "start_x_events", float(out.iloc[i].get("start_x", np.nan)))
    _set(out, i, "start_y_events", float(out.iloc[i].get("start_y", np.nan)))
    _set(out, i, "type_id", _SHOT)
    _set(out, i, "start_x", np.nan)
    _set(out, i, "start_y", np.nan)


def _apply_penalties(
    out: pd.DataFrame,
    penalties: pd.DataFrame,
    data_profile: DataProfile | None,
    gate_table: Mapping | None = None,
    *,
    native: bool = False,
) -> pd.DataFrame:
    """Add / annotate penalty rows from the detector table (sections 11.3, 12).

    Detection + attribution gates must pass, else nothing ships (commit 2). Per detection:
    - attribution ``goal``: the reconciliation row is annotated ``penalty_crosscheck="agree"``; no row, no retype.
    - ``rebound`` / ``later_goal``: the missed row ships (or an existing kick row is annotated) with that crosscheck,
      and the reconciliation goal row is retyped ``shot_penalty -> shot``.
    - no score change: the missed row ships (or annotates); ``penalty_outcome`` = saved / off_target only when the
      split gate passes, else ``missed``. A gated save adds a ``keeper_save`` row unless a keeper action exists.
    - unattributed with a reconciliation pair: ``disagree``; no row, no retype.
    One kick, one row (C-56): an existing shot row of the attacking team whose time lies in the penalty's own
    window [rest-window start - tau, release + W + tau] is annotated instead of adding a second row (time and
    team only, never event location, B-51).
    ``native=True`` (a native provider): annotate the native ``shot_penalty`` within the ``tau_event`` ceiling of
    the release; never add a row and never retype."""
    det = gate(data_profile, "penalty", "penalty", "detection", table=gate_table) if data_profile is not None else None
    attr = (
        gate(data_profile, "penalty", "penalty", "attribution", table=gate_table) if data_profile is not None else None
    )
    if det is None or attr is None:
        return out  # withheld in commit 2
    split = gate(data_profile, "penalty", "penalty", "split", table=gate_table) if data_profile is not None else None
    w = RATIFIED["penalty_W_s"]
    tau = RATIFIED["tau_event_ceiling_s"]
    eval_s = RATIFIED["penalty_position_eval_s"]
    new_rows: list[dict] = []
    for _, p in penalties.iterrows():
        period = int(p["period_id"])
        team = p.get("attacking_team_id")
        release = float(p["release_time"])
        rest_start = float(p["rest_start"]) if _known(p.get("rest_start")) else release - eval_s
        outcome = p.get("outcome")
        attribution = p.get("attribution")
        outcome = outcome if _known(outcome) else None
        attribution = attribution if _known(attribution) else None
        if native:
            hits = _rows_near(out, period, team, release - tau, release + tau, {_SHOT_PENALTY})
            if hits and (outcome is not None or attribution is not None):
                split_ok = outcome in ("saved", "off_target") and split is not None
                _annotate(out, hits[0], det, outcome if (outcome == "goal" or split_ok) else "missed", pd.NA)
            continue

        if outcome == "goal" or attribution == "goal":
            rec = _reconciliation_row(out, period, team, release)
            if rec is not None:
                _annotate(out, rec, det, "goal", "agree")
            continue
        if outcome is None and attribution is None:
            rec = _reconciliation_row(out, period, team, release)
            if rec is not None:
                _set(out, rec, "penalty_crosscheck", "disagree")
            continue

        pen_outcome = outcome if (outcome in ("saved", "off_target") and split is not None) else "missed"
        crosscheck = attribution if attribution in ("rebound", "later_goal") else pd.NA
        rec = None
        if attribution in ("rebound", "later_goal"):
            rec = _reconciliation_row(out, period, team, release)
            if rec is not None:
                _retype_to_shot(out, rec)

        # The reconciliation goal row is the episode's GOAL, never the penalty kick: exclude it from the match.
        existing = [
            i for i in _rows_near(out, period, team, rest_start - tau, release + w + tau, _SHOT_LIKE) if i != rec
        ]
        if existing:
            _annotate(out, existing[0], det, pen_outcome, crosscheck)
        else:
            new_rows.append(
                {
                    "game_id": p["game_id"],
                    "period_id": period,
                    "time_seconds": release,
                    "team_id": team,
                    "player_id": p.get("taker_player_id"),
                    "type_id": _SHOT_PENALTY,
                    "result_id": _FAIL,
                    "result_source": "inferred",
                    "type_source": "tracking",
                    "action_provenance": "derived",
                    "refine_created": True,
                    "penalty_evidence": det.tier,
                    "penalty_gate_version": GATE_TABLE_VERSION,
                    "penalty_cell_resolution": det.cell_resolution,
                    "penalty_outcome": pen_outcome,
                    "penalty_crosscheck": crosscheck,
                }
            )

        keeper = p.get("keeper_player_id")
        if pen_outcome == "saved" and split is not None and _known(keeper):
            keeper_team = _defending_team(out, period, team)
            already = [
                i
                for i in range(len(out))
                if out.iloc[i]["type_id"] == _KEEPER_SAVE
                and int(out.iloc[i]["period_id"]) == period
                and abs(float(out.iloc[i]["time_seconds"]) - release) <= w + tau
            ]
            if not already:
                new_rows.append(
                    {
                        "game_id": p["game_id"],
                        "period_id": period,
                        "time_seconds": release,
                        "team_id": keeper_team,
                        "player_id": keeper,
                        "type_id": _KEEPER_SAVE,
                        "result_id": _SUCCESS,
                        "result_source": "inferred",
                        "type_source": "tracking",
                        "action_provenance": "derived",
                        "refine_created": True,
                        "penalty_evidence": split.tier,
                        "penalty_gate_version": GATE_TABLE_VERSION,
                        "penalty_cell_resolution": split.cell_resolution,
                        "penalty_outcome": "saved",
                    }
                )
    if new_rows:
        add = pd.DataFrame(new_rows)
        out = pd.concat([out, add], ignore_index=True)
        out["refine_created"] = out["refine_created"].astype("boolean").fillna(False)
    return out


def _defending_team(out: pd.DataFrame, period: int, attacking_team):
    """The other team seen in the period (the defending team of the penalty); ``<NA>`` when unknown."""
    if "team_id" not in out:
        return pd.NA
    for t in out.loc[out["period_id"] == period, "team_id"].dropna().unique():
        if not _same_team(t, attacking_team):
            return t
    return pd.NA
