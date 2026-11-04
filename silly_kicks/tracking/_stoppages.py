"""Event-derived stoppage intervals with a per-interval provenance, and the frame stoppage state.

A provider outside ``_DEAD_BALL_OBSERVED_PROVIDERS`` has no observed dead-ball state: its ``ball_state`` is the constant
``"alive"`` (TF-46 Part 2 section 27.15, G4). The dead-ball spans then come from the events: every restart closes the
span that the previous action opened, and every goal opens a span that the next action closes. This module is the
single source of those spans for a frame mask (DL rulings G7 C and R-C1, 2026-10-26; spec section 27.19).

The start (DL ruling R-C1 item 1). The span is half-open at the previous action: that action's own instant is live
(a shot or cross that goes out of play is a live action). When frames exist, the start is the first ball frame of a run
of at least ``MIN_OUT_RUN_FRAMES`` frames with the ball centre beyond a line by at least the ball radius in real metres
(Law 9: the ball has wholly crossed the line). A goal span reads the goal lines only. Without such a run, or without
frames, the start stays at the previous action (``start_source = "after_previous_action"``).

The end (DL ruling R-C1 item 2). A native restart event carries its own time (IDSSE, Gradient Sports):
``end_source = "event"``. A SkillCorner restart is a derived row whose time is an ordering placeholder (the next
possession's time minus 0.02 s, ``spadl._skillcorner_restarts``), so its span ends at the resolved kick time when the
Part 2 resolver gives one (``t_kick``; ``"resolved_kick"``), else at the placeholder as an UNKNOWN span (the kick lies
somewhere inside it; ``"placeholder"``). A reader treats an unknown span as not live; a training preparer excludes it
and counts the labels inside it.

Rule version 2 (DL rulings 2026-10-27, R-C1 final and the Q1 ruling). P is the previous action's instant, R the
restart row's own time (the placeholder), K the resolved ``t_kick``, and B the first ball-out frame of start rule (b)
in (P, R). A kick with K <= P (``"before_previous_action"``) or K > R (``"after_restart_row"``, after the receiver's
possession began) is rejected: the span keeps its start and ends at R as an UNKNOWN span. With P < K <= R the kick is
accepted when B is absent (the span is (P, K), dead) or B < K (the span is [B, K), dead). When B >= K, the only
ball-out seen follows the kick: it is a later event, or the true exit and K is wrong, and the frames cannot tell the
two apart. The kick is rejected (``"ball_out_after_kick"``) and the span is (P, R), UNKNOWN, started after P. So the
rule never marks a dead frame live; it costs only live frames, and ``resolved_kick_reject_reason`` counts them. No
action instant is ever inside a span.

The mask never writes into ``ball_state``. :func:`derive_stoppage_state` writes it as two frame columns,
``stoppage_state`` and ``stoppage_source`` (owner decision 2026-10-26, R-C2); every ``ball_state`` reader on a provider
without an observed dead-ball state requires them (:func:`has_stoppage_state`, fail closed).
"""

from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.id_compat import canonical_id, ids_match
from silly_kicks.tracking._law_geometry import resolve_pitch_dims
from silly_kicks.tracking._provider_visibility import dead_ball_observed, validate_provider

__all__ = [
    "END_SOURCES",
    "FRAME_STOPPAGE_STATES",
    "KICK_REJECT_REASONS",
    "MIN_OUT_RUN_FRAMES",
    "PENALTY_MARKER_TAG",
    "RESTART_TYPES",
    "SHOT_TYPES",
    "START_SOURCES",
    "STOPPAGE_INTERVAL_COLUMNS",
    "STOPPAGE_PROVENANCE_KEYS",
    "STOPPAGE_RULE_VERSION",
    "STOPPAGE_SOURCES",
    "THIN_FRAME_FACTOR",
    "UNKNOWN_END_SOURCES",
    "StoppageProvenanceWarning",
    "combine_stoppage_provenance",
    "derive_stoppage_state",
    "event_stoppage_intervals",
    "frame_stoppage_state",
    "has_stoppage_state",
    "start_is_open",
    "stoppage_not_live",
    "stoppage_provenance",
    "stoppage_provenance_problem",
    "verify_stoppage_provenance",
]

#: SPADL restart types: each closes the stoppage that the previous action opened (TF-58 D20).
RESTART_TYPES = (
    "throw_in",
    "freekick_crossed",
    "freekick_short",
    "corner_crossed",
    "corner_short",
    "goalkick",
    "shot_freekick",
    "shot_penalty",
)
SHOT_TYPES = ("shot", "shot_freekick", "shot_penalty")
#: SkillCorner restart types whose NATIVE row time is measured, not a placeholder: a native ``shot_penalty`` is a
#: native shot that the goals reconciliation (``spadl._skillcorner_goals``) re-typed, at the shot's own time. Its span
#: ends there (``"event"``) and its ``t_kick`` is never read. A DERIVED ``shot_penalty`` (a scored penalty that GI
#: does not record) sits at the end of the possession that won the penalty (the foul), not at the kick: it ends no
#: restart span, and the goal rule alone covers it (rule version 3; DL ruling 2026-10-27, pen_measure item 1; owner
#: decision 2026-10-27). Every other SkillCorner restart is a derived row with a placeholder time.
_SC_OWN_TIME_RESTARTS = ("shot_penalty",)
#: The GI tag that marks the possession that won a penalty (``gi_interruption_after``; fix (A), owner decision
#: 2026-10-27).
PENALTY_MARKER_TAG = "penalty_for"

#: The fewest consecutive ball frames beyond a line that open a stoppage: one frame of tracker noise must not
#: (DL ruling R-C1 item 1, 2026-10-26).
MIN_OUT_RUN_FRAMES = 2

#: ``resolved_kick_reject_reason`` values (DL ruling 2026-10-27, Q1): ``"none"`` (no kick, or the kick is accepted),
#: else why a resolved kick is not the span end.
KICK_REJECT_REASONS = ("none", "before_previous_action", "ball_out_after_kick", "after_restart_row")

#: ``start_source`` values: ``ball_out`` (the first ball frame of a ball-out run; the span is CLOSED there), and the
#: open starts: ``after_previous_action`` (the previous action's instant, or a carrier row's own time) and
#: ``gi_possession_end`` (the end of the possession that won a penalty, mapped through the frames; rule version 3).
START_SOURCES = ("ball_out", "after_previous_action", "gi_possession_end")
#: ``end_source`` values. ``next_action``: a penalty-marker span that ends at the next action row with no measured kick
#: (rule version 3).
END_SOURCES = ("event", "resolved_kick", "placeholder", "next_action")
#: The end source of an UNKNOWN span placeholder: the SkillCorner placeholder time (the kick lies somewhere inside).
_PLACEHOLDER_END = END_SOURCES[2]
#: The end sources of an UNKNOWN span (the kick lies somewhere inside it): every reader reads THIS set
#: (:func:`frame_stoppage_state`), never one value (DL ruling 2026-10-27, marker end time, build note 2).
UNKNOWN_END_SOURCES: tuple[str, ...] = ("placeholder", "next_action")
FRAME_STOPPAGE_STATES = ("live", "dead", "unknown")

#: The version of the event-span rules above. A ``stoppage_state`` column built by another version is refused by
#: every reader (owner decision 2026-10-26, R-C2 item 4); bump it whenever a span rule changes. Version 1: DL ruling
#: R-C1, 2026-10-26. Version 2: the resolved-kick rule with its reject reasons (DL rulings 2026-10-27). Version 3:
#: a derived SkillCorner ``shot_penalty`` ends no restart span, and the penalty-marker span (owner decisions
#: 2026-10-27; DL rulings 2026-10-27, pen_measure and the marker end time).
STOPPAGE_RULE_VERSION = 3
#: ``stoppage_source`` values: the observed ``ball_state`` (IDSSE, Sportec); the event spans of this rule version; or
#: ``event_instant`` for SB360 freeze frames (owner decision 2026-10-26: each freeze frame sits at an action instant,
#: and under the span rules no action instant is inside a span, so it is live by the same rules).
STOPPAGE_SOURCES = ("ball_state", f"events_v{STOPPAGE_RULE_VERSION}", "event_instant")
_EVENTS_SOURCE = STOPPAGE_SOURCES[1]
_EVENT_INSTANT_SOURCE = STOPPAGE_SOURCES[2]
#: The ``source_provider`` of SB360 freeze frames (``snapshot_to_tracking_frames``).
_SNAPSHOT_PROVIDER = "snapshot"
#: The column is built on full-rate frames only (start rule (b) needs the ball-out frame): a game whose ball rows step
#: by more than this many frame intervals (the median step of a period) is thinner than its ``frame_rate``.
THIN_FRAME_FACTOR = 1.5
#: The manifest keys of a product a reader built (owner decision 2026-10-26, saved products, condition 6): its loader
#: refuses a product without them (:func:`verify_stoppage_provenance`).
STOPPAGE_PROVENANCE_KEYS = ("stoppage_rule_version", "stoppage_sources")


class StoppageProvenanceWarning(UserWarning):
    """A reader-built model without a current stoppage provenance was loaded under ``legacy_override=True``.

    Examples
    --------
    >>> issubclass(StoppageProvenanceWarning, UserWarning)
    True
    """


STOPPAGE_INTERVAL_COLUMNS = (
    "game_id",
    "period_id",
    "kind",
    "anchor_action_id",
    "start",
    "end",
    "start_source",
    "end_source",
    "resolved_kick_reject_reason",
    "start_ball_detected",
    "rejected_ball_out_detected",
)


def start_is_open(start_source: object) -> bool:
    """Whether a span is OPEN at its start (the start instant is live): every start source but ``"ball_out"`` (a
    ball-out run starts at its first out-of-play frame, which is dead). The ONE rule both readers use
    (:func:`frame_stoppage_state` and ``coordination``; DL ruling 2026-10-27, marker end time, build note 1).

    Examples
    --------
    >>> start_is_open("after_previous_action"), start_is_open("gi_possession_end"), start_is_open("ball_out")
    (True, True, False)
    """
    return str(start_source) != "ball_out"


def _int(v) -> int:
    """pandas groupby key -> int; the untyped argument sidesteps the Hashable -> int stub friction."""
    return int(v)


def _frame_time(frame_id, frame_lookup: dict) -> float:
    """Period-relative time for a GI frame id via the period frame-id lookup; NaN when the id is missing or
    unmapped."""
    if frame_id is None or pd.isna(frame_id):
        return float("nan")
    return frame_lookup.get(int(frame_id), float("nan"))


def _pairs(indices) -> list[tuple[tuple, np.ndarray]]:
    """``groupby(...).indices`` items with their two-part keys typed as tuples."""
    return [(tuple(k), np.asarray(v)) for k, v in indices.items()]


def _ball_tracks(frames: pd.DataFrame | None, mx: float, my: float) -> dict[tuple[object, int], tuple]:
    """(canonical game, period) -> (t, out of any line, out of a goal line, detected) for the ball rows, by time."""
    if frames is None or not len(frames):
        return {}
    ball = frames[frames["is_ball"].to_numpy(dtype=bool)]
    length, width = float(spadlconfig.field_length), float(spadlconfig.field_width)
    out: dict[tuple[object, int], tuple] = {}
    for (game, period), grp in ball.groupby(["game_id", "period_id"], sort=True, observed=True):
        grp = grp.sort_values("time_seconds", kind="stable")
        x = grp["x"].to_numpy(dtype=np.float64)
        y = grp["y"].to_numpy(dtype=np.float64)
        out_goal = (x < -mx) | (x > length + mx)
        out_any = out_goal | (y < -my) | (y > width + my)
        det = grp["visibility"].astype(object).to_numpy() if "visibility" in grp.columns else None
        t = grp["time_seconds"].to_numpy(dtype=np.float64)
        out[(canonical_id(game), _int(period))] = (t, out_any, out_goal, det)
    return out


def _first_out_run(track: tuple | None, lo: float, hi: float, *, goal_line_only: bool) -> tuple[float, object] | None:
    """The time (and the detection flag) of the first ball frame in (lo, hi) that opens a run of at least
    ``MIN_OUT_RUN_FRAMES`` frames beyond a line, all inside (lo, hi); else None."""
    if track is None:
        return None
    t, out_any, out_goal, det = track
    out = out_goal if goal_line_only else out_any
    i0, i1 = int(np.searchsorted(t, lo, side="right")), int(np.searchsorted(t, hi, side="left"))
    seg = out[i0:i1]
    n = MIN_OUT_RUN_FRAMES
    if len(seg) < n:
        return None
    run = np.ones(len(seg) - n + 1, dtype=bool)
    for k in range(n):
        run &= seg[k : len(seg) - n + 1 + k]
    hit = np.flatnonzero(run)
    if not len(hit):
        return None
    j = i0 + int(hit[0])
    flag = det[j] if det is not None else None
    return float(t[j]), (pd.NA if flag is None or pd.isna(flag) else bool(flag))


def _span_end(
    j, name, prov, t, t_kick, *, lo: float, b: float | None, is_sc: bool, game, aid
) -> tuple[float, str, str]:
    """The span end at action row ``j``, its source, and the reject reason of a resolved kick (DL rulings R-C1 item 2,
    2026-10-26, and 2026-10-27): a native event time; on SkillCorner a derived restart ends at its resolved ``t_kick``
    K when ``lo`` (P) < K <= ``t[j]`` (R) and the first ball-out ``b`` (B) is absent or before K, else at R as an
    UNKNOWN span (``KICK_REJECT_REASONS``)."""
    if not (is_sc and name[j] in RESTART_TYPES):
        return float(t[j]), "event", "none"
    if name[j] in _SC_OWN_TIME_RESTARTS and prov is not None and prov[j] == "native":
        return float(t[j]), "event", "none"  # the re-typed native shot's own time: never a kick end
    if prov is None or prov[j] != "derived":
        raise ValueError(
            "event_stoppage_intervals: a SkillCorner restart row is not action_provenance 'derived' (game "
            f"{game!r}, action {aid[j]!r}): the actions come from a converter older than the marker-keyed "
            "restarts; reconvert the events"
        )
    if t_kick is not None and np.isfinite(t_kick[j]):
        k = float(t_kick[j])
        if k <= lo:
            return float(t[j]), _PLACEHOLDER_END, "before_previous_action"
        if k > float(t[j]):
            return float(t[j]), _PLACEHOLDER_END, "after_restart_row"
        if b is not None and b >= k:
            return float(t[j]), _PLACEHOLDER_END, "ball_out_after_kick"
        return k, "resolved_kick", "none"
    return float(t[j]), _PLACEHOLDER_END, "none"


def event_stoppage_intervals(
    actions: pd.DataFrame,
    frames: pd.DataFrame | None = None,
    *,
    provider: str,
    pitch_length: float | None = None,
    pitch_width: float | None = None,
) -> pd.DataFrame:
    """The event-derived stoppage intervals with their start and end sources (``STOPPAGE_INTERVAL_COLUMNS``). Pure.

    One row per restart (``kind`` = the restart's SPADL type name) and per goal (``kind = "goal"``), per
    ``(game_id, period_id)``. ``start_source = "after_previous_action"`` means the span is OPEN at ``start`` (the
    previous action's instant is live); ``"ball_out"`` means it is closed at ``start`` (the first out-of-play ball
    frame). Every span is open at ``end``. ``start_ball_detected`` is the detection flag of the ball frame that set a
    ``ball_out`` start (``<NA>`` otherwise, or where the provider has no flag); on SkillCorner a False flag is an
    extrapolated ball.

    ``actions`` may carry the Part 2 kick resolver's ``t_kick`` (joint release); a SkillCorner restart then ends at
    its finite ``t_kick`` under rule version 2 (the module docstring). A rejected kick sets
    ``resolved_kick_reject_reason`` (``KICK_REJECT_REASONS``) and ends the span at the placeholder
    (``end_source = "placeholder"``); a ``"ball_out_after_kick"`` reject also starts the span after the previous
    action, and ``rejected_ball_out_detected`` keeps the detection flag of the ball-out frame that follows the kick.
    The start rule reads (P, R), never up to the kick. A native SkillCorner ``shot_penalty`` (a re-typed native
    shot) ends at its own time, ``"event"``, and its ``t_kick`` is never read; a DERIVED one ends no restart span (the
    goal rule covers it; rule version 3). Any other SkillCorner restart row that is not
    ``action_provenance == "derived"`` comes from a converter older than the marker-keyed restarts (it dropped the
    restarts that the defending side won) and raises ``ValueError``: reconvert the events.

    The penalty-marker span (rule version 3; fix (A), owner decision 2026-10-27): a SkillCorner row whose
    ``gi_interruption_after`` is ``"penalty_for"`` (the possession that won a penalty) opens a span (``kind =
    "penalty_marker"``) at that possession's end -- ``gi_possession_frame_end`` mapped through the frames with the
    resolver's exact frame-id lookup (``start_source = "gi_possession_end"``) -- or, when the frame id does not map
    (no frames, a dropped frame, or a time before the row's own: inconsistent), at the row's own time
    (``"after_previous_action"``); open either way. It ends at
    the next action row after that start: DEAD (``"event"``) when that row is a native shot by the same team (the
    taker's shot), else UNKNOWN (``"next_action"``). A SkillCorner action table without ``gi_interruption_after``
    comes from a converter older than the marker and raises ``ValueError``: reconvert the events.

    ``pitch_length`` / ``pitch_width`` are the provider pitch in real metres (``resolve_pitch_dims``: the declared
    keywords, else ``frames.attrs``, else 105 x 68); the ball radius margin is in real metres.

    Examples
    --------
    ::

        ivs = event_stoppage_intervals(actions, frames, provider="skillcorner")
        ivs.groupby("end_source").size()  # "placeholder" until the resolver gives a kick time
    """
    validate_provider(provider)
    dims = resolve_pitch_dims(frames, pitch_length, pitch_width)
    mx = spadlconfig.ball_radius * spadlconfig.field_length / dims.length
    my = spadlconfig.ball_radius * spadlconfig.field_width / dims.width
    tracks = _ball_tracks(frames, mx, my)
    is_sc = provider == "skillcorner"
    types, results = spadlconfig.actiontypes, spadlconfig.results
    from silly_kicks.id_compat import same_id
    from silly_kicks.spadl.base import sort_actions_chronologically

    rows: list[dict] = []
    if not len(actions):
        return pd.DataFrame(columns=list(STOPPAGE_INTERVAL_COLUMNS))
    if is_sc and "gi_interruption_after" not in actions.columns:
        raise ValueError(
            "event_stoppage_intervals: SkillCorner actions without 'gi_interruption_after' come from a converter older "
            "than the penalty marker (stoppage rule version 3); reconvert the events"
        )
    # the per-(game, period) frame-id -> time lookup (the resolver's exact match; built ONCE, ADR-068)
    frame_lut: dict[tuple[object, int], dict] = {}
    if is_sc and frames is not None and len(frames):
        for (g_, p_), fg in frames.groupby(["game_id", "period_id"], sort=False, observed=True):
            frame_lut[(canonical_id(g_), _int(p_))] = dict(
                zip(fg["frame_id"].astype(int), fg["time_seconds"].astype(float), strict=False)
            )
    for (game, period), grp in actions.groupby(["game_id", "period_id"], sort=True, observed=True):
        grp = sort_actions_chronologically(grp, tiebreak=("action_id",)).reset_index(drop=True)
        t = grp["time_seconds"].to_numpy(dtype=np.float64)
        name = [types[int(v)] for v in grp["type_id"]]
        res = [results[int(v)] for v in grp["result_id"]]
        prov = grp["action_provenance"].astype(object).to_numpy() if "action_provenance" in grp.columns else None
        t_kick = grp["t_kick"].to_numpy(dtype=np.float64) if "t_kick" in grp.columns else None
        aid = grp["action_id"].to_numpy()
        track = tracks.get((canonical_id(game), _int(period)))
        marker = grp["gi_interruption_after"].astype(object).to_numpy() if is_sc else None
        fe = (
            grp["gi_possession_frame_end"].to_numpy(dtype=object)
            if is_sc and "gi_possession_frame_end" in grp
            else None
        )
        team = grp["team_id"].to_numpy(dtype=object) if "team_id" in grp.columns else None
        lut = frame_lut.get((canonical_id(game), _int(period)), {})
        for i in range(len(grp)):
            spans: list[tuple[str, float, int, bool]] = []  # (kind, the previous / goal action's time, end row, goal)
            derived_pen = is_sc and name[i] == "shot_penalty" and prov is not None and prov[i] == "derived"
            if name[i] in RESTART_TYPES and i > 0 and not derived_pen:  # rule v3: the goal rule covers a derived one
                spans.append((name[i], float(t[i - 1]), i, False))
            is_goal = (name[i] in SHOT_TYPES and res[i] == "success") or res[i] == "owngoal"
            if is_goal and i + 1 < len(grp):
                spans.append(("goal", float(t[i]), i + 1, True))
            for kind, lo, j_end, goal_line_only in spans:
                # B: the first ball-out of start rule (b) in (P, R), read up to the restart row, never up to the kick
                hit = _first_out_run(track, lo, float(t[j_end]), goal_line_only=goal_line_only)
                hi, end_source, reason = _span_end(
                    j_end, name, prov, t, t_kick, lo=lo, b=hit[0] if hit else None, is_sc=is_sc, game=game, aid=aid
                )
                rej_det = pd.NA
                if reason == "ball_out_after_kick":  # the only ball-out follows the kick: (P, R), UNKNOWN
                    start, start_source, det = lo, "after_previous_action", pd.NA
                    rej_det = hit[1] if hit else pd.NA
                elif hit:
                    start, start_source, det = hit[0], "ball_out", hit[1]
                else:
                    start, start_source, det = lo, "after_previous_action", pd.NA
                if hi > start:
                    rows.append(
                        {
                            "game_id": game,
                            "period_id": _int(period),
                            "kind": kind,
                            "anchor_action_id": aid[i],
                            "start": start,
                            "end": hi,
                            "start_source": start_source,
                            "end_source": end_source,
                            "resolved_kick_reject_reason": reason,
                            "start_ball_detected": det,
                            "rejected_ball_out_detected": rej_det,
                        }
                    )
            if marker is not None and isinstance(marker[i], str) and marker[i] == PENALTY_MARKER_TAG:
                t_me = _frame_time(fe[i] if fe is not None else None, lut)
                # a mapped end before the row's own time (the possession start) is inconsistent: it does not map
                mapped = bool(np.isfinite(t_me)) and float(t_me) >= float(t[i])
                start, start_source = (
                    (float(t_me), "gi_possession_end") if mapped else (float(t[i]), "after_previous_action")
                )
                nxt = next((j for j in range(i + 1, len(grp)) if float(t[j]) > start), None)
                if nxt is not None:
                    taker_shot = (
                        name[nxt] in SHOT_TYPES
                        and prov is not None
                        and prov[nxt] == "native"
                        and team is not None
                        and same_id(team[nxt], team[i])
                    )
                    rows.append(
                        {
                            "game_id": game,
                            "period_id": _int(period),
                            "kind": "penalty_marker",
                            "anchor_action_id": aid[i],
                            "start": start,
                            "end": float(t[nxt]),
                            "start_source": start_source,
                            "end_source": "event" if taker_shot else "next_action",
                            "resolved_kick_reject_reason": "none",
                            "start_ball_detected": pd.NA,
                            "rejected_ball_out_detected": pd.NA,
                        }
                    )
    out = pd.DataFrame(rows, columns=list(STOPPAGE_INTERVAL_COLUMNS))
    out["resolved_kick_reject_reason"] = out["resolved_kick_reject_reason"].astype(object)
    out["start_ball_detected"] = out["start_ball_detected"].astype("boolean")
    out["rejected_ball_out_detected"] = out["rejected_ball_out_detected"].astype("boolean")
    return out


def frame_stoppage_state(frames: pd.DataFrame, intervals: pd.DataFrame) -> np.ndarray:
    """The stoppage state of every frame row (``FRAME_STOPPAGE_STATES``), aligned to ``frames``. Pure.

    ``"dead"`` inside a span with a known end, ``"unknown"`` inside an unknown span only (``end_source`` in
    :data:`UNKNOWN_END_SOURCES`; a known dead span wins an overlap), ``"live"`` elsewhere. The span bounds follow
    :func:`event_stoppage_intervals`: open at every start but a ``ball_out`` one (:func:`start_is_open`), closed at a
    ``ball_out`` start, open at every end.

    Examples
    --------
    ::

        state = frame_stoppage_state(frames, event_stoppage_intervals(actions, frames, provider="skillcorner"))
        live = frames[state == "live"]
    """
    state = np.full(len(frames), "live", dtype=object)
    if not len(frames) or not len(intervals):
        return state
    t = frames["time_seconds"].to_numpy(dtype=np.float64)
    # the frame rows of each (game, period), grouped ONCE (ADR-068: no rescan per interval group)
    rows_by_key = {
        (canonical_id(key[0]), _int(key[1])): np.asarray(ix)
        for key, ix in _pairs(frames.groupby(["game_id", "period_id"], sort=False, observed=True).indices)
    }
    for (game, period), grp in intervals.groupby(["game_id", "period_id"], sort=True, observed=True):
        idx = rows_by_key.get((canonical_id(game), _int(period)))
        if idx is None or not len(idx):
            continue
        tt = t[idx]
        dead = np.zeros(len(idx), dtype=bool)
        unknown = np.zeros(len(idx), dtype=bool)
        for start, end, s_src, e_src in zip(
            grp["start"].to_numpy(dtype=np.float64),
            grp["end"].to_numpy(dtype=np.float64),
            grp["start_source"].to_numpy(),
            grp["end_source"].to_numpy(),
            strict=True,
        ):
            inside = ((tt > start) if start_is_open(s_src) else (tt >= start)) & (tt < end)
            if e_src in UNKNOWN_END_SOURCES:
                unknown |= inside
            else:
                dead |= inside
        sub = state[idx]
        sub[unknown] = "unknown"
        sub[dead] = "dead"
        state[idx] = sub
    return state


def _assert_full_rate(frames: pd.DataFrame, game: object) -> None:
    """Refuse a game whose ball rows are thinner than their ``frame_rate`` (owner decision R-C2 item 6)."""
    if "frame_rate" not in frames.columns:
        raise ValueError(f"derive_stoppage_state: game {game!r} frames carry no frame_rate (the full-rate check)")
    rate = pd.to_numeric(frames["frame_rate"], errors="coerce").to_numpy(dtype=np.float64)
    rate = rate[np.isfinite(rate) & (rate > 0)]
    if not len(rate):
        raise ValueError(f"derive_stoppage_state: game {game!r} has no positive frame_rate")
    step = 1.0 / float(np.median(rate))
    ball = frames[frames["is_ball"].to_numpy(dtype=bool)]
    for period, grp in ball.groupby("period_id", sort=True, observed=True):
        t = np.unique(grp["time_seconds"].to_numpy(dtype=np.float64))
        if len(t) >= 2 and float(np.median(np.diff(t))) > THIN_FRAME_FACTOR * step:
            raise ValueError(
                f"derive_stoppage_state: game {game!r} period {period!r} ball rows step by "
                f"{float(np.median(np.diff(t))):.3f} s against a frame interval of {step:.3f} s: build the column on "
                "full-rate frames, before any subsampling or filtering"
            )


def derive_stoppage_state(
    frames: pd.DataFrame,
    *,
    actions: pd.DataFrame | None = None,
    pitch_length: float | None = None,
    pitch_width: float | None = None,
) -> pd.DataFrame:
    """Return a copy of ``frames`` with the stoppage columns ``stoppage_state`` and ``stoppage_source``. Pure.

    One value per frame row, the same rows as ``ball_state`` (owner decision 2026-10-26, R-C2):

    - ``stoppage_state`` in ``FRAME_STOPPAGE_STATES``: ``"live"``, ``"dead"`` or ``"unknown"`` (an UNKNOWN span: the
      SkillCorner kick lies somewhere inside it). Every reader treats ``"dead"`` and ``"unknown"`` as not live.
    - ``stoppage_source`` in ``STOPPAGE_SOURCES``: ``"ball_state"`` for a provider in
      ``_DEAD_BALL_OBSERVED_PROVIDERS`` (IDSSE, Sportec: alive -> live, dead -> dead, any other value -> unknown), else
      ``"events_v<STOPPAGE_RULE_VERSION>"``: the spans of :func:`event_stoppage_intervals` over ``actions``; or
      ``"event_instant"`` for SB360 freeze frames (``source_provider = "snapshot"``): every row is ``"live"``. Each
      freeze frame sits at an action instant, and under the span rules no action instant lies inside a span (a span
      is open at the previous action, open at the restart, half-open after a goal), so the same rules give live; no
      ``actions`` are needed.

    The provider is the game's ``source_provider``; a game with more than one provider raises ``ValueError``, and an
    unclassified provider raises (fail closed). A provider without an observed dead-ball state needs ``actions``
    (else ``ValueError``) and full-rate frames: start rule (b) reads the first ball-out frame, so a game whose ball
    rows step by more than ``THIN_FRAME_FACTOR`` frame intervals raises. Build the column BEFORE any subsampling or
    filtering. ``actions`` that carry a ``t_kick`` column (the Part 2 resolver's kick times) raise ``ValueError``:
    ``events_v<N>`` does not record the kicks, so a build with resolved kicks would carry the same label as a build
    without them (owner decision 2026-10-27; a labelled source for such builds comes with R.2). The core
    (:func:`event_stoppage_intervals`) keeps ``t_kick`` for the R.2 probes. ``pitch_length`` / ``pitch_width``: as
    :func:`event_stoppage_intervals` (one pitch per call). ``ball_state`` is never written.

    ``out.attrs["stoppage_report"]`` holds the rule version, per game the provider, the source and the span counts by
    ``start_source`` / ``end_source``, and the spans themselves (``span_columns`` + ``spans``, plain tuples).

    Examples
    --------
    ::

        frames = derive_stoppage_state(frames, actions=actions)  # full-rate frames, once per match
        live = frames[frames["stoppage_state"] == "live"]
    """
    required = ("game_id", "period_id", "time_seconds", "is_ball", "source_provider")
    need = [c for c in required if c not in frames.columns]
    if need:
        raise ValueError(f"derive_stoppage_state: frames lack the columns {need}")
    if actions is not None and "t_kick" in actions.columns:
        raise ValueError(
            "derive_stoppage_state: the actions carry t_kick (the Part 2 resolver's kick times). The column labels its "
            f"build {_EVENTS_SOURCE!r}, which does not record the kicks, so a build with resolved kicks is refused "
            "(owner decision 2026-10-27). Drop t_kick (the load seam does), or read the spans with "
            "event_stoppage_intervals."
        )
    dims = resolve_pitch_dims(frames, pitch_length, pitch_width)
    out = frames.copy()
    state = np.full(len(out), "live", dtype=object)
    source = np.full(len(out), None, dtype=object)
    games: dict[object, dict] = {}
    spans: list[tuple] = []
    for game, ix in out.groupby("game_id", sort=True, observed=True).indices.items():
        ix = np.asarray(ix)
        g = out.iloc[ix]
        provs = sorted({str(p) for p in g["source_provider"].dropna().unique()})
        if len(provs) != 1:
            raise ValueError(f"derive_stoppage_state: game {game!r} has source_provider {provs}; one provider per game")
        prov = provs[0]
        if prov == _SNAPSHOT_PROVIDER:  # a freeze frame is an action instant: live by the span rules (see above)
            state[ix] = "live"
            source[ix] = _EVENT_INSTANT_SOURCE
            games[canonical_id(game)] = {"provider": prov, "source": _EVENT_INSTANT_SOURCE}
            continue
        if dead_ball_observed(prov):  # raises for an unclassified provider (fail closed)
            bs = g["ball_state"].astype(object).to_numpy() if "ball_state" in g.columns else np.full(len(g), None)
            state[ix] = np.where(bs == "dead", "dead", np.where(bs == "alive", "live", "unknown"))
            source[ix] = "ball_state"
            games[canonical_id(game)] = {"provider": prov, "source": "ball_state"}
            continue
        if actions is None:
            raise ValueError(
                f"derive_stoppage_state: provider {prov!r} (game {game!r}) has no observed dead-ball state; pass the "
                "actions"
            )
        _assert_full_rate(g, game)
        acts = actions[ids_match(actions["game_id"], game).to_numpy(dtype=bool)]
        ivs = event_stoppage_intervals(acts, g, provider=prov, pitch_length=dims.length, pitch_width=dims.width)
        state[ix] = frame_stoppage_state(g, ivs)
        source[ix] = _EVENTS_SOURCE
        games[canonical_id(game)] = {
            "provider": prov,
            "source": _EVENTS_SOURCE,
            "n_spans": len(ivs),
            "start_source": dict(Counter(ivs["start_source"])),
            "end_source": dict(Counter(ivs["end_source"])),
        }
        for rec in ivs.itertuples(index=False):
            # `stoppage_report` is a JSON-serializable structure (a provenance mirror; the parquet writer
            # also json-dumps `df.attrs`). A pandas nullable `boolean` column -- `start_ball_detected` /
            # `rejected_ball_out_detected`, populated only for a detection-aware provider such as Gradient
            # Sports -- yields `numpy.bool_` on `itertuples`, which `json.dumps` rejects under numpy 2.x
            # (its type name is `bool`). Coerce every numpy scalar to its Python native so the report
            # round-trips for ALL providers, not only the ones that leave the flag null.
            spans.append(tuple(None if pd.isna(v) else (v.item() if isinstance(v, np.generic) else v) for v in rec))
    out["stoppage_state"] = pd.Categorical(state, categories=list(FRAME_STOPPAGE_STATES))
    out["stoppage_source"] = pd.Categorical(source, categories=list(STOPPAGE_SOURCES))
    out.attrs["stoppage_report"] = {
        "rule_version": STOPPAGE_RULE_VERSION,
        "games": games,
        "span_columns": STOPPAGE_INTERVAL_COLUMNS,
        "spans": tuple(spans),
    }
    return out


def has_stoppage_state(frames: pd.DataFrame, *, where: str) -> bool:
    """The reader gate (owner decision 2026-10-26, R-C2 items 3-5): True when ``frames`` carry a current
    ``stoppage_state`` (read it with :func:`stoppage_not_live`); False when the column is absent and every
    ``source_provider`` observes ``ball_state`` (the reader keeps its ``ball_state`` logic, unchanged).

    Raises ``ValueError`` (fail closed, like the G detection gate) when the column is absent on a provider without an
    observed dead-ball state (SkillCorner, Gradient Sports, Metrica: their ``ball_state`` is the constant
    ``"alive"``), when ``stoppage_source`` names another rule version, or when ``frames`` carry no
    ``source_provider``. An unclassified provider raises (``dead_ball_observed``). ``where`` names the reader.

    Examples
    --------
    ::

        if has_stoppage_state(frames, where="my_reader"):
            live_rows = ~stoppage_not_live(frames)
    """
    if "stoppage_state" in frames.columns:
        if "stoppage_source" not in frames.columns:
            raise ValueError(f"{where}: stoppage_state without stoppage_source; build both with derive_stoppage_state")
        bad = sorted({str(v) for v in frames["stoppage_source"].dropna().unique()} - set(STOPPAGE_SOURCES))
        if bad:
            raise ValueError(
                f"{where}: stoppage_source {bad} is not a current rule ({list(STOPPAGE_SOURCES)}); rebuild the column "
                "with derive_stoppage_state"
            )
        return True
    if not len(frames):
        return False
    if "source_provider" not in frames.columns:
        raise ValueError(
            f"{where}: frames carry no source_provider, so the dead-ball source is unknown; call derive_stoppage_state"
        )
    provs = sorted({str(p) for p in frames["source_provider"].dropna().unique()})
    if _SNAPSHOT_PROVIDER in provs:
        raise ValueError(
            f"{where}: SB360 freeze frames (source_provider 'snapshot') carry no stoppage column; call "
            "derive_stoppage_state(frames) first (each freeze frame is an action instant: live)"
        )
    unobserved = [p for p in provs if not dead_ball_observed(p)]  # an unclassified provider raises (fail closed)
    if unobserved:
        raise ValueError(
            f"{where}: provider(s) {unobserved} have no observed dead-ball state (ball_state is the constant 'alive'); "
            "call derive_stoppage_state(frames, actions=actions) on the full-rate frames first"
        )
    return False


def stoppage_not_live(frames: pd.DataFrame) -> np.ndarray:
    """True where a frame row is not live (``stoppage_state`` dead, unknown or missing); aligned to ``frames``.

    Examples
    --------
    ::

        rows = frames[~stoppage_not_live(frames)]
    """
    return frames["stoppage_state"].astype(object).to_numpy() != "live"


def stoppage_provenance(frames: pd.DataFrame, *, where: str) -> dict:
    """The stoppage provenance that a product built from ``frames`` records in its manifest (owner decision
    2026-10-26, saved products, condition 6): ``{"stoppage_rule_version", "stoppage_sources"}``.

    ``stoppage_sources`` are the sorted ``stoppage_source`` values of the frames, or ``["ball_state"]`` when the
    column is absent and every provider observes ``ball_state`` (the reader then keeps its ``ball_state`` logic, the
    same mapping). It runs the reader gate first (:func:`has_stoppage_state`), so it raises where a reader would.
    Empty frames give no sources.

    Examples
    --------
    >>> import pandas as pd
    >>> f = pd.DataFrame(
    ...     {"source_provider": ["sportec"], "stoppage_state": ["live"], "stoppage_source": ["ball_state"]}
    ... )
    >>> stoppage_provenance(f, where="doc")["stoppage_sources"]
    ['ball_state']
    """
    if not len(frames):
        sources: list[str] = []
    elif has_stoppage_state(frames, where=where):
        sources = sorted({str(v) for v in frames["stoppage_source"].dropna().unique()})
    else:
        sources = [STOPPAGE_SOURCES[0]]
    return {"stoppage_rule_version": STOPPAGE_RULE_VERSION, "stoppage_sources": sources}


def combine_stoppage_provenance(provenances: Iterable[Mapping]) -> dict:
    """One provenance for a product built from several parts (for example a model trained on many matches): the
    union of the sources. Raises ``ValueError`` when a part fails :func:`stoppage_provenance_problem` (another rule
    version, an unknown source, a missing key) or when there are no parts.

    Examples
    --------
    >>> a = {"stoppage_rule_version": STOPPAGE_RULE_VERSION, "stoppage_sources": ["ball_state"]}
    >>> b = {"stoppage_rule_version": STOPPAGE_RULE_VERSION, "stoppage_sources": ["event_instant"]}
    >>> combine_stoppage_provenance([a, b])["stoppage_sources"]
    ['ball_state', 'event_instant']
    """
    parts = list(provenances)
    if not parts:
        raise ValueError("combine_stoppage_provenance: no parts")
    for p in parts:
        problem = stoppage_provenance_problem(p)
        if problem is not None:
            raise ValueError(f"combine_stoppage_provenance: a part {problem}")
    sources = sorted({str(s) for p in parts for s in p["stoppage_sources"]})
    return {"stoppage_rule_version": STOPPAGE_RULE_VERSION, "stoppage_sources": sources}


def stoppage_provenance_problem(recorded: Mapping | None) -> str | None:
    """Why ``recorded`` (a manifest, or the dict of :func:`stoppage_provenance`) is not a current stoppage
    provenance, or ``None`` when it is: both keys present, the current rule version, and a non-empty list of current
    ``stoppage_source`` values.

    Examples
    --------
    >>> current = {"stoppage_rule_version": STOPPAGE_RULE_VERSION, "stoppage_sources": ["ball_state"]}
    >>> stoppage_provenance_problem(current)
    >>> stoppage_provenance_problem({})
    'records no stoppage provenance'
    """
    if recorded is None or any(k not in recorded for k in STOPPAGE_PROVENANCE_KEYS):
        return "records no stoppage provenance"
    version, sources = recorded["stoppage_rule_version"], recorded["stoppage_sources"]
    if version != STOPPAGE_RULE_VERSION:
        return f"records stoppage rule version {version!r}; the library is version {STOPPAGE_RULE_VERSION}"
    if isinstance(sources, str) or not isinstance(sources, (list, tuple)) or not sources:
        return f"records stoppage_sources {sources!r}; expected a non-empty list of {list(STOPPAGE_SOURCES)}"
    unknown = sorted({str(s) for s in sources} - set(STOPPAGE_SOURCES))
    if unknown:
        return f"records stoppage_sources {unknown} that are not current ({list(STOPPAGE_SOURCES)})"
    return None


def verify_stoppage_provenance(
    recorded: Mapping | None,
    *,
    model_name: str,
    legacy_override: bool,
    error_cls: type[Exception],
) -> None:
    """The load-time refusal of a reader-built trained model (owner decision 2026-10-26: refuse, retrain first).

    Raises ``error_cls`` when :func:`stoppage_provenance_problem` finds a problem: the model may have trained on
    dead-ball or unknown frames. ``legacy_override=True`` loads it with a :class:`StoppageProvenanceWarning`
    instead, only to audit the old model (for example a head-to-head against its retrain), as for the chirality and
    feature-contract guards.

    Examples
    --------
    ::

        verify_stoppage_provenance(meta, model_name="xShotOccurrence", legacy_override=False, error_cls=IntegrityError)
    """
    problem = stoppage_provenance_problem(recorded)
    if problem is None:
        return
    msg = (
        f"{model_name}: the weights artifact {problem} (a model trained before the stoppage column, or by a stale "
        "rule), so its training rows may include dead-ball or unknown frames (TF-46 Part 2 spec section 27.19)."
    )
    if legacy_override:
        warnings.warn(
            f"{msg} Loading it under legacy_override=True: audit use only.", StoppageProvenanceWarning, stacklevel=3
        )
        return
    raise error_cls(f"{msg} Refusing to load; retrain it, or pass legacy_override=True only to audit the old model.")
