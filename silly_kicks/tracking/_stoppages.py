"""Dead-ball state of tracking frames, for providers whose frames do not record it (ADR-130).

SkillCorner and Gradient Sports frames carry ``ball_state = "alive"`` on every frame. A frame where the ball is out of
play must not become a training row of the xShot/xCross models. This module marks each frame ``live``, ``dead`` or
``unknown`` (``unknown`` = maybe dead; a training preparer drops it like ``dead``). It never marks a frame with a
tracked ball out of play ``live``; it pays for that with live frames marked ``dead`` or ``unknown``.

The rules (spec ``docs/superpowers/specs/2026-11-11-stoppage-gate-refit-design.md``, section 2):

* **Events.** SkillCorner: the ``game_interruption_after`` / ``game_interruption_before`` markers of consecutive
  possessions (:func:`skillcorner_stoppage_spans`). Gradient Sports: its native restart and goal actions
  (:func:`event_stoppage_spans`).
* **Ball out (Law 9).** A frame whose ball is beyond a line by more than :data:`BALL_OUT_MARGIN_M` is ``dead``, and so
  is every frame up to the next on-ball event; that event's own span is ``unknown`` (the restart lies inside it).
* **Period edges.** Frames of a period before its first and after its last on-ball event are ``dead``; a period with
  no on-ball event is ``unknown``. The period is the frames' own ``period_id``, never an event label.
* **Observed providers** (``spans=None``): the state comes from ``ball_state``.

A frame takes the most restrictive state (``unknown`` over ``dead`` over ``live``). The result goes into two new
columns, ``stoppage_state`` and ``stoppage_source``; ``ball_state`` is never written.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import pandas as pd

import silly_kicks.spadl.config as spadlconfig

#: The rule set of this module; recorded in a trained model's metadata and ``metrics.json``.
STOPPAGE_RULE_VERSION = 1
#: Metres beyond a line (the ball radius): the whole ball has crossed it (Law 9). Smaller is the safe side.
BALL_OUT_MARGIN_M = 0.11
STOPPAGE_STATES = ("live", "dead", "unknown")
STOPPAGE_SOURCES = ("events", "ball_out", "period_edge", "no_coverage", "ball_state")

_LIVE, _DEAD, _UNKNOWN = 0, 1, 2
_RESTART_TYPES = (
    "throw_in",
    "freekick_crossed",
    "freekick_short",
    "corner_crossed",
    "corner_short",
    "goalkick",
    "shot_penalty",
    "shot_freekick",
)
_SHOT_TYPES = ("shot", "shot_penalty", "shot_freekick")


@dataclass(frozen=True)
class StoppageSpans:
    """Event-derived spans of one match, and its on-ball events (the anchors of the ball-out and edge rules).

    ``key`` is the frame column that ``start``/``end`` refer to: ``"frame_id"`` (SkillCorner; frame ids run on across
    the periods) or ``"time_seconds"`` (Gradient Sports; period-relative, so both tables carry ``period_id``).
    ``spans``: ``start``, ``end`` (half-open ``[start, end)``), ``state`` (``"dead"``/``"unknown"``).
    ``anchors``: ``start``, ``end`` of each on-ball event in time order (``end`` = the end of its own span).
    """

    key: Literal["frame_id", "time_seconds"]
    spans: pd.DataFrame
    anchors: pd.DataFrame


def skillcorner_stoppage_spans(events: pd.DataFrame) -> StoppageSpans:
    """Spans from the game-interruption markers of a SkillCorner ``dynamic_events`` table.

    For each possession ``a`` and the next possession ``b`` (frame order): when ``a`` has
    ``game_interruption_after`` and ``b`` has ``game_interruption_before``, the frames ``[a.frame_end, b.frame_start)``
    are ``dead`` and the taker's possession ``[b.frame_start, b.frame_end]`` is ``unknown``; when only one of the two
    markers is set, ``[a.frame_end, b.frame_end]`` is ``unknown``.

    Examples
    --------
    >>> ev = pd.DataFrame({
    ...     "event_type": ["player_possession"] * 2, "frame_start": [10, 40], "frame_end": [20, 50],
    ...     "game_interruption_after": ["throw_in_against", None],
    ...     "game_interruption_before": [None, "throw_in_for"]})
    >>> skillcorner_stoppage_spans(ev).spans.to_dict("records")
    [{'start': 20, 'end': 40, 'state': 'dead'}, {'start': 40, 'end': 51, 'state': 'unknown'}]
    """
    if "event_type" not in events.columns:  # an empty table: no possession, no span
        empty = pd.DataFrame({"start": pd.Series(dtype="int64"), "end": pd.Series(dtype="int64")})
        return StoppageSpans("frame_id", empty.assign(state=pd.Series(dtype=object)), empty)
    all_pp = events[events["event_type"] == "player_possession"]
    located = (all_pp["frame_start"].notna() & all_pp["frame_end"].notna()).to_numpy()
    lost = _lost_marker_spans(all_pp, located)  # rule 3: a marked possession without frame ids
    pp = all_pp[located].sort_values("frame_start", kind="mergesort")
    start = pp["frame_start"].to_numpy(dtype="int64")
    end = pp["frame_end"].to_numpy(dtype="int64") + 1  # half-open
    after = _marked(pp, "game_interruption_after")
    before = _marked(pp, "game_interruption_before")
    a_end, b_start, b_end = end[:-1] - 1, start[1:], end[1:]
    both = after[:-1] & before[1:]
    one = after[:-1] ^ before[1:]
    spans = pd.concat(
        [
            pd.DataFrame({"start": a_end[both], "end": b_start[both], "state": "dead"}),
            pd.DataFrame({"start": b_start[both], "end": b_end[both], "state": "unknown"}),
            pd.DataFrame({"start": a_end[one], "end": b_end[one], "state": "unknown"}),
            lost,
        ],
        ignore_index=True,
    )
    spans = spans.sort_values(["start", "end"], kind="mergesort").reset_index(drop=True)
    return StoppageSpans("frame_id", spans, pd.DataFrame({"start": start, "end": end}))


def event_stoppage_spans(actions: pd.DataFrame) -> StoppageSpans:
    """Spans from a SPADL actions table whose restart actions carry their own time (Gradient Sports).

    A restart action ``r`` makes ``[previous action time, r.time)`` ``dead``; a goal (a shot type with result
    ``success``, or result ``owngoal``) makes ``[goal time, next action time)`` ``dead``. Times are period-relative.

    Examples
    --------
    >>> import silly_kicks.spadl.config as c
    >>> acts = pd.DataFrame({"period_id": [1, 1, 1], "time_seconds": [10.0, 20.0, 30.0],
    ...     "type_id": [c.actiontype_id["pass"], c.actiontype_id["throw_in"], c.actiontype_id["pass"]],
    ...     "result_id": [1, 1, 1]})
    >>> event_stoppage_spans(acts).spans.to_dict("records")
    [{'period_id': 1, 'start': 10.0, 'end': 20.0, 'state': 'dead'}]
    """
    a = actions.sort_values(["period_id", "time_seconds"], kind="mergesort")
    per = a["period_id"].to_numpy()
    t = a["time_seconds"].to_numpy(dtype="float64")
    type_id, result_id = a["type_id"].to_numpy(), a["result_id"].to_numpy()
    same_prev = np.r_[False, per[1:] == per[:-1]]
    same_next = np.r_[per[:-1] == per[1:], False]
    t_prev, t_next = np.r_[np.nan, t[:-1]], np.r_[t[1:], np.nan]
    restart = np.isin(type_id, [spadlconfig.actiontype_id[n] for n in _RESTART_TYPES]) & same_prev
    goal = (
        (
            np.isin(type_id, [spadlconfig.actiontype_id[n] for n in _SHOT_TYPES])
            & (result_id == spadlconfig.result_id["success"])
        )
        | (result_id == spadlconfig.result_id["owngoal"])
    ) & same_next
    spans = (
        pd.concat(
            [
                pd.DataFrame({"period_id": per[restart], "start": t_prev[restart], "end": t[restart], "state": "dead"}),
                pd.DataFrame({"period_id": per[goal], "start": t[goal], "end": t_next[goal], "state": "dead"}),
            ],
            ignore_index=True,
        )
        .sort_values(["period_id", "start"], kind="mergesort")
        .reset_index(drop=True)
    )
    anchors = pd.DataFrame({"period_id": per, "start": t, "end": np.where(same_next, t_next, t)})
    return StoppageSpans("time_seconds", spans, anchors)


def with_stoppage_state(
    frames: pd.DataFrame, *, spans: StoppageSpans | None, margin_m: float = BALL_OUT_MARGIN_M
) -> pd.DataFrame:
    """Return a copy of ``frames`` with ``stoppage_state`` and ``stoppage_source`` (pure: the input is not changed).

    ``spans=None`` is an observed provider: ``dead`` where ``ball_state == "dead"``, else ``live``. Otherwise the
    event spans, the ball-out rule and the period-edge rule of the module docstring apply. The frames are canonical
    tracking frames in SPADL coordinates (105 x 68 m) with ``game_id``, ``period_id``, ``frame_id``,
    ``time_seconds``, ``is_ball``, ``x``, ``y``.

    Examples
    --------
    >>> fr = pd.DataFrame({"game_id": 1, "period_id": 1, "frame_id": [1, 2], "time_seconds": [0.0, 0.1],
    ...     "is_ball": True, "x": [50.0, 50.0], "y": [30.0, 30.0], "ball_state": ["alive", "dead"]})
    >>> with_stoppage_state(fr, spans=None)["stoppage_state"].tolist()
    ['live', 'dead']
    """
    out = frames.copy()
    if spans is None:
        dead = out["ball_state"].astype(str).to_numpy() == "dead"
        out["stoppage_state"] = np.where(dead, "dead", "live")
        out["stoppage_source"] = "ball_state"
        return out
    if out["game_id"].nunique(dropna=True) > 1:
        raise ValueError("with_stoppage_state takes the frames of ONE match with that match's spans")
    keys = ["game_id", "period_id", "frame_id"]
    uniq = out[[*keys, "time_seconds"]].drop_duplicates(keys).sort_values(keys, kind="mergesort")
    ball = out[out["is_ball"].astype(bool)].drop_duplicates(keys)[[*keys, "x", "y"]]
    uniq = uniq.merge(ball, on=keys, how="left")
    code = np.full(len(uniq), _LIVE, dtype=np.int8)
    src = np.zeros(len(uniq), dtype=np.int8)  # index into STOPPAGE_SOURCES
    for key, idx in uniq.groupby(["game_id", "period_id"], sort=True).indices.items():
        per = cast("tuple", key)[1]
        idx = np.asarray(idx)
        pos = uniq[spans.key].to_numpy(dtype="float64")[idx]
        order = np.argsort(pos, kind="mergesort")
        idx, pos = idx[order], pos[order]
        sp = spans.spans if spans.key == "frame_id" else spans.spans[spans.spans["period_id"] == per]
        an = spans.anchors if spans.key == "frame_id" else spans.anchors[spans.anchors["period_id"] == per]
        if spans.key == "frame_id":  # anchors that fall inside this period's frames
            an = an[(an["start"] >= pos[0]) & (an["start"] <= pos[-1])]
        if an.empty:
            _mark(code, src, idx, _UNKNOWN, 3)
            continue
        for s, e, st in zip(sp["start"], sp["end"], sp["state"], strict=True):
            lo, hi = np.searchsorted(pos, s, "left"), np.searchsorted(pos, e, "left")
            _mark(code, src, idx[lo:hi], _DEAD if st == "dead" else _UNKNOWN, 0)
        a_start = an["start"].to_numpy(dtype="float64")
        a_end = an["end"].to_numpy(dtype="float64")
        _mark(code, src, idx[pos < a_start.min()], _DEAD, 2)
        # After the last on-ball event: SkillCorner ends are exclusive frame ids; a Gradient Sports action is an
        # instant, so the frames after the last action's own time are dead.
        after_last = pos >= a_end.max() if spans.key == "frame_id" else pos > a_start.max()
        _mark(code, src, idx[after_last], _DEAD, 2)
        bx = uniq["x"].to_numpy(dtype="float64")[idx]
        by = uniq["y"].to_numpy(dtype="float64")[idx]
        out_ = (
            (bx < -margin_m)
            | (bx > spadlconfig.field_length + margin_m)
            | (by < -margin_m)
            | (by > spadlconfig.field_width + margin_m)
        )  # NaN compares False: a frame without a ball takes no state from this rule
        if out_.any():
            # Every ball-out frame is dead up to the next on-ball event, whose own span is unknown. The frames are
            # sorted, so the first ball-out frame before each next event binds; a ball-out run that crosses an event
            # (a player holding the ball outside the line) binds again for the event after it.
            k_out = np.flatnonzero(out_)
            nxt_out = np.searchsorted(a_start, pos[k_out], "right")
            _events, first = np.unique(nxt_out, return_index=True)
            for k, nxt in zip(k_out[first], nxt_out[first], strict=True):
                if nxt >= len(a_start):
                    _mark(code, src, idx[k:], _DEAD, 1)
                    continue
                hi = np.searchsorted(pos, a_start[nxt], "left")
                _mark(code, src, idx[k:hi], _DEAD, 1)
                _mark(code, src, idx[hi : np.searchsorted(pos, a_end[nxt], "left")], _UNKNOWN, 1)
            _mark(code, src, idx[out_], _DEAD, 1)
    uniq["stoppage_state"] = np.asarray(STOPPAGE_STATES)[code]
    uniq["stoppage_source"] = np.asarray(STOPPAGE_SOURCES)[src]
    merged = out.merge(uniq[[*keys, "stoppage_state", "stoppage_source"]], on=keys, how="left", validate="many_to_one")
    merged.index = out.index  # a left merge keeps the row order; keep the caller's index too
    return merged


def training_frame_mask(frames: pd.DataFrame, *, horizon_seconds: float) -> np.ndarray:
    """Per row: True for a ``live`` frame with no ``unknown`` frame of the same game and period in
    ``[t, t + horizon_seconds]`` (its label is not known there). A ``dead`` frame inside the horizon keeps the row:
    no shot or cross starts in dead time.

    Examples
    --------
    >>> fr = pd.DataFrame({"game_id": 1, "period_id": 1, "time_seconds": [0.0, 0.5, 2.0],
    ...     "stoppage_state": ["live", "unknown", "live"]})
    >>> training_frame_mask(fr, horizon_seconds=1.0).tolist()
    [False, False, True]
    """
    state = frames["stoppage_state"].to_numpy()
    t = frames["time_seconds"].to_numpy(dtype="float64")
    ok = state == "live"
    for _key, idx in frames.groupby(["game_id", "period_id"], sort=False).indices.items():
        idx = np.asarray(idx)
        unk = np.sort(t[idx][state[idx] == "unknown"])
        if not len(unk):
            continue
        j = np.searchsorted(unk, t[idx], "left")
        hit = j < len(unk)
        hit[hit] = unk[j[hit]] <= t[idx][hit] + horizon_seconds
        ok[idx[hit]] = False
    return ok


def _lost_marker_spans(pp: pd.DataFrame, located: np.ndarray) -> pd.DataFrame:
    """``unknown`` spans for marked possessions without frame ids (rule 3): from the frame end of the located
    possession before it (event order) to the frame end of the located possession after it. Without a located
    neighbour on one side, the span runs from the other neighbour to the edge rule of that period."""
    marked = _marked(pp, "game_interruption_after") | _marked(pp, "game_interruption_before")
    rows = np.flatnonzero(marked & ~located)
    if not len(rows):
        return pd.DataFrame({"start": pd.Series(dtype="int64"), "end": pd.Series(dtype="int64"), "state": []})
    fs = pp["frame_start"].to_numpy(dtype="float64")
    fe = pp["frame_end"].to_numpy(dtype="float64")
    prev_end = pd.Series(np.where(located, fe, np.nan)).ffill().to_numpy()
    next_end = pd.Series(np.where(located, fe, np.nan)).bfill().to_numpy()
    next_start = pd.Series(np.where(located, fs, np.nan)).bfill().to_numpy()
    last = np.nanmax(fe) if located.any() else 0.0
    start = np.where(np.isnan(prev_end[rows]), next_start[rows], prev_end[rows])
    end = np.where(np.isnan(next_end[rows]), last, next_end[rows]) + 1
    ok = ~np.isnan(start)
    return pd.DataFrame({"start": start[ok].astype("int64"), "end": end[ok].astype("int64"), "state": "unknown"})


def _marked(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.zeros(len(df), dtype=bool)
    v = df[col]
    return (v.notna() & (v.astype(str).str.strip() != "")).to_numpy()


def _mark(code: np.ndarray, src: np.ndarray, idx: np.ndarray, level: int, source: int) -> None:
    """Raise ``code`` to ``level`` at ``idx`` (never lower it); the rule that raised it becomes the source."""
    if len(idx) == 0:
        return
    up = code[idx] < level
    code[idx[up]] = level
    src[idx[up]] = source


def require_stoppage_state(frames: pd.DataFrame, provider: str) -> None:
    """Fail closed: a match of a provider whose frames do not record the dead ball must carry ``stoppage_state``.

    A training preparer would otherwise fall back to ``ball_state``, which is constant ``"alive"`` there, and train
    on dead-ball frames without a sign (ADR-130).

    Examples
    --------
    >>> require_stoppage_state(pd.DataFrame({"stoppage_state": ["live"]}), "skillcorner")
    >>> require_stoppage_state(pd.DataFrame({"ball_state": ["alive"]}), "sportec")
    """
    from silly_kicks.tracking._provider_visibility import dead_ball_observed

    # A staged match needs no provider lookup; an unclassified provider without the columns still raises there.
    if "stoppage_state" not in frames.columns and not dead_ball_observed(provider):
        raise ValueError(
            f"{provider} frames carry no stoppage_state: their ball_state does not record the dead ball, so the "
            "dataset must be staged with the stoppage columns (tc3-frames-4 or later; ADR-130)"
        )


#: The keys of :func:`stoppage_summary`, in output order.
STOPPAGE_SUMMARY_KEYS = (
    "frames_dead",
    "frames_unknown",
    "frames_horizon",
    "labels_in_dropped",
    *(f"source_{s}" for s in STOPPAGE_SOURCES),
)


def stoppage_summary(frames: pd.DataFrame, *, horizon_seconds: float, label_times: pd.DataFrame) -> dict[str, int]:
    """Per-match counts for a trained model's records: frames dropped as ``dead``, as ``unknown`` and by the label
    horizon (:func:`training_frame_mask`), the labels (``label_times``: ``period_id``, ``time_seconds``) whose own
    frame is dropped, and the frames per ``stoppage_source``. All zero without a ``stoppage_state`` column.

    Examples
    --------
    >>> fr = pd.DataFrame({"game_id": 1, "period_id": 1, "frame_id": [1, 2, 3], "time_seconds": [0.0, 1.0, 5.0],
    ...     "stoppage_state": ["live", "dead", "unknown"], "stoppage_source": ["events", "events", "events"]})
    >>> lt = pd.DataFrame({"period_id": [1], "time_seconds": [1.2]})
    >>> s = stoppage_summary(fr, horizon_seconds=0.5, label_times=lt)
    >>> (s["frames_dead"], s["frames_unknown"], s["frames_horizon"], s["labels_in_dropped"], s["source_events"])
    (1, 1, 0, 1, 3)
    """
    out = dict.fromkeys(STOPPAGE_SUMMARY_KEYS, 0)
    if "stoppage_state" not in frames.columns:
        return out
    keys = ["game_id", "period_id", "frame_id"]
    one = frames.drop_duplicates(keys).sort_values(["period_id", "time_seconds"], kind="mergesort")
    state = one["stoppage_state"].to_numpy()
    keep = training_frame_mask(one, horizon_seconds=horizon_seconds)
    out["frames_dead"] = int((state == "dead").sum())
    out["frames_unknown"] = int((state == "unknown").sum())
    out["frames_horizon"] = int(((state == "live") & ~keep).sum())
    for s, n in one["stoppage_source"].value_counts().items():
        if f"source_{s}" in out:
            out[f"source_{s}"] = int(n)
    dropped = 0
    for per, lt in label_times.groupby("period_id", sort=False):
        fr = one[one["period_id"] == per]
        if fr.empty:
            continue
        t = fr["time_seconds"].to_numpy(dtype="float64")
        j = np.clip(np.searchsorted(t, lt["time_seconds"].to_numpy(dtype="float64"), "right") - 1, 0, len(t) - 1)
        dropped += int((~keep[np.flatnonzero(one["period_id"].to_numpy() == per)][j]).sum())
    out["labels_in_dropped"] = dropped
    return out
