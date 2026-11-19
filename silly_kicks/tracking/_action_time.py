"""One step sets the action time from the tracking (action-time repair spec sections 4.3, 4.4, 4.17).

``sync_action_times(actions, frames)`` applies four rules in order, and ``restart_gaps(actions, frames)`` defines
the one gap of a derived restart or penalty row that both the step and the loader's kick check read:

1. ELASTIC (``align_events_to_frames``) on every row of a match that has an ``"event"`` row; a matched ``"event"``
   row takes the time of its ``elastic_frame_id`` and becomes ``"elastic"``. A match with no ``"event"`` row skips it.
2. The corner kick and the penalty kick: a derived corner or penalty row with ``"placeholder"`` or
   ``"order_only"`` and a kick found in its gap takes the kick time and the rest point; it becomes ``"kick"``.
3. A synthetic dribble (``"derived"``) takes the midpoint of its neighbours.
4. The rows are sorted chronologically and ``action_id`` is renumbered.

The kick rule (ported, spec section 4.4): a run of at least 0.5 s of DETECTED ball frames that stay within 0.5 m of
the run's first point, that point within 2 m of a corner flag (corner) or 1.5 m of a penalty mark (penalty), and a
detected ball frame more than 2 m from it within 1 s after the run. The kick time is the run's last frame.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

from silly_kicks.id_compat import canonical_id, ids_match
from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl._skillcorner_inference import restart_subtype
from silly_kicks.spadl.base import sort_actions_chronologically
from silly_kicks.spadl.schema import _RESTART_KINDS, is_time_unmeasured, restart_kind

from ._elastic_sync import align_events_to_frames
from ._provider_visibility import (
    _DETECTION_AWARE_PROVIDERS,
    _FULLY_OBSERVED_PROVIDERS,
    validate_provider,
)

__all__ = ["restart_gaps", "sync_action_times"]

REST_RADIUS_M: float = 0.5  # the TF-46 design section 6.4 rest condition
REST_MIN_S: float = 0.5
RELEASE_DISTANCE_M: float = 2.0  # restart_release_reach.py
RELEASE_WITHIN_S: float = 1.0
FLAG_RADIUS_M: float = 2.0  # the author's choice inside the TF-46 bound (spec section 4.4)
MARK_RADIUS_M: float = 1.5  # measured by sc_penalty_check.py (spec section 4.17)
_GAP_OFFSET_S = {"placeholder": 0.02, "order_only": 0.03}
_KICK_KINDS = ("corner", "penalty")
_FOLLOWERS_THAT_CANNOT_REST_AT_A_FLAG = frozenset({"goal_kick", "penalty"})
_RESTART_KIND_NAMES = frozenset(_RESTART_KINDS.values())  # the kinds of a restart or penalty row

_L, _W = float(spadlconfig.field_length), float(spadlconfig.field_width)
_FLAGS = np.array([[0.0, 0.0], [0.0, _W], [_L, 0.0], [_L, _W]])
_MARKS = np.array([[11.0, _W / 2], [_L - 11.0, _W / 2]])


def _cell(df: pd.DataFrame, label: Any, col: str) -> Any:
    """One cell by label, typed ``Any`` (the pandas stubs type ``.at`` reads as a scalar union)."""
    return df.at[label, col]


def _kinds(actions: pd.DataFrame) -> pd.Series:
    return actions["type_id"].map(restart_kind)


def _is_derived(actions: pd.DataFrame) -> np.ndarray:
    if "action_provenance" not in actions.columns:
        return np.zeros(len(actions), dtype=bool)
    return (actions["action_provenance"] == "derived").to_numpy()


def _period_bounds(frames: pd.DataFrame) -> dict[tuple[Any, int], tuple[float, float]]:
    if len(frames) == 0:
        return {}
    g = frames.groupby(["game_id", "period_id"], sort=False)["time_seconds"]
    lo, hi = g.min(), g.max()
    return {(canonical_id(k[0]), int(k[1])): (float(lo[k]), float(hi[k])) for k in lo.index}


def restart_gaps(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    """The gap of each derived restart or penalty row with an unmeasured time (spec section 4.4).

    Read from ``actions`` BEFORE any rule of the step changes a time, and from ``frames``:
    ``gap_start`` is the time of the previous non-derived row of the same period (``a.frame_end``), or the
    period's first frame time when no such row exists (the row's ``b`` opens its period); ``gap_end`` is the
    row's own unmeasured time plus its offset (0.02 s ``"placeholder"``, 0.03 s ``"order_only"``), which is ``b``'s
    reception, or the period's last frame time when no non-derived row follows in the period (a period-end
    penalty).

    Returns
    -------
    pd.DataFrame
        Columns ``game_id``, ``original_event_id``, ``action_provenance``, ``kind``, ``time_source``,
        ``period_id``, ``gap_start``, ``gap_end``, ``period_end`` (True when no non-derived row follows in the
        period: a period-end penalty); one row per derived restart or penalty row with
        ``"placeholder"`` or ``"order_only"``. The first four columns are the key of spec section 4.18.

    Examples
    --------
    >>> import pandas as pd
    >>> restart_gaps(pd.DataFrame(columns=["game_id", "period_id", "time_seconds", "action_id", "type_id",
    ...     "original_event_id", "action_provenance", "time_source"]), pd.DataFrame()).empty
    True
    """
    cols = [
        "game_id",
        "original_event_id",
        "action_provenance",
        "kind",
        "time_source",
        "period_id",
        "gap_start",
        "gap_end",
        "period_end",
    ]
    if len(actions) == 0 or "time_source" not in actions.columns:
        return pd.DataFrame(columns=cols)
    a = actions.sort_values(["game_id", "period_id", "time_seconds", "action_id"], kind="mergesort")
    a = a.reset_index(drop=True)
    derived = _is_derived(a)
    unmeasured = is_time_unmeasured(a) & derived
    bounds = _period_bounds(frames)
    rows = []
    for key, idx in a.groupby(["game_id", "period_id"], sort=False).indices.items():
        gid, pid = cast(tuple, key)
        idx = np.asarray(idx)
        times = a["time_seconds"].to_numpy(dtype="float64")[idx]
        anchor = ~derived[idx]
        lo_frame, hi_frame = bounds.get((canonical_id(gid), int(pid)), (np.nan, np.nan))
        for j, row_pos in enumerate(idx):
            if not unmeasured[row_pos]:
                continue
            kind = restart_kind(int(_cell(a, row_pos, "type_id")))
            prev = np.flatnonzero(anchor[:j])
            nxt = np.flatnonzero(anchor[j + 1 :])
            gap_start = float(times[prev[-1]]) if len(prev) else lo_frame
            src = str(a.at[row_pos, "time_source"])
            gap_end = float(times[j]) + _GAP_OFFSET_S[src] if len(nxt) else hi_frame
            rows.append(
                {
                    "game_id": gid,
                    "original_event_id": a.at[row_pos, "original_event_id"],
                    "action_provenance": a.at[row_pos, "action_provenance"],
                    "kind": kind,
                    "time_source": src,
                    "period_id": int(pid),
                    "gap_start": gap_start,
                    "gap_end": gap_end,
                    "period_end": not len(nxt),
                }
            )
    return pd.DataFrame(rows, columns=cols)


def _detected(ball: pd.DataFrame) -> np.ndarray:
    """Detected ball frames only (ADR-109). An all-null ``visibility`` of a detection-aware feed: none detected."""
    if len(ball) == 0:
        return np.zeros(0, dtype=bool)
    provider = str(ball["source_provider"].iloc[0]) if "source_provider" in ball.columns else None
    if provider is not None:
        validate_provider(provider)
        if provider in _FULLY_OBSERVED_PROVIDERS:
            return np.ones(len(ball), dtype=bool)
    if "visibility" not in ball.columns:
        return np.zeros(len(ball), dtype=bool) if provider in _DETECTION_AWARE_PROVIDERS else np.ones(len(ball), bool)
    vis = ball["visibility"]
    if vis.isna().all():
        return np.zeros(len(ball), dtype=bool)
    # nullable boolean first: fillna on an object column with missing values is a pandas downcast (FutureWarning)
    return vis.astype("boolean").fillna(False).to_numpy(dtype=bool)


def _rest_runs(t: np.ndarray, x: np.ndarray, y: np.ndarray, fps: float) -> list[tuple[int, int]]:
    """Maximal rest runs followed by a release: (first index, last index), in time order."""
    n_min = max(2, round(REST_MIN_S * fps))
    max_step = 1.5 / fps + 1e-6
    runs: list[tuple[int, int]] = []
    s = 0
    while s <= len(t) - n_min:
        e = s + n_min - 1
        window_ok = (t[e] - t[s] <= REST_MIN_S + 1.5 / fps) and np.all(
            np.hypot(x[s : e + 1] - x[s], y[s : e + 1] - y[s]) < REST_RADIUS_M
        )
        if not window_ok:
            s += 1
            continue
        while (
            e + 1 < len(t)
            and t[e + 1] - t[e] <= max_step
            and np.hypot(x[e + 1] - x[s], y[e + 1] - y[s]) < (REST_RADIUS_M)
        ):
            e += 1
        after = (t > t[e]) & (t <= t[e] + RELEASE_WITHIN_S)
        if np.any(np.hypot(x[after] - x[s], y[after] - y[s]) > RELEASE_DISTANCE_M):
            runs.append((s, e))
        s = e + 1
    return runs


def _near(points: np.ndarray, x: float, y: float, radius: float) -> bool:
    return bool(np.min(np.hypot(points[:, 0] - x, points[:, 1] - y)) <= radius)


def _find_kick(
    ball: pd.DataFrame, lo: float, hi: float, kind: str, *, first_run_rule: bool, follower_cannot_rest: bool
) -> tuple[float, float, float] | None:
    """(kick time, rest x, rest y) in the frames' own coordinates, or None."""
    m = (ball["time_seconds"].to_numpy() >= lo) & (ball["time_seconds"].to_numpy() <= hi)
    sub = ball[m]
    if len(sub) == 0:
        return None
    det = _detected(sub)
    sub = sub[det].sort_values("time_seconds", kind="mergesort")
    t = sub["time_seconds"].to_numpy(dtype="float64")
    x = sub["x"].to_numpy(dtype="float64")
    y = sub["y"].to_numpy(dtype="float64")
    if len(t) < 2:
        return None
    steps = np.diff(t)
    fps = 1.0 / float(np.median(steps[steps > 0])) if np.any(steps > 0) else 10.0
    runs = _rest_runs(t, x, y, fps)
    if not runs:
        return None
    points, radius = (_FLAGS, FLAG_RADIUS_M) if kind == "corner" else (_MARKS, MARK_RADIUS_M)
    if first_run_rule:
        s, e = runs[0]
        if not _near(points, x[s], y[s], radius):
            return None
        later = any(rs > e for rs, _ in runs[1:])
        if not (later or follower_cannot_rest):
            return None
        return float(t[e]), float(x[s]), float(y[s])
    for s, e in reversed(runs):
        if _near(points, x[s], y[s], radius):
            return float(t[e]), float(x[s]), float(y[s])
    return None


def _to_attacking_end(x: float, y: float) -> tuple[float, float]:
    """A corner or penalty rest point in the restart team's frame: it lies at the end that team attacks."""
    if x < _L / 2:
        return _L - x, _W - y
    return x, y


def sync_action_times(actions: pd.DataFrame, frames: pd.DataFrame, *, params: object | None = None) -> pd.DataFrame:
    """Set each action time from the tracking where it can be measured (spec section 4.3). Pure.

    Parameters
    ----------
    actions : pd.DataFrame
        SPADL actions of one or more matches, with ``time_source``. A table without the column is refused,
        because a stored provider time would otherwise read as ``"event"``.
    frames : pd.DataFrame
        Long-form tracking frames of the same matches (``game_id``, ``period_id``, ``frame_id``,
        ``time_seconds``, ``is_ball``, ``x``, ``y``, ``visibility``, ``source_provider``).
    params : object or None
        Reserved for the ELASTIC parameters (the default ``ElasticSyncParams`` is used).

    Returns
    -------
    pd.DataFrame
        A new actions table: times and ``time_source`` set by the four rules, rows in chronological order,
        ``action_id`` renumbered. A derived corner or penalty row with a measured kick also takes the rest point
        as its start (and as its end where its end is its start); a corner with a first possession is re-typed by
        the 15 m rule. ``type_id``'s restart kind and every key of spec section 4.18 are unchanged.

    Raises
    ------
    ValueError
        If ``actions`` has no ``time_source`` column.

    Notes
    -----
    The ``"elastic"`` accuracy is measured on idsse only (spec section 4.3); metrica or kloppy actions with frames
    would also get ``"elastic"``, with no measurement in this cycle.

    Examples
    --------
    >>> import pandas as pd
    >>> sync_action_times(pd.DataFrame({"time_seconds": [1.0]}), pd.DataFrame())  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: sync_action_times: actions has no time_source column
    """
    if "time_source" not in actions.columns:
        raise ValueError(
            "sync_action_times: actions has no time_source column; a stored provider time would otherwise read as "
            "'event' and be re-synced (action-time repair spec section 4.3)"
        )
    out = actions.copy()
    gaps = restart_gaps(out, frames)
    is_ball = frames["is_ball"].astype(bool).to_numpy() if len(frames) and "is_ball" in frames.columns else None

    # Rule 1: ELASTIC, once per match that has an "event" row.
    for gid, idx in out.groupby("game_id", sort=False).indices.items():
        idx = np.asarray(idx)
        if not (out["time_source"].to_numpy()[idx] == "event").any() or len(frames) == 0:
            continue
        match = out.iloc[idx]
        match_frames = frames[ids_match(frames["game_id"], gid).to_numpy()]
        aligned = align_events_to_frames(match, match_frames)
        if aligned is None or len(aligned) == 0:
            continue
        aligned = aligned[aligned["elastic_frame_id"].notna()]
        ft = match_frames.drop_duplicates(["period_id", "frame_id"]).set_index(["period_id", "frame_id"])[
            "time_seconds"
        ]
        fid = dict(zip(aligned["action_id"].to_numpy(), aligned["elastic_frame_id"].to_numpy(), strict=True))
        labels = list(out.index)
        for pos in idx:
            lbl = labels[pos]
            if _cell(out, lbl, "time_source") != "event":
                continue
            aid = _cell(out, lbl, "action_id")
            if aid not in fid:
                continue
            key = (_cell(out, lbl, "period_id"), int(fid[aid]))
            if key in ft.index:
                out.at[lbl, "time_seconds"] = float(ft.loc[key])
                out.at[lbl, "time_source"] = "elastic"

    # Rule 2: the corner kick and the penalty kick, inside the gap.
    if len(gaps) and is_ball is not None:
        ball = frames[is_ball]
        kinds = _kinds(out)
        derived = _is_derived(out)
        gap_key: dict[tuple[Any, Any, str], dict[Any, Any]] = {  # canonical ids (str, or NA for a missing id)
            (canonical_id(r["game_id"]), canonical_id(r["original_event_id"]), str(r["kind"])): r
            for r in gaps.to_dict("records")
        }
        for i in range(len(out)):
            label = out.index[i]
            kind = kinds.iat[i]
            src = _cell(out, label, "time_source")
            if not derived[i] or kind not in _KICK_KINDS or src not in _GAP_OFFSET_S:
                continue
            gid, oid = canonical_id(_cell(out, label, "game_id")), canonical_id(_cell(out, label, "original_event_id"))
            g = gap_key.get((gid, oid, str(kind)))
            if g is None:
                continue
            in_match = ids_match(ball["game_id"], gid).to_numpy()
            match_ball = ball[in_match & (ball["period_id"] == g["period_id"]).to_numpy()]
            first_run_rule = kind == "corner" and src == "order_only"
            follower_cannot_rest = True
            if first_run_rule:
                # The follower is the restart or penalty row of the same gap, never another derived row: an
                # inferred defensive row of b carries b's event id too (spec section 4.4).
                same_gap = derived & ids_match(out["original_event_id"], oid).to_numpy()
                followers = set(kinds[same_gap & kinds.isin(_RESTART_KIND_NAMES).to_numpy()]) - {kind}
                follower_cannot_rest = not followers or followers <= _FOLLOWERS_THAT_CANNOT_REST_AT_A_FLAG
            kick = _find_kick(
                match_ball,
                float(g["gap_start"]),
                float(g["gap_end"]),
                str(kind),
                first_run_rule=first_run_rule,
                follower_cannot_rest=follower_cannot_rest,
            )
            if kick is None:
                continue
            t_kick, rx, ry = kick
            sx, sy = _to_attacking_end(rx, ry)
            out.at[label, "time_seconds"] = t_kick
            out.at[label, "start_x"], out.at[label, "start_y"] = sx, sy
            if src == "order_only" or kind == "penalty":
                out.at[label, "end_x"], out.at[label, "end_y"] = sx, sy  # a row whose end is its start
            else:
                success = _cell(out, label, "result_id") == spadlconfig.result_id["success"]
                out.at[label, "type_id"] = restart_subtype(
                    int(_cell(out, label, "type_id")),
                    (sx, sy),
                    (float(_cell(out, label, "end_x")), float(_cell(out, label, "end_y"))),
                    success=bool(success),
                )
            out.at[label, "time_source"] = "kick"

    # Rule 3: a synthetic dribble takes the midpoint of its neighbours (the _add_dribbles rule).
    is_dribble_type = (out["type_id"] == spadlconfig.actiontype_id["dribble"]).to_numpy()
    dribble = (out["time_source"] == "derived").to_numpy() & is_dribble_type
    if dribble.any():
        by_id = out.sort_values(["game_id", "period_id", "action_id"], kind="mergesort")
        pos = {lab: k for k, lab in enumerate(by_id.index)}
        times = by_id["time_seconds"].to_numpy(dtype="float64")
        same = by_id["period_id"].to_numpy()
        for label in out.index[dribble]:
            k = pos[label]
            if 0 < k < len(by_id) - 1 and same[k - 1] == same[k] == same[k + 1]:
                out.at[label, "time_seconds"] = (times[k - 1] + times[k + 1]) / 2

    # Rule 4: chronological order (the ADR-065 sorter) on the old action_id, then renumber.
    out = sort_actions_chronologically(out, by=("game_id", "period_id", "time_seconds"), tiebreak=("action_id",))
    out = out.reset_index(drop=True)
    out["action_id"] = out.groupby("game_id", sort=False).cumcount().astype("int64") if len(out) else out["action_id"]
    return out
