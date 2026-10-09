"""Coordination analysis windows and stoppage evidence (spec 7.6).

Three window builders (period/sliding, possession-from-events, possession-from-tracking) produce the shared
COORD_WINDOW_COLUMNS schema; ``validate_windows`` enforces the contract and the source-mix rule (C25).
``resolve_stoppages`` derives the dead-ball intervals used to split a window, from ``ball_state`` (observed
providers) or from restart/goal events, with the D20 precedence.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.coordination._columns import COORD_WINDOW_COLUMNS, COORD_WINDOW_KINDS, COORD_WINDOW_SOURCES
from silly_kicks.coordination._config import CoordinationParams
from silly_kicks.coordination._report import CoordinationCoverageWarning
from silly_kicks.id_compat import canonical_id, same_id
from silly_kicks.tracking._provider_visibility import dead_ball_observed

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

_NA_TERMINALS = {"attacking_team_id": pd.NA, "terminal_action": pd.NA, "terminal_team_id": pd.NA, "n_phases": pd.NA}


def _f(v) -> float:
    """pandas Scalar (itertuples/iloc) -> float; the untyped arg sidesteps the Scalar->float stub friction."""
    return float(v)


def _i(v) -> int:
    """pandas Scalar -> int (see :func:`_f`)."""
    return int(v)


#: The fewest phases a subdivision may have: n = 1 would only duplicate the window's own row (owner ruling
#: 2026-10-04, review A-22). A window's n_phases is NA (no subdivision) or an integer of at least this.
MIN_N_PHASES = 2


def _valid_n_phases(v: object) -> bool:
    """True iff ``v`` is an integer of at least :data:`MIN_N_PHASES` (bool is not an integer here)."""
    if isinstance(v, bool | np.bool_) or not isinstance(v, int | float | np.integer | np.floating):
        return False
    return bool(float(v).is_integer() and v >= MIN_N_PHASES)


def check_n_phases(n_phases: object, *, where: str) -> None:
    """Refuse a builder's ``n_phases`` unless it is an integer of at least :data:`MIN_N_PHASES`.

    Examples
    --------
    A valid phase count is accepted (returns ``None``); anything below the minimum is refused:

    >>> check_n_phases(3, where="build_pair_phase") is None
    True
    >>> check_n_phases(1, where="build_pair_phase")  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: ...n_phases must be an integer >= ...
    """
    if not _valid_n_phases(n_phases):
        raise ValueError(
            f"{where}: n_phases must be an integer >= {MIN_N_PHASES} (one phase only duplicates the window), got "
            f"{n_phases!r}"
        )


def _check_window_n_phases(values: pd.Series) -> None:
    """Each window's ``n_phases`` is NA (no subdivision) or an integer >= :data:`MIN_N_PHASES` (spec 7.6; A-22)."""
    present = values[values.notna()]
    bad = sorted({repr(v) for v in present.unique().tolist() if not _valid_n_phases(v)})
    if bad:
        raise ValueError(
            f"windows n_phases must be NA (no subdivision) or an integer >= {MIN_N_PHASES}; got {', '.join(bad)}"
        )


def phase_assignment(m: int, n_phases: int) -> np.ndarray:
    """Assign ``m`` rank-ordered samples to phases ``1..n_phases`` (C5: ``k = ((i+1)*n + m - 1) // m``).

    Examples
    --------
    >>> phase_assignment(6, 3).tolist()
    [1, 1, 2, 2, 3, 3]
    """
    i = np.arange(m)
    return ((i + 1) * n_phases + m - 1) // m


# --------------------------------------------------------------------------- window frame assembly
def _window_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=list(COORD_WINDOW_COLUMNS))
    for col, dtype in COORD_WINDOW_COLUMNS.items():
        if dtype == "Int64":
            df[col] = df[col].astype("Int64")
        elif dtype == "int64":
            df[col] = df[col].astype("int64")
        elif dtype == "float64":
            df[col] = df[col].astype("float64")
        else:
            df[col] = df[col].astype(object)
    return df


def _row(game, period, kind, source, start, end, **extra) -> dict[str, Any]:
    row: dict[str, Any] = {
        "game_id": game,
        "period_id": int(period),
        "window_kind": kind,
        "window_id": pd.NA,  # filled after sorting
        "window_source": source,
        "start_time_s": float(start),
        "end_time_s": float(end),
        **_NA_TERMINALS,
    }
    row.update(extra)
    return row


def _assign_window_ids(df: pd.DataFrame) -> pd.DataFrame:
    order = df.groupby(["game_id", "period_id", "window_kind"], observed=True)["start_time_s"].rank(method="first")
    df["window_id"] = pd.array((order - 1).to_numpy(), dtype="Int64")
    return df


def _period_bounds(frames: pd.DataFrame) -> pd.DataFrame:
    g = frames.groupby(["game_id", "period_id"], sort=True, observed=True)
    out = g.agg(start=("time_seconds", "min"), last=("time_seconds", "max"), fr=("frame_rate", "first")).reset_index()
    out["end"] = out["last"] + 1.0 / out["fr"]
    return out


def period_windows(frames: pd.DataFrame, *, length_s: float | None = None, step_s: float | None = None) -> pd.DataFrame:
    """One ``period`` window per (game, period), or full-length ``sliding`` windows when ``length_s`` is given.

    Examples
    --------
    Build coordination windows over a tracking-frames table (needs a real match's ``frames``)::

        windows = period_windows(frames)                                # one "period" window per (game, period)
        sliding = period_windows(frames, length_s=120.0, step_s=60.0)   # 2-min windows, 1-min hop
    """
    bounds = _period_bounds(frames)
    rows: list[dict[str, Any]] = []
    for b in bounds.itertuples(index=False):
        start, end = _f(b.start), _f(b.end)
        if length_s is None:
            rows.append(_row(b.game_id, b.period_id, "period", "period", start, end))
        else:
            if step_s is None:
                raise ValueError("step_s is required when length_s is given")
            j = 0
            while start + j * step_s + length_s <= end + 1e-9:  # full-length only (C21)
                s = start + j * step_s
                rows.append(_row(b.game_id, b.period_id, "sliding", "period", s, s + length_s))
                j += 1
    return _assign_window_ids(_window_df(rows))


def _frame_key(game: object, period: object, frame: object) -> tuple[str, int, str]:
    """A frame's lookup key on CANONICAL ids (ADR-019), like :func:`_period_key`: a game or frame id that is a
    string on one table and an int on the other is the same id, never a silent miss."""
    return str(canonical_id(game)), _i(period), str(canonical_id(frame))


def _frame_time_lookup(frames: pd.DataFrame) -> dict[tuple[str, int, str], float]:
    uniq = frames.drop_duplicates(["game_id", "period_id", "frame_id"])
    keys = [_frame_key(g, p, f) for g, p, f in zip(uniq["game_id"], uniq["period_id"], uniq["frame_id"], strict=True)]
    vals = uniq["time_seconds"].to_numpy(dtype=np.float64)
    return {k: float(v) for k, v in zip(keys, vals, strict=True)}


def _period_key(game: object, period: object) -> tuple[str, int]:
    """A (game, period) key that survives dtype drift: a row pulled out of a mixed-dtype frame upcasts an int
    ``game_id`` to float, which a raw-value dict key would then miss (ADR-019)."""
    return str(canonical_id(game)), _i(period)


def _actions_with_tracking(actions: pd.DataFrame, frames: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The actions in a (game, period) the frames cover, plus what was left out (empty when nothing was).

    A possession in a period with no tracking -- a per-match data hole, e.g. Gradient Sports 10510/10511, whose
    tracking stops after period 2 while the events run through extra time -- has no samples to score and no period
    end to close it, so it gets no window; the caller warns, never a silent drop.
    """
    tracked = {_period_key(g, p) for g, p in frames[["game_id", "period_id"]].drop_duplicates().itertuples(index=False)}
    keys = [_period_key(g, p) for g, p in zip(actions["game_id"], actions["period_id"], strict=True)]
    keep = np.fromiter((k in tracked for k in keys), dtype=bool, count=len(keys))
    if keep.all():
        return actions, {}
    missing = sorted({k for k, ok in zip(keys, keep, strict=True) if not ok})
    where = ", ".join(f"game {g} period {p}" for g, p in missing)
    return actions.loc[keep], {"n_actions": int((~keep).sum()), "where": where}


def possession_windows_from_actions(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    links: pd.DataFrame | None = None,
    n_phases: int = 3,
    possession_kwargs: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Possession windows from event data: one window per possession, ending at the event that ends it.

    Examples
    --------
    Possession-scoped windows from SPADL actions + frames (needs a real match)::

        windows = possession_windows_from_actions(actions, frames)
        # windows.window_kind == "possession"; each row spans one team's possession
    """
    from silly_kicks.spadl import add_possessions
    from silly_kicks.spadl.base import sort_actions_chronologically

    check_n_phases(n_phases, where="possession_windows_from_actions")
    actions, skipped = _actions_with_tracking(actions, frames)
    if skipped:
        warnings.warn(
            f"possession_windows_from_actions: {skipped['n_actions']} action(s) in a (game, period) the frames do not "
            f"cover ({skipped['where']}) get no possession window -- there are no samples to score them on",
            CoordinationCoverageWarning,
            stacklevel=2,
        )
    # ADR-065 §3d: the robust (game, period, time_seconds, action_id) order, never action_id alone -- a persisted mart
    # may carry a non-chronological action_id (a no-op on converter output, whose ids are chronological)
    ctx = sort_actions_chronologically(
        add_possessions(actions, **(possession_kwargs or {})), tiebreak=("action_id",)
    ).reset_index(drop=True)
    times = ctx["time_seconds"].to_numpy(dtype=np.float64).copy()
    if links is not None:
        time_of = _frame_time_lookup(frames)
        frame_by_action = {
            str(canonical_id(a)): fid for a, fid in zip(links["action_id"], links["frame_id"], strict=True)
        }
        for pos in range(len(ctx)):
            fid = frame_by_action.get(str(canonical_id(ctx.at[pos, "action_id"])))
            if fid is not None:
                key = _frame_key(ctx.at[pos, "game_id"], ctx.at[pos, "period_id"], fid)
                if key in time_of:
                    times[pos] = time_of[key]
    ctx = ctx.assign(_t=times)

    bounds = _period_bounds(frames).itertuples(index=False)
    period_end = {_period_key(b.game_id, b.period_id): _f(b.end) for b in bounds}
    types = spadlconfig.actiontypes
    # per-possession summary in possession order (first-action time)
    summary = []
    # add_possessions restarts its counter per game, so a possession is (game, possession_id) -- never possession_id
    # alone, which would merge possession k of every game. Rows keep ctx's chronological order within each group; each
    # value is read per COLUMN (an `iloc` row of a mixed-dtype frame upcasts an int game_id to float).
    for _key, grp in ctx.groupby(["game_id", "possession_id"], sort=False, observed=True):
        summary.append(
            {
                "game_id": grp["game_id"].iloc[0],
                "period_id": _i(grp["period_id"].iloc[0]),
                "att_team": grp["team_id"].iloc[0],
                "start": _f(grp["_t"].iloc[0]),
                "first_type": types[_i(grp["type_id"].iloc[0])],
                "last_type": types[_i(grp["type_id"].iloc[-1])],
                "last_t": _f(grp["_t"].iloc[-1]),
            }
        )
    summary.sort(key=lambda s: (*_period_key(s["game_id"], s["period_id"]), s["start"]))
    rows: list[dict[str, Any]] = []
    for idx, s in enumerate(summary):
        nxt = summary[idx + 1] if idx + 1 < len(summary) else None
        if s["last_type"] in SHOT_TYPES:
            end, term_action, term_team = s["last_t"], s["last_type"], s["att_team"]
        elif nxt is not None and same_id(nxt["game_id"], s["game_id"]) and nxt["period_id"] == s["period_id"]:
            # the next possession's FIRST action (the regaining tackle, interception, ...) is what ended this one
            end, term_action, term_team = nxt["start"], nxt["first_type"], nxt["att_team"]
        else:
            end, term_action, term_team = period_end[_period_key(s["game_id"], s["period_id"])], pd.NA, pd.NA
        rows.append(
            _row(
                s["game_id"],
                s["period_id"],
                "possession",
                "possession_events",
                s["start"],
                end,
                attacking_team_id=s["att_team"],
                terminal_action=term_action,
                terminal_team_id=term_team,
                n_phases=int(n_phases),
            )
        )
    return _assign_window_ids(_window_df(rows))


def possession_windows_from_frames(
    frames: pd.DataFrame,
    *,
    carrier: pd.DataFrame | None = None,
    n_phases: int = 3,
    params: CoordinationParams | None = None,
) -> pd.DataFrame:
    """Possession windows from tracking: maximal one-team spells, bridging same-team NA gaps up to the gap.

    Examples
    --------
    Possession windows inferred from the ball carrier when no event data is available::

        windows = possession_windows_from_frames(frames)
        # one row per maximal one-team spell; same-team NA gaps <= possession_gap_s are bridged
    """
    from silly_kicks.tracking import derive_team_in_possession, infer_ball_carrier

    check_n_phases(n_phases, where="possession_windows_from_frames")
    gap_s = (params or CoordinationParams()).possession_gap_s
    car = carrier if carrier is not None else infer_ball_carrier(frames)
    pos = derive_team_in_possession(frames, car)
    per = (
        pos.drop_duplicates(["game_id", "period_id", "frame_id"])
        .sort_values(["game_id", "period_id", "time_seconds"])
        .reset_index(drop=True)
    )
    rows: list[dict[str, Any]] = []
    for (game, period), grp in per.groupby(["game_id", "period_id"], sort=True, observed=True):
        t = grp["time_seconds"].to_numpy(dtype=np.float64)
        team = grp["team_in_possession"].to_numpy(dtype=object)
        fr = _f(grp["frame_rate"].iloc[0])
        rows.extend(_spells(game, _i(period), t, team, gap_s, fr, n_phases))
    return _assign_window_ids(_window_df(rows))


def _spells(
    game: object, period: int, t: np.ndarray, team: np.ndarray, gap_s: float, fr: float, n_phases: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cur_team: Any = None
    start_t: Any = None
    last_t: Any = None
    for i in range(len(t)):
        tm: Any = team[i]
        valid = tm is not None and not (isinstance(tm, float) and np.isnan(tm)) and not pd.isna(tm)
        if not valid:
            continue
        if cur_team is not None and tm == cur_team and (t[i] - last_t) <= gap_s + 1e-9:
            last_t = t[i]
            continue
        if cur_team is not None:
            rows.append(
                _row(
                    game,
                    period,
                    "possession",
                    "possession_tracking",
                    start_t,
                    last_t + 1.0 / fr,
                    attacking_team_id=cur_team,
                    n_phases=int(n_phases),
                )
            )
        cur_team, start_t, last_t = tm, t[i], t[i]
    if cur_team is not None:
        rows.append(
            _row(
                game,
                period,
                "possession",
                "possession_tracking",
                start_t,
                last_t + 1.0 / fr,
                attacking_team_id=cur_team,
                n_phases=int(n_phases),
            )
        )
    return rows


def validate_windows(windows: pd.DataFrame) -> str:
    """Validate the window contract and return the call's regime; raise on a violation or forbidden mix (C25).

    Examples
    --------
    Validate a windows table and read back its regime token::

        regime = validate_windows(period_windows(frames))   # -> "period"
        # raises ValueError on a missing column, unknown window_kind, or a forbidden source mix
    """
    missing = set(COORD_WINDOW_COLUMNS) - set(windows.columns)
    if missing:
        raise ValueError(f"windows missing columns: {sorted(missing)}")
    if not windows.index.is_unique:
        # the compute addresses each window by its index label (``signals.windows.loc[row_idx]``); a duplicate label
        # would return several rows and corrupt the per-window slice -- refuse it here (review A-50).
        raise ValueError("windows index must be unique (reset_index before passing a concatenated windows table)")
    bad_kind = set(windows["window_kind"].dropna()) - set(COORD_WINDOW_KINDS)
    if bad_kind:
        raise ValueError(f"unknown window_kind tokens: {sorted(bad_kind)}")
    sources = set(windows["window_source"].dropna())
    bad_src = sources - set(COORD_WINDOW_SOURCES)
    if bad_src:
        raise ValueError(f"unknown window_source tokens: {sorted(bad_src)}")
    if "caller" in sources and sources != {"caller"}:
        raise ValueError("caller windows cannot mix with any builder source (C25)")
    if {"possession_events", "possession_tracking"} <= sources:
        raise ValueError("event and tracking possession windows are never mixed (C25)")
    _check_window_n_phases(windows["n_phases"])
    return "|".join(sorted(sources))


# --------------------------------------------------------------------------- stoppage evidence
@dataclass(frozen=True)
class StoppageEvidence:
    """Dead-ball intervals (longer than the threshold) that split a coordination window.

    Examples
    --------
    The result of :func:`resolve_stoppages`; read its provenance + per-period intervals::

        ev = resolve_stoppages(frames, actions=None, provider="sportec", max_stoppage_s=25.0)
        ev.source        # "ball_state" / "events" / "unavailable"
        ev.intervals[(game_id, period_id)]   # (S, 2) array of [start, end) dead spans > threshold
    """

    source: Literal["ball_state", "events", "unavailable"]
    intervals: Mapping[tuple[object, object], np.ndarray]  # (game, period) -> (S, 2) [start, end)
    dead_seconds: float
    n_splits: int


def _runs_longer_than(intervals: list[tuple[float, float]], threshold: float) -> np.ndarray:
    kept = [(lo, hi) for lo, hi in intervals if (hi - lo) > threshold]
    return np.array(kept, dtype=np.float64).reshape(-1, 2)


def _ball_state_intervals(frames: pd.DataFrame, max_stoppage_s: float) -> dict[tuple[object, object], np.ndarray]:
    ball = frames[frames["is_ball"]].sort_values(["game_id", "period_id", "time_seconds"])
    out: dict[tuple[object, object], np.ndarray] = {}
    for (game, period), grp in ball.groupby(["game_id", "period_id"], sort=True, observed=True):
        t = grp["time_seconds"].to_numpy(dtype=np.float64)
        dt = float(np.median(np.diff(t))) if len(t) >= 2 else 0.0  # one frame, to close a run at the period end (A-49)
        dead = grp["ball_state"].astype(object).to_numpy() == "dead"
        raw: list[tuple[float, float]] = []
        i = 0
        n = len(t)
        while i < n:
            if dead[i]:
                j = i
                while j < n and dead[j]:
                    j += 1
                # [t[i], t[j]) up to the next alive frame; a run reaching the last frame extends one frame past it
                # (the last dead sample's own duration), so it is not one sample short at the period end (review A-49).
                end = t[j] if j < n else t[n - 1] + dt
                raw.append((float(t[i]), float(end)))
                i = j
            else:
                i += 1
        kept = _runs_longer_than(raw, max_stoppage_s)
        if len(kept):
            out[(game, period)] = kept
    return out


def _event_intervals(actions: pd.DataFrame, max_stoppage_s: float) -> dict[tuple[object, object], np.ndarray]:
    types = spadlconfig.actiontypes
    results = spadlconfig.results
    out: dict[tuple[object, object], np.ndarray] = {}
    from silly_kicks.spadl.base import sort_actions_chronologically

    for (game, period), grp in actions.groupby(["game_id", "period_id"], sort=True, observed=True):
        # ADR-065: the robust chronological order (time then action_id), never an unstable sort on time alone -- a
        # persisted mart may carry tied times whose action_id breaks the tie (review A-49).
        grp = sort_actions_chronologically(grp, tiebreak=("action_id",)).reset_index(drop=True)
        t = grp["time_seconds"].to_numpy(dtype=np.float64)
        name = [types[_i(x)] for x in grp["type_id"]]
        res = [results[_i(x)] for x in grp["result_id"]]
        raw: list[tuple[float, float]] = []
        for i in range(len(grp)):
            if name[i] in RESTART_TYPES and i > 0:
                raw.append((float(t[i - 1]), float(t[i])))
            is_goal = (name[i] in SHOT_TYPES and res[i] == "success") or res[i] == "owngoal"
            if is_goal and i + 1 < len(grp):
                raw.append((float(t[i]), float(t[i + 1])))
        kept = _runs_longer_than(raw, max_stoppage_s)
        if len(kept):
            out[(game, period)] = kept
    return out


def _evidence(source, intervals) -> StoppageEvidence:
    dead = float(sum(float((arr[:, 1] - arr[:, 0]).sum()) for arr in intervals.values()))
    n = int(sum(len(arr) for arr in intervals.values()))
    return StoppageEvidence(source=source, intervals=intervals, dead_seconds=dead, n_splits=n)


def resolve_stoppages(
    frames: pd.DataFrame,
    *,
    actions: pd.DataFrame | None,
    provider: str,
    max_stoppage_s: float,
    mode: Literal["auto", "ball_state", "events", "none"] = "auto",
) -> StoppageEvidence:
    """Resolve dead-ball splitting intervals; ``auto`` is the D20 precedence ball_state -> events -> unavailable.

    Examples
    --------
    Resolve the dead-ball evidence used to split coordination segments::

        ev = resolve_stoppages(frames, actions=actions, provider="skillcorner", max_stoppage_s=25.0)
        # ev.source is the chosen evidence; ev.n_splits counts the dead spans longer than max_stoppage_s
    """
    if mode == "none":
        return StoppageEvidence("unavailable", {}, 0.0, 0)
    if mode in ("auto", "ball_state"):
        if dead_ball_observed(provider):  # raises for an unclassified provider (fail-closed)
            return _evidence("ball_state", _ball_state_intervals(frames, max_stoppage_s))
        if mode == "ball_state":
            raise ValueError(f"mode='ball_state' cannot be honoured for provider {provider!r} (no dead-ball signal)")
    if mode in ("auto", "events"):
        if actions is not None:
            return _evidence("events", _event_intervals(actions, max_stoppage_s))
        if mode == "events":
            raise ValueError("mode='events' requires actions")
    return StoppageEvidence("unavailable", {}, 0.0, 0)
