"""SkillCorner derived action inference.

Produces SPADL actions that are not directly mapped from ``player_possession``
dispatch but inferred from cross-referencing event types:

- Defensive actions (interceptions, tackles) from ``start_type`` + OBE
- Keeper saves from shot -> opponent-possession sequences
- Restart rows (corner, free kick, goal kick, throw-in) and penalty rows from the interruption markers
  (action-time repair spec sections 4.2, 4.17)

All returned DataFrames have partial SPADL columns (at minimum:
``period_id``, ``time_seconds``, ``team_id``, ``player_id``, ``start_x``,
``start_y``, ``type_id``, ``result_id``, ``bodypart_id``,
``action_provenance``). The caller merges them into the main action stream.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from silly_kicks._frame_index import group_rows

from . import config as spadlconfig

_OBE_TEMPORAL_WINDOW: float = 2.0  # seconds


def _is_defensive_start_type(start_type: pd.Series) -> pd.Series:
    """True for start_type values that indicate a possession win."""
    return start_type.str.endswith("_interception", na=False) | (start_type == "recovery")


def infer_defensive_actions(
    pp: pd.DataFrame,
    obe: pd.DataFrame,
) -> pd.DataFrame:
    """Infer interceptions and tackles from start_type + OBE cross-referencing.

    Parameters
    ----------
    pp : pd.DataFrame
        ``player_possession`` rows, sorted chronologically.
        Required columns: ``event_id``, ``period``, ``time_seconds``,
        ``team_id``, ``player_id``, ``start_type``, ``x_start``, ``y_start``.
    obe : pd.DataFrame
        ``on_ball_engagement`` rows.
        Required columns: ``period``, ``time_seconds``, ``team_id``,
        ``player_id``, ``end_type``, ``x_start``, ``y_start``.

    Returns
    -------
    pd.DataFrame
        Derived defensive actions with partial SPADL columns.
    """
    defensive_mask = _is_defensive_start_type(pp["start_type"])
    if not defensive_mask.any():
        return pd.DataFrame()

    rows: list[dict] = []
    obe_regains = obe[obe["end_type"] == "direct_regain"] if len(obe) > 0 else pd.DataFrame()
    # ADR-068: pre-group the direct_regain OBE rows by (period, team_id) ONCE, instead of
    # re-filtering the whole obe_regains table per defensive row (was O(n_defensive * n_obe)). The
    # per-row candidate set is then an O(1) group lookup + the temporal-window filter; the exact
    # (|dt|, time_seconds, player_id) sort/pick below is unchanged -> byte-identical.
    obe_groups = group_rows(obe_regains, ("period", "team_id")) if len(obe_regains) > 0 else None

    for idx in pp.index[defensive_mask]:
        row = pp.loc[idx]
        period = row["period"]
        t = row["time_seconds"]

        # Default: interception from start_type
        action_type = spadlconfig.actiontype_id["interception"]
        player = row["player_id"]
        team = row["team_id"]
        x = row["x_start"]
        y = row["y_start"]

        # OBE upgrade: check for direct_regain within temporal window + same team
        if obe_groups is not None:
            same_period_team = obe_groups.get(period, row["team_id"])  # O(1): same (period, team_id)
            candidates = same_period_team[(same_period_team["time_seconds"] - t).abs() <= _OBE_TEMPORAL_WINDOW]
            if len(candidates) > 0:
                # Order-insensitive pick (ADR-065): nearest in time, ties broken by CONTENT columns
                # that are always present on an OBE row (``time_seconds`` then ``player_id``), NOT by
                # positional ``argmin()`` (first-on-tie in obe's INPUT row order, so two equidistant
                # same-team ``direct_regain`` rows would flip the tackle's attributed
                # player/team/coords when the input is permuted). ``event_id`` is deliberately NOT
                # used: it is not a guaranteed OBE column at this seam.
                best = (
                    candidates.assign(_dt=(candidates["time_seconds"] - t).abs())
                    .sort_values(["_dt", "time_seconds", "player_id"], kind="mergesort")
                    .iloc[0]
                )
                action_type = spadlconfig.actiontype_id["tackle"]
                player = best["player_id"]
                team = best["team_id"]
                x = best["x_start"]
                y = best["y_start"]

        rows.append(
            {
                "event_id": row["event_id"],
                "period_id": int(period),
                "time_seconds": float(t) - 0.01,  # just before the native action
                "team_id": team,
                "player_id": player,
                "start_x": float(x),
                "start_y": float(y),
                "end_x": float(x),
                "end_y": float(y),
                "type_id": action_type,
                "result_id": spadlconfig.result_id["success"],
                "bodypart_id": spadlconfig.bodypart_id["foot"],
                "action_provenance": "derived",
            }
        )

    return pd.DataFrame(rows)


def infer_keeper_saves(pp: pd.DataFrame) -> pd.DataFrame:
    """Infer keeper saves from shot -> opponent-possession sequences.

    Parameters
    ----------
    pp : pd.DataFrame
        ``player_possession`` rows, sorted chronologically.
        Required columns: ``period``, ``time_seconds``, ``team_id``,
        ``player_id``, ``end_type``, ``x_start``, ``y_start``.
        Optional: ``game_interruption_after``.

    Returns
    -------
    pd.DataFrame
        Derived keeper_save actions with partial SPADL columns.

    Notes
    -----
    Keeper save is attributed to whoever starts the next possession, which is
    typically an outfield player (e.g. CB taking the goal kick), NOT the actual
    GK. This is a data limitation -- SkillCorner does not tag saves natively.
    """
    rows: list[dict] = []
    end_types = pp["end_type"].to_numpy()
    team_ids = pp["team_id"].to_numpy()
    gia = (
        pp["game_interruption_after"].to_numpy() if "game_interruption_after" in pp.columns else np.full(len(pp), None)
    )

    for i in range(len(pp) - 1):
        if end_types[i] != "shot":
            continue
        # Only infer save for on-target shots: gi_after is NaN (plausible save)
        # or corner_for (deflected behind goal line by GK/defender).
        # Skip: goal_for (scored), goal_kick_against (missed wide/over bar),
        # free_kick_against (foul), throw_in_* (unusual -- not a save).
        if gia[i] == "goal_for":
            continue
        if not (pd.isna(gia[i]) or gia[i] == "corner_for"):
            continue
        if team_ids[i] == team_ids[i + 1]:
            continue

        next_row = pp.iloc[i + 1]
        rows.append(
            {
                "period_id": int(next_row["period"]),
                "time_seconds": float(next_row["time_seconds"]) - 0.01,
                "team_id": next_row["team_id"],
                "player_id": next_row["player_id"],
                "start_x": float(next_row["x_start"]),
                "start_y": float(next_row["y_start"]),
                "end_x": float(next_row["x_start"]),
                "end_y": float(next_row["y_start"]),
                "type_id": spadlconfig.actiontype_id["keeper_save"],
                "result_id": spadlconfig.result_id["success"],
                "bodypart_id": spadlconfig.bodypart_id["foot"],
                "action_provenance": "derived",
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Derived restart and penalty rows (action-time repair spec sections 4.2, 4.17, 4.18)
# ---------------------------------------------------------------------------

_RESTART_KINDS: tuple[str, ...] = ("corner", "free_kick", "goal_kick", "throw_in")
_ALL_MARKER_KINDS: tuple[str, ...] = (*_RESTART_KINDS, "penalty")
_SHORT_DISTANCE_M: float = 15.0  # the threshold of the converter's former is_short rule
_PLACEHOLDER_OFFSET_S: float = 0.02  # before a start-of-possession interception (minus 0.01 s)
_ORDER_ONLY_OFFSET_S: float = 0.03  # before the next restart row
_PERIOD_END_OFFSET_S: float = 0.01


def _marker_kind(marker: object, kinds: tuple[str, ...]) -> tuple[str, bool] | None:
    """(kind, is_for) of a ``<kind>_for`` / ``<kind>_against`` marker, else None."""
    if not isinstance(marker, str):
        return None
    for kind in kinds:
        if marker == f"{kind}_for":
            return kind, True
        if marker == f"{kind}_against":
            return kind, False
    return None


def _restart_gaps(pp: pd.DataFrame, kinds: tuple[str, ...], *, period_end: bool) -> list[tuple]:
    """Enumerate the restarts by the count rule: (kind, a, b, cls, marked, is_for), positional indices.

    A marker is the possession's ``game_interruption_before`` (on ``b``) or the previous possession's
    ``game_interruption_after`` (on ``a``) in the same period; an "after" and a "before" marker of the same kind on
    ``a`` and ``b`` are one restart. A period-end "after" marker counts only when ``period_end`` is True.
    """
    n = len(pp)
    per = pp["period"].to_numpy()
    gib = [_marker_kind(m, kinds) for m in pp["game_interruption_before"].tolist()]
    gia = [_marker_kind(m, kinds) for m in pp["game_interruption_after"].tolist()]
    out: list[tuple] = []
    for i in range(n):
        marker_b = gib[i]
        if marker_b is not None:
            kind, is_for = marker_b
            a = i - 1 if i > 0 and per[i - 1] == per[i] else None
            marker_a = gia[a] if a is not None else None
            paired = marker_a is not None and marker_a[0] == kind
            out.append((kind, a, i, "paired" if paired else "before_only", i, is_for))
        marker_after = gia[i]
        if marker_after is not None:
            kind, is_for = marker_after
            nxt = i + 1 if i + 1 < n and per[i + 1] == per[i] else None
            marker_next = gib[nxt] if nxt is not None else None
            if marker_next is not None and marker_next[0] == kind:
                continue  # the paired restart, counted at its "before" marker
            if nxt is None:
                if period_end:
                    out.append((kind, i, None, "period_end", i, is_for))
                continue
            out.append((kind, i, nxt, "after_only", i, is_for))
    return out


class _Gaps:
    """Positional views of ``pp`` that the row builders share."""

    def __init__(self, pp: pd.DataFrame, teams: tuple[str, str]):
        self.pp = pp
        self.teams = teams
        self.team = pp["team_id"].astype(str).to_numpy()
        self.period = pp["period"].to_numpy()
        self.event_id = pp["event_id"].to_numpy()
        self.start_s = pp["start_seconds"].to_numpy(dtype="float64")
        self.end_s = pp["time_seconds"].to_numpy(dtype="float64")
        self.sx_end = pp["sx_end"].to_numpy(dtype="float64")
        self.sy_end = pp["sy_end"].to_numpy(dtype="float64")
        self.sx_start = pp["sx_start"].to_numpy(dtype="float64")
        self.sy_start = pp["sy_start"].to_numpy(dtype="float64")
        self.gib = pp["game_interruption_before"].tolist()
        self.gia = pp["game_interruption_after"].tolist()
        self.has_score = "team_score" in pp.columns and "opponent_team_score" in pp.columns

    def other(self, team: str) -> str:
        home, away = self.teams
        return away if team == home else home

    def in_frame(self, idx: int, x: float, y: float, team: str) -> tuple[float, float]:
        """A point of possession ``idx`` (its team's attacking frame) in ``team``'s frame (a 180-degree turn)."""
        if self.team[idx] == team:
            return x, y
        return spadlconfig.field_length - x, spadlconfig.field_width - y

    def score_of(self, idx: int, team: str) -> float:
        if not self.has_score:
            return float("nan")
        col = "team_score" if self.team[idx] == team else "opponent_team_score"
        v = self.pp[col].iloc[idx]
        return float(v) if pd.notna(v) else float("nan")

    def goal_scorer(self, idx: int) -> str | None:
        """The scoring team named by a goal marker on possession ``idx`` (goal_for: its own team scored)."""
        m = self.gib[idx]
        if m == "goal_for":
            return str(self.team[idx])
        if m == "goal_against":
            return self.other(str(self.team[idx]))
        return None


def _row_time(g: _Gaps, a: int | None, b: int | None, cls: str) -> tuple[float, str]:
    if cls in ("paired", "before_only"):
        if b is None:  # a paired or "before"-only gap has a possession after it
            raise ValueError(f"_row_time: gap class {cls!r} needs the possession after it")
        return float(g.start_s[b]) - _PLACEHOLDER_OFFSET_S, "placeholder"
    if cls == "after_only":
        if b is None:
            raise ValueError(f"_row_time: gap class {cls!r} needs the possession after it")
        return float(g.start_s[b]) - _ORDER_ONLY_OFFSET_S, "order_only"
    if a is None:  # a period-end gap has the possession before it
        raise ValueError(f"_row_time: gap class {cls!r} needs the possession before it")
    t = float(g.end_s[a]) if np.isfinite(g.end_s[a]) else float(g.start_s[a])
    return t + _PERIOD_END_OFFSET_S, "order_only"


def _base_row(g: _Gaps, type_id: int, a: int | None, b: int | None, cls: str, team: str) -> dict:
    t, source = _row_time(g, a, b, cls)
    key = b if b is not None else a
    return {
        "event_id": g.event_id[key],
        "period_id": int(g.period[key]),
        "time_seconds": float(t),
        "time_source": source,
        "team_id": team,
        "player_id": pd.NA,
        "type_id": type_id,
        "bodypart_id": spadlconfig.bodypart_id["foot"],
        "action_provenance": "derived",
        "result_source": "inferred",
    }


def _short_or_crossed(
    short_id: int,
    crossed_id: int,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    success: bool,
    has_first_possession: bool,
) -> int:
    """The 15 m subtype rule of spec section 4.2; a corner with no known side measures to the nearer flag."""
    if not (success and has_first_possession) or not np.isfinite(start[0]) or not np.isfinite(end[0]):
        return crossed_id
    if np.isfinite(start[1]):
        dist = float(np.hypot(end[0] - start[0], end[1] - start[1]))
    else:
        dist = min(float(np.hypot(end[0] - start[0], end[1] - fy)) for fy in (0.0, spadlconfig.field_width))
    return short_id if dist < _SHORT_DISTANCE_M else crossed_id


def restart_subtype(type_id: int, start: tuple[float, float], end: tuple[float, float], *, success: bool) -> int:
    """Re-set a corner or free-kick subtype of a row with a first possession (spec sections 4.2, 4.3 rule 2).

    Examples
    --------
    >>> from silly_kicks.spadl import config as c
    >>> restart_subtype(c.actiontype_id["corner_crossed"], (105.0, 0.0), (100.0, 5.0), success=True) == (
    ...     c.actiontype_id["corner_short"])
    True
    """
    tid = spadlconfig.actiontype_id
    if type_id in (tid["corner_short"], tid["corner_crossed"]):
        pair = (tid["corner_short"], tid["corner_crossed"])
    elif type_id in (tid["freekick_short"], tid["freekick_crossed"]):
        pair = (tid["freekick_short"], tid["freekick_crossed"])
    else:
        return type_id
    return _short_or_crossed(*pair, start, end, success=success, has_first_possession=True)


def infer_restarts(pp: pd.DataFrame, teams: tuple[str, str]) -> pd.DataFrame:
    """One derived row for each corner, free kick, goal kick and throw-in by the count rule (spec section 4.2).

    Parameters
    ----------
    pp : pd.DataFrame
        ``player_possession`` rows in reception order, with ``event_id``, ``period``, ``start_seconds`` (the
        reception), ``time_seconds`` (the possession end), ``team_id`` (str), ``game_interruption_before``,
        ``game_interruption_after``, and the SPADL-frame points ``sx_start``, ``sy_start``, ``sx_end``, ``sy_end``
        in the possession team's attacking frame.
    teams : tuple[str, str]
        The two team ids (home, away).

    Returns
    -------
    pd.DataFrame
        Derived rows with partial SPADL columns plus ``time_source`` (``"placeholder"`` or ``"order_only"``). The
        ``event_id`` is that of ``b``, the possession after the row's gap (the key of spec section 4.18).
    """
    from silly_kicks.tracking._gk_geometry import _CORNER_X, _RULE_POINT

    g = _Gaps(pp, teams)
    tid = spadlconfig.actiontype_id
    half_width = spadlconfig.field_width / 2
    rows: list[dict] = []
    for kind, a, b, cls, marked, is_for in _restart_gaps(pp, _RESTART_KINDS, period_end=False):
        has_fp = cls in ("paired", "before_only")
        anchor = b if has_fp else marked
        team = str(g.team[anchor]) if is_for else g.other(str(g.team[anchor]))
        # A restart that follows an "after"-only restart: the end of `a` belongs to the earlier stoppage.
        after_on_a = _marker_kind(g.gia[a], _ALL_MARKER_KINDS) if a is not None else None
        prev_known = a is not None and not (has_fp and after_on_a is not None and after_on_a[0] != kind)
        if prev_known:
            prev = g.in_frame(cast(int, a), float(g.sx_end[a]), float(g.sy_end[a]), team)
            # a missing end of `a` gives an unknown side (spec section 4.2: NaN start_y), never the y = 68 touchline
            side = np.nan if np.isnan(prev[1]) else (0.0 if prev[1] < half_width else spadlconfig.field_width)
        else:
            prev, side = (np.nan, np.nan), np.nan
        if kind == "corner":
            start = (float(_CORNER_X), side)
        elif kind == "goal_kick":
            start = (float(_RULE_POINT[0]), float(_RULE_POINT[1]))
        elif kind == "throw_in":
            start = (prev[0], side)
        else:
            start = prev
        if has_fp:
            end = g.in_frame(b, float(g.sx_start[b]), float(g.sy_start[b]), team)
            success = bool(g.team[b] == team)
        else:
            end = start
            success = b is not None and g.goal_scorer(b) == team
        if kind == "corner":
            type_id = _short_or_crossed(
                tid["corner_short"], tid["corner_crossed"], start, end, success=success, has_first_possession=has_fp
            )
        elif kind == "free_kick":
            type_id = _short_or_crossed(
                tid["freekick_short"], tid["freekick_crossed"], start, end, success=success, has_first_possession=has_fp
            )
        elif kind == "goal_kick":
            type_id = tid["goalkick"]
        else:
            type_id = tid["throw_in"]
        row = _base_row(g, type_id, a, b, cls, team)
        if kind == "throw_in":
            row["bodypart_id"] = spadlconfig.bodypart_id["other"]
        row.update(
            start_x=float(start[0]),
            start_y=float(start[1]),
            end_x=float(end[0]),
            end_y=float(end[1]),
            result_id=spadlconfig.result_id["success" if success else "fail"],
        )
        rows.append(row)
    return pd.DataFrame(rows)


def infer_penalties(pp: pd.DataFrame, teams: tuple[str, str], match_metadata: dict) -> pd.DataFrame:
    """One derived ``shot_penalty`` row for each penalty marker (spec section 4.17).

    The count rule is that of :func:`infer_restarts`, and a period-end "after" marker counts too: a penalty is
    taken even when time is up (Law 14). The start and the end are the penalty mark of the attacked end; there is
    no taker (``player_id`` ``<NA>``). The result: a paired or "before"-only penalty is ``fail`` (the next
    possession carries the penalty marker, so the ball stayed in play; a rebound goal is the rebound's own row);
    an "after"-only penalty is ``success`` when the team's running score is higher at the next possession than at
    ``a``, else (without the running-score fields) when the next possession carries a goal marker of the team; at
    a period end the next possession is the first of the next period; a penalty at the end of the last period
    with no later event compares the metadata final score with the running score at ``a`` (``fail`` without it).

    The time is that of a restart row: the first reception minus 0.02 s (``"placeholder"``) or the next reception
    minus 0.03 s (``"order_only"``); a period-end penalty takes ``a``'s end plus 0.01 s (``"order_only"``). The time
    step :func:`~silly_kicks.tracking.sync_action_times` measures the kick at the mark where the tracking shows it.
    """
    from silly_kicks.tracking._gk_geometry import _PENALTY_SPOT

    g = _Gaps(pp, teams)
    home = teams[0]
    n = len(pp)
    rows: list[dict] = []
    for _kind, a, b, cls, marked, is_for in _restart_gaps(pp, ("penalty",), period_end=True):
        if cls in ("paired", "before_only"):
            team = str(g.team[b]) if is_for else g.other(str(g.team[b]))
            success = False
        else:
            team = str(g.team[marked]) if is_for else g.other(str(g.team[marked]))
            nxt = b if b is not None else (a + 1 if a + 1 < n else None)
            if nxt is not None:
                before, after = g.score_of(a, team), g.score_of(nxt, team)
                if np.isfinite(before) and np.isfinite(after):
                    success = after > before
                else:
                    success = g.goal_scorer(nxt) == team
            else:
                final = match_metadata.get("home_team_score" if team == home else "away_team_score")
                before = g.score_of(a, team)
                success = final is not None and np.isfinite(before) and float(final) > before
        row = _base_row(g, spadlconfig.actiontype_id["shot_penalty"], a, b, cls, team)
        row.update(
            start_x=float(_PENALTY_SPOT[0]),
            start_y=float(_PENALTY_SPOT[1]),
            end_x=float(_PENALTY_SPOT[0]),
            end_y=float(_PENALTY_SPOT[1]),
            result_id=spadlconfig.result_id["success" if success else "fail"],
        )
        rows.append(row)
    return pd.DataFrame(rows)
