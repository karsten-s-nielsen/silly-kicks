"""DFL event period, time and actor derivation (spec 2026-11-08 tracking-batch section 4.8, W8.1-W8.4).

One period rule and one actor rule, single-sourced: ``parse_dfl_events`` applies them to the XML, and the public
adapters below apply them to any DFL consumer's own bronze (the column contract of W8.3). The rules are exactly the
parser's:

- **Period** (``_dfl_period_rule``): an event's period is the one whose kickoff time is the LATEST kickoff time at or
  before the event time; on equal kickoff times the first recorded wins. A period's kickoff is its first KickOff
  (document order) with a parseable time; a later KickOff of the same period is ignored (``skipped_kickoffs``); a
  KickOff of an unrecognised GameSection is skipped. A KickOff with NO GameSection after a recognised period
  KickOff (document order) is a restart after a goal: never a period start, counted, never warned (owner ruling
  2026-11-12). ``timestamp_seconds`` is ``round((event time - kickoff time).total_seconds(), 4)`` with Python
  ``round``.
- **Actor** (``_dfl_actor_rule``): the first non-empty attribute of ``_PLAYER_ATTR_ORDER[event_type]`` (default
  ``["Player"]``) on the search element -- for a KickOff the FIRST nested ``Play`` whenever one exists (even without
  a ``Player``), else the KickOff itself; for every other type the first child.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ._dfl_tables import (
    _DEFAULT_PLAYER_ATTRS,
    _EVENT_TYPE_PREFIX,
    _NESTED_PREFIX_MAP,
    _PLAYER_ATTR_ORDER,
    _SECTION_TO_PERIOD,
    _to_snake_case,
)

if TYPE_CHECKING:  # annotations only: parse.py imports this module, so a runtime import would be a cycle
    from .parse import MatchInfo

__all__ = [
    "DFL_EVENT_TYPE_PREFIX",
    "DFL_NESTED_PREFIX",
    "DflEventTimeWarning",
    "DflGameSectionWarning",
    "DflPeriodReport",
    "complete_dfl_event_bronze",
    "derive_dfl_event_periods",
    "dfl_actor_columns",
    "dfl_event_column",
    "resolve_dfl_event_actors",
]

#: First-child DFL tag -> bronze column prefix (read-only view of the parser's map).
DFL_EVENT_TYPE_PREFIX: Mapping[str, str] = MappingProxyType(_EVENT_TYPE_PREFIX)
#: Nested DFL tag -> bronze column prefix (read-only view of the parser's map).
DFL_NESTED_PREFIX: Mapping[str, str] = MappingProxyType(_NESTED_PREFIX_MAP)

#: The KickOff actor columns of the contract (W8.3), in fallback order.
_KICKOFF_ACTOR_COLUMNS = ("kickoff_play_present", "kickoff_first_play_player", "kickoff_player")


class DflEventTimeWarning(UserWarning):
    """A DFL event time that cannot be parsed, or cannot be compared with the kickoff times (mixed tz-aware and
    naive). The row gets no period. Not a pre-kickoff row, which is routine and never warned.

    Examples
    --------
    Turn the defect into an error in a strict pipeline::

        import warnings
        from silly_kicks.providers.sportec import DflEventTimeWarning
        warnings.simplefilter("error", DflEventTimeWarning)
    """


class DflGameSectionWarning(UserWarning):
    """A KickOff whose ``GameSection`` is not a recognised DFL section (that KickOff is skipped), or a period start
    that may lack its section: a section-less KickOff before any period started, or restarts with no recognised
    ``secondHalf`` KickOff. A section-less KickOff after a period started is a routine restart and never warns.

    Examples
    --------
    Silence it for a feed known to carry a custom section::

        import warnings
        from silly_kicks.providers.sportec import DflGameSectionWarning
        warnings.simplefilter("ignore", DflGameSectionWarning)
    """


@dataclass(frozen=True)
class DflPeriodReport:
    """What :func:`derive_dfl_event_periods` resolved and what it could not.

    Examples
    --------
    >>> report = DflPeriodReport({}, 0, 0, 0, (), ())
    >>> report.n_pre_kickoff, report.n_restart_kickoffs
    (0, 0)

    ``n_restart_kickoffs`` counts the section-less KickOffs after a period started (restarts after a goal).
    """

    kickoff_times: dict[int, datetime] = field(default_factory=dict)
    n_pre_kickoff: int = 0
    n_unparseable_time: int = 0
    n_incomparable_time: int = 0
    unrecognized_sections: tuple[str, ...] = ()
    skipped_kickoffs: tuple[int, ...] = ()
    n_restart_kickoffs: int = 0


# --------------------------------------------------------------------------------------------------------------------
# The rules (pure; shared with the parser)
# --------------------------------------------------------------------------------------------------------------------


def _parse_event_time(value) -> datetime | None:
    """The parser's time rule: ``datetime.fromisoformat``, tz-aware -> UTC, naive kept naive; None when missing or
    unparseable (the running Python's ``fromisoformat`` decides: 3.10 rejects ``Z`` and 7+ fractional digits)."""
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    return dt.astimezone(timezone.utc) if dt.tzinfo is not None else dt


def _collect_kickoffs(
    event_types: Sequence, sections: Sequence, times: Sequence[datetime | None]
) -> tuple[list[tuple[int, datetime]], tuple[int, ...], list[str], int]:
    """``(kickoffs, skipped_kickoffs, unrecognised sections, restart count)`` in document order: per recognised
    period, its first KickOff with a parseable time. A section-less KickOff after a recognised period KickOff is a
    restart (counted); one before any is a period start without its section (``""`` in the unrecognised list)."""
    kickoffs: list[tuple[int, datetime]] = []
    seen: set[int] = set()
    skipped: list[int] = []
    unrecognized: list[str] = []
    started = False
    n_restart = 0
    for event_type, section, t in zip(event_types, sections, times, strict=True):
        if event_type != "KickOff":
            continue
        section_name = section if isinstance(section, str) else ""
        period = _SECTION_TO_PERIOD.get(section_name)
        if period is None:
            if section_name == "" and started:
                n_restart += 1  # a restart after a goal: never a period start
            else:
                unrecognized.append(section_name)
            continue
        started = True
        if t is None:
            continue  # an unparseable KickOff time: skipped, so a later KickOff of the period can win
        if period in seen:
            skipped.append(period)
            continue
        seen.add(period)
        kickoffs.append((period, t))
    return kickoffs, tuple(skipped), unrecognized, n_restart


def _second_half_unseen(kickoffs: list[tuple[int, datetime]], n_restart: int) -> bool:
    """Restarts but no recognised ``secondHalf`` KickOff: one section-less KickOff may be the second-half start."""
    return n_restart > 0 and 2 not in {period for period, _ in kickoffs}


def _period_outcome(
    t: datetime | None, kickoffs: list[tuple[int, datetime]]
) -> tuple[int | None, datetime | None, str]:
    """``(period, kickoff time, outcome)``; outcome is ``"ok"``, ``"unparseable"``, ``"incomparable"`` or
    ``"pre_kickoff"``. The latest kickoff at or before ``t`` wins; on equal kickoff times the first recorded."""
    if t is None:
        return None, None, "unparseable"
    best_period: int | None = None
    best_start: datetime | None = None
    try:
        for period, start in kickoffs:
            if t >= start and (best_start is None or start > best_start):
                best_period, best_start = period, start
    except TypeError:  # a tz-aware time against a naive one (owner ruling R3)
        return None, None, "incomparable"
    if best_period is None:
        return None, None, "pre_kickoff"
    return best_period, best_start, "ok"


def _dfl_period_rule(
    times: list[datetime | None], kickoffs: list[tuple[int, datetime]]
) -> list[tuple[int | None, datetime | None]]:
    """The period and kickoff time of each event time (``(None, None)`` when it has none)."""
    return [_period_outcome(t, kickoffs)[:2] for t in times]


def _timestamp_seconds(t: datetime, start: datetime) -> float:
    """Period-relative seconds, rounded to 4 decimals with Python ``round`` (the parser's arithmetic)."""
    return round((t - start).total_seconds(), 4)


def _dfl_actor_rule(event_type: str, actor_attrs: Mapping[str, str]) -> str:
    """The acting player's id: the first non-empty attribute of the type's ``_PLAYER_ATTR_ORDER``; ``""`` if none."""
    for attr in _PLAYER_ATTR_ORDER.get(event_type, _DEFAULT_PLAYER_ATTRS):
        pid = actor_attrs.get(attr, "")
        if pid:
            return pid
    return ""


# --------------------------------------------------------------------------------------------------------------------
# The column contract (W8.3)
# --------------------------------------------------------------------------------------------------------------------


def dfl_event_column(event_type: str, attribute: str, *, nested: str | None = None) -> str:
    """The bronze column of a DFL attribute: ``{prefix}_{snake(attribute)}``.

    The prefix is the event type's (``DFL_EVENT_TYPE_PREFIX``, else ``snake(event_type)``), or with ``nested`` the
    nested tag's (``DFL_NESTED_PREFIX``, else ``snake(nested)``) -- the parser's bronze naming rule, defined for
    every DFL tag so a consumer bronze can carry it.

    Examples
    --------
    >>> from silly_kicks.providers.sportec import dfl_event_column
    >>> dfl_event_column("TacklingGame", "Winner")
    'tackle_winner'
    >>> dfl_event_column("KickOff", "Player", nested="Play")
    'play_player'
    """
    if nested is not None:
        prefix = _NESTED_PREFIX_MAP.get(nested, _to_snake_case(nested))
    else:
        prefix = _EVENT_TYPE_PREFIX.get(event_type, _to_snake_case(event_type))
    return f"{prefix}_{_to_snake_case(attribute)}"


def dfl_actor_columns(event_type: str) -> tuple[str, ...]:
    """The ordered actor columns of one DFL event type.

    For a KickOff: ``kickoff_play_present`` (boolean; required whenever events hold a KickOff),
    ``kickoff_first_play_player`` (the FIRST nested ``Play``'s ``Player``) and ``kickoff_player`` (the KickOff's own
    ``Player``, read only when no nested ``Play`` exists). They are contract-only: the parser bronze's ``play_*``
    columns hold the LAST nested ``Play``. For every other type: one column per ``_PLAYER_ATTR_ORDER`` attribute.

    Examples
    --------
    >>> from silly_kicks.providers.sportec import dfl_actor_columns
    >>> dfl_actor_columns("TacklingGame")
    ('tackle_winner', 'tackle_player')
    >>> dfl_actor_columns("KickOff")
    ('kickoff_play_present', 'kickoff_first_play_player', 'kickoff_player')
    """
    if event_type == "KickOff":
        return _KICKOFF_ACTOR_COLUMNS
    return tuple(dfl_event_column(event_type, a) for a in _PLAYER_ATTR_ORDER.get(event_type, _DEFAULT_PLAYER_ATTRS))


# --------------------------------------------------------------------------------------------------------------------
# The public adapters (W8.1, W8.2, W8.4)
# --------------------------------------------------------------------------------------------------------------------


def _times_of(event_time: pd.Series) -> list[datetime | None]:
    """Each value as the parser's ``datetime`` (strings: :func:`_parse_event_time`; datetime64: floored to whole
    microseconds, tz-aware -> UTC, naive kept naive; a missing value is None)."""
    if pd.api.types.is_datetime64_any_dtype(event_time.dtype):
        s = event_time.dt.floor("us")
        if s.dt.tz is not None:
            s = s.dt.tz_convert("UTC")
        out: list[datetime | None] = []
        for stamp in s.astype(object).tolist():  # Timestamps (whole microseconds after the floor) or NaT
            if pd.isna(stamp):
                out.append(None)
                continue
            t = stamp.to_pydatetime()
            out.append(t.astimezone(timezone.utc) if t.tzinfo is not None else t)
        return out
    return [_parse_event_time(v) for v in event_time.tolist()]


def derive_dfl_event_periods(
    event_time: pd.Series, event_type: pd.Series, kickoff_game_section: pd.Series
) -> tuple[pd.DataFrame, DflPeriodReport]:
    """The parser's period and ``timestamp_seconds`` for a DFL consumer's own events.

    The three Series are index-aligned and in DOCUMENT order (the kickoffs are recorded in that order).
    ``event_time`` holds ISO-8601 strings, parsed as the parser does (``datetime.fromisoformat``, tz-aware -> UTC;
    the running Python decides what parses: Python 3.10 rejects a ``Z`` suffix and more than 6 fractional digits, and
    such a row is an unparseable time), or a datetime64 Series (tz-aware -> UTC, naive kept naive). A datetime64 value
    is floored to whole microseconds first; the dropped part is below the 4-decimal rounding of
    ``timestamp_seconds``. ``kickoff_game_section`` is read on KickOff rows only.

    Returns the columns ``period`` (``Int64``) and ``timestamp_seconds`` (``float64``), index-aligned, and a
    :class:`DflPeriodReport`. A row with no period gets ``<NA>`` / NaN and is counted: pre-kickoff (routine, never
    warned), an unparseable time, or a time incomparable with the kickoffs (mixed tz-aware and naive). The two defects
    warn once per call (:class:`DflEventTimeWarning`). An unrecognised GameSection, or a period start that may lack
    its section, warns once (:class:`DflGameSectionWarning`); a section-less KickOff after a period started is a
    restart, counted in ``n_restart_kickoffs`` and never warned. With no KickOff at all every row is ``<NA>``.

    Examples
    --------
    >>> import pandas as pd
    >>> from silly_kicks.providers.sportec import derive_dfl_event_periods
    >>> times = pd.Series(["2024-03-09T15:00:00+00:00", "2024-03-09T15:10:00.5+00:00"])
    >>> types = pd.Series(["KickOff", "Play"])
    >>> sections = pd.Series(["firstHalf", None])
    >>> out, report = derive_dfl_event_periods(times, types, sections)
    >>> out["period"].tolist(), out["timestamp_seconds"].tolist()
    ([1, 1], [0.0, 600.5])
    """
    times = _times_of(event_time)
    kickoffs, skipped, unrecognized, n_restart = _collect_kickoffs(
        event_type.tolist(), kickoff_game_section.tolist(), times
    )
    periods: list = []
    stamps: list[float] = []
    counts = {"unparseable": 0, "incomparable": 0, "pre_kickoff": 0}
    for t in times:
        period, start, outcome = _period_outcome(t, kickoffs)
        if outcome == "ok":
            periods.append(period)
            stamps.append(_timestamp_seconds(t, start))  # type: ignore[arg-type]
        else:
            counts[outcome] += 1
            periods.append(pd.NA)
            stamps.append(np.nan)
    report = DflPeriodReport(
        kickoff_times=dict(kickoffs),
        n_pre_kickoff=counts["pre_kickoff"],
        n_unparseable_time=counts["unparseable"],
        n_incomparable_time=counts["incomparable"],
        unrecognized_sections=tuple(unrecognized),
        skipped_kickoffs=skipped,
        n_restart_kickoffs=n_restart,
    )
    if counts["unparseable"] or counts["incomparable"]:
        warnings.warn(
            f"derive_dfl_event_periods: {counts['unparseable']} event(s) with an unparseable time and "
            f"{counts['incomparable']} with a time incomparable with the kickoffs got no period",
            DflEventTimeWarning,
            stacklevel=2,
        )
    problems = []
    if unrecognized:
        problems.append(f"KickOff(s) with an unrecognised GameSection skipped: {sorted(set(unrecognized))}")
    if _second_half_unseen(kickoffs, n_restart):
        problems.append(
            f"no recognised secondHalf KickOff, but {n_restart} section-less KickOff(s): one may be the "
            "second-half start without its GameSection"
        )
    if problems:
        warnings.warn(f"derive_dfl_event_periods: {'; '.join(problems)}", DflGameSectionWarning, stacklevel=2)
    out = pd.DataFrame(
        {
            "period": pd.array(periods, dtype="Int64"),
            "timestamp_seconds": np.asarray(stamps, dtype=np.float64),
        },
        index=event_time.index,
    )
    return out, report


def _cell(events: pd.DataFrame, column: str, i: int) -> str:
    """A contract cell as the parser's ``get(attr, "")``: a missing column or an empty value is ``""``."""
    if column not in events.columns:
        return ""
    value = events[column].iloc[i]
    return value if isinstance(value, str) and value else ""


def resolve_dfl_event_actors(events: pd.DataFrame, match_info: MatchInfo) -> pd.DataFrame:
    """The parser's ``player_id`` / ``team`` / ``team_id_native`` for a DFL consumer's own events.

    ``events`` carries ``event_type`` and the actor columns of the contract (:func:`dfl_actor_columns`). A missing
    column or an empty value is an absent attribute. ``team`` is ``"home"`` / ``"away"`` / ``"unknown"`` through
    ``match_info.player_team_map``; ``team_id_native`` is the home or away club id, else None. A KickOff row needs
    ``kickoff_play_present`` (a missing column raises ``ValueError``). The silly-kicks parser bronze already carries
    these columns and never needs this resolver.

    Examples
    --------
    Resolve the actors of a consumer bronze::

        from silly_kicks.providers.sportec import resolve_dfl_event_actors
        actors = resolve_dfl_event_actors(events, match_info)
    """
    types = events["event_type"].tolist()
    if "KickOff" in types and "kickoff_play_present" not in events.columns:
        raise ValueError(
            "resolve_dfl_event_actors: events hold a KickOff but no 'kickoff_play_present' column "
            "(the DFL column contract, dfl_actor_columns('KickOff'))"
        )
    ids, teams, natives = [], [], []
    for i, event_type in enumerate(types):
        if event_type == "KickOff":
            present = events["kickoff_play_present"].iloc[i]
            column = "kickoff_first_play_player" if (not pd.isna(present) and bool(present)) else "kickoff_player"
            attrs = {"Player": _cell(events, column, i)}
        else:
            order = _PLAYER_ATTR_ORDER.get(event_type, _DEFAULT_PLAYER_ATTRS)
            attrs = {a: _cell(events, dfl_event_column(event_type, a), i) for a in order}
        pid = _dfl_actor_rule(event_type, attrs)
        team = match_info.player_team_map.get(pid, "unknown") if pid else "unknown"
        ids.append(pid)
        teams.append(team)
        natives.append(
            match_info.home_team_id if team == "home" else match_info.away_team_id if team == "away" else None
        )
    return pd.DataFrame(
        {
            "player_id": pd.Series(ids, index=events.index, dtype=object),
            "team": pd.Series(teams, index=events.index, dtype=object),
            "team_id_native": pd.Series(natives, index=events.index, dtype=object),
        }
    )


def complete_dfl_event_bronze(events: pd.DataFrame, match_info: MatchInfo) -> tuple[pd.DataFrame, DflPeriodReport]:
    """A consumer bronze completed for ``shape_events_to_native``: ``period``, ``timestamp_seconds``, ``player_id``,
    ``team``, ``team_id_native``, the match-level ``home_team_id_native`` / ``away_team_id_native``, and
    ``competition_native_id`` / ``season_native_id`` where ``match_info`` has them.

    ``events`` has ``event_time``, ``event_type``, ``kickoff_game_section`` and the contract's actor columns, in
    document order. Rows without a period are KEPT; the caller decides. The parser drops them::

        completed = completed[completed["period"].notna()]

    Examples
    --------
    Complete a consumer bronze and drop the rows the parser drops::

        from silly_kicks.providers.sportec import complete_dfl_event_bronze
        completed, report = complete_dfl_event_bronze(events, match_info)
        completed = completed[completed["period"].notna()]
    """
    periods, report = derive_dfl_event_periods(
        events["event_time"], events["event_type"], events["kickoff_game_section"]
    )
    actors = resolve_dfl_event_actors(events, match_info)
    out = events.copy()
    for col in periods.columns:
        out[col] = periods[col]
    for col in actors.columns:
        out[col] = actors[col]
    out["home_team_id_native"] = match_info.home_team_id
    out["away_team_id_native"] = match_info.away_team_id
    if match_info.competition_id:
        out["competition_native_id"] = match_info.competition_id
    if match_info.season_id:
        out["season_native_id"] = match_info.season_id
    return out, report
