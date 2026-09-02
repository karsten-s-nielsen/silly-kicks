"""StatsBomb events-only keeper-appearance extractor (TF-59 PR1, spec §5.5).

Produces the :func:`~silly_kicks.keeper_identity.validate_keeper_appearances` port -- one row per
keeper on-pitch interval PER PERIOD -- from StatsBomb RAW events ALONE (no separate ``lineups``
artifact; controller ruling). Four event types drive it:

* ``Starting XI`` -- one per team; the ``tactics.lineup`` entry whose ``position.name ==
  "Goalkeeper"`` is that team's opening keeper. It starts an on-pitch SEGMENT at ``(period 1,
  0.0s)``.
* ``Substitution`` -- a keeper substitution is one whose OUTGOING ``player.id`` equals the team's
  CURRENT keeper (tracked forward from the Starting XI). It ENDS the outgoing keeper's segment at the
  event's ``(period, period-relative time)`` and STARTS the incoming keeper's segment there. A
  non-keeper substitution is ignored.
* ``Player Off`` -- a ``Player Off`` of the CURRENT keeper (a red card / injury with NO
  ``Substitution``) ENDS that keeper's segment at the event time. No replacement is named on a Player
  Off, so none is fabricated; the emergency keeper is picked up by the next ``Tactical Shift``.
* ``Tactical Shift`` -- carries a full ``tactics.lineup`` like ``Starting XI``; its ``Goalkeeper``
  slot names the team's keeper AT THAT MOMENT. A slot naming a keeper the team is not currently
  attributing is an EMERGENCY keeper change (an outfielder donning the gloves after a Player Off, or a
  reshape after a keeper sub) -> it closes any open segment and opens the emergency keeper's
  (``source="emergency_keeper"``). An UNCHANGED slot is a no-op, so a normal formation change stays
  byte-identical.

**Per-period decomposition (the port contract).** The port keys appearances per ``(game,
period_id, team)`` and the consumer looks up the covering interval WITHIN a single period, so each
keeper's ``(from_period, from_time) -> (to_period, to_time)`` SEGMENT is decomposed into one row per
period it spans: the entry period starts at ``from_time`` (else ``0.0``) and the exit period ends at
``to_time`` (else ``+inf``, i.e. "to the period end"). A period slice with ``start >= end`` (e.g. a
keeper subbed at the START of a period has no tenure in it) is dropped. This is exactly the shape
SkillCorner's ``playing_time.by_period[]`` emits, and it means an unsubbed keeper in a two-period
match gets TWO rows (one per period), a half-time keeper change splits cleanly at the period
boundary, and extra-time periods are handled by construction (the segment simply spans more periods).

``timestamp`` (``"HH:MM:SS.mmm"``) is PERIOD-RELATIVE (it resets each period; ADR-017), matching the
port's period-relative time base, so it is parsed straight to seconds with no cross-period offset.

A goal-kick-taker / tracking-frame cross-check (spec §5.4, the consumer) supplies finer resolution
when frames are available -- e.g. the exact frame an emergency keeper first appears, ahead of the
``Tactical Shift`` that names them in the events feed.

Imports are deliberately confined to :mod:`silly_kicks.keeper_identity` (the port) +
:mod:`silly_kicks.id_compat` (ADR-019 dtype-safe ids) + pandas/stdlib -- NOT
:mod:`silly_kicks.tracking`.

See NOTICE for the StatsBomb Public Data License (non-commercial).
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from silly_kicks.id_compat import canonical_id, same_id
from silly_kicks.keeper_identity import (
    KeeperSegment,
    build_keeper_appearances_from_segments,
)

__all__ = ["extract_keeper_appearances"]

#: StatsBomb ``position.name`` for the goalkeeper lineup slot.
_GOALKEEPER_POSITION = "Goalkeeper"

#: Fallback period for an event missing its ``period`` key. Single-sourced so the Starting XI seed,
#: the substitution ordering and the substitution period all agree (M3: no split default).
_DEFAULT_PERIOD = 1


def _timestamp_to_seconds(timestamp: str) -> float:
    """Parse a StatsBomb ``"HH:MM:SS.mmm"`` period-relative timestamp to float seconds.

    Times reset each period, so the result is the seconds-into-period the port stores (ADR-017); no
    period-start offset is applied.
    """
    hours, minutes, seconds = timestamp.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _event_period(event: dict) -> int:
    """The event's integer ``period``, defaulting to :data:`_DEFAULT_PERIOD` when absent."""
    return int(event.get("period", _DEFAULT_PERIOD))


def _lineup_keeper_id(event: dict) -> object | None:
    """The ``player.id`` of the ``Goalkeeper`` slot in an event's ``tactics.lineup``, or ``None``.

    Both ``Starting XI`` and ``Tactical Shift`` carry a full ``tactics.lineup`` whose ``Goalkeeper``
    position slot names the team's keeper AT THAT MOMENT -- so this reads the opening keeper from a
    ``Starting XI`` AND the current (possibly emergency) keeper from a later ``Tactical Shift``. Returns
    ``None`` when the lineup names no goalkeeper (a malformed / partial event) -- the caller then seeds
    nothing / keeps the current keeper rather than crashing the extractor.
    """
    lineup = event.get("tactics", {}).get("lineup", [])
    for entry in lineup:
        if entry.get("position", {}).get("name") == _GOALKEEPER_POSITION:
            return entry.get("player", {}).get("id")
    return None


def extract_keeper_appearances(events: list[dict], *, game_id: object) -> pd.DataFrame:
    """Extract per-period keeper on-pitch intervals from StatsBomb RAW events (the TF-59 port, spec §5.5).

    Builds an on-pitch SEGMENT per keeper tenure -- opened by a ``Starting XI`` Goalkeeper entry or a
    keeper ``Substitution``, closed by the next keeper ``Substitution`` (or left open to the last
    period) -- then DECOMPOSES each segment into one row per period it spans (see the module
    docstring). Returns a validated
    :func:`~silly_kicks.keeper_identity.validate_keeper_appearances` frame (columns in
    :data:`~silly_kicks.keeper_identity.KEEPER_APPEARANCE_COLUMNS` order) with keeper rows only.

    Ids (``game_id`` passed by the caller; team / player ids read from the events) are consumed
    AS-IS -- StatsBomb ids are ints, and the port's id columns are ``object``, so they are stored
    un-coerced; the ``current keeper`` bookkeeping compares ids through :func:`id_compat.same_id`
    (ADR-019), never raw ``==``. PURE -- ``events`` is never mutated.

    Examples
    --------
    Two opening keepers, one period, no substitution -> two intervals open to the period end:

    >>> import numpy as np
    >>> from silly_kicks.providers.statsbomb.appearances import extract_keeper_appearances
    >>> events = [
    ...     {"type": {"name": "Starting XI"}, "period": 1, "timestamp": "00:00:00.000",
    ...      "team": {"id": 10},
    ...      "tactics": {"lineup": [{"player": {"id": 901},
    ...                              "position": {"name": "Goalkeeper"}}]}},
    ...     {"type": {"name": "Starting XI"}, "period": 1, "timestamp": "00:00:00.000",
    ...      "team": {"id": 20},
    ...      "tactics": {"lineup": [{"player": {"id": 902},
    ...                              "position": {"name": "Goalkeeper"}}]}},
    ... ]
    >>> ap = extract_keeper_appearances(events, game_id="g1")
    >>> len(ap)
    2
    >>> sorted(ap["source"].unique().tolist())
    ['starting_xi']
    >>> bool(np.isinf(ap["end_time_seconds"]).all())
    True

    See NOTICE for full bibliographic citations.
    """
    # Periods present across the WHOLE stream (not just Starting XI / subs), so an unsubbed keeper is
    # decomposed across every period the match actually has -- incl. extra time. `_DEFAULT_PERIOD` is
    # the degenerate single-period fallback when no event carries a period.
    periods = sorted({_event_period(e) for e in events if e.get("period") is not None})
    if not periods:
        periods = [_DEFAULT_PERIOD]
    last_period = periods[-1]

    # A keeper tenure is an on-pitch SEGMENT: (from_period, from_time) -> (to_period, to_time). Every
    # segment opens with `to = (last_period, +inf)` ("still on at match end") and a later keeper sub
    # overwrites the outgoing segment's `to`.
    segments: list[dict[str, Any]] = []
    #: canonical team_id -> the currently-open (mutable) segment dict for that team's keeper.
    open_segment: dict[object, dict[str, Any]] = {}
    #: canonical team_id -> the raw id of the team's CURRENT keeper (tracked forward from Starting XI).
    current_keeper: dict[object, object] = {}

    # 1) Starting XI -> a starter segment per team, opened at (period 1, 0.0).
    for event in events:
        if event.get("type", {}).get("name") != "Starting XI":
            continue
        team_id = event.get("team", {}).get("id")
        team_key = canonical_id(team_id)
        if team_key is pd.NA or team_key in open_segment:
            continue  # no team id, or this team already seeded (only the first Starting XI counts)
        keeper_id = _lineup_keeper_id(event)
        if keeper_id is None:
            continue  # lineup names no goalkeeper -> seed nothing (never fabricate a keeper)
        segment: dict[str, Any] = {
            "team_id": team_id,
            "player_id": keeper_id,
            "from_period": _event_period(event),
            "from_time": 0.0,
            "to_period": last_period,
            "to_time": math.inf,
            "source": "starting_xi",
        }
        segments.append(segment)
        open_segment[team_key] = segment
        current_keeper[team_key] = keeper_id

    # 2) Keeper-tenure events chronologically. `timestamp` is period-relative, so (period, seconds) IS
    #    chronological order; a keeper `Substitution`, a `Tactical Shift` naming a new keeper, and a
    #    keeper `Player Off` each open or close a tenure. Period splitting is done entirely by the
    #    decomposition in step 3 (no same-row closing/degenerate-skip logic here).
    def _tenure_sort_key(event: dict) -> tuple[int, float]:
        return (_event_period(event), _timestamp_to_seconds(event.get("timestamp", "00:00:00.000")))

    def _end_current(team_key: object, period: int, time: float) -> None:
        """Close the team's open keeper segment at ``(period, time)`` (the keeper left the pitch)."""
        seg = open_segment.pop(team_key, None)
        if seg is not None:
            seg["to_period"] = period
            seg["to_time"] = time
        current_keeper.pop(team_key, None)

    def _start_keeper(
        team_id: object, team_key: object, keeper_id: object, period: int, time: float, source: str
    ) -> None:
        """Open a new keeper segment at ``(period, time)``, open to the last period until superseded."""
        segment: dict[str, Any] = {
            "team_id": team_id,
            "player_id": keeper_id,
            "from_period": period,
            "from_time": time,
            "to_period": last_period,
            "to_time": math.inf,
            "source": source,
        }
        segments.append(segment)
        open_segment[team_key] = segment
        current_keeper[team_key] = keeper_id

    _tenure_types = {"Substitution", "Tactical Shift", "Player Off"}
    tenure_events = [e for e in events if e.get("type", {}).get("name") in _tenure_types]
    for event in sorted(tenure_events, key=_tenure_sort_key):
        etype = event.get("type", {}).get("name")
        team_id = event.get("team", {}).get("id")
        team_key = canonical_id(team_id)
        if team_key is pd.NA:
            continue  # no team id -> not attributable
        period = _event_period(event)
        time = _timestamp_to_seconds(event.get("timestamp", "00:00:00.000"))

        if etype == "Substitution":
            # A KEEPER sub: the outgoing player IS the team's current keeper. Ends the tenure and (when a
            # replacement is named, as a formal sub always is) opens the replacement's there.
            if team_key not in current_keeper or not same_id(
                event.get("player", {}).get("id"), current_keeper[team_key]
            ):
                continue  # a non-keeper substitution, or a team with no tracked keeper
            replacement_id = event.get("substitution", {}).get("replacement", {}).get("id")
            _end_current(team_key, period, time)
            if replacement_id is not None:
                _start_keeper(team_id, team_key, replacement_id, period, time, "sub_events")

        elif etype == "Player Off":
            # A `Player Off` of the CURRENT keeper (a red card / injury with NO `Substitution`) ends the
            # tenure. No replacement is named on a Player Off, so none is fabricated -- the emergency
            # keeper is identified by a subsequent `Tactical Shift`, if the feed carries one.
            if team_key in current_keeper and same_id(event.get("player", {}).get("id"), current_keeper[team_key]):
                _end_current(team_key, period, time)

        elif etype == "Tactical Shift":
            # A `Tactical Shift` carries a full tactics.lineup; its `Goalkeeper` slot names the team's
            # keeper at that moment. A slot naming a keeper the team is NOT currently attributing is an
            # emergency keeper change (an outfielder donning the gloves after a Player Off, or a reshape
            # after a keeper sub) -> close any open segment and open the emergency keeper's there. An
            # UNCHANGED slot is a no-op, so a normal formation change stays byte-identical.
            new_keeper = _lineup_keeper_id(event)
            if new_keeper is None:
                continue  # lineup names no goalkeeper -> no keeper information, keep the current one
            if team_key in current_keeper and same_id(new_keeper, current_keeper[team_key]):
                continue  # keeper unchanged
            _end_current(team_key, period, time)
            _start_keeper(team_id, team_key, new_keeper, period, time, "emergency_keeper")

    # 3) Hand the finished tenures to the shared per-period decomposition (spec §5.5, Part A). Each
    #    mutable segment dict (its `to_*` may have been overwritten by a later keeper sub in step 2)
    #    becomes an immutable `KeeperSegment`; the builder splits it into one row per period it spans,
    #    dropping any `start >= end` slice, and returns the validated port.
    keeper_segments = [
        KeeperSegment(
            team_id=segment["team_id"],
            player_id=segment["player_id"],
            source=segment["source"],
            start_period=segment["from_period"],
            start_time=segment["from_time"],
            end_period=segment["to_period"],
            end_time=segment["to_time"],
        )
        for segment in segments
    ]
    return build_keeper_appearances_from_segments(keeper_segments, periods, game_id=game_id)
