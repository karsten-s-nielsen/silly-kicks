"""Gradient Sports keeper-appearance extractor (TF-59 PR1, spec §5.5).

Produces the :func:`~silly_kicks.keeper_identity.validate_keeper_appearances` port -- one row per
keeper on-pitch interval PER PERIOD -- from a Gradient Sports RAW ``events`` list + ``roster`` list
(the shapes the owner-tier pining feed carries, verified on real WC2022 data 2026-09-01). Two signals
drive it:

* **Starting keepers.** The ``roster`` names each player's ``positionGroupType``; a value of ``"GK"``
  / ``"GOALKEEPER"`` (case-insensitive) is a keeper. A team carries MULTIPLE roster keepers (real
  WC2022 rosters list THREE GKs per team, and the substitutions are outfielders -- so NO GK is ever a
  ``playerOnId``), so "the GK not seen as a ``playerOnId``" does NOT distinguish the starter from an
  unused bench keeper. The STARTER is therefore resolved from the ACTION signal (the
  ``_earliest_acting`` idiom sportec uses): among the team GKs that are NOT introduced by a
  substitution, the starter is the one with the EARLIEST on-ball action -- the smallest
  ``(period, startGameClock)`` over the ordinary (non-``SUB``) ``gameEvents`` rows whose
  ``playerId`` is that GK. A roster GK with NO on-ball action is an unused bench keeper and is
  EXCLUDED (never seeds a starter); a team with no acting, non-subbed-on GK seeds none (honest gap).
  Each resolved starter opens an on-pitch SEGMENT at ``(first period, 0.0s)`` (the keeper is on from
  kickoff; the earliest-action is only the SIGNAL identifying WHICH GK started, not when they came
  on).

* **Keeper substitution.** A ``gameEvents`` envelope with ``gameEventType == "SUB"`` carries
  ``playerOffId`` (outgoing) / ``playerOnId`` (incoming) + ``startGameClock`` (period-relative
  seconds) + ``period``. It is a KEEPER change iff the outgoing player is the team's CURRENT keeper
  (tracked forward from the starter); it then ENDS the outgoing keeper's segment at
  ``(period, startGameClock)`` and OPENS the incoming (``playerOnId``, ``source="sub_events"``) there.
  A non-keeper substitution (the outgoing player is not the current keeper) is ignored. **The SUB's
  ``teamId`` may be null** -- the sub's team is then derived from the OUTGOING player's roster team.

``startGameClock`` is PERIOD-RELATIVE (it resets each period; ADR-017), matching the port's time
base, so it is used straight (no cross-period offset). ``periods`` = the sorted distinct
``gameEvents["period"]`` present in the events, so extra-time is handled by construction. Still-on
keepers end at ``(last period, +inf)``. The per-period decomposition is the shared
:func:`~silly_kicks.keeper_identity.build_keeper_appearances_from_segments`.

Gradient Sports ids are INTEGERS; the port's id columns are ``object``, so they are stored un-coerced
and every id comparison routes through :mod:`silly_kicks.id_compat` (ADR-019). The roster is read
tolerantly -- nested (``it["player"]["id"]`` / ``it["team"]["id"]``) OR flat (``it["playerId"]`` /
``it["teamId"]``). Imports are confined to :mod:`silly_kicks.keeper_identity` +
:mod:`silly_kicks.id_compat` + pandas/stdlib -- NOT :mod:`silly_kicks.tracking`.

**Starting-keeper-selection rule + roster-shape tolerance (reported).** The starter is the team GK
that (a) is NOT introduced as a ``SUB`` ``playerOnId`` and (b) has the EARLIEST on-ball action in the
event stream; a non-subbed-on GK with no action is an unused bench keeper and is excluded. If (a
degenerate quirk) more than one non-subbed-on GK still ties on earliest action, the tie is broken by
the smallest canonical id (last resort, noted). A team with no acting, non-subbed-on GK seeds no
starter (an honest gap, never a fabricated keeper). The roster tolerates both the nested and flat id
shapes and accepts ``positionGroupType`` in ``{"GK", "GOALKEEPER"}`` case-insensitively.

See NOTICE for full bibliographic citations.
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

#: Gradient Sports ``gameEventType`` for a substitution -- carries ``playerOffId`` / ``playerOnId``.
_SUB_EVENT_TYPE = "SUB"

#: ``positionGroupType`` values (upper-cased) that mark a goalkeeper on the roster.
_GK_POSITION_GROUPS = frozenset({"GK", "GOALKEEPER"})

#: The match's first period -- the starter is on from here (no explicit "starting XI" event exists in
#: the GS stream, unlike StatsBomb). It is ALWAYS spanned (forced into ``periods``), so a
#: substitution-only event stream that never witnesses period 1 still emits the starter's period-1
#: interval. Single-sourced so the starter seed and the per-period decomposition agree.
_STARTER_PERIOD = 1


def _player_id(item: dict) -> object | None:
    """The roster record's player id -- nested ``item["player"]["id"]`` or flat ``item["playerId"]``."""
    player = item.get("player")
    if isinstance(player, dict) and player.get("id") is not None:
        return player["id"]
    return item.get("playerId")


def _team_id(item: dict) -> object | None:
    """The roster record's team id -- nested ``item["team"]["id"]`` or flat ``item["teamId"]``."""
    team = item.get("team")
    if isinstance(team, dict) and team.get("id") is not None:
        return team["id"]
    return item.get("teamId")


def _is_goalkeeper(item: dict) -> bool:
    """True iff the roster record's ``positionGroupType`` names a keeper (case-insensitive)."""
    pos = item.get("positionGroupType")
    return isinstance(pos, str) and pos.strip().upper() in _GK_POSITION_GROUPS


def _roster_index(roster: list[dict]) -> tuple[dict[object, dict[str, Any]], dict[object, object]]:
    """Index the roster into ``(gk_by_team, player_team)``.

    ``gk_by_team`` maps a CANONICAL team key -> ``{"raw_team": <raw team id>, "gks": [<raw gk id>...]}``
    (the raw team id is carried through for the stored ``team_id``; the key is canonical so the sub
    events -- keyed on the same team id -- line up, ADR-019). ``player_team`` maps EVERY canonical
    player id -> its raw team id, so a ``SUB`` with a null ``teamId`` can derive its team from the
    outgoing player. A record with no resolvable player id is skipped.
    """
    gk_by_team: dict[object, dict[str, Any]] = {}
    player_team: dict[object, object] = {}
    for item in roster:
        pid = _player_id(item)
        if pid is None:
            continue
        tid = _team_id(item)
        if tid is not None:
            player_team[canonical_id(pid)] = tid
        if _is_goalkeeper(item) and tid is not None:
            entry = gk_by_team.setdefault(canonical_id(tid), {"raw_team": tid, "gks": []})
            entry["gks"].append(pid)
    return gk_by_team, player_team


def _sub_events(events: list[dict], player_team: dict[object, object]) -> list[dict[str, Any]]:
    """The chronologically-sortable ``SUB`` gameEvents, normalized.

    Each is ``{"period", "time", "team_raw", "off", "on"}``. A SUB missing its outgoing/incoming
    player or its period is skipped (it cannot be resolved / dated); a null ``teamId`` derives the
    team from the OUTGOING player's roster team (``None`` when the outgoing player is unknown -- such a
    sub cannot be attributed and is filtered by the walk). ``startGameClock`` is period-relative
    seconds; a missing one defaults to ``0.0`` (period start) so the change is still placed.
    """
    subs: list[dict[str, Any]] = []
    for event in events:
        ge = event.get("gameEvents") or {}
        if ge.get("gameEventType") != _SUB_EVENT_TYPE:
            continue
        off = ge.get("playerOffId")
        on = ge.get("playerOnId")
        period = ge.get("period")
        if off is None or on is None or period is None:
            continue  # cannot resolve the change / date it to a period
        time = ge.get("startGameClock")
        team_raw = ge.get("teamId")
        if team_raw is None or (not isinstance(team_raw, str) and pd.isna(team_raw)):
            # Null teamId -> derive from the OUTGOING player's roster team (may be None -> unattributable).
            team_raw = player_team.get(canonical_id(off))
        subs.append(
            {
                "period": int(period),
                "time": float(time) if time is not None and not pd.isna(time) else 0.0,
                "team_raw": team_raw,
                "off": off,
                "on": on,
            }
        )
    return subs


def _earliest_actions(events: list[dict], candidate_keys: set[Any]) -> dict[object, tuple[float, float]]:
    """``{canonical player id -> earliest (period, startGameClock)}`` over on-ball gameEvents.

    The starter action signal (the sportec ``_earliest_acting`` idiom, but keyed on GS
    ``gameEvents.playerId`` + ``startGameClock``): the smallest ``(period, time)`` at which each
    candidate GK is the acting ``playerId`` of an ORDINARY (non-``SUB``) event -- SUB rows are
    substitution markers, not on-ball actions. Restricted to ``candidate_keys`` (the non-subbed-on GK
    ids) so an unused bench keeper simply never appears here. A NaN/absent ``period`` or ``time`` sorts
    LAST (``math.inf``), so a dated action always beats an undated one.
    """
    earliest: dict[object, tuple[float, float]] = {}
    for event in events:
        ge = event.get("gameEvents") or {}
        if ge.get("gameEventType") == _SUB_EVENT_TYPE:
            continue  # a substitution marker is not an on-ball action
        pid = ge.get("playerId")
        if pid is None:
            continue
        key = canonical_id(pid)
        if key not in candidate_keys:
            continue
        period = ge.get("period")
        time = ge.get("startGameClock")
        stamp = (
            float(period) if period is not None and not pd.isna(period) else math.inf,
            float(time) if time is not None and not pd.isna(time) else math.inf,
        )
        if key not in earliest or stamp < earliest[key]:
            earliest[key] = stamp
    return earliest


def _resolve_starters(
    gk_by_team: dict[object, dict[str, Any]], subs: list[dict[str, Any]], events: list[dict]
) -> dict[object, tuple[object, object]]:
    """``{canonical team key -> (raw team id, starting keeper id)}``.

    The starter is the team GK that is NOT introduced by a substitution AND actually PLAYED -- i.e.
    among the roster GKs whose id is never a ``SUB`` ``playerOnId``, the one with the EARLIEST on-ball
    action (:func:`_earliest_actions`). A never-acting non-subbed-on GK is an unused bench keeper and
    is excluded (real WC2022 rosters carry 3 GKs/team, 2 of them never play). A residual earliest-action
    tie (degenerate) is broken by the smallest canonical id. A team with no acting, non-subbed-on GK
    seeds no starter (honest gap). See the module docstring's reported selection rule.
    """
    subbed_on = {canonical_id(s["on"]) for s in subs}
    candidate_keys = {canonical_id(g) for entry in gk_by_team.values() for g in entry["gks"]} - subbed_on
    earliest = _earliest_actions(events, candidate_keys)
    starters: dict[object, tuple[object, object]] = {}
    for team_key, entry in gk_by_team.items():
        # A starter candidate is a roster GK that is NOT introduced by a sub AND actually acted on the
        # ball; `earliest` already excludes subbed-on ids (its `candidate_keys` filter), so membership
        # in `earliest` is the acting-and-not-subbed-on test.
        acting = [g for g in entry["gks"] if canonical_id(g) in earliest]
        if not acting:
            continue  # no acting, non-subbed-on GK -> seed no starter (honest gap)
        # Earliest on-ball action wins; a residual tie falls back to the smallest canonical id.
        starter = min(acting, key=lambda g: (earliest[canonical_id(g)], str(canonical_id(g))))
        starters[team_key] = (entry["raw_team"], starter)
    return starters


def extract_keeper_appearances(
    events: list[dict],
    roster: list[dict],
    *,
    game_id: object,
) -> pd.DataFrame:
    """Extract per-period keeper on-pitch intervals from Gradient Sports raw data (the TF-59 port, §5.5).

    See the module docstring for the two driving signals (starting keepers from the roster; keeper
    ``SUB`` gameEvents) and the reported starting-keeper-selection rule. Returns a validated
    :func:`~silly_kicks.keeper_identity.validate_keeper_appearances` frame (columns in
    :data:`~silly_kicks.keeper_identity.KEEPER_APPEARANCE_COLUMNS` order). PURE -- neither ``events``
    nor ``roster`` is mutated (only read).

    Examples
    --------
    Two roster keepers, a two-period match, a mid-second-half keeper sub -> one row per period the
    keeper is on. The real GS shapes (nested ``roster`` records + ``gameEvents`` SUB envelopes) are
    what the owner pining feed carries -- see ``tests/providers/gradientsports/test_appearances.py``:

        from silly_kicks.providers.gradientsports.appearances import extract_keeper_appearances

        appearances = extract_keeper_appearances(events, roster, game_id="gs1")

    See NOTICE for full bibliographic citations.
    """
    gk_by_team, player_team = _roster_index(roster)
    subs = _sub_events(events, player_team)

    # Periods present across the WHOLE event stream (not just SUBs), so an unsubbed keeper is
    # decomposed across every period the match actually has -- incl. extra time. `_STARTER_PERIOD`
    # (1) is UNIONED in because the starter is on from the first period whether or not any event
    # witnesses it (a substitution-only synthetic stream carries no period-1 event yet the starter's
    # period-1 interval must still emit); real GS data always has period-1 events, so this is a no-op
    # there. When NO event carries a period, `periods` degenerates to just `[_STARTER_PERIOD]`.
    observed = {int(ge["period"]) for e in events if (ge := (e.get("gameEvents") or {})).get("period") is not None}
    periods = sorted(observed | {_STARTER_PERIOD})
    first_period, last_period = periods[0], periods[-1]

    # 1) Open a starter tenure per team at (first period, 0.0). `open_seg` is keyed by CANONICAL team
    #    id so the change events (keyed on the same team id) close the right tenure (ADR-019);
    #    `current_keeper` tracks each team's on-pitch keeper forward so a SUB is classified.
    open_seg: dict[object, dict[str, Any]] = {}
    current_keeper: dict[object, object] = {}
    finished: list[KeeperSegment] = []
    for team_key, (raw_team, starter) in _resolve_starters(gk_by_team, subs, events).items():
        open_seg[team_key] = {
            "team_id": raw_team,
            "player_id": starter,
            "source": "starting_xi",
            "start_period": first_period,
            "start_time": 0.0,
        }
        current_keeper[team_key] = starter

    # 2) Apply keeper substitutions chronologically. A SUB is a KEEPER change iff the outgoing player
    #    is the team's CURRENT keeper; it then closes that tenure at (period, time) and opens the
    #    incoming keeper there. A SUB whose team is unattributable (null teamId + outgoing not in the
    #    roster) or whose outgoing player is not the current keeper (a non-keeper sub) is ignored.
    for ch in sorted(subs, key=lambda c: (c["period"], c["time"])):
        team_raw = ch["team_raw"]
        if team_raw is None or (not isinstance(team_raw, str) and pd.isna(team_raw)):
            continue  # team unresolvable -> cannot attribute this substitution
        team_key = canonical_id(team_raw)
        cur = current_keeper.get(team_key)
        if cur is None or not same_id(ch["off"], cur):
            continue  # the outgoing player is not this team's current keeper -> a non-keeper sub
        prior = open_seg.pop(team_key, None)
        if prior is not None:
            finished.append(
                KeeperSegment(
                    team_id=prior["team_id"],
                    player_id=prior["player_id"],
                    source=prior["source"],
                    start_period=prior["start_period"],
                    start_time=prior["start_time"],
                    end_period=ch["period"],
                    end_time=ch["time"],
                )
            )
        open_seg[team_key] = {
            "team_id": team_raw,
            "player_id": ch["on"],
            "source": "sub_events",
            "start_period": ch["period"],
            "start_time": ch["time"],
        }
        current_keeper[team_key] = ch["on"]

    # 3) Close every still-open tenure at (last period, +inf) ("still on at the match end").
    for seg in open_seg.values():
        finished.append(
            KeeperSegment(
                team_id=seg["team_id"],
                player_id=seg["player_id"],
                source=seg["source"],
                start_period=seg["start_period"],
                start_time=seg["start_time"],
                end_period=last_period,
                end_time=math.inf,
            )
        )

    return build_keeper_appearances_from_segments(finished, periods, game_id=game_id)
