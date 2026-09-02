"""Sportec / DFL keeper-appearance extractor (TF-59 PR1, spec §5.5).

Produces the :func:`~silly_kicks.keeper_identity.validate_keeper_appearances` port -- one row per
keeper on-pitch interval PER PERIOD -- from a parsed DFL bronze DataFrame (from
:func:`~silly_kicks.providers.sportec.parse.parse_dfl_events`) plus its
:class:`~silly_kicks.providers.sportec.parse.MatchInfo`. It reads bronze COLUMNS, never raw XML (no
re-parse). Three signals drive it:

* **Starting keepers.** ``MatchInfo.gk_player_ids`` are the PersonIds with DFL
  ``PlayingPosition="TW"``, which the two committed DFL fixtures show is starters-only (both are
  ``Starting="true"`` and there is exactly one per team). Each is mapped to its team's CLU id via
  ``MatchInfo.player_team_map`` (``person_id -> "home"|"away"``) + ``home_team_id`` / ``away_team_id``,
  and opens an on-pitch SEGMENT at ``(first period, 0.0s)``. **Concern (reported):** ``MatchInfo``
  carries no explicit starting-XI flag, so the starter is derived from ``gk_player_ids``; should a
  future match ever tag a bench keeper ``PlayingPosition="TW"`` (bench players CAN carry a
  ``PlayingPosition`` -- 9 did in one fixture, none a keeper), a team would have >1 candidate. That
  ambiguity is broken by the earliest keeper ACTION in the bronze (the starter acts before any
  backup enters); a team whose keepers never act seeds no starter (honest gap, never fabricated).

* **``TW`` substitution.** A bronze row with ``sub_playing_position == "TW"`` is a keeper change: it
  ENDS the outgoing keeper's segment at ``(period, timestamp_seconds)`` and OPENS the incoming
  (``sub_player_in``, ``source="sub_events"``) there, for the team ``sub_team``.

* **Emergency keeper.** A bronze row with a truthy ``other_action_player_becomes_goalkeeper`` opens an
  ``emergency_keeper`` segment for ``other_action_player`` (the acting player who takes the gloves --
  e.g. after a keeper red card with no sub) for the team ``other_action_team``, ending the prior
  keeper's segment.

``timestamp_seconds`` is PERIOD-RELATIVE (it resets each period; ADR-017), matching the port's time
base, so it is used straight (no cross-period offset). ``periods`` = the sorted distinct bronze
``period`` values, so extra-time is handled by construction. Still-on keepers end at ``(last period,
+inf)``. The per-period decomposition is the shared
:func:`~silly_kicks.keeper_identity.build_keeper_appearances_from_segments`.

DFL ids are STRINGS (``MatchInfo.gk_player_ids: frozenset[str]``); the port's id columns are
``object``, so they are stored un-coerced and every id comparison routes through
:mod:`silly_kicks.id_compat` (ADR-019). Imports are confined to
:mod:`silly_kicks.keeper_identity` + :mod:`silly_kicks.id_compat` + pandas/stdlib (the ``MatchInfo``
type is imported ``TYPE_CHECKING``-only) -- NOT :mod:`silly_kicks.tracking`.

See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import pandas as pd

from silly_kicks.id_compat import canonical_id, ids_equal
from silly_kicks.keeper_identity import (
    KeeperSegment,
    build_keeper_appearances_from_segments,
)

if TYPE_CHECKING:  # type-only import -- keeps the runtime module free of the heavy `parse` chain.
    from silly_kicks.providers.sportec.parse import MatchInfo

__all__ = ["extract_keeper_appearances"]

#: DFL ``PlayingPosition`` value for the goalkeeper slot (Torwart). A ``sub_playing_position == "TW"``
#: substitution is therefore a keeper change.
_TW_POSITION = "TW"

#: Fallback period when the bronze carries no ``period`` at all (degenerate). Single-sourced so the
#: starter seed and the decomposition agree.
_DEFAULT_PERIOD = 1

#: Values of ``other_action_player_becomes_goalkeeper`` that are NOT a "the player became keeper"
#: signal. Anything else non-null (a truthy flag ``"true"`` or, defensively, a player id) triggers.
_FALSEY_TOKENS = frozenset({"", "false", "0", "no", "none"})


def _series(df: pd.DataFrame, name: str) -> pd.Series:
    """The named column, or an all-``None`` Series aligned to ``df`` when the column is absent.

    The real DFL bronze always carries every ``_IDSSE_EVENTS_BRONZE_COLS`` column; this keeps the
    extractor robust to a slim/synthetic bronze that omits an unused one.
    """
    if name in df.columns:
        return df[name]
    return pd.Series([None] * len(df), index=df.index, dtype="object")


def _is_truthy_flag(value: Any) -> bool:
    """True iff ``value`` names a "player became goalkeeper" signal (non-null, non-falsey).

    ``value`` is a provider-dtype pandas scalar (a bronze cell); ``Any`` so ``pd.isna`` accepts it
    (the ``add_defending_gk_player_id`` scalar idiom).
    """
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return False
    return str(value).strip().lower() not in _FALSEY_TOKENS


def _resolve_starters(match_info: MatchInfo, df: pd.DataFrame) -> dict[object, tuple[object, object]]:
    """``{canonical team CLU id -> (raw team CLU id, starting keeper player id)}``.

    One entry per team with a resolvable starter. ``gk_player_ids`` are the ``PlayingPosition="TW"``
    PersonIds (starters-only in real DFL); each is mapped to its CLU id via ``player_team_map``
    (``person -> "home"|"away"``) + the two team ids. When a team has exactly one such keeper it is the
    starter; a team with MORE than one (a hypothetical bench keeper also tagged ``TW``) is
    disambiguated by the earliest keeper ACTION in the bronze; a team with none seeds no starter
    (honest gap). The key is canonical (ADR-019) so the change events -- keyed on the same CLU id --
    close the right tenure; the raw CLU id is carried through for the stored ``team_id``.
    """
    label_to_clu: dict[str, object] = {"home": match_info.home_team_id, "away": match_info.away_team_id}
    keepers_by_team: dict[object, list[object]] = {}
    raw_by_key: dict[object, object] = {}
    for gk in match_info.gk_player_ids:
        label = match_info.player_team_map.get(gk)
        clu = label_to_clu.get(label) if label is not None else None
        if clu is None:
            continue
        key = canonical_id(clu)
        keepers_by_team.setdefault(key, []).append(gk)
        raw_by_key[key] = clu

    starters: dict[object, tuple[object, object]] = {}
    for clu_key, candidates in keepers_by_team.items():
        starter = candidates[0] if len(candidates) == 1 else _earliest_acting(candidates, df)
        if starter is not None:
            starters[clu_key] = (raw_by_key[clu_key], starter)
    return starters


def _earliest_acting(candidates: list[object], df: pd.DataFrame) -> object | None:
    """The candidate keeper whose earliest ``(period, timestamp_seconds)`` bronze action is smallest.

    A deterministic tie-break for the (rare) case of >1 ``TW`` keeper per team: the starting keeper
    acts before any backup enters. Returns ``None`` when none of the candidates ever acts (so the team
    seeds no starter rather than fabricating one).
    """
    player_ids = _series(df, "player_id")
    periods = _series(df, "period")
    times = _series(df, "timestamp_seconds")
    best: tuple[float, float] | None = None
    winner: object | None = None
    for cand in candidates:
        acted = ids_equal(player_ids, pd.Series([cand] * len(df), index=df.index))
        for i in range(len(df)):
            if not bool(acted.iloc[i]):
                continue
            p = periods.iloc[i]
            t = times.iloc[i]
            key = (float(p) if not pd.isna(p) else math.inf, float(t) if not pd.isna(t) else math.inf)
            if best is None or key < best:
                best = key
                winner = cand
    return winner


def _keeper_change_events(df: pd.DataFrame) -> list[dict[str, Any]]:
    """The chronologically-sortable keeper CHANGE events in the bronze.

    Each is ``{"period", "time", "team", "incoming", "source"}``: a ``TW`` substitution
    (``source="sub_events"``, incoming ``sub_player_in``, team ``sub_team``) or an emergency keeper
    (``source="emergency_keeper"``, incoming ``other_action_player``, team ``other_action_team``). A
    row with no attributable ``period`` is skipped (it cannot be dated); a NaN ``timestamp_seconds``
    defaults to ``0.0`` (period start) so the change is still placed.
    """
    period = _series(df, "period")
    time = _series(df, "timestamp_seconds")
    sub_pos = _series(df, "sub_playing_position")
    sub_in = _series(df, "sub_player_in")
    sub_team = _series(df, "sub_team")
    oa_player = _series(df, "other_action_player")
    oa_team = _series(df, "other_action_team")
    oa_becomes = _series(df, "other_action_player_becomes_goalkeeper")

    changes: list[dict[str, Any]] = []
    for i in range(len(df)):
        p_raw = period.iloc[i]
        if pd.isna(p_raw):
            continue  # cannot attribute to a period
        p = int(p_raw)
        t_raw = time.iloc[i]
        t = float(t_raw) if not pd.isna(t_raw) else 0.0

        pos = sub_pos.iloc[i]
        if not pd.isna(pos) and str(pos) == _TW_POSITION:
            incoming = sub_in.iloc[i]
            team = sub_team.iloc[i]
            if not pd.isna(incoming) and not pd.isna(team):
                changes.append({"period": p, "time": t, "team": team, "incoming": incoming, "source": "sub_events"})

        if _is_truthy_flag(oa_becomes.iloc[i]):
            player = oa_player.iloc[i]
            team = oa_team.iloc[i]
            # Defensive: if `other_action_player` is null but the flag column itself carries a player
            # id (an alternative bronze encoding), fall back to it.
            if pd.isna(player) and not _is_truthy_flag_only(oa_becomes.iloc[i]):
                player = oa_becomes.iloc[i]
            if not pd.isna(player) and not pd.isna(team):
                changes.append({"period": p, "time": t, "team": team, "incoming": player, "source": "emergency_keeper"})
    return changes


def _is_truthy_flag_only(value: object) -> bool:
    """True iff ``value`` is a boolean-ish flag token (``"true"``/``"1"``/...), i.e. NOT a player id."""
    return isinstance(value, str) and value.strip().lower() in {"true", "1", "yes"}


def extract_keeper_appearances(
    match_info: MatchInfo,
    events_bronze: pd.DataFrame,
    *,
    game_id: object,
) -> pd.DataFrame:
    """Extract per-period keeper on-pitch intervals from a DFL bronze DataFrame (the TF-59 port, §5.5).

    See the module docstring for the three driving signals (starting keepers, ``TW`` substitutions,
    emergency keepers). Returns a validated
    :func:`~silly_kicks.keeper_identity.validate_keeper_appearances` frame (columns in
    :data:`~silly_kicks.keeper_identity.KEEPER_APPEARANCE_COLUMNS` order). PURE -- neither
    ``match_info`` nor ``events_bronze`` is mutated (only read).

    Examples
    --------
    A two-team, no-substitution match yields one open starting-keeper interval per team-period. The
    real bronze shape (``period`` / ``timestamp_seconds`` / ``sub_*`` / ``other_action_*``) is what
    :func:`~silly_kicks.providers.sportec.parse.parse_dfl_events` returns, so a realistic example
    needs a real match's events -- see ``tests/providers/sportec/test_appearances.py``:

        from silly_kicks.providers.sportec.appearances import extract_keeper_appearances

        appearances = extract_keeper_appearances(match_info, events_bronze, game_id="DFL-MAT-...")

    See NOTICE for full bibliographic citations.
    """
    df = events_bronze
    periods = sorted({int(p) for p in _series(df, "period").dropna().unique()})
    if not periods:
        periods = [_DEFAULT_PERIOD]
    first_period, last_period = periods[0], periods[-1]

    # 1) Open a starter tenure per team at (first period, 0.0). `open_seg` is keyed by CANONICAL team
    #    id so the change events (keyed on the same CLU id) close the right tenure (ADR-019).
    open_seg: dict[object, dict[str, Any]] = {}
    finished: list[KeeperSegment] = []
    for clu_key, (raw_clu, starter) in _resolve_starters(match_info, df).items():
        open_seg[clu_key] = {
            "team_id": raw_clu,
            "player_id": starter,
            "source": "starting_xi",
            "start_period": first_period,
            "start_time": 0.0,
        }

    # 2) Apply keeper changes chronologically: each closes the team's open tenure at (period, time) and
    #    opens the incoming keeper there. A change on a team with no open tenure (unresolved starter or
    #    a second change) simply opens the incoming keeper -- nothing to close.
    for ch in sorted(_keeper_change_events(df), key=lambda c: (c["period"], c["time"])):
        team_key = canonical_id(ch["team"])
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
            "team_id": ch["team"],
            "player_id": ch["incoming"],
            "source": ch["source"],
            "start_period": ch["period"],
            "start_time": ch["time"],
        }

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
