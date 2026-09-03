"""Keeper-identity resolution for tracking frames (ADR-055 single-source).

The tracking GK families (``add_pre_shot_gk_*`` / ``add_xt_gk`` / ``add_ghost_gk``) need the REAL
keeper identity, which SB360 freeze-frames do not carry (rows are numbered). This module is the ONE
resolver. Its ``identity="native"`` path DELEGATES to ``defending_gk_from_frames`` /
``acting_gk_from_frames`` (which already return the keeper ``player_id`` from the frame); only the
``identity="roster"`` path (SB360's injected-roster + goal-kick-event ladder) is new work.

See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any, Literal, NamedTuple, TypeAlias

import numpy as np
import pandas as pd

from silly_kicks._nan_safety import nan_safe_enrichment
from silly_kicks.id_compat import canonical_id, ids_match, same_id

# NOTE (ADR-055 single-source): the native path DELEGATES to the TF-13 frame resolvers
# ``defending_gk_from_frames`` / ``acting_gk_from_frames`` (and links via ``link_actions_to_frames``)
# rather than re-deriving keeper identity from frames. Those imports are LAZY -- made inside
# ``_resolve_from_native`` -- because importing ``silly_kicks.tracking._gk_resolve`` /
# ``silly_kicks.tracking.utils`` runs ``silly_kicks.tracking.__init__`` (numba + ~30 submodules), and
# ``import silly_kicks.keeper_identity`` must stay tracking-free. A test that needs to prove the
# delegation patches the DEFINITION SITE ``silly_kicks.tracking._gk_resolve.defending_gk_from_frames``
# (a module attribute of ``_gk_resolve``), which the lazy per-call import resolves afresh.

__all__ = [
    "DEFENDING_GK_SOURCE_VALUES",
    "KEEPER_APPEARANCE_COLUMNS",
    "KEEPER_APPEARANCE_SOURCE_VALUES",
    "KEEPER_ID_SOURCE_DERIVED",
    "KEEPER_ID_SOURCE_EVENT",
    "KEEPER_ID_SOURCE_NATIVE",
    "KEEPER_ID_SOURCE_ROSTER",
    "KEEPER_ID_SOURCE_UNRESOLVED",
    "KEEPER_ID_SOURCE_VALUES",
    "KeeperIdentity",
    "KeeperIdentityMap",
    "KeeperIdentityReport",
    "KeeperSegment",
    "add_defending_gk_player_id",
    "apply_keeper_identities_to_frames",
    "build_keeper_appearances_from_segments",
    "resolve_keeper_identities",
    "validate_keeper_appearances",
]

#: SPADL ``type_name`` for a goal kick (``spadl/config.py`` ``actiontypes[22]``). A goal-kick actor
#: event names the acting team's keeper, so it is the most authoritative rung for that team-period.
_GOALKICK_TYPE_NAME = "goalkick"

#: The keeper identity was named by a goal-kick actor event (the SB360 acting keeper; most
#: authoritative for that team-period, and beats a stale roster after a substitution).
KEEPER_ID_SOURCE_EVENT = "event"
#: The keeper identity came from the injected ``{team_id: gk_id}`` roster (the SB360 defending
#: keeper, whom no event names).
KEEPER_ID_SOURCE_ROSTER = "roster"
#: The keeper identity came from the frame's ``is_goalkeeper`` row carrying a real provider-assigned
#: ``player_id`` (non-SB360 providers), whose ``is_goalkeeper_source`` was ``"native"``.
KEEPER_ID_SOURCE_NATIVE = "native"
#: As ``native``, but the frame's ``is_goalkeeper`` was set by positional derivation
#: (``is_goalkeeper_source == "derived"``).
KEEPER_ID_SOURCE_DERIVED = "derived"
#: No rung named this team's keeper -> the identity is NA, counted (never fabricated).
KEEPER_ID_SOURCE_UNRESOLVED = "unresolved"

#: Closed vocabulary for the ``source`` field of a resolved keeper identity.
KEEPER_ID_SOURCE_VALUES: tuple[str, ...] = (
    KEEPER_ID_SOURCE_EVENT,
    KEEPER_ID_SOURCE_ROSTER,
    KEEPER_ID_SOURCE_NATIVE,
    KEEPER_ID_SOURCE_DERIVED,
    KEEPER_ID_SOURCE_UNRESOLVED,
)


# --- Keeper appearance-interval port (TF-59, spec §5.3) --------------------------------------------
# A normalized ``KeeperAppearances`` table: one row per keeper's on-pitch interval, injected by an
# extractor (Tasks 4-8 produce it; the interval-resolution consumer reads it). Plain-dict schema
# (house style; no pandera). Contracts:
#   * Times are PERIOD-RELATIVE (they reset each period), matching SPADL ``time_seconds`` (ADR-017);
#     an extractor converts its provider's native time (frames / clock / minute) into this base.
#     ``end_time_seconds`` may be ``+inf`` / NaN, meaning "to the period end".
#   * The three ids are ``object`` (string-tolerant), NOT ``Int64``: DFL ids are strings
#     (``MatchInfo.gk_player_ids: frozenset[str]``) and SkillCorner ids are strings, so ``Int64`` would
#     drop them. The port tolerates string AND numeric ids un-coerced; comparisons downstream route
#     through ``id_compat`` (ADR-019).
#: Column -> dtype schema for the ``KeeperAppearances`` port. ids ``object`` (see the note above).
KEEPER_APPEARANCE_COLUMNS: dict[str, str] = {
    "game_id": "object",
    "team_id": "object",
    "player_id": "object",
    "period_id": "int64",
    "start_time_seconds": "float64",
    "end_time_seconds": "float64",
    "source": "object",
}

#: Closed vocabulary for the ``source`` provenance token of a keeper appearance interval.
KEEPER_APPEARANCE_SOURCE_VALUES: tuple[str, ...] = (
    "native_intervals",
    "sub_events",
    "starting_xi",
    "emergency_keeper",
)

#: Closed vocabulary for the ``defending_gk_source`` provenance column, emitted ONLY on the
#: appearance-resolution path of :func:`add_defending_gk_player_id` (the ADR-054 source-column
#: pattern). ``appearance`` -- a covering interval named the keeper and it AGREES with the coarse
#: map; ``appearance_map_conflict`` -- a covering interval named a keeper who DISAGREES with the
#: coarse map (spec §5.4 appearance-vs-map cross-check); ``map_fallback`` -- no interval covered the
#: action's ``time_seconds`` (an appearance gap), so the coarse ``keeper_map`` governed; ``unresolved``
#: -- neither rung named a keeper (never a fabricated id; ADR-027).
DEFENDING_GK_SOURCE_VALUES: tuple[str, ...] = (
    "appearance",
    "map_fallback",
    "appearance_map_conflict",
    "unresolved",
)


def validate_keeper_appearances(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a ``KeeperAppearances`` frame against the port contract; return it unchanged.

    Raises ``ValueError`` on a missing column, a negative ``start_time_seconds`` (times are
    period-relative, so ``>= 0``; ADR-017), any interval with ``start >= end`` (a NaN / ``+inf`` end
    passes -- it means "to the period end"), or an out-of-vocabulary ``source`` token. Does NOT coerce
    id dtypes: ids stay whatever the caller supplied (string or numeric), so DFL/SkillCorner string
    ids survive un-coerced and downstream comparisons route through ``id_compat`` (ADR-019).

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from silly_kicks.keeper_identity import validate_keeper_appearances
    >>> appearances = pd.DataFrame(
    ...     {
    ...         "game_id": ["g1"], "team_id": ["DFL-CLU-00000G"], "player_id": ["DFL-OBJ-0027AX"],
    ...         "period_id": [1], "start_time_seconds": [0.0], "end_time_seconds": [np.inf],
    ...         "source": ["starting_xi"],
    ...     }
    ... )
    >>> validate_keeper_appearances(appearances)["player_id"].tolist()
    ['DFL-OBJ-0027AX']

    See NOTICE for full bibliographic citations.
    """
    missing = [c for c in KEEPER_APPEARANCE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"keeper appearances missing column(s): {missing}")
    start = df["start_time_seconds"]
    end = df["end_time_seconds"]
    if (start < 0).any():
        raise ValueError("start_time_seconds must be period-relative (>= 0)")
    # A NaN / +inf end never satisfies `start >= end` (means "to period end"), so only a finite
    # end that precedes its start raises here.
    if (start >= end).any():
        raise ValueError("each appearance needs start < end")
    bad_sources = set(df["source"].dropna()) - set(KEEPER_APPEARANCE_SOURCE_VALUES)
    if bad_sources:
        raise ValueError(f"unknown appearance source(s): {sorted(bad_sources)}")
    return df


class KeeperSegment(NamedTuple):
    """One keeper's on-pitch TENURE, spanning ``(start_period, start_time) -> (end_period, end_time)``.

    Provider-agnostic input to :func:`build_keeper_appearances_from_segments`: an extractor builds a
    list of these (one per keeper tenure -- opened by a starter / substitution / emergency-keeper
    event, closed by the next keeper change or left open to the match end) and the builder decomposes
    them into the per-period port rows. Times are PERIOD-RELATIVE (ADR-017), matching the port; ids are
    stored RAW (string or numeric), so DFL/SkillCorner string ids survive un-coerced. ``end_time`` may
    be ``float("inf")`` -- "still on at the match end". ``source`` is one of
    :data:`KEEPER_APPEARANCE_SOURCE_VALUES`.

    Examples
    --------
    >>> import math
    >>> from silly_kicks.keeper_identity import KeeperSegment
    >>> seg = KeeperSegment(
    ...     team_id="DFL-CLU-0", player_id="DFL-OBJ-1", source="starting_xi",
    ...     start_period=1, start_time=0.0, end_period=2, end_time=math.inf,
    ... )
    >>> (seg.player_id, seg.source, math.isinf(seg.end_time))
    ('DFL-OBJ-1', 'starting_xi', True)
    """

    team_id: object
    player_id: object
    source: str
    start_period: int
    start_time: float
    end_period: int
    end_time: float


def build_keeper_appearances_from_segments(
    segments: Iterable[KeeperSegment],
    periods: Sequence[int],
    *,
    game_id: object,
) -> pd.DataFrame:
    """Decompose provider-agnostic keeper SEGMENTS into the per-``(game, period_id, team)`` port rows.

    This is the ONE per-period decomposition every extractor reuses (spec §5.5). The port keys
    appearances per ``(game, period_id, team)`` and the consumer looks up the covering interval WITHIN
    a single period, so each :class:`KeeperSegment` -- a ``(start_period, start_time) ->
    (end_period, end_time)`` tenure -- is split into one row per period ``p`` in ``periods`` it spans
    (``start_period <= p <= end_period``): the entry period starts at ``start_time`` (else ``0.0``) and
    the exit period ends at ``end_time`` (else ``+inf``, i.e. "to the period end"). A period slice with
    ``start >= end`` (no tenure -- e.g. a keeper subbed at the very start of a period) is DROPPED, which
    also elides a segment that opens at a period boundary it has no time in. So an unsubbed keeper in a
    two-period match gets TWO rows, a half-time change splits cleanly at the boundary, and extra-time
    periods are handled by construction (the segment simply spans more periods).

    Builds the DataFrame in :data:`KEEPER_APPEARANCE_COLUMNS` order and returns
    :func:`validate_keeper_appearances` of it (raises on a malformed segment set). PURE -- ``segments``
    is only read.

    Examples
    --------
    A full-match starter over two periods decomposes into one open row per period:

    >>> import math
    >>> from silly_kicks.keeper_identity import (
    ...     KeeperSegment, build_keeper_appearances_from_segments,
    ... )
    >>> seg = KeeperSegment(
    ...     team_id=10, player_id=901, source="starting_xi",
    ...     start_period=1, start_time=0.0, end_period=2, end_time=math.inf,
    ... )
    >>> ap = build_keeper_appearances_from_segments([seg], [1, 2], game_id="g1")
    >>> list(ap["period_id"])
    [1, 2]

    See NOTICE for full bibliographic citations.
    """
    inf = float("inf")
    rows: list[dict[str, object]] = []
    for seg in segments:
        for period in periods:
            if not (seg.start_period <= period <= seg.end_period):
                continue
            start = seg.start_time if period == seg.start_period else 0.0
            end = seg.end_time if period == seg.end_period else inf
            if start >= end:
                continue
            rows.append(
                {
                    "game_id": game_id,
                    "team_id": seg.team_id,
                    "player_id": seg.player_id,
                    "period_id": period,
                    "start_time_seconds": start,
                    "end_time_seconds": end,
                    "source": seg.source,
                }
            )
    appearances = pd.DataFrame(rows, columns=list(KEEPER_APPEARANCE_COLUMNS)).astype(KEEPER_APPEARANCE_COLUMNS)
    return validate_keeper_appearances(appearances)


class KeeperIdentity(NamedTuple):
    """One resolved keeper identity for a ``(game, period, team)``.

    ``conflict`` records a roster-vs-event disagreement (both named a keeper and they differed);
    ``source`` still records the WINNING rung per precedence, so the disagreement is a separate,
    durable signal, never a lost warning.

    ``team_id`` is the keeper's OWN team in its RAW representation -- the same team the (canonical)
    map key names, but stored so a consumer recovers the provider-native id without re-deriving it
    (ADR-085; the map key is canonical, so the raw team would otherwise be lost). Defaulted to
    ``pd.NA`` for back-compat: :func:`resolve_keeper_identities` populates it from the actions/frames
    it enumerates, and a hand-built map that omits it reads NA. It is why
    :func:`add_defending_gk_player_id` can stamp the authoritative ``defending_gk_team_id`` even for
    an opponent that never appears in the actions (a frame-seeded map).

    Examples
    --------
    >>> from silly_kicks.keeper_identity import KeeperIdentity, KEEPER_ID_SOURCE_NATIVE
    >>> ident = KeeperIdentity(gk_id=920, source=KEEPER_ID_SOURCE_NATIVE, conflict=False, team_id=7)
    >>> (ident.gk_id, ident.source, ident.conflict, ident.team_id)
    (920, 'native', False, 7)
    """

    gk_id: object
    source: str
    conflict: bool
    team_id: object = pd.NA


#: ``{(canonical game_id, period_id, canonical team_id) -> KeeperIdentity}``. ``game_id`` and
#: ``team_id`` are canonical (ADR-055 rule 2 -- look up via ``canonical_id``); ``period_id`` is used
#: AS-IS (raw). So a lookup is ``m[(canonical_id(game), period, canonical_id(team))]`` -- canonicalize
#: game/team, NOT period.
KeeperIdentityMap: TypeAlias = dict[tuple[object, object, object], KeeperIdentity]


@dataclasses.dataclass(frozen=True)
class KeeperIdentityReport:
    """Run-level audit of keeper-identity resolution. Conserves: ``n_resolved + n_unresolved ==
    n_teams_in`` (ADR-052).

    Examples
    --------
    >>> from silly_kicks.keeper_identity import KeeperIdentityReport
    >>> report = KeeperIdentityReport(
    ...     n_teams_in=2, n_resolved=2, n_unresolved=0, n_conflict=0, source_counts={"native": 2}
    ... )
    >>> report.n_resolved + report.n_unresolved == report.n_teams_in
    True
    """

    n_teams_in: int
    n_resolved: int
    n_unresolved: int
    n_conflict: int
    source_counts: dict[str, int]


def resolve_keeper_identities(
    actions: pd.DataFrame,
    frames: pd.DataFrame | None = None,
    *,
    identity: Literal["native", "roster"],
    roster: dict | None = None,
) -> tuple[KeeperIdentityMap, KeeperIdentityReport]:
    """Resolve the real keeper identity per ``(game, period, team)``. See module docstring.

    ``frames`` is OPTIONAL for ``identity="roster"``: when omitted, the resolver enumerates the
    ``(game, period, team)`` triples from ``actions`` themselves (the event-only path, for callers
    with no tracking frames). ``identity="native"`` reads frame positions, so it REQUIRES ``frames``
    -- calling it with ``frames=None`` raises ``ValueError``.

    Examples
    --------
    Native providers carry a real keeper ``player_id`` on their ``is_goalkeeper`` frame rows, so
    ``identity="native"`` DELEGATES to the TF-13 frame resolvers (ADR-055 single-source)::

        from silly_kicks.keeper_identity import resolve_keeper_identities

        identities, report = resolve_keeper_identities(actions, frames, identity="native")
        keeper = identities[(game_id, period_id, team_id)]
        keeper.gk_id, keeper.source  # -> e.g. (920, "native")

    SB360 freeze-frames carry no player identity, so ``identity="roster"`` resolves from an
    injected ``{team_id: gk_id}`` roster, with a goal-kick actor event overriding a stale
    starter::

        identities, report = resolve_keeper_identities(
            actions, frames, identity="roster", roster={10: 901, 20: 902}
        )

    See NOTICE for full bibliographic citations.
    """
    if identity == "roster":
        return _resolve_from_roster(actions, frames, roster)
    if identity == "native":
        return _resolve_from_native(actions, frames)
    raise ValueError(f"unknown identity mode: {identity!r}")


@nan_safe_enrichment
def add_defending_gk_player_id(
    actions: pd.DataFrame,
    keeper_map: KeeperIdentityMap,
    *,
    appearances: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Stamp each action's DEFENDING keeper id (the opponent's keeper) from a resolved map.

    The PLACEMENT half of :func:`resolve_keeper_identities` on the ACTION grain: the resolver
    returns a pure ``(game, period, team) -> KeeperIdentity`` map and mutates nothing; this applies
    it. In a two-team match the defending keeper of an action is the OTHER team's keeper, so the
    opponent is derived per ``(game, period)`` from the map itself (the single other team present)
    and looked up in it. Every id build/lookup routes through ``id_compat`` (ADR-019), so the
    action ``team_id`` dtype need not match the map-key dtype.

    Returns a COPY with new ``defending_gk_player_id`` + ``defending_gk_team_id`` columns -- the
    defending keeper and its AUTHORITATIVE team, the opponent the resolver used to select the keeper.
    BOTH are read from ``keeper_map``: the team is the opponent entry's ``KeeperIdentity.team_id`` in
    its RAW provider representation (ADR-085), so it resolves even for an opponent that never appears
    in ``actions`` (a frame-seeded map) and joins via ``id_compat`` (ADR-019) when the consumer's dim
    table uses another representation. ``defending_gk_player_id`` is ``pd.NA`` where the opponent is
    unresolvable (no ``(game, period)`` entry, a NaN action team, not exactly one other team, or the
    resolved opponent's ``gk_id`` is itself NA); ``defending_gk_team_id`` is NA for the SAME structural
    reasons OR where the map value carries no ``team_id`` (a hand-built map that omitted it) -- INDEPENDENT
    of the keeper (a row can carry a known team with an NA keeper). PURE -- ``actions`` is never mutated.

    Interval-granular resolution (``appearances``, TF-59 spec §5.4). When a validated
    :func:`validate_keeper_appearances` table is supplied, each action's defending keeper is the
    opponent's keeper whose ``[start_time_seconds, end_time_seconds)`` interval COVERS the action's
    ``time_seconds`` -- so a mid-period keeper substitution is dated exactly and attribution flips at
    the sub minute. An action whose time falls in an appearance GAP falls back to the coarse
    ``keeper_map``. The defending team is still derived from ``keeper_map`` exactly as the coarse path
    does (the opponent within ``(game, period)``). The appearance path ALSO emits a
    ``defending_gk_source`` provenance column over :data:`DEFENDING_GK_SOURCE_VALUES` (the ADR-054
    source-column pattern), which records the appearance-vs-map cross-check. **When ``appearances`` is
    omitted the output carries ``defending_gk_player_id`` + ``defending_gk_team_id`` but NO
    ``defending_gk_source`` column** -- ADR-085 amends ADR-084's byte-identity-when-omitted contract
    with the additive team column; existing column VALUES are unchanged.

    Examples
    --------
    Stamp the defending keeper's id from a resolved identity map. The map keys are canonical
    ``(game, period, team)`` tuples, so the opponent lookup is dtype-safe (ADR-019) -- the action's
    ``team_id`` here is a plain ``int`` while the map keys are canonical strings:

    >>> import pandas as pd
    >>> from silly_kicks.id_compat import canonical_id
    >>> from silly_kicks.keeper_identity import KeeperIdentity, add_defending_gk_player_id
    >>> actions = pd.DataFrame(
    ...     {"action_id": [0], "game_id": [1], "period_id": [1], "team_id": [10], "type_name": ["shot"]}
    ... )
    >>> keeper_map = {
    ...     (canonical_id(1), 1, canonical_id(10)): KeeperIdentity(
    ...         gk_id=901, source="roster", conflict=False, team_id=10
    ...     ),
    ...     (canonical_id(1), 1, canonical_id(20)): KeeperIdentity(
    ...         gk_id=902, source="roster", conflict=False, team_id=20
    ...     ),
    ... }
    >>> out = add_defending_gk_player_id(actions, keeper_map)
    >>> out["defending_gk_player_id"].tolist()  # team 10 is defended by team 20's keeper
    [902]
    >>> out["defending_gk_team_id"].tolist()  # the authoritative opponent team, from the map VALUE
    [20]

    See NOTICE for full bibliographic citations.
    """
    out = actions.copy()
    # The coarse per-action opponent-derivation + map lookup is factored into ``_coarse_defending_gk``
    # so the omit path can return it UNCHANGED (byte-identical to the historical output) and the
    # appearance path can reuse it as the interval-gap fallback + the cross-check comparand.
    fallback = _coarse_defending_gk(actions, keeper_map)
    if appearances is None:
        # Omit path: the coarse keeper Series + the authoritative defending team (ADR-085), and NO
        # ``defending_gk_source`` provenance. (ADR-085 amends ADR-084's byte-identity-when-omitted
        # contract: the omit path now carries the additive ``defending_gk_team_id`` column too --
        # existing column VALUES are unchanged.)
        out["defending_gk_player_id"] = fallback
        out["defending_gk_team_id"] = _coarse_defending_team(actions, keeper_map)
        return out

    # Interval-granular path (spec §5.4). The defending team is the SAME opponent the coarse path
    # derives from ``keeper_map``; the covering interval (if any) supersedes the coarse map and the
    # appearance-vs-map cross-check is recorded in ``defending_gk_source``.
    intervals = _index_appearances(appearances)
    # Group ONCE before the loop (the sibling ``_coarse_defending_gk`` already does this) so
    # ``_defending_team_for`` does not rebuild ``_by_game_period(keeper_map)`` per action.
    by_gp = _by_game_period(keeper_map)
    vals: list[object] = []
    srcs: list[str] = []
    for i, (game, period, team, t) in enumerate(
        zip(actions["game_id"], actions["period_id"], actions["team_id"], actions["time_seconds"], strict=True)
    ):
        opp = _defending_team_for(game, period, team, by_gp)
        # ``period`` (raw) matches the raw period ``_index_appearances`` keys on; ``opp`` is already a
        # canonical key (``canonical_id`` is idempotent). ``None`` opponent -> no interval lookup.
        key = (canonical_id(game), period, canonical_id(opp)) if opp is not None else None
        gk = _keeper_covering(intervals.get(key, ()), t) if key is not None else pd.NA
        coarse = fallback.iloc[i]
        if gk is not pd.NA:
            # A covering interval governs; ``appearance_map_conflict`` iff it disagrees with the
            # coarse map (both present). ``same_id`` is the dtype-safe comparand (ADR-019).
            src = "appearance_map_conflict" if (coarse is not pd.NA and not same_id(gk, coarse)) else "appearance"
            vals.append(gk)
        elif coarse is not pd.NA:
            src = "map_fallback"
            vals.append(coarse)
        else:
            src = "unresolved"
            vals.append(pd.NA)
        srcs.append(src)

    # object dtype: the stored gk_id keeps its provider representation (the roster/native/appearance
    # paths store the RAW id) and a heterogeneous NA is representable; downstream matching routes
    # through the dtype-safe ``id_compat`` seam (ADR-019).
    out["defending_gk_player_id"] = pd.Series(vals, index=out.index, dtype="object")
    out["defending_gk_team_id"] = _coarse_defending_team(actions, keeper_map)
    out["defending_gk_source"] = pd.Series(srcs, index=out.index, dtype="object")
    return out


def _by_game_period(
    keeper_map: KeeperIdentityMap,
) -> dict[tuple[object, object], dict[object, KeeperIdentity]]:
    """Group a resolved ``keeper_map`` into ``{(canonical game, canonical period) -> {canonical team
    -> KeeperIdentity}}``.

    Period is canonicalized on BOTH the map-key and (downstream) the lookup side, so a raw-int map
    period matches a raw-int action period regardless of representation (int vs np.int64 vs Int64);
    game/team are re-canonicalized defensively (``canonical_id`` is idempotent on an already-canonical
    key). A NA-team map key names no team and can never be an opponent, so it is skipped.
    """
    by_gp: dict[tuple[object, object], dict[object, KeeperIdentity]] = {}
    for (g, p, t), ident in keeper_map.items():
        ct = canonical_id(t)
        if ct is pd.NA:
            continue
        by_gp.setdefault((canonical_id(g), canonical_id(p)), {})[ct] = ident
    return by_gp


def _coarse_defending_gk(actions: pd.DataFrame, keeper_map: KeeperIdentityMap) -> pd.Series:
    """The coarse per-action DEFENDING keeper id (the historical ``add_defending_gk_player_id`` body).

    For each action the defending keeper is the SINGLE OTHER team's keeper within the action's
    ``(game, period)`` group of ``keeper_map``; ``pd.NA`` where the opponent is unresolvable (no
    ``(game, period)`` entry, a NaN action team, or not exactly one other team). Returned as an
    ``object`` Series indexed by ``actions.index`` -- byte-identical to the pre-refactor output, which
    the omit path returns unchanged.
    """
    by_gp = _by_game_period(keeper_map)
    values: list[object] = []
    for game, period, team in zip(actions["game_id"], actions["period_id"], actions["team_id"], strict=True):
        ct = canonical_id(team)
        group = by_gp.get((canonical_id(game), canonical_id(period)))
        if group is None or ct is pd.NA:
            values.append(pd.NA)  # no map entry for this (game, period), or a NaN action team
            continue
        # The opponent is the SINGLE other team in this (game, period). `same_id` compares two
        # already-canonical keys dtype-safely; a match on more/fewer than one is unresolvable.
        opponents = [k for k in group if not same_id(k, ct)]
        values.append(group[opponents[0]].gk_id if len(opponents) == 1 else pd.NA)
    return pd.Series(values, index=actions.index, dtype="object")


def _defending_team_for(
    g: object,
    p: object,
    team: object,
    by_gp: dict[tuple[object, object], dict[object, KeeperIdentity]],
) -> object | None:
    """The single OTHER team present in ``keeper_map`` for this action's ``(game, period)``.

    The SAME opponent derivation :func:`_coarse_defending_gk` uses (the opponent is the one canonical
    team key that is not the acting team). Takes the pre-grouped ``by_gp`` -- built ONCE by the caller
    via :func:`_by_game_period` -- so the grouping is not rebuilt per action. Returns that canonical
    team key, or ``None`` when the group is absent, the acting team is NA, or there is not exactly one
    opponent (so the interval lookup is skipped and the coarse fallback governs). Consistent with the
    coarse path by construction (both read the shared ``_by_game_period`` grouping).
    """
    ct = canonical_id(team)
    if ct is pd.NA:
        return None
    group = by_gp.get((canonical_id(g), canonical_id(p)))
    if group is None:
        return None
    opponents = [k for k in group if not same_id(k, ct)]
    return opponents[0] if len(opponents) == 1 else None


def _coarse_defending_team(actions: pd.DataFrame, keeper_map: KeeperIdentityMap) -> pd.Series:
    """The DEFENDING (opponent) team id per action -- the authoritative team the resolver used to pick
    the defending keeper (ADR-085).

    The opponent IDENTITY is derived from ``keeper_map`` exactly as :func:`_coarse_defending_gk` /
    :func:`_defending_team_for` do (the single OTHER canonical team in the action's ``(game, period)``),
    and the RAW team is read straight from that opponent's ``KeeperIdentity.team_id`` -- the resolver's
    own provider representation, populated by :func:`resolve_keeper_identities`. Reading the team from
    the map VALUE (rather than recovering it from ``actions``) makes it available even for an opponent
    that never appears in ``actions`` (a frame-seeded map) -- the case a from-``actions`` recovery
    returned NA for. ``pd.NA`` where the opponent is unresolvable (as :func:`_coarse_defending_gk`) OR
    where the opponent's map entry carries no ``team_id`` (a hand-built map that omitted it). Independent
    of the keeper: a row can carry a KNOWN team with an NA keeper.
    """
    by_gp = _by_game_period(keeper_map)
    vals: list[object] = []
    for g, p, team in zip(actions["game_id"], actions["period_id"], actions["team_id"], strict=True):
        opp = _defending_team_for(g, p, team, by_gp)
        if opp is None:
            vals.append(pd.NA)
        else:
            # ``opp`` is a canonical team key drawn FROM this (game, period) group, so the lookup exists;
            # ``.team_id`` is the resolver's raw team (or NA if a hand-built map omitted it).
            vals.append(by_gp[(canonical_id(g), canonical_id(p))][opp].team_id)
    return pd.Series(vals, index=actions.index, dtype="object")


def _index_appearances(
    appearances: pd.DataFrame,
) -> dict[tuple[object, object, object], list[tuple[float, float, object]]]:
    """Group a validated ``KeeperAppearances`` table into ``{(canonical game, period, canonical team)
    -> [(start, end, gk), ...] sorted by start}``.

    Game and team keys are canonical (ADR-055 rule 2 / ADR-019), so an ``Int64`` appearances id
    matches a python-int action id; ``period_id`` is used AS-IS (the ``KeeperIdentityMap`` "period
    used as-is" contract), matching the raw action period the lookup keys on. ``gk`` is stored RAW
    (its provider representation) like the roster/native paths -- canonicalization is for keys and
    comparisons, never the stored value. A NA-team appearance names no keeper interval and is skipped.
    """
    index: dict[tuple[object, object, object], list[tuple[float, float, object]]] = {}
    for g, p, team, gk, start, end in zip(
        appearances["game_id"],
        appearances["period_id"],
        appearances["team_id"],
        appearances["player_id"],
        appearances["start_time_seconds"],
        appearances["end_time_seconds"],
        strict=True,
    ):
        ct = canonical_id(team)
        if ct is pd.NA:
            continue
        index.setdefault((canonical_id(g), p, ct), []).append((float(start), float(end), gk))
    for rows in index.values():
        rows.sort(key=lambda r: r[0])
    return index


def _keeper_covering(rows: Sequence[tuple[float, float, object]], t: float) -> object:
    """Return the ``gk`` of the first interval covering ``t`` (``start <= t < end``), else ``pd.NA``.

    ``rows`` are pre-sorted by start (:func:`_index_appearances`), so the first match is the covering
    interval. An ``end`` of ``+inf`` / NaN is OPEN (unbounded to the period end). A NaN ``t`` (no
    action time) covers nothing -> ``pd.NA``.
    """
    if pd.isna(t):
        return pd.NA
    for start, end, gk in rows:
        open_end = pd.isna(end) or end == float("inf")
        if start <= t and (open_end or t < end):
            return gk
    return pd.NA


def apply_keeper_identities_to_frames(
    frames: pd.DataFrame,
    keeper_map: KeeperIdentityMap,
) -> pd.DataFrame:
    """Bridge resolved keeper ids onto the frames' ``is_goalkeeper`` rows (the R1 identity->frame bridge).

    SB360 freeze-frames carry SYNTHETIC numbered ``player_id`` values (``snapshot_to_tracking_frames``
    numbers rows), so the real keeper id resolved from a roster/goal-kick event matches NO frame row
    and every GK-position feature (``add_pre_shot_gk_position`` matches ``frame.player_id ==
    defending_gk_player_id``) is silently NaN. This stamps the resolved real id onto each non-ball
    ``is_goalkeeper`` row so the match succeeds. Callers apply it only where the frame ids are NOT
    already real (the roster/SB360 path).

    Returns a COPY with the bridged ``player_id``; a row is left UNCHANGED where the map has no entry
    for its ``(game, period, team)`` or the resolved ``gk_id`` is NA. PURE -- ``frames`` is never
    mutated. Identifier keys route through ``id_compat`` (ADR-019); the ``is_goalkeeper`` / ``is_ball``
    masks use ``astype("boolean").fillna(False)`` to avoid the string-qualifier ``astype(bool)`` trap.

    Examples
    --------
    Stamp the resolved keeper id onto the synthetic keeper row; the ball row (``is_ball``) is left
    untouched:

    >>> import pandas as pd
    >>> from silly_kicks.id_compat import canonical_id
    >>> from silly_kicks.keeper_identity import KeeperIdentity, apply_keeper_identities_to_frames
    >>> frames = pd.DataFrame(
    ...     {
    ...         "game_id": [1, 1], "period_id": [1, 1], "frame_id": [0, 0],
    ...         "team_id": [20, pd.NA], "player_id": [7, pd.NA],
    ...         "is_ball": [False, True], "is_goalkeeper": [True, False],
    ...     }
    ... ).astype({"team_id": "Int64", "player_id": "Int64"})
    >>> keeper_map = {
    ...     (canonical_id(1), 1, canonical_id(20)): KeeperIdentity(gk_id=902, source="roster", conflict=False)
    ... }
    >>> apply_keeper_identities_to_frames(frames, keeper_map)["player_id"].tolist()
    [902, <NA>]

    See NOTICE for full bibliographic citations.
    """
    out = frames.copy()

    # `astype("boolean").fillna(False)` guards a string/object qualifier (ADR-019 astype-bool trap)
    # before negating; a bare `astype(bool)` renders the string "false" as True.
    is_gk = out["is_goalkeeper"].astype("boolean").fillna(False)
    is_ball = out["is_ball"].astype("boolean").fillna(False)
    target = (is_gk & ~is_ball).to_numpy(dtype=bool)
    if not target.any():
        return out

    # Fully-canonical map keys, so period/game/team representation never causes a silent miss.
    by_key: dict[tuple[object, object, object], KeeperIdentity] = {
        (canonical_id(g), canonical_id(p), canonical_id(t)): ident for (g, p, t), ident in keeper_map.items()
    }

    new_pid = out["player_id"].copy()
    game_col = out["game_id"].to_numpy()
    period_col = out["period_id"].to_numpy()
    team_col = out["team_id"].to_numpy()
    for i in np.flatnonzero(target):
        pos = int(i)  # np.intp -> int for the .iat positional set
        ident = by_key.get((canonical_id(game_col[pos]), canonical_id(period_col[pos]), canonical_id(team_col[pos])))
        if ident is None:
            continue  # no resolved identity for this keeper's (game, period, team) -> leave as-is
        gk_id: Any = ident.gk_id  # provider-dtype scalar (int/Int64/str); Any so pd.isna + .iat accept it
        if pd.isna(gk_id):
            continue  # a counted-but-unresolved keeper (gk_id is NA) -> leave as-is
        try:
            new_pid.iat[pos] = gk_id
        except (TypeError, ValueError):
            # A roster gk_id whose dtype the player_id column cannot hold (e.g. a str id into an Int64
            # column -- NOT the live SB360 path, where ids are ints). Promote to object so ANY id type
            # bridges; the downstream `frame.player_id == defending_gk_player_id` match is dtype-safe via
            # id_compat regardless of the stored dtype.
            new_pid = new_pid.astype("object")
            new_pid.iat[pos] = gk_id
    out["player_id"] = new_pid
    return out


def _build_report(result_map: KeeperIdentityMap) -> KeeperIdentityReport:
    """Assemble the conserving run-level audit from the resolved map.

    ``source_counts`` is initialised over the FULL ``KEEPER_ID_SOURCE_VALUES`` vocabulary so an
    absent rung reports ``0`` rather than a missing key; ``n_resolved + n_unresolved`` equals the
    number of teams in by construction (every entry is exactly one of the two).
    """
    source_counts = {v: 0 for v in KEEPER_ID_SOURCE_VALUES}
    n_resolved = 0
    n_unresolved = 0
    n_conflict = 0
    for ident in result_map.values():
        source_counts[ident.source] += 1
        if ident.source == KEEPER_ID_SOURCE_UNRESOLVED:
            n_unresolved += 1
        else:
            n_resolved += 1
        if ident.conflict:
            n_conflict += 1
    return KeeperIdentityReport(
        n_teams_in=len(result_map),
        n_resolved=n_resolved,
        n_unresolved=n_unresolved,
        n_conflict=n_conflict,
        source_counts=source_counts,
    )


def _resolve_from_roster(
    actions: pd.DataFrame,
    frames: pd.DataFrame | None,
    roster: dict | None,
) -> tuple[KeeperIdentityMap, KeeperIdentityReport]:
    """SB360 injected-roster + goal-kick-event source-precedence ladder (event > roster).

    ``frames`` may be ``None`` (the event-only path): the seed ``(game, period, team)`` triples and
    the match's teams are then enumerated from ``actions`` themselves instead of the frames' non-ball
    rows. Both branches feed the SAME downstream applicability guard, seed loop and goal-kick
    override, so the frames-supplied path stays byte-identical.

    Pure: never mutates ``actions`` or ``frames`` (reads derived Series / operates on local copies).
    """
    roster = {} if roster is None else roster
    #: ``{canonical team_id -> gk_id}`` -- an ADR-019 id-safe lookup: the frame ``team_id`` dtype
    #: (e.g. ``Int64``) need not match the roster key dtype (e.g. Python ``int``).
    canonical_roster = {canonical_id(k): v for k, v in roster.items()}

    if frames is None:
        # Event-only path: the match's teams and the seed (game, period, team) triples both come
        # from the actions' non-NA team rows. `frame_team_values` and `seed_df` are the SAME shapes
        # the frames branch builds, so every downstream consumer below is source-agnostic.
        has_team = actions["team_id"].notna()
        frame_team_values = actions.loc[has_team, "team_id"].dropna().unique()
        seed_df = (
            actions.loc[has_team, ["game_id", "period_id", "team_id"]].dropna(subset=["team_id"]).drop_duplicates()
        )
    else:
        # The match's teams come from the frames' non-ball rows (ADR-062 numbers SB360 rows, so a
        # real team_id is what an applicable roster is keyed on). `.astype("boolean").fillna(False)`
        # guards a string/object `is_ball` qualifier (ADR-019 astype-bool trap) before negating.
        non_ball = ~frames["is_ball"].astype("boolean").fillna(False)
        frame_team_values = frames.loc[non_ball, "team_id"].dropna().unique()
        seed_df = frames.loc[non_ball, ["game_id", "period_id", "team_id"]].dropna(subset=["team_id"]).drop_duplicates()

    # Roster-APPLICABILITY guard (P3): if NONE of the frame team-ids intersects a roster key, the
    # roster does not describe THIS match -- the synthetic {0,1} fallback (parse.py) is the primary
    # instance, but a wrong-match roster or an unbridgeable dtype land here too. `ids_match` is the
    # only dtype-safe test (raw `==` would silently match nothing). A passing guard proves the
    # roster APPLIES, not that the frames are non-synthetic.
    roster_key_series = pd.Series(list(roster.keys()))
    applies = any(ids_match(roster_key_series, t).any() for t in frame_team_values)
    if not applies:
        present_teams = {canonical_id(t) for t in frame_team_values}
        roster_keys = set(canonical_roster.keys())
        # The teams came from the frames' non-ball rows OR (event-only path) the actions' team rows;
        # name the actual source so the message is accurate regardless of which branch ran above.
        source = "frame" if frames is not None else "action"
        raise ValueError(
            f"roster names none of this match's teams: {source} teams {present_teams}, "
            f"roster keys {roster_keys} (the synthetic {{0,1}} fallback is one instance)"
        )

    # Seed every (game, period, team) witnessed above from the roster; a team with no roster entry
    # stays NA + "unresolved" (counted, never fabricated). `seed_df` was built source-agnostically
    # above (frames' non-ball rows, or actions' team rows on the event-only path).
    result_map: KeeperIdentityMap = {}
    for gid_raw, pid, tid_raw in zip(seed_df["game_id"], seed_df["period_id"], seed_df["team_id"], strict=True):
        tid = canonical_id(tid_raw)
        key = (canonical_id(gid_raw), pid, tid)
        # ``team_id=tid_raw`` stores the keeper's OWN team in its RAW representation (ADR-085) so
        # ``add_defending_gk_player_id`` recovers the authoritative opponent team from the map value
        # rather than the actions -- the map key is canonical, which would otherwise lose the raw id.
        if tid in canonical_roster:
            result_map[key] = KeeperIdentity(
                gk_id=canonical_roster[tid], source=KEEPER_ID_SOURCE_ROSTER, conflict=False, team_id=tid_raw
            )
        else:
            result_map[key] = KeeperIdentity(
                gk_id=pd.NA, source=KEEPER_ID_SOURCE_UNRESOLVED, conflict=False, team_id=tid_raw
            )

    # Goal-kick event override (event > roster). A goal-kick actor names the acting team's keeper,
    # so it beats a stale roster after a substitution. Restricted to (game, period, team) tuples the
    # frames already witnessed -- an event cannot introduce a team the match's frames never showed.
    is_goalkick = (actions["type_name"] == _GOALKICK_TYPE_NAME) & actions["player_id"].notna()
    goalkicks = actions.loc[is_goalkick, ["game_id", "period_id", "team_id", "player_id", "time_seconds"]].copy()
    if not goalkicks.empty:
        # Sort ascending so the LATEST goal kick per group is last (a mid-period sub: the later
        # taker wins). game/team group keys are canonical (ADR-055 rule 2) to MIRROR the seed keys;
        # period is compared RAW on BOTH sides (seed period from frames, override period from actions).
        # INVARIANT (why raw is safe): a match's frames and actions share one numeric period
        # representation (SB360 frames derive period from `actions`; native providers use a single
        # converter), and numeric periods hash-consistently -- so the raw seed/override periods key-match.
        # This is the one place period stays raw, matching the map's public "period_id used as-is" contract.
        goalkicks = goalkicks.sort_values("time_seconds", kind="stable")
        goalkicks["_gid"] = goalkicks["game_id"].map(canonical_id)
        goalkicks["_tid"] = goalkicks["team_id"].map(canonical_id)
        for (gid, pid, tid), grp in goalkicks.groupby(["_gid", "period_id", "_tid"], sort=False):
            key = (gid, pid, tid)
            if key not in result_map:
                continue
            winner_gk = grp["player_id"].iloc[-1]
            distinct_takers = {canonical_id(v) for v in grp["player_id"]}
            prior = result_map[key]
            conflict = False
            if len(distinct_takers) > 1:
                # event-vs-event: two different takers named this keeper in the same period.
                conflict = True
            if prior.source == KEEPER_ID_SOURCE_ROSTER and not same_id(prior.gk_id, winner_gk):
                # roster-vs-event: both named a keeper and they differ.
                conflict = True
            # The override replaces this key's seed entry with the event-sourced keeper but the TEAM
            # is unchanged (same ``(game, period, team)`` key) -- carry the seed's raw ``team_id`` (ADR-085).
            result_map[key] = KeeperIdentity(
                gk_id=winner_gk, source=KEEPER_ID_SOURCE_EVENT, conflict=conflict, team_id=prior.team_id
            )

    return result_map, _build_report(result_map)


def _resolve_from_native(
    actions: pd.DataFrame,
    frames: pd.DataFrame | None,
) -> tuple[KeeperIdentityMap, KeeperIdentityReport]:
    """Native-provider keeper identity by DELEGATING to the TF-13 frame resolvers (ADR-055).

    ``defending_gk_from_frames`` gives each action's OPPONENT keeper ``player_id`` and
    ``acting_gk_from_frames`` its ACTING keeper ``player_id`` -- both frame-linked, both aligned to
    ``actions.index``. Each action therefore witnesses TWO ``(game, period, team) -> keeper`` facts
    (its own team from the acting resolver; the match's OTHER team from the defending resolver). The
    witnesses are reduced per ``(game, period, team)`` to a modal keeper (later-time breaks a tie), and
    the frame's ``is_goalkeeper_source`` for the winning keeper labels the rung ``native`` vs
    ``derived``. A team witnessed by an action but with no resolvable keeper stays NA + ``unresolved``,
    counted (never fabricated).

    Pure: never mutates ``actions`` or ``frames`` (reads derived Series / builds local structures).
    """
    if frames is None:
        # The native path reads keeper POSITIONS from frames (it delegates to the TF-13 frame
        # resolvers), so it cannot run without them -- unlike the event-only roster path.
        raise ValueError("native identity requires frames")

    # Lazy import (see the module-top NOTE): importing these from ``silly_kicks.tracking`` runs the
    # heavy tracking ``__init__``, so it is deferred to the native path -- keeping
    # ``import silly_kicks.keeper_identity`` tracking-free. Patch the definition site
    # ``silly_kicks.tracking._gk_resolve.{defending,acting}_gk_from_frames`` to intercept the delegation.
    from silly_kicks.tracking._gk_resolve import (
        acting_gk_from_frames,
        defending_gk_from_frames,
    )
    from silly_kicks.tracking.utils import link_actions_to_frames

    # The TF-13 resolvers link through ``link_actions_to_frames``, which requires ``source_provider``
    # (used ONLY for the discarded per-provider link-rate report, never for the nearest-frame match or
    # the resolved keeper ids). A minimal native frame set may omit it, so add a placeholder on a LOCAL
    # COPY -- keeping this function pure and the link result byte-identical.
    linkable_frames = frames
    if "source_provider" not in frames.columns:
        linkable_frames = frames.copy()
        linkable_frames["source_provider"] = None

    # Pre-link ONCE and thread the pointers into BOTH TF-13 resolvers (the codebase's `links`
    # thread-through pattern) so delegating to both does not link the same actions to the same frames
    # twice. Byte-identical to letting each resolver link internally: `link_actions_to_frames` is
    # deterministic and both resolvers use the same default tolerance.
    links, _link_report = link_actions_to_frames(actions, linkable_frames)
    def_keeper = defending_gk_from_frames(actions, linkable_frames, links=links)
    act_keeper = acting_gk_from_frames(actions, linkable_frames, links=links)

    # Non-ball frame rows carry the match's real team ids. `.astype("boolean").fillna(False)` guards a
    # string/object `is_ball` qualifier (ADR-019 astype-bool trap) before negating.
    non_ball = ~frames["is_ball"].astype("boolean").fillna(False)

    # {canonical game_id -> set of canonical team_ids}: the opponent of an acting team is the match's
    # OTHER team (2-team match; derived per action from this map).
    teams_by_game: dict[object, set] = {}
    team_frames = frames.loc[non_ball, ["game_id", "team_id"]].dropna(subset=["team_id"])
    for gid_raw, tid_raw in zip(team_frames["game_id"], team_frames["team_id"], strict=True):
        teams_by_game.setdefault(canonical_id(gid_raw), set()).add(canonical_id(tid_raw))

    # {canonical team_id -> RAW team_id}: recover the provider-native team representation from the
    # frames' non-ball rows so a resolved KeeperIdentity carries its own raw team (ADR-085). Every
    # opponent an action derives comes from ``teams_by_game`` (these same rows), so the opponent's raw
    # team is always present here -- the map value is self-sufficient for ``add_defending_gk_player_id``.
    raw_team_by_canonical: dict[object, object] = {}
    for tid_raw in team_frames["team_id"]:
        ct = canonical_id(tid_raw)
        if ct is not pd.NA and ct not in raw_team_by_canonical:
            raw_team_by_canonical[ct] = tid_raw

    # {(canonical game_id, canonical team_id, canonical player_id) -> is_goalkeeper_source}. First
    # non-null wins per keeper (a keeper's source is constant across the frames it appears in). Absent
    # column -> empty map -> every resolved keeper defaults to "native" (a real provider player_id).
    is_gk = frames["is_goalkeeper"].astype("boolean").fillna(False)
    gk_frames = frames[non_ball & is_gk]
    source_by_gtp: dict[tuple[object, object, object], str] = {}
    if "is_goalkeeper_source" in gk_frames.columns:
        for gid_raw, tid_raw, pid_raw, src in zip(
            gk_frames["game_id"],
            gk_frames["team_id"],
            gk_frames["player_id"],
            gk_frames["is_goalkeeper_source"],
            strict=True,
        ):
            if pd.isna(tid_raw) or pd.isna(pid_raw) or pd.isna(src):
                continue
            skey = (canonical_id(gid_raw), canonical_id(tid_raw), canonical_id(pid_raw))
            source_by_gtp.setdefault(skey, src)

    # Collect per-(game, period, team) keeper witnesses. A key is TOUCHED for every acting team AND its
    # derived opponent even when the resolver returned NA, so a witnessed team with no resolvable keeper
    # is counted "unresolved" rather than silently dropped. Each observation is (time, raw_id, canon_id):
    # canonical for grouping/counting (dtype-safe, ADR-019), raw for the STORED gk_id (the roster path
    # stores raw ids too -- canonicalization is for keys/comparison, never the value).
    observations: dict[tuple[object, object, object], list[tuple[float, object, object]]] = {}
    has_time = "time_seconds" in actions.columns

    def _observe(key: tuple[object, object, object], kid: Any, t: float) -> None:
        # ``kid`` is a provider-dtype pandas scalar (``Series.iloc[int]`` -> ``Any``): int/Int64/str.
        obs = observations.setdefault(key, [])
        if not pd.isna(kid):
            obs.append((t, kid, canonical_id(kid)))

    game_ids = actions["game_id"]
    period_ids = actions["period_id"]
    team_ids = actions["team_id"]
    time_vals = actions["time_seconds"] if has_time else None
    for pos, (gid_raw, pid, acting_raw) in enumerate(zip(game_ids, period_ids, team_ids, strict=True)):
        acting = canonical_id(acting_raw)
        if acting is pd.NA:
            continue  # acting team unknown -> cannot key any witness for this action
        gid = canonical_id(gid_raw)
        t_raw = time_vals.iloc[pos] if time_vals is not None else 0.0
        t = float(t_raw) if not pd.isna(t_raw) else 0.0

        acting_key = (gid, pid, acting)
        _observe(acting_key, act_keeper.iloc[pos], t)

        others = teams_by_game.get(gid, set()) - {acting}
        if len(others) == 1:
            (opponent,) = others
            _observe((gid, pid, opponent), def_keeper.iloc[pos], t)

    result_map: KeeperIdentityMap = {}
    for key, obs in observations.items():
        raw_team = raw_team_by_canonical.get(key[2], pd.NA)  # key[2] is the canonical team (ADR-085)
        if not obs:
            result_map[key] = KeeperIdentity(
                gk_id=pd.NA, source=KEEPER_ID_SOURCE_UNRESOLVED, conflict=False, team_id=raw_team
            )
            continue
        canon_ids = [c for _, _, c in obs]
        counts = Counter(canon_ids)
        latest: dict[object, float] = {}
        raw_of: dict[object, object] = {}
        for t, raw, c in obs:
            if c not in latest or t > latest[c]:
                latest[c] = t
            raw_of.setdefault(c, raw)
        conflict = len(counts) > 1
        # Modal keeper wins; a tie in count is broken by the later-time witness (a mid-period sub).
        winner_c = max(counts, key=lambda c: (counts[c], latest[c]))
        winner_raw = raw_of[winner_c]

        game, _period, team = key
        src = source_by_gtp.get((game, team, winner_c))
        source = KEEPER_ID_SOURCE_DERIVED if src == KEEPER_ID_SOURCE_DERIVED else KEEPER_ID_SOURCE_NATIVE
        result_map[key] = KeeperIdentity(gk_id=winner_raw, source=source, conflict=conflict, team_id=raw_team)

    return result_map, _build_report(result_map)
