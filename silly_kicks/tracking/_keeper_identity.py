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
from typing import Any, Literal, NamedTuple, TypeAlias

import numpy as np
import pandas as pd

from silly_kicks._nan_safety import nan_safe_enrichment
from silly_kicks.id_compat import canonical_id, ids_match, same_id

# Bound at MODULE TOP by NAME (ADR-055 single-source): the native path DELEGATES to these TF-13
# frame resolvers rather than re-deriving keeper identity from frames. Importing the names here (not
# ``from . import _gk_resolve`` + attribute access) is what lets a consumer/test patch
# ``silly_kicks.tracking._keeper_identity.defending_gk_from_frames`` as a module attribute.
from ._gk_resolve import acting_gk_from_frames, defending_gk_from_frames
from .utils import link_actions_to_frames

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


class KeeperIdentity(NamedTuple):
    """One resolved keeper identity for a ``(game, period, team)``.

    ``conflict`` records a roster-vs-event disagreement (both named a keeper and they differed);
    ``source`` still records the WINNING rung per precedence, so the disagreement is a separate,
    durable signal, never a lost warning.

    Examples
    --------
    >>> from silly_kicks.tracking import KeeperIdentity, KEEPER_ID_SOURCE_NATIVE
    >>> ident = KeeperIdentity(gk_id=920, source=KEEPER_ID_SOURCE_NATIVE, conflict=False)
    >>> (ident.gk_id, ident.source, ident.conflict)
    (920, 'native', False)
    """

    gk_id: object
    source: str
    conflict: bool


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
    >>> from silly_kicks.tracking import KeeperIdentityReport
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
    frames: pd.DataFrame,
    *,
    identity: Literal["native", "roster"],
    roster: dict | None = None,
) -> tuple[KeeperIdentityMap, KeeperIdentityReport]:
    """Resolve the real keeper identity per ``(game, period, team)``. See module docstring.

    Examples
    --------
    Native providers carry a real keeper ``player_id`` on their ``is_goalkeeper`` frame rows, so
    ``identity="native"`` DELEGATES to the TF-13 frame resolvers (ADR-055 single-source)::

        from silly_kicks.tracking import resolve_keeper_identities

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
) -> pd.DataFrame:
    """Stamp each action's DEFENDING keeper id (the opponent's keeper) from a resolved map.

    The PLACEMENT half of :func:`resolve_keeper_identities` on the ACTION grain: the resolver
    returns a pure ``(game, period, team) -> KeeperIdentity`` map and mutates nothing; this applies
    it. In a two-team match the defending keeper of an action is the OTHER team's keeper, so the
    opponent is derived per ``(game, period)`` from the map itself (the single other team present)
    and looked up in it. Every id build/lookup routes through ``id_compat`` (ADR-019), so the
    action ``team_id`` dtype need not match the map-key dtype.

    Returns a COPY with a new ``defending_gk_player_id`` column; ``pd.NA`` where the opponent is
    unresolvable (no ``(game, period)`` entry, a NaN action team, not exactly one other team, or the
    resolved opponent's ``gk_id`` is itself NA). PURE -- ``actions`` is never mutated.

    Examples
    --------
    Stamp the defending keeper's id from a resolved identity map. The map keys are canonical
    ``(game, period, team)`` tuples, so the opponent lookup is dtype-safe (ADR-019) -- the action's
    ``team_id`` here is a plain ``int`` while the map keys are canonical strings:

    >>> import pandas as pd
    >>> from silly_kicks.id_compat import canonical_id
    >>> from silly_kicks.tracking import KeeperIdentity, add_defending_gk_player_id
    >>> actions = pd.DataFrame(
    ...     {"action_id": [0], "game_id": [1], "period_id": [1], "team_id": [10], "type_name": ["shot"]}
    ... )
    >>> keeper_map = {
    ...     (canonical_id(1), 1, canonical_id(10)): KeeperIdentity(gk_id=901, source="roster", conflict=False),
    ...     (canonical_id(1), 1, canonical_id(20)): KeeperIdentity(gk_id=902, source="roster", conflict=False),
    ... }
    >>> add_defending_gk_player_id(actions, keeper_map)["defending_gk_player_id"].tolist()
    [902]

    See NOTICE for full bibliographic citations.
    """
    # {(canonical game, canonical period) -> {canonical team -> KeeperIdentity}}. Period is
    # canonicalized on BOTH the map-key and the lookup side, so a raw-int map period matches a
    # raw-int action period regardless of representation (int vs np.int64 vs Int64); game/team are
    # re-canonicalized defensively (canonical_id is idempotent on an already-canonical key).
    by_gp: dict[tuple[object, object], dict[object, KeeperIdentity]] = {}
    for (g, p, t), ident in keeper_map.items():
        ct = canonical_id(t)
        if ct is pd.NA:
            continue  # a NA-team map key names no team and can never be an opponent
        by_gp.setdefault((canonical_id(g), canonical_id(p)), {})[ct] = ident

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

    out = actions.copy()
    # object dtype: the stored gk_id keeps its provider representation (roster/native paths store
    # the RAW id) and a heterogeneous NA is representable. Downstream matching against a frame's
    # player_id routes through the dtype-safe `id_compat` seam (ADR-019), so no numeric dtype is
    # required here.
    out["defending_gk_player_id"] = pd.Series(values, index=out.index, dtype="object")
    return out


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
    >>> from silly_kicks.tracking import KeeperIdentity, apply_keeper_identities_to_frames
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
    frames: pd.DataFrame,
    roster: dict | None,
) -> tuple[KeeperIdentityMap, KeeperIdentityReport]:
    """SB360 injected-roster + goal-kick-event source-precedence ladder (event > roster).

    Pure: never mutates ``actions`` or ``frames`` (reads derived Series / operates on local copies).
    """
    roster = {} if roster is None else roster
    #: ``{canonical team_id -> gk_id}`` -- an ADR-019 id-safe lookup: the frame ``team_id`` dtype
    #: (e.g. ``Int64``) need not match the roster key dtype (e.g. Python ``int``).
    canonical_roster = {canonical_id(k): v for k, v in roster.items()}

    # The match's teams come from the frames' non-ball rows (ADR-062 numbers SB360 rows, so a real
    # team_id is what an applicable roster is keyed on). `.astype("boolean").fillna(False)` guards a
    # string/object `is_ball` qualifier (ADR-019 astype-bool trap) before negating.
    non_ball = ~frames["is_ball"].astype("boolean").fillna(False)
    frame_team_values = frames.loc[non_ball, "team_id"].dropna().unique()

    # Roster-APPLICABILITY guard (P3): if NONE of the frame team-ids intersects a roster key, the
    # roster does not describe THIS match -- the synthetic {0,1} fallback (parse.py) is the primary
    # instance, but a wrong-match roster or an unbridgeable dtype land here too. `ids_match` is the
    # only dtype-safe test (raw `==` would silently match nothing). A passing guard proves the
    # roster APPLIES, not that the frames are non-synthetic.
    roster_key_series = pd.Series(list(roster.keys()))
    applies = any(ids_match(roster_key_series, t).any() for t in frame_team_values)
    if not applies:
        frame_teams = {canonical_id(t) for t in frame_team_values}
        roster_keys = set(canonical_roster.keys())
        raise ValueError(
            f"roster names none of this match's teams: frame teams {frame_teams}, "
            f"roster keys {roster_keys} (the synthetic {{0,1}} fallback is one instance)"
        )

    # Seed every (game, period, team) present in the frames from the roster; a team with no roster
    # entry stays NA + "unresolved" (counted, never fabricated).
    result_map: KeeperIdentityMap = {}
    seed_df = frames.loc[non_ball, ["game_id", "period_id", "team_id"]].dropna(subset=["team_id"]).drop_duplicates()
    for gid_raw, pid, tid_raw in zip(seed_df["game_id"], seed_df["period_id"], seed_df["team_id"], strict=True):
        tid = canonical_id(tid_raw)
        key = (canonical_id(gid_raw), pid, tid)
        if tid in canonical_roster:
            result_map[key] = KeeperIdentity(
                gk_id=canonical_roster[tid], source=KEEPER_ID_SOURCE_ROSTER, conflict=False
            )
        else:
            result_map[key] = KeeperIdentity(gk_id=pd.NA, source=KEEPER_ID_SOURCE_UNRESOLVED, conflict=False)

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
            result_map[key] = KeeperIdentity(gk_id=winner_gk, source=KEEPER_ID_SOURCE_EVENT, conflict=conflict)

    return result_map, _build_report(result_map)


def _resolve_from_native(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
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
        if not obs:
            result_map[key] = KeeperIdentity(gk_id=pd.NA, source=KEEPER_ID_SOURCE_UNRESOLVED, conflict=False)
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
        result_map[key] = KeeperIdentity(gk_id=winner_raw, source=source, conflict=conflict)

    return result_map, _build_report(result_map)
