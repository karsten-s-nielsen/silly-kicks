"""Frame-based defending-GK resolution (TF-13).

Resolves the defending team's goalkeeper player_id from tracking frames
for every action. Standalone composable utility -- callers use for fillna
on events-based defending_gk_player_id or as direct lookup.

See spec: docs/superpowers/specs/2026-05-04-tf13-tf14-defensive-line-design.md s2.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from silly_kicks.id_compat import (
    canonical_id,
    canonical_id_series,
    ids_differ,
    ids_equal,
    ids_match,
    restore_id_dtype,
)
from silly_kicks.spadl import config as spadlconfig

from ._gk_geometry import _truthy_bool
from .utils import link_actions_to_frames

_GOALKICK = spadlconfig.actiontype_id["goalkick"]
_PASS = spadlconfig.actiontype_id["pass"]
_THROW_IN = spadlconfig.actiontype_id["throw_in"]


def gk_distribution_mask(
    actions: pd.DataFrame,
    frames: pd.DataFrame | None = None,
    *,
    resolve_gk: Literal["native", "robust"] = "robust",
    tolerance_seconds: float = 0.2,
) -> pd.Series:
    """Per-action boolean: is this a GK distribution? (goal-kick OR pass/throw-in by the acting GK).

    True for any ``goalkick`` (actor-independent), OR a ``pass``/``throw_in`` whose actor is the
    acting team's goalkeeper. Returns a bool ``pd.Series`` aligned to ``actions.index``.

    Parameters
    ----------
    actions : pd.DataFrame
        SPADL actions. Required columns: ``type_id``, ``player_id``, ``team_id`` (and
        ``period_id``/``time_seconds``/``game_id`` used by the frame link when ``frames`` is given).
    frames : pd.DataFrame | None, default None
        Long-form tracking frames (TRACKING_FRAMES_COLUMNS). ``None`` -> goal-kicks-only (the GK
        open-play-pass term is undetectable without frames; both modes degrade to goal-kicks).
    resolve_gk : {"native", "robust"}, default "robust"
        ``"robust"`` resolves the acting GK per action via :func:`acting_gk_from_frames` (time-accurate,
        roster-identity fallback) -- the default and the resolver the lakehouse pins for its goal-kick
        taker override. ``"native"`` uses a global ``frames[is_goalkeeper]`` (game,team,player)
        set-membership (reproduces the frozen v1 mask byte-for-byte; used by the v1 shim). For the GK-pass
        term ``robust`` is a subset of ``native`` (it tightens stale/substituted keepers, never broadens).
    tolerance_seconds : float, default 0.2
        Frame-link tolerance passed to :func:`acting_gk_from_frames` (robust only).

    Notes
    -----
    Pure (never mutates ``actions``); dtype-safe id matching (ADR-019); NaN actor -> not in scope.
    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> import pandas as pd
    >>> from silly_kicks.tracking import gk_distribution_mask
    >>> actions = pd.DataFrame(
    ...     {"type_id": [22, 0], "player_id": [10, 1], "team_id": [5, 5]}
    ... )
    >>> gk_distribution_mask(actions, frames=None).tolist()  # goal-kicks-only without frames
    [True, False]
    """
    missing = [c for c in ("type_id", "player_id", "team_id") if c not in actions.columns]
    if missing:
        raise ValueError(f"gk_distribution_mask: actions missing required column(s) {missing}")

    type_id = actions["type_id"].to_numpy()
    is_goalkick = type_id == _GOALKICK
    is_open = np.isin(type_id, (_PASS, _THROW_IN))

    if frames is None:
        return pd.Series(is_goalkick, index=actions.index)

    if resolve_gk == "native":
        actor_is_gk = _native_actor_is_gk(actions, frames)
    elif resolve_gk == "robust":
        acting_gk = acting_gk_from_frames(actions, frames, tolerance_seconds=tolerance_seconds)
        actor_is_gk = ids_equal(actions["player_id"], acting_gk).to_numpy()
    else:
        raise ValueError(f"resolve_gk must be 'native' or 'robust', got {resolve_gk!r}")

    mask = is_goalkick | (is_open & actor_is_gk)
    return pd.Series(mask, index=actions.index)


def _native_actor_is_gk(actions: pd.DataFrame, frames: pd.DataFrame) -> npt.NDArray[np.bool_]:
    """Positional bool array: is each action's (game,team,player) in the frames' is_goalkeeper set?

    Byte-identical to the frozen v1 ``_gk_distribution_mask`` set-membership block (global over all
    frames, NOT the linked frame). dtype-safe via ``canonical_id_series`` (ADR-019).
    """
    gk = frames[frames["is_goalkeeper"].astype(bool) & (~frames["is_ball"].astype(bool))]
    keyed_by_game = "game_id" in actions.columns and "game_id" in frames.columns

    gk_team = canonical_id_series(gk["team_id"]).to_numpy()
    gk_player = canonical_id_series(gk["player_id"]).to_numpy()
    act_team = canonical_id_series(actions["team_id"]).to_numpy()
    act_player = canonical_id_series(actions["player_id"]).to_numpy()
    if keyed_by_game:
        gk_game = canonical_id_series(gk["game_id"]).to_numpy()
        act_game = canonical_id_series(actions["game_id"]).to_numpy()
        gk_set = set(zip(gk_game, gk_team, gk_player, strict=True))
        return np.array([(g, t, p) in gk_set for g, t, p in zip(act_game, act_team, act_player, strict=True)])
    gk_set = set(zip(gk_team, gk_player, strict=True))
    return np.array([(t, p) in gk_set for t, p in zip(act_team, act_player, strict=True)])


def _gk_from_frames_linked(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    same_team: bool,
    tolerance_seconds: float,
) -> pd.Series:
    """Shared body of the two public resolvers: per-action GK player_id from the LINKED frame.

    Links each action to the nearest frame (within tolerance) and returns the ``is_goalkeeper`` row
    whose team matches (``same_team=True``, the ACTING team's GK) or differs (``same_team=False``, the
    OPPOSING/defending GK) from the action's team. Deterministic lowest-player_id tiebreak; output cast
    to frames' ``player_id`` dtype. ``same_team=False`` reproduces the original
    ``defending_gk_from_frames`` byte-for-byte (only the team predicate is parameterized)."""
    pid_dtype = frames["player_id"].dtype

    n = len(actions)
    out = pd.Series(np.full(n, np.nan), index=actions.index, dtype="object")

    if n == 0 or len(frames) == 0:
        return out

    pointers, _report = link_actions_to_frames(actions, frames, tolerance_seconds=tolerance_seconds)

    gk_rows = frames[(frames["is_goalkeeper"] == True) & (~frames["is_ball"])].copy()  # noqa: E712
    if gk_rows.empty:
        return out

    ptr = pointers.merge(
        actions[["action_id", "team_id", "period_id"]],
        on="action_id",
        how="left",
    )
    linked = ptr[ptr["frame_id"].notna()].copy()
    if linked.empty:
        return out

    linked["frame_id_int"] = linked["frame_id"].astype("int64")
    gk_in_frame = linked.merge(
        gk_rows[["period_id", "frame_id", "team_id", "player_id"]].rename(
            columns={"team_id": "gk_team_id", "player_id": "gk_player_id"}
        ),
        left_on=["period_id", "frame_id_int"],
        right_on=["period_id", "frame_id"],
        how="inner",
    )

    # Team predicate, dtype-safe (ADR-019): ids_equal for the acting (same-team) GK, ids_differ for the
    # opposing (defending) GK. ids_differ requires BOTH ids present (NA -> not-differ -> False), which
    # preserves the "unresolved -> NaN" semantics for a NaN action team AND fixes cross-dtype (canonicalizes
    # int-vs-str) so defending picks the true opponent, not the acting team's own keeper. A raw `==` here
    # left the opposing branch (~match_team over an all-False mismatch) selecting every GK -> own keeper.
    # Both helpers are POSITIONAL / non-nullable np.bool_; .to_numpy() masks gk_in_frame positionally
    # (fresh inner-merge -> RangeIndex, so this matches the old index-aligned == on matched dtypes). Do NOT
    # reindex gk_in_frame above without revisiting this.
    if same_team:
        keep = ids_equal(gk_in_frame["gk_team_id"], gk_in_frame["team_id"]).to_numpy()
    else:
        keep = ids_differ(gk_in_frame["gk_team_id"], gk_in_frame["team_id"]).to_numpy()
    picked = gk_in_frame[keep]

    if picked.empty:
        return out

    # Deterministic tiebreak: lowest player_id per action
    best = picked.sort_values("gk_player_id").drop_duplicates("action_id", keep="first")

    action_to_idx = pd.Series(actions.index, index=actions["action_id"].to_numpy())
    for _, row in best.iterrows():
        aid = row["action_id"]
        if aid in action_to_idx.index:
            out.loc[action_to_idx.loc[aid]] = row["gk_player_id"]

    # Restore the frames dtype (shared rule -- see restore_id_dtype).
    out = restore_id_dtype(out, pid_dtype)

    return out


def defending_gk_from_frames(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    tolerance_seconds: float = 0.2,
) -> pd.Series:
    """Per-action defending-GK player_id resolved from tracking frames.

    For each action, links to the nearest frame (within tolerance), finds
    the opposing team's is_goalkeeper=True row, and returns that player_id.

    Parameters
    ----------
    actions : pd.DataFrame
        SPADL actions with action_id, period_id, time_seconds, team_id.
    frames : pd.DataFrame
        Long-form tracking frames (TRACKING_FRAMES_COLUMNS shape).
    tolerance_seconds : float, default 0.2
        Maximum |time_offset| for a valid link.

    Returns
    -------
    pd.Series
        Aligned with actions.index. dtype matches frames' player_id dtype
        (object for kloppy/sportec, int64/Int64 for Gradient Sports).
        NaN where action couldn't link, no opposing-team GK in linked frame,
        or action.team_id is NaN.

    Examples
    --------
    Fill NaN from events-based GK resolution::

        from silly_kicks.tracking.features import defending_gk_from_frames
        gk_series = defending_gk_from_frames(actions, frames)
        actions["defending_gk_player_id"] = (
            actions["defending_gk_player_id"].fillna(gk_series)
        )

    See NOTICE for full bibliographic citations.
    """
    return _gk_from_frames_linked(actions, frames, same_team=False, tolerance_seconds=tolerance_seconds)


def acting_gk_from_frames(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    tolerance_seconds: float = 0.2,
) -> pd.Series:
    """Per-action ACTING-team GK player_id resolved from tracking frames (mirror of
    :func:`defending_gk_from_frames` with the team predicate inverted, CR 2026-07-01).

    For each action, returns the acting team's goalkeeper ``player_id``. Unlike the pure per-frame link
    (which returns NaN whenever the keeper is not detected in the linked frame), this adds an **identity
    fallback**: the acting team's GK is resolved from the roster-stable ``is_goalkeeper`` identity for
    that ``(game_id, team_id)`` even when the keeper is undetected at the linked frame — essential for
    goal-kicks on broadcast tracking, where the keeper is missing at ~40% of event frames. When a
    ``(game, team)`` has more than one ``is_goalkeeper`` identity (a keeper substitution), the one whose
    frames are **nearest-in-time** to the action is chosen.

    Parameters
    ----------
    actions : pd.DataFrame
        SPADL actions with action_id, period_id, time_seconds, team_id (game_id used when present).
    frames : pd.DataFrame
        Long-form tracking frames (TRACKING_FRAMES_COLUMNS shape).
    tolerance_seconds : float, default 0.2
        Maximum |time_offset| for the per-frame link (the identity fallback is time-tolerance-free).

    Returns
    -------
    pd.Series
        Aligned with actions.index; dtype matches frames' player_id dtype. NaN only where the acting
        team has no ``is_goalkeeper`` identity anywhere in the frames, or ``team_id`` is NaN.

    Notes
    -----
    Pure resolver — it never mutates ``actions``. Deciding WHEN to apply it (e.g. overriding a
    goal-kick's NULL taker with the keeper) is the consumer's synthesis step, not this function's.

    See NOTICE for full bibliographic citations.
    """
    out = _gk_from_frames_linked(actions, frames, same_team=True, tolerance_seconds=tolerance_seconds)
    need = out.isna().to_numpy()
    if not need.any() or len(frames) == 0:
        return out

    gk_rows = frames[(frames["is_goalkeeper"] == True) & (~frames["is_ball"])]  # noqa: E712
    if gk_rows.empty:
        return out

    use_game = "game_id" in actions.columns and "game_id" in gk_rows.columns
    # game arrays are dummy (empty) when use_game is False -- only read inside the ``if use_game`` guard,
    # but kept as ndarrays (not None) so the game filter stays statically subscriptable.
    gk_team = gk_rows["team_id"].to_numpy()
    gk_game = gk_rows["game_id"].to_numpy() if use_game else np.zeros(len(gk_rows))
    gk_time = gk_rows["time_seconds"].to_numpy(float)
    gk_pid = gk_rows["player_id"].to_numpy()

    act_team = actions["team_id"].to_numpy()
    act_game = actions["game_id"].to_numpy() if use_game else np.zeros(len(actions))
    act_time = actions["time_seconds"].to_numpy(float)

    # Series views for ids_match, built ONCE (not per loop iteration).
    gk_team_s = pd.Series(gk_team)
    gk_game_s = pd.Series(gk_game)

    # int(): np.where yields np.intp; `out.iloc[i] = ...` below wants a plain int index.
    for i in (int(_i) for _i in np.where(need)[0]):
        t = act_team[i]
        if pd.isna(t):
            continue  # NaN team -> unresolvable (stays NaN)
        sel = ids_match(gk_team_s, t).to_numpy()
        if use_game:
            sel = sel & ids_match(gk_game_s, act_game[i]).to_numpy()
        if not sel.any():
            continue  # no acting-team GK identity anywhere
        cand_pid = gk_pid[sel]
        distinct = pd.unique(cand_pid)
        if len(distinct) == 1:
            out.iloc[i] = distinct[0]
        else:  # GK sub: nearest-in-time identity
            cand_time = gk_time[sel]
            out.iloc[i] = cand_pid[np.abs(cand_time - act_time[i]).argmin()]

    return out


class GoalEndUnresolvedError(ValueError):
    """The goal map cannot resolve the end a computation needs.

    Raised by per-frame functions, which REQUIRE a resolvable map; caught by name at the
    ``add_*`` edge, which emits a NaN row plus provenance. Keeping the decision in one place
    matters: having the aggregator pre-check would duplicate the exact lookup the callee is
    about to perform, and the two copies can drift on which accessor and which ``allow_guess``.

    Examples
    --------
    A ``ValueError`` subclass, so existing handlers keep working while new code can catch the
    specific case by name:

    >>> from silly_kicks.tracking import GoalEndUnresolvedError
    >>> issubclass(GoalEndUnresolvedError, ValueError)
    True
    >>> try:
    ...     raise GoalEndUnresolvedError("team 2 has no end in (game=1, period=1)")
    ... except ValueError as exc:
    ...     print(type(exc).__name__)
    GoalEndUnresolvedError
    """


@dataclass(frozen=True)
class GoalMap:
    """Defended goal end per ``(game_id, period_id, team_id)``, with provenance.

    Three MUTUALLY EXCLUSIVE states -- the ladder:

    * ``resolved``   -- GK mean-x is finite
    * ``guessed``    -- GK mean-x is not finite AND outfield mean-x IS finite (N1 coverage)
    * ``unresolved`` -- NA team identity, or every x NaN: in NEITHER mapping

    Guessed ends are opt-in via ``allow_guess`` rather than merged in, because a caller that
    discards provenance is the defect this seam exists to end.

    **Keys are canonical, and canonical means STRING** (``canonical_id(1) == "1"``). Never
    hold the mappings as a plain dict and index them with raw ids -- ``("1", "1", "2")`` does
    not equal ``(1, 1, 2)``, so such a lookup MISSES silently. Use the accessors, which
    canonicalize on the way in.

    ``frozen=True`` freezes the binding, not the mapping, hence ``MappingProxyType``.

    Examples
    --------
    Resolve once per match, then ask for an own end and an attacked end::

        goal_map = resolve_defended_goals(frames)
        own = goal_map.get(game_id, period_id, team_id, allow_guess=True)
        att = goal_map.attacked_goal(game_id, period_id, team_id, allow_guess=True)
    """

    resolved: Mapping[tuple, float]
    guessed: Mapping[tuple, float]
    unresolved: frozenset

    def __post_init__(self) -> None:
        strict = dict(self.resolved)
        loose = {**dict(self.guessed), **strict}
        by_period: dict[str, dict[tuple, dict]] = {}
        for label, pool in (("strict", strict), ("loose", loose)):
            idx: dict[tuple, dict] = {}
            for (game, period, team), end in pool.items():
                idx.setdefault((game, period), {})[team] = end
            by_period[label] = idx
        # Derived caches, not state: computed once so `attacked_goal`/`ends_in_period` do not
        # rebuild and linearly scan a merged dict on every call inside a per-frame loop.
        object.__setattr__(self, "_strict", strict)
        object.__setattr__(self, "_loose", loose)
        object.__setattr__(self, "_by_period", by_period)

    @staticmethod
    def _key(game_id, period_id, team_id) -> tuple:
        return (canonical_id(game_id), canonical_id(period_id), canonical_id(team_id))

    def _pool(self, allow_guess: bool) -> dict:
        return self._loose if allow_guess else self._strict  # type: ignore[attr-defined]

    def _period(self, game, period, allow_guess: bool) -> dict:
        label = "loose" if allow_guess else "strict"
        return self._by_period[label].get((game, period), {})  # type: ignore[attr-defined]

    def get(self, game_id, period_id, team_id, *, allow_guess: bool = False) -> float | None:
        """The end THIS team defends, or ``None`` when it does not resolve.

        Examples
        --------
        Team 1's keeper stands at x=4, so team 1 defends the x=0 end. The lookup canonicalizes,
        so the id may arrive in any dtype:

        >>> import pandas as pd
        >>> from silly_kicks.tracking import resolve_defended_goals
        >>> frames = pd.DataFrame(
        ...     {
        ...         "game_id": [1] * 4,
        ...         "period_id": [1] * 4,
        ...         "team_id": [1, 1, 2, 2],
        ...         "is_ball": [False] * 4,
        ...         "is_goalkeeper": [True, False, True, False],
        ...         "x": [4.0, 40.0, 101.0, 65.0],
        ...         "y": [34.0] * 4,
        ...     }
        ... )
        >>> goal_map = resolve_defended_goals(frames)
        >>> goal_map.get(1, 1, 1, allow_guess=True)
        0.0
        >>> goal_map.get("1", "1", "1", allow_guess=True)
        0.0
        >>> goal_map.get(1, 1, 99, allow_guess=True) is None
        True
        """
        key = self._key(game_id, period_id, team_id)
        if key[2] is pd.NA:
            return None
        return self._pool(allow_guess).get(key)

    def attacked_goal(self, game_id, period_id, team_id, *, allow_guess: bool = False) -> float | None:
        """The end this team ATTACKS -- i.e. the end its OPPONENT defends.

        A real lookup of the opponent's entry, never ``105.0 - get(...)``: the arithmetic
        identity is a second implementation of the rule, and it is wrong on a degenerate map.

        Returns ``None`` when the ``(game, period)`` does not resolve to exactly one opponent,
        **or when that opponent's end equals this team's own end**. The second guard is not
        redundant -- in the degenerate case there IS exactly one opponent, so a count-only
        check passes and the answer would say this team attacks the goal it defends.

        Examples
        --------
        Team 1 defends x=0, so it ATTACKS x=105 -- read off its opponent's entry, not by
        subtracting its own from the pitch length:

        >>> import pandas as pd
        >>> from silly_kicks.tracking import resolve_defended_goals
        >>> frames = pd.DataFrame(
        ...     {
        ...         "game_id": [1] * 4,
        ...         "period_id": [1] * 4,
        ...         "team_id": [1, 1, 2, 2],
        ...         "is_ball": [False] * 4,
        ...         "is_goalkeeper": [True, False, True, False],
        ...         "x": [4.0, 40.0, 101.0, 65.0],
        ...         "y": [34.0] * 4,
        ...     }
        ... )
        >>> goal_map = resolve_defended_goals(frames)
        >>> goal_map.attacked_goal(1, 1, 1, allow_guess=True)
        105.0
        >>> goal_map.attacked_goal(1, 1, 2, allow_guess=True)
        0.0
        """
        game, period, team = self._key(game_id, period_id, team_id)
        if team is pd.NA:
            return None
        ends = self._period(game, period, allow_guess)
        opponents = [end for other, end in ends.items() if other != team]
        if len(opponents) != 1:
            return None
        own = ends.get(team)
        if own is not None and opponents[0] == own:
            return None
        return opponents[0]

    def ends_in_period(self, game_id, period_id, *, allow_guess: bool = False) -> dict:
        """``{team_id: defended_end}`` for one ``(game, period)``.

        Examples
        --------
        Keys come back CANONICAL (strings), which is what makes a raw-tuple lookup against the
        underlying mappings miss:

        >>> import pandas as pd
        >>> from silly_kicks.tracking import resolve_defended_goals
        >>> frames = pd.DataFrame(
        ...     {
        ...         "game_id": [1] * 4,
        ...         "period_id": [1] * 4,
        ...         "team_id": [1, 1, 2, 2],
        ...         "is_ball": [False] * 4,
        ...         "is_goalkeeper": [True, False, True, False],
        ...         "x": [4.0, 40.0, 101.0, 65.0],
        ...         "y": [34.0] * 4,
        ...     }
        ... )
        >>> goal_map = resolve_defended_goals(frames)
        >>> goal_map.ends_in_period(1, 1, allow_guess=True)
        {'1': 0.0, '2': 105.0}
        """
        return dict(self._period(canonical_id(game_id), canonical_id(period_id), allow_guess))

    @property
    def n_resolved(self) -> int:
        """How many ``(game, period, team)`` ends came from a finite GK mean-x.

        Examples
        --------
        Both teams have a keeper with finite coordinates here, so both ends are RESOLVED
        rather than guessed from outfield positions:

        >>> import pandas as pd
        >>> from silly_kicks.tracking import resolve_defended_goals
        >>> frames = pd.DataFrame(
        ...     {
        ...         "game_id": [1] * 4,
        ...         "period_id": [1] * 4,
        ...         "team_id": [1, 1, 2, 2],
        ...         "is_ball": [False] * 4,
        ...         "is_goalkeeper": [True, False, True, False],
        ...         "x": [4.0, 40.0, 101.0, 65.0],
        ...         "y": [34.0] * 4,
        ...     }
        ... )
        >>> goal_map = resolve_defended_goals(frames)
        >>> goal_map.n_resolved
        2
        """
        return len(self.resolved)

    @property
    def n_guessed(self) -> int:
        """How many ends fell back to the outfield mean-x (the ladder's N1 rung).

        Examples
        --------
        Zero on frames where every keeper is tracked; a non-zero count is the signal that
        ``allow_guess=True`` is doing load-bearing work for this match:

        >>> import pandas as pd
        >>> from silly_kicks.tracking import resolve_defended_goals
        >>> frames = pd.DataFrame(
        ...     {
        ...         "game_id": [1] * 4,
        ...         "period_id": [1] * 4,
        ...         "team_id": [1, 1, 2, 2],
        ...         "is_ball": [False] * 4,
        ...         "is_goalkeeper": [True, False, True, False],
        ...         "x": [4.0, 40.0, 101.0, 65.0],
        ...         "y": [34.0] * 4,
        ...     }
        ... )
        >>> goal_map = resolve_defended_goals(frames)
        >>> goal_map.n_guessed
        0
        """
        return len(self.guessed)


def _end_from_mean_x(mean_x: float) -> float:
    """The defended end implied by a mean x. THE one place this choice is made.

    Both the GK-derived and the outfield-guessed branches route through here, so the goal-end
    population gate sees exactly one derivation for the whole package.
    """
    return 0.0 if mean_x < spadlconfig.field_length / 2.0 else spadlconfig.field_length


def resolve_defended_goals(frames: pd.DataFrame) -> GoalMap:
    """Build the pinned goal map. THE single implementation of the rule.

    Consumers that need a goal side must call this rather than re-derive it: a second
    implementation is a fork that can disagree with the first, and this repo carried ten of
    them. ``tests/tracking/test_goal_map_population.py`` pins the population.

    **Build ONCE per match, from the FULL frames.** The quantity is the MEAN GK x per
    ``(game, period, team)`` and the mean is the robustness; building from a single frame is a
    different estimator, and the cost is PROVIDER-DEPENDENT. Measured on the committed slim
    fixtures, a per-frame map disagrees with the per-match one on 7.1% of team-frames
    (skillcorner) / 2.2% (metrica) / 0.0% (sportec, gradientsports), and :meth:`GoalMap.attacked_goal`
    is unresolvable for 35.7% / 61.7% / 0.0%. The damage concentrates in sparse broadcast
    detection -- SkillCorner sees a keeper in ~19.6% of frames -- which is exactly the provider
    class this seam serves. (ADR-055 records why the spec's 78.8% headline is not cited here: it
    does not reproduce on these fixtures.)

    N1 (retained from the original): GK identification quality is provider-variable, so a
    ``(game, period, team)`` with no GK rows falls back to the team's mean outfield x -- but as
    ``guessed``, so consuming the guess is a decision the caller makes explicitly.

    Examples
    --------
    Resolve once, then thread it into the per-frame functions::

        goal_map = resolve_defended_goals(frames)
        influence = compute_gk_influence(frame, team_id, gk_id, xt, goal_map=goal_map)
    """
    is_ball = _truthy_bool(frames["is_ball"])
    players = frames[~is_ball]
    if players.empty:
        return GoalMap(MappingProxyType({}), MappingProxyType({}), frozenset())

    is_gk = _truthy_bool(players["is_goalkeeper"])
    keys = ["game_id", "period_id", "team_id"]
    gk_mean = players[is_gk].groupby(keys, dropna=False)["x"].mean()
    all_mean = players.groupby(keys, dropna=False)["x"].mean()

    resolved: dict = {}
    guessed: dict = {}
    unresolved: set = set()
    for raw_key, outfield_mean in all_mean.items():
        # `Series.items()` is typed `Hashable`, but a groupby over THREE keys always yields a
        # 3-tuple; the cast states that rather than restructuring the loop around the type stub.
        key = tuple(canonical_id(part) for part in cast("tuple", raw_key))
        if key[2] is pd.NA:
            unresolved.add(key)
            continue
        gk_x = gk_mean.get(raw_key, np.nan)
        if np.isfinite(gk_x):
            resolved[key] = _end_from_mean_x(gk_x)
        elif np.isfinite(outfield_mean):
            guessed[key] = _end_from_mean_x(outfield_mean)
        else:
            # Every x NaN: `nan < 52.5` is False, so the old code returned 105.0 silently.
            unresolved.add(key)
    return GoalMap(MappingProxyType(resolved), MappingProxyType(guessed), frozenset(unresolved))
