"""Composable objective functions for the positioning solver (spec section 5).

"The objective function is the tactics." A defensive-shape objective is a ``frame -> float``
scorer where **lower = defensively safer**. The shipped ``positioning_gap`` column commits to
ONE canonical objective (:class:`ThreatObjective`); the others are exploratory (composite / coach
use) per the repo's raw-primitives convention (a composite weighting IS a tactical choice).

See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from typing import Protocol, runtime_checkable

import pandas as pd

# Public seams only (compute_threat_pc computes the surface DIRECTLY, never via PitchControlCache;
# id_compat is the repo-wide dtype-safe id seam, ADR-019).
from silly_kicks.id_compat import canonical_id, ids_match
from silly_kicks.tracking import compute_threat_pc, get_das, pressure_on_target


@runtime_checkable
class Objective(Protocol):
    """A defensive-shape scorer. ``score(frame) -> float`` where lower = defensively safer.

    Examples
    --------
    Any object with a ``score(frame) -> float`` is an ``Objective`` (composable, no God-object)::

        class MyObjective:
            def score(self, frame):
                return float(...)  # lower = defensively safer

        solver = optimise_positions(frame, movable=ids, objective=MyObjective(), constraints=cs)
    """

    def score(self, frame: pd.DataFrame) -> float:
        """The shape's score for this objective (lower = defensively safer).

        Examples
        --------
        ::

            safety = objective.score(frame)  # a float; the solver minimises it
        """
        ...


class ThreatObjective:
    """xT-weighted pitch-control threat the DEFENCE concedes (the shipped-column objective).

    Wraps :func:`~silly_kicks.tracking.compute_threat_pc` with ``method="spearman"`` (GK-aware via
    ``lambda_gk``), oriented by the injected ``goal_map`` and the attacking (opponent) team id.

    Computes the surface DIRECTLY, NEVER via ``PitchControlCache``: the cache key excludes player
    positions, so a moved-defender frame carrying its twin's ``frame_id`` would be served the
    factual surface and every counterfactual delta would collapse to exactly 0 (ADR-043).

    See NOTICE for full bibliographic citations.

    Examples
    --------
    The shipped-column objective, oriented by an injected fitted xT + the resolved goal map::

        from silly_kicks.tracking import resolve_defended_goals

        goal_map = resolve_defended_goals(frame)
        objective = ThreatObjective(xt=xt, goal_map=goal_map, attacking_team_id=opponent_id)
        conceded = objective.score(frame)  # xT-weighted pitch-control threat (lower = safer)
    """

    def __init__(self, *, xt, goal_map, attacking_team_id, params=None) -> None:
        self._xt = xt
        self._goal_map = goal_map
        self._attacking_team_id = attacking_team_id
        self._params = params

    def score(self, frame: pd.DataFrame) -> float:
        """xT-weighted pitch-control threat the defence concedes at ``frame`` (lower = safer).

        Examples
        --------
        ::

            conceded = ThreatObjective(xt=xt, goal_map=goal_map, attacking_team_id=opp).score(frame)
        """
        return float(
            compute_threat_pc(
                frame,
                attacking_team_id=self._attacking_team_id,
                xt=self._xt,
                goal_map=self._goal_map,
                method="spearman",
                params=self._params,
            )
        )


def _nearest_to_ball_player_id(frame: pd.DataFrame):
    """The non-ball player nearest the ball (the presumed carrier), or None."""
    if frame is None or "is_ball" not in frame.columns:
        return None
    ball = frame[frame["is_ball"].astype(bool)]
    players = frame[~frame["is_ball"].astype(bool)].dropna(subset=["x", "y"])
    if len(ball) == 0 or len(players) == 0:
        return None
    bx = float(ball.iloc[0]["x"])
    by = float(ball.iloc[0]["y"])
    d2 = (players["x"].astype(float) - bx) ** 2 + (players["y"].astype(float) - by) ** 2
    return players.loc[d2.idxmin(), "player_id"]


class PressureObjective:
    """Pressing intensity on the ball-carrier (EXPLORATORY -- not the shipped column).

    Locates the carrier (nearest player to the ball) and returns ``-pressure_on_target(carrier)``:
    higher pressure is defensively SAFER, so it maps to a LOWER score under the shared
    "lower = better" contract, and composes with :class:`ThreatObjective` in a :class:`WeightedSum`.
    NaN carrier / NaN pressure propagate.

    ``bekkers_pi`` (default) is velocity-gated -- exploratory, so velocity handling is the compute
    layer's responsibility (Task 8), not this adapter's. See NOTICE for full bibliographic citations.

    Examples
    --------
    Compose a pressing term with the threat objective (both "lower = safer")::

        ws = WeightedSum([(ThreatObjective(xt=xt, goal_map=gm, attacking_team_id=opp), 1.0),
                          (PressureObjective(method="bekkers_pi"), 0.5)])
    """

    def __init__(self, *, method: str = "bekkers_pi", params=None) -> None:
        self._method = method
        self._params = params

    def score(self, frame: pd.DataFrame) -> float:
        """``-pressure_on_target(carrier)`` -- higher pressure = safer = lower score.

        Examples
        --------
        ::

            score = PressureObjective(method="bekkers_pi").score(frame)  # negative of the carrier's pressure
        """
        pid = _nearest_to_ball_player_id(frame)
        if pid is None:
            return float("nan")
        return -float(pressure_on_target(frame, pid, method=self._method, params=self._params))


class DasObjective:
    """Dangerous accessible space the defence concedes (EXPLORATORY -- not the shipped column).

    Wraps :func:`~silly_kicks.tracking.get_das` on the single frame and returns the in-possession
    (attacking) team's DAS scalar -- the dangerous space the defence conceded. More conceded space
    is defensively WORSE, so it maps to a HIGHER score under the "lower = better" contract (no
    negation, unlike :class:`PressureObjective`).

    Requires the ``accessible-space`` optional dependency and velocity (``vx``/``vy`` +
    ``team_in_possession``). Exploratory, so velocity gating is the compute layer's responsibility
    (Task 8). See NOTICE for full bibliographic citations.

    Examples
    --------
    The conceded dangerous-accessible-space of the in-possession team, as a shape objective::

        conceded_space = DasObjective().score(frame)  # higher = more space conceded = worse
    """

    def __init__(self, *, player_in_possession_col: str | None = None) -> None:
        self._ppc = player_in_possession_col

    def score(self, frame: pd.DataFrame) -> float:
        """The in-possession team's dangerous accessible space at ``frame`` (higher = worse).

        Examples
        --------
        ::

            conceded = DasObjective().score(frame)  # attacking team's DAS scalar
        """
        result = get_das(frame) if self._ppc is None else get_das(frame, player_in_possession_col=self._ppc)
        das = result["DAS"].dropna()
        return float(das.iloc[0]) if len(das) else float("nan")


class WeightedSum:
    """A weighted sum of objectives -- itself an :class:`Objective` (composable, no God-object).

    ``score(frame) = sum(weight * objective.score(frame))``. "The objective function is the
    tactics": a counterpress / low-block / man-orientation philosophy is a different weighting,
    not hard-coded engine behaviour.

    Examples
    --------
    >>> class _C:
    ...     def __init__(self, v):
    ...         self.v = v
    ...     def score(self, frame):
    ...         return self.v
    >>> WeightedSum([(_C(2.0), 1.0), (_C(4.0), 0.5)]).score(None)
    4.0
    """

    def __init__(self, terms: Iterable[tuple[Objective, float]]) -> None:
        self._terms = list(terms)

    def score(self, frame: pd.DataFrame) -> float:
        """The weighted sum ``sum(weight * objective.score(frame))``.

        Examples
        --------
        ::

            ws = WeightedSum([(threat_obj, 1.0), (pressure_obj, 0.5)])
            combined = ws.score(frame)
        """
        return float(sum(float(weight) * float(obj.score(frame)) for obj, weight in self._terms))


def _non_ball_player_ids(frame, *, agents: Collection | None) -> list:
    """Non-ball player ids in *frame*, optionally restricted to *agents* (canonical match).

    Returns ``[]`` for ``None`` / an empty / column-less frame, so the caller degrades to the
    whole-value clip rather than fabricating a decomposition.
    """
    if frame is None or not hasattr(frame, "columns") or "player_id" not in frame.columns:
        return []
    players = frame[~frame["is_ball"].astype(bool)] if "is_ball" in frame.columns else frame
    ids = list(players["player_id"].dropna())
    if agents is not None:
        wanted = {canonical_id(a) for a in agents}
        ids = [pid for pid in ids if canonical_id(pid) in wanted]
    return ids


class CappedContribution:
    """Cap any single agent's marginal contribution to a wrapped objective (spec section 2/5).

    The aggregate-averaging fix: a threat+pressure ``WeightedSum`` optimum can abandon a marked
    man because the aggregate averages across agents, so over-committing to a few masks abandoning
    one. Capping how much any single agent's leave-one-out marginal can move the score removes the
    incentive to bank surplus from one zone against a hole in another.

    ``score(frame)``:

    * With decomposable agents -- each agent ``p``'s leave-one-out marginal
      ``m_p = objective(frame) - objective(frame without p)`` is clipped to ``[-cap, +cap]``, and
      the score is ``residual + sum(clip(m_p))`` where ``residual = objective(frame) - sum(m_p)``
      keeps the un-decomposable interaction intact. As ``cap -> inf`` the clip is a no-op and the
      wrapped objective is recovered EXACTLY.
    * With no decomposable agents (``frame is None`` / no player rows) -- the whole value is one
      contribution, clipped to ``[-cap, +cap]``.

    ``agents`` restricts the decomposition to specific ``player_id``s (e.g. the movable defenders);
    ``None`` decomposes over all non-ball players. Itself an :class:`Objective`.

    Examples
    --------
    >>> class _C:
    ...     def __init__(self, v):
    ...         self.v = v
    ...     def score(self, frame):
    ...         return self.v
    >>> CappedContribution(_C(100.0), cap=1.0).score(None)
    1.0
    """

    def __init__(self, objective: Objective, *, cap: float, agents: Collection | None = None) -> None:
        self._obj = objective
        self._cap = float(cap)
        self._agents = None if agents is None else list(agents)

    def _clip(self, v: float) -> float:
        return float(min(max(v, -self._cap), self._cap))

    def score(self, frame: pd.DataFrame) -> float:
        """The wrapped objective with each agent's leave-one-out marginal clipped to +/-cap.

        Examples
        --------
        Cap the movable defenders' contributions to prevent abandoning a marked man::

            capped = CappedContribution(weighted_sum, cap=0.5, agents=movable_ids)
            score = capped.score(frame)
        """
        total = float(self._obj.score(frame))
        agents = _non_ball_player_ids(frame, agents=self._agents)
        if not agents:
            return self._clip(total)
        marginals = [total - float(self._obj.score(frame[~ids_match(frame["player_id"], pid)])) for pid in agents]
        residual = total - sum(marginals)
        return float(residual + sum(self._clip(m) for m in marginals))
