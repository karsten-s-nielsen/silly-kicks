"""The pure prescriptive solver entry point (spec section 5)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ._constraints import Constraint
from ._objectives import Objective
from ._optimizer import Optimizer, OptimizeResult, SimulatedAnnealing


def optimise_positions(
    frame: pd.DataFrame,
    *,
    movable: Sequence,
    objective: Objective,
    constraints: Sequence[Constraint],
    optimizer: Optimizer | None = None,
    seed: int | None = None,
) -> OptimizeResult:
    """Search for the best *reachable* defensive shape for a single tracking ``frame``.

    PURE: never mutates ``frame`` (the optimizer is handed a copy). Deterministic when ``seed`` is
    supplied -- the metric layer seeds from ``(game_id, period_id, frame_id)`` so ``positioning_gap``
    is a pure function of inputs.

    ``movable`` is the set of ``player_id``s the search may reposition (typically the defending
    outfielders; the keeper stays a fixed agent). ``objective`` scores a shape (lower = safer);
    ``constraints`` reject infeasible candidates (e.g. :class:`ReachabilityConstraint`). The
    default optimizer is :class:`SimulatedAnnealing`.

    Returns an :class:`OptimizeResult` whose ``best_score <= actual_score`` by construction, so
    ``actual_score - best_score >= 0`` is a path identity.

    Examples
    --------
    Optimise the reachable defensive shape for one frame::

        from silly_kicks.positioning import (
            optimise_positions, ThreatObjective, ReachabilityConstraint, ReachabilityParams,
        )
        from silly_kicks.tracking import resolve_defended_goals

        goal_map = resolve_defended_goals(frame)
        objective = ThreatObjective(xt=xt, goal_map=goal_map, attacking_team_id=opponent_id)
        result = optimise_positions(
            frame,
            movable=defending_outfielder_ids,
            objective=objective,
            constraints=[ReachabilityConstraint(ReachabilityParams.default())],
            seed=1234,
        )
        gap = result.actual_score - result.best_score  # >= 0
    """
    movable = list(movable)
    if not movable:
        raise ValueError("optimise_positions requires a non-empty `movable` set of player_ids.")
    optimizer = optimizer if optimizer is not None else SimulatedAnnealing()
    rng = np.random.default_rng(seed)
    return optimizer.optimize(
        frame.copy(),
        movable=movable,
        objective=objective,
        constraints=list(constraints),
        rng=rng,
    )
