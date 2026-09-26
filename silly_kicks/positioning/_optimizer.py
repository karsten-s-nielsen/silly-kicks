"""Simulated-annealing optimizer for the positioning solver (spec section 5).

The prescriptive search: perturb ONE movable player per step (Gaussian, decaying sigma), reject a
candidate that fails any constraint (never scored), Metropolis accept-worse with a decaying
temperature, patience early-stop on a scored-step plateau.

The ACTUAL shape is the iteration-0 incumbent: ``actual_score`` is that incumbent-0 evaluation --
the SAME code path as ``best_score`` -- so ``best_score <= actual_score`` holds by CONSTRUCTION,
and the metric's ``gap = actual_score - best_score >= 0`` is a path identity, not a comparison of
two independent evaluations. Feasibility is always tested against the player's REAL position +
velocity (the input frame), never a mid-search intermediate.

See NOTICE for full bibliographic citations (Oonk & Shah, databallpy ``optimization``, MIT).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from silly_kicks.id_compat import ids_match

from ._config import SAParams
from ._constraints import Constraint
from ._objectives import Objective


@dataclass(frozen=True)
class OptimizeResult:
    """Result of one :meth:`Optimizer.optimize` call.

    ``best_frame`` -- the input frame with the movable players at their best-found positions.
    ``best_score`` -- the objective at ``best_frame`` (``<= actual_score`` by construction).
    ``actual_score`` -- the objective at the FACTUAL shape (the incumbent-0 evaluation).
    ``n_iter`` -- iterations run (``<= num_iterations``; fewer if patience early-stopped).
    ``n_feasible_proposals`` -- proposals that passed every constraint and were scored.
    ``converged`` -- True iff the run stopped on the patience plateau (else it ran out of iters).

    Examples
    --------
    >>> OptimizeResult(None, 1.0, 2.0, 10, 3, True).best_score
    1.0
    """

    best_frame: pd.DataFrame
    best_score: float
    actual_score: float
    n_iter: int
    n_feasible_proposals: int
    converged: bool


@runtime_checkable
class Optimizer(Protocol):
    """Searches for the best reachable shape under an objective + constraints.

    Examples
    --------
    Any object with an ``optimize(frame, *, movable, objective, constraints, rng)`` is an
    ``Optimizer``; :class:`SimulatedAnnealing` is the built-in::

        optimizer = SimulatedAnnealing()
        result = optimise_positions(frame, movable=ids, objective=obj, constraints=cs, optimizer=optimizer)
    """

    def optimize(
        self,
        frame: pd.DataFrame,
        *,
        movable: Sequence,
        objective: Objective,
        constraints: Sequence[Constraint],
        rng: np.random.Generator,
    ) -> OptimizeResult:
        """Search for the best reachable shape; return an :class:`OptimizeResult`.

        Examples
        --------
        Driven by :func:`optimise_positions`, which supplies a seeded ``rng``::

            import numpy as np

            result = SimulatedAnnealing().optimize(
                frame, movable=ids, objective=obj, constraints=cs, rng=np.random.default_rng(0)
            )
            gap = result.actual_score - result.best_score  # >= 0
        """
        ...


class SimulatedAnnealing:
    """Metropolis simulated annealing over movable-player positions.

    Deterministic given ``rng``: the metric layer seeds it from ``(game_id, period_id, frame_id)``
    so ``positioning_gap`` is a pure function of inputs.

    Examples
    --------
    >>> SimulatedAnnealing(SAParams(num_iterations=10, patience=5))  # doctest: +ELLIPSIS
    <silly_kicks.positioning._optimizer.SimulatedAnnealing object at ...>
    """

    def __init__(self, params: SAParams = SAParams.default()) -> None:  # noqa: B008 - frozen singleton default
        self._params = params

    def optimize(
        self,
        frame: pd.DataFrame,
        *,
        movable: Sequence,
        objective: Objective,
        constraints: Sequence[Constraint],
        rng: np.random.Generator,
    ) -> OptimizeResult:
        """Run the SA search; ``best_score <= actual_score`` by construction (gap >= 0).

        Examples
        --------
        Seed the ``rng`` for a deterministic result::

            import numpy as np

            result = SimulatedAnnealing(SAParams(num_iterations=500)).optimize(
                frame, movable=ids, objective=obj, constraints=cs, rng=np.random.default_rng(1)
            )
        """
        p = self._params
        real = frame  # read-only: feasibility is ALWAYS vs the real position + velocity
        movable = list(movable)

        # Iteration-0 incumbent: the factual shape, via the SAME code path as every trial score.
        actual_score = float(objective.score(real))

        # F1b (ADR-106): `current`/`trial` are scratch COMPUTE frames the SA loop mutates via `.at`.
        # If `frame` stores float32 coords, a float64 `.at` assign raises pandas 3's LossySetitemError,
        # so work the mutable copy in float64 (compute dtype; nothing here is persisted). `real` is left
        # untouched (read-only, feasibility vs the real position).
        current = real.copy()
        for _c in ("x", "y"):
            if _c in current.columns:
                current[_c] = current[_c].astype("float64")
        current_score = actual_score
        best = real.copy()
        best_score = actual_score

        # Pre-resolve each movable player's row index in `current` (stable across the run).
        row_of: dict = {}
        for pid in movable:
            idx = current.index[ids_match(current["player_id"], pid) & ~current["is_ball"].astype(bool)]
            if len(idx):
                row_of[pid] = idx[0]
        movable = [pid for pid in movable if pid in row_of]

        temperature = float(p.init_temperature)
        sigma = float(p.init_sigma_m)
        n_feasible = 0
        n_iter = 0
        since_improve = 0
        converged = False

        if not movable:
            return OptimizeResult(best, best_score, actual_score, 0, 0, False)

        for _ in range(int(p.num_iterations)):
            n_iter += 1
            pid = movable[int(rng.integers(len(movable)))]
            di = row_of[pid]
            cx = float(current.at[di, "x"])  # type: ignore[arg-type]  (pandas .at -> Scalar; a real float here)
            cy = float(current.at[di, "y"])  # type: ignore[arg-type]
            dx, dy = rng.normal(0.0, sigma, 2)
            candidate = (cx + float(dx), cy + float(dy))

            temperature *= p.cooling
            sigma *= p.cooling

            # Feasibility vs the REAL position + velocity, never the mid-search `current`.
            if not all(c.is_feasible(pid, candidate, real) for c in constraints):
                continue  # rejected proposal -- does not advance the patience plateau

            n_feasible += 1
            trial = current.copy()  # float64 compute frame (see `current` above) -> `.at` is lossless
            trial.at[di, "x"] = candidate[0]
            trial.at[di, "y"] = candidate[1]
            new_score = float(objective.score(trial))

            delta = new_score - current_score
            if delta <= 0.0 or rng.random() < math.exp(-delta / max(temperature, 1e-12)):
                current = trial
                current_score = new_score

            if new_score < best_score:
                best = trial.copy()
                best_score = new_score
                since_improve = 0
            else:
                since_improve += 1
                if since_improve >= int(p.patience):
                    converged = True
                    break

        return OptimizeResult(best, best_score, actual_score, n_iter, n_feasible, converged)
