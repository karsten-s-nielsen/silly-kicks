"""TF-56 -- prescriptive defensive-positioning optimiser + measured ``positioning_gap``.

Two layers on one engine:

1. A pure prescriptive solver -- :func:`optimise_positions` searches (simulated annealing)
   for the best *reachable* defensive shape for a single tracking frame under a
   caller-supplied composable objective, subject to per-player time-to-intercept
   reachability. The prescriptive sibling of GKDV's descriptive counterfactual.
2. A measured metric -- :func:`compute_positioning_gap` scores, per defensive frame,
   ``positioning_gap = threat(actual shape) - threat(reachable optimum) >= 0``.

Hexagonal: imports ``silly_kicks.tracking`` PUBLIC seams + ``silly_kicks.id_compat`` /
``silly_kicks.reflection`` ONLY -- never a ``tracking._*`` private. ``xt`` is INJECTED.
Nothing imports ``positioning``.

A ``compute_*``, NOT an ``add_*`` -- the action-coupled aggregator count stays 33; in no
default xfn list; no ``*_xfns``.

See NOTICE for full bibliographic citations (Oonk & Shah databallpy ``optimization`` MIT;
Le et al. 2017 ghosting; Spearman pitch control; Bekkers pressure; Pleuler TTI).
"""

from ._compute import (
    POSITIONING_KEYS,
    POSITIONING_METRIC_COLUMNS,
    POSITIONING_SAMPLE_COLUMNS,
    compute_positioning_gap,
    summarize_positioning_gap,
)
from ._config import PositioningParams, ReachabilityParams, SAParams
from ._constraints import Constraint, ReachabilityConstraint
from ._objectives import CappedContribution, DasObjective, Objective, PressureObjective, ThreatObjective, WeightedSum
from ._optimizer import Optimizer, OptimizeResult, SimulatedAnnealing
from ._report import PositioningReport
from ._solve import optimise_positions

__all__ = [
    "POSITIONING_KEYS",
    "POSITIONING_METRIC_COLUMNS",
    "POSITIONING_SAMPLE_COLUMNS",
    "CappedContribution",
    "Constraint",
    "DasObjective",
    "Objective",
    "OptimizeResult",
    "Optimizer",
    "PositioningParams",
    "PositioningReport",
    "PressureObjective",
    "ReachabilityConstraint",
    "ReachabilityParams",
    "SAParams",
    "SimulatedAnnealing",
    "ThreatObjective",
    "WeightedSum",
    "compute_positioning_gap",
    "optimise_positions",
    "summarize_positioning_gap",
]
