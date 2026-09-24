"""Validation-battery machinery for the commit-2 GO/NO-GO report (spec section 8, AMENDED 2026-09-22).

The original dose / responsiveness instrument was RETIRED. A commit-1 spike MEASURED
``positioning_gap`` NON-MONOTONIC in any dose -- it re-anchors to each defender's REACHABLE set, so a
dose of the actual shape (or of the optimum) shifts that set and the recoverable gap does not rise
with the displacement (dose-actual |gap change| median 0.73 < placebo p95 1.57 on the fixture) -- and
ORTHOGONAL to concurrent shape-badness (corr(threat_actual, gap) = -0.20). "worse position -> larger
gap" is therefore invalid as an instrument. The battery is now:

* Instrument validity (cheap; fixture, no corpus): optimizer STABILITY (the SA optimum is seed- AND
  iteration-invariant, which requires ``SAParams.init_sigma_m`` matched to the reachable radius --
  measured 2.0 m) + DISCRIMINATION (the gap is non-degenerate across a shape spectrum).
* Construct validity (DGX corpus; the ship gate): PREDICTIVE -- a higher ``positioning_gap`` at frame
  t predicts MORE conceded threat over the next window. This is the meaningful construct test PRECISELY
  because the gap is orthogonal to concurrent threat: a positive, significant
  ``corr(positioning_gap_t, conceded_threat_{t+dt})`` is what shows the metric means something.

The predictive verdict is a POOLED-corpus statistic computed in a REDUCE over ALL shards, NEVER per
shard. ``arm_unscoreable`` is a FIRST-CLASS verdict (thin / degenerate domain), distinct from
``not_predictive``.

See NOTICE for full bibliographic citations (Le et al. 2017 ghosting; Oonk & Shah databallpy).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pandas as pd

from silly_kicks.id_compat import ids_match
from silly_kicks.tracking import resolve_defended_goals

from ._compute import compute_positioning_gap
from ._config import PositioningParams, ReachabilityParams, SAParams
from ._constraints import Constraint, ReachabilityConstraint
from ._objectives import CappedContribution, Objective, PressureObjective, ThreatObjective, WeightedSum
from ._optimizer import SimulatedAnnealing
from ._solve import optimise_positions

# --- Pooled-verdict constants (LOCAL registrations; the gkdv values are model-specific) ------------

#: Per-frame gap std below this across a seed set == the SA optimum is SEED-INVARIANT.
STABILITY_TOL: float = 1e-6
#: A discriminating gap distribution has std above this across a shape spectrum.
DISCRIMINATION_MIN_STD: float = 1e-6
#: Two-sided significance level for the predictive correlation.
PREDICTIVE_ALPHA: float = 0.05
#: Pooled-domain floor below which the predictive verdict is `arm_unscoreable` (not "broken").
MIN_DOMAIN_FRAMES: int = 200
#: Seeds swept by :func:`optimizer_stability_verdict` (the SA optimum must not move across them).
DEFAULT_STABILITY_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)
#: Horizons swept by :func:`horizon_sensitivity` -- the gap SCALE rides on the reachability horizon.
HORIZONS_S: tuple[float, float, float] = (0.5, 0.7, 1.0)

STABILITY_VERDICTS: tuple[str, str] = ("stable", "seed_unstable")
DISCRIMINATION_VERDICTS: tuple[str, str] = ("discriminating", "degenerate")
PREDICTIVE_VERDICTS: tuple[str, str, str] = ("predictive", "not_predictive", "arm_unscoreable")


def optimizer_stability_verdict(
    frame: pd.DataFrame,
    *,
    movable,
    objective: Objective,
    constraints,
    seeds: tuple[int, ...] = DEFAULT_STABILITY_SEEDS,
    params: SAParams = SAParams.default(),  # noqa: B008 - frozen singleton default
    tol: float = STABILITY_TOL,
) -> dict:
    """Instrument validity: the SA optimum is seed- AND iteration-invariant (spec section 8).

    Runs :func:`optimise_positions` for each seed (fixed ``params``) and once more at ``seeds[0]`` with
    a doubled iteration budget; returns ``{"verdict", "gap_std", "iter_delta", "gaps"}``. ``stable``
    iff the per-seed gap std ``<= tol`` AND doubling ``num_iterations`` leaves the gap unchanged
    (``<= tol``). At an ``init_sigma_m`` matched to the reachable radius the SA is seed-invariant; a
    too-large ``init_sigma_m`` makes it ``seed_unstable`` (the amended-battery gate; §8).

    Examples
    --------
    >>> # v = optimizer_stability_verdict(frame, movable=[10, 11], objective=obj, constraints=cs)
    """
    seeds = tuple(int(s) for s in seeds)
    gaps: list[float] = []
    for s in seeds:
        r = optimise_positions(
            frame,
            movable=list(movable),
            objective=objective,
            constraints=list(constraints),
            optimizer=SimulatedAnnealing(params),
            seed=s,
        )
        gaps.append(float(r.actual_score - r.best_score))
    gap_std = float(np.std(gaps)) if len(gaps) > 1 else 0.0

    r1 = optimise_positions(
        frame,
        movable=list(movable),
        objective=objective,
        constraints=list(constraints),
        optimizer=SimulatedAnnealing(params),
        seed=seeds[0],
    )
    doubled = dataclasses.replace(params, num_iterations=int(params.num_iterations) * 2)
    r2 = optimise_positions(
        frame,
        movable=list(movable),
        objective=objective,
        constraints=list(constraints),
        optimizer=SimulatedAnnealing(doubled),
        seed=seeds[0],
    )
    iter_delta = abs(float(r1.actual_score - r1.best_score) - float(r2.actual_score - r2.best_score))
    stable = gap_std <= float(tol) and iter_delta <= float(tol)
    return {
        "verdict": "stable" if stable else "seed_unstable",
        "gap_std": gap_std,
        "iter_delta": iter_delta,
        "gaps": gaps,
    }


def discrimination_verdict(gaps, *, min_std: float = DISCRIMINATION_MIN_STD) -> str:
    """Instrument validity: the gap is non-degenerate across a shape spectrum (spec section 8).

    ``discriminating`` iff the finite gaps' std ``> min_std`` (>= 2 finite values); else ``degenerate``
    (all-identical / all-pinned / too few).

    Examples
    --------
    >>> discrimination_verdict([0.5, 1.5, 3.0])
    'discriminating'
    >>> discrimination_verdict([2.0, 2.0, 2.0])
    'degenerate'
    """
    arr = np.asarray(list(gaps), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return "degenerate"
    return "discriminating" if float(np.std(arr)) > float(min_std) else "degenerate"


def predictive_verdict(gap, conceded, *, n_min: int = MIN_DOMAIN_FRAMES, alpha: float = PREDICTIVE_ALPHA) -> dict:
    """Construct validity: does ``positioning_gap`` at t predict conceded threat over ``t+dt``?

    Pooled ``corr(gap, conceded)`` (Pearson) with effect size + significance, over the finite pairs.
    ``predictive`` iff ``r > 0`` AND ``p < alpha``; else ``not_predictive``. ``arm_unscoreable`` (thin
    domain OR a degenerate constant leg) is a FIRST-CLASS verdict distinct from ``not_predictive``.
    Returns ``{"verdict", "r", "p", "n"}``.

    Examples
    --------
    >>> import numpy as np
    >>> g = np.arange(300.0)
    >>> c = 0.5 * g + 1.0
    >>> predictive_verdict(g, c, n_min=100)["verdict"]
    'predictive'
    """
    g = np.asarray(gap, dtype=float)
    c = np.asarray(conceded, dtype=float)
    n = min(g.size, c.size)
    g, c = g[:n], c[:n]
    mask = np.isfinite(g) & np.isfinite(c)
    g, c = g[mask], c[mask]
    if g.size < int(n_min) or float(np.std(g)) == 0.0 or float(np.std(c)) == 0.0:
        return {"verdict": "arm_unscoreable", "r": float("nan"), "p": float("nan"), "n": int(g.size)}
    from scipy.stats import pearsonr

    result: Any = pearsonr(g, c)  # PearsonRResult (scipy >= 1.9); Any: the stub omits .pvalue
    r = float(result.statistic)
    p = float(result.pvalue)
    verdict = "predictive" if (r > 0.0 and p < float(alpha)) else "not_predictive"
    return {"verdict": verdict, "r": r, "p": p, "n": int(g.size)}


def horizon_sensitivity(
    frames: pd.DataFrame,
    *,
    xt,
    horizons: tuple[float, ...] = HORIZONS_S,
    movable=None,
    params: PositioningParams = PositioningParams.default(),  # noqa: B008 - frozen singleton default
) -> dict[float, np.ndarray]:
    """Scored-gap distribution at each reachability horizon (reported, not gated; spec section 8).

    The whole gap SCALE rides on ``max_reach_seconds`` (an intent-set, uncalibrated horizon), so the
    battery reports the distribution at several horizons -- evidence the metric is not an artifact of
    one horizon. Returns ``{horizon_seconds: array of scored positioning_gap values}``.

    Examples
    --------
    >>> # dists = horizon_sensitivity(frames, xt=fitted_xt)  # -> {0.5: array, 0.7: array, 1.0: array}
    """
    out: dict[float, np.ndarray] = {}
    for h in horizons:
        reach = dataclasses.replace(params.reachability, max_reach_seconds=float(h))
        p = dataclasses.replace(params, reachability=reach)
        samples, _ = compute_positioning_gap(frames, xt=xt, movable=movable, params=p)
        scored = samples.loc[samples["positioning_gap_source"] == "scored", "positioning_gap"]
        out[float(h)] = scored.to_numpy(dtype=float)
    return out


def _worst_attacker_exposure(frame: pd.DataFrame, *, xt, goal_map, attacking_team_id) -> float:
    """Max single-attacker threat contribution (leave-one-out) -- the "abandoned man" proxy.

    A defender abandoned to chase an aggregate pressure gain leaves one attacker with a large
    marginal threat contribution; capping each agent's marginal is meant to prevent that trade.
    """
    threat = ThreatObjective(xt=xt, goal_map=goal_map, attacking_team_id=attacking_team_id)
    total = threat.score(frame)
    nonball = frame[~frame["is_ball"].astype(bool)]
    attackers = nonball.loc[ids_match(nonball["team_id"], attacking_team_id), "player_id"].dropna()
    if len(attackers) == 0:
        return float("nan")
    marginals = [total - float(threat.score(frame[~ids_match(frame["player_id"], pid)])) for pid in attackers]
    return float(max(marginals)) if marginals else float("nan")


def averaging_artifact_demo(
    frame: pd.DataFrame,
    *,
    xt,
    attacking_team_id,
    movable,
    pressure_weight: float = 1.0,
    cap: float = 0.5,
    seed: int = 0,
    reachability: ReachabilityParams = ReachabilityParams.default(),  # noqa: B008 - frozen singleton default
) -> dict:
    """Reported (not gated) demonstration of the aggregate-averaging fix (spec section 8).

    Optimises a threat+pressure :class:`WeightedSum` WITHOUT and WITH :class:`CappedContribution`
    over ``movable`` and returns both optima plus the worst single-attacker exposure at each -- the
    capped optimum should not abandon a marked man that the uncapped one does. The cap being
    load-bearing in the SEARCH (the two optima differ) is the non-vacuity check.

    Examples
    --------
    >>> # demo = averaging_artifact_demo(frame, xt=xt, attacking_team_id=2, movable=[10, 11])
    """
    goal_map = resolve_defended_goals(frame)
    threat = ThreatObjective(xt=xt, goal_map=goal_map, attacking_team_id=attacking_team_id)
    pressure = PressureObjective()
    uncapped = WeightedSum([(threat, 1.0), (pressure, float(pressure_weight))])
    capped = CappedContribution(uncapped, cap=float(cap), agents=list(movable))
    constraints: list[Constraint] = [ReachabilityConstraint(reachability)]

    res_uncapped = optimise_positions(
        frame, movable=list(movable), objective=uncapped, constraints=constraints, seed=seed
    )
    res_capped = optimise_positions(frame, movable=list(movable), objective=capped, constraints=constraints, seed=seed)
    return {
        "uncapped_best_frame": res_uncapped.best_frame,
        "capped_best_frame": res_capped.best_frame,
        "uncapped_worst_attacker_exposure": _worst_attacker_exposure(
            res_uncapped.best_frame, xt=xt, goal_map=goal_map, attacking_team_id=attacking_team_id
        ),
        "capped_worst_attacker_exposure": _worst_attacker_exposure(
            res_capped.best_frame, xt=xt, goal_map=goal_map, attacking_team_id=attacking_team_id
        ),
        "optima_differ": not res_uncapped.best_frame.equals(res_capped.best_frame),
    }
