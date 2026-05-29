"""Stage-1 carrier-accuracy objective (ruthless Objective; spec §3).

Maximizes the fraction of carrier-actor actions (pass/cross/shot/dribble — actor == ball carrier
by definition) whose inferred ball carrier matches the SPADL actor, averaged with EQUAL WEIGHT per
provider (so match-count imbalance can't dominate). Providers with ~0 matched carrier events are
loudly EXCLUDED (signal_sanity), never silently averaged in.

Examples
--------
>>> from silly_kicks.calibration._carrier_objective import CarrierAccuracyObjective
>>> from ruthless import Candidate
>>> # obj = CarrierAccuracyObjective(fold)  # fold: {provider: [(actions, frames, home_team_id)]}
>>> # obj.evaluate(Candidate(id="t0", params={"tolerance_m": 3.0, "beta": 0.5, "gamma": 1.0}))
"""

from __future__ import annotations

import numpy as np
from ruthless.result import Candidate, Metrics

from silly_kicks.calibration._gates import signal_sanity

_CARRIER_ACTION_TYPES = {"pass", "cross", "shot", "dribble"}


def _match_accuracy(actions, frames, *, tolerance_m, beta, gamma) -> tuple[float, int]:
    """Carrier accuracy for one match + the number of carrier-actor actions compared."""
    from silly_kicks.tracking.features import ball_carrier_at_action

    if "type_name" in actions.columns:
        mask = actions["type_name"].isin(_CARRIER_ACTION_TYPES)
    else:
        from silly_kicks.spadl.config import actiontypes

        type_ids = {i for i, name in enumerate(actiontypes) if name in _CARRIER_ACTION_TYPES}
        mask = actions["type_id"].isin(type_ids)

    filtered = actions[mask]
    if filtered.empty:
        return float("nan"), 0
    inferred = ball_carrier_at_action(filtered, frames, tolerance_m=tolerance_m, beta=beta, gamma=gamma)
    # Compare as strings to avoid Int64/int64/object dtype mismatch (provider-asymmetric ids).
    # np.asarray-wrap the .values (ExtensionArray) so numpy ops + pyright are both happy.
    matched = np.asarray(inferred.astype(str).to_numpy()) == np.asarray(filtered["player_id"].astype(str).to_numpy())
    valid = np.asarray(inferred.notna().to_numpy())
    n = int(valid.sum())
    if n == 0:
        return float("nan"), 0
    return float(matched[valid].mean()), n


class CarrierAccuracyObjective:
    """ruthless ``Objective`` (maximize ``carrier_accuracy``).

    Examples
    --------
    >>> from silly_kicks.calibration._carrier_objective import CarrierAccuracyObjective
    >>> from ruthless import Candidate
    >>> # obj = CarrierAccuracyObjective({"skillcorner": [(actions, frames, home)]})  # doctest: +SKIP
    >>> # obj.evaluate(Candidate(id="t0", params={"tolerance_m": 3.0, "beta": 0.5, "gamma": 1.0}))
    """

    def __init__(self, fold: dict[str, list[tuple]]) -> None:
        # fold: {provider: [(actions, frames, home_team_id), ...]}
        self._fold = fold
        self.diagnostics: dict = {}  # surfaced into the manifest (M1)

    def evaluate(self, candidate: Candidate) -> Metrics:
        """Equal-provider-weight carrier accuracy for a candidate's (tolerance_m, beta, gamma).

        Examples
        --------
        >>> # obj.evaluate(Candidate(id="t0", params={"tolerance_m": 3.0,  # doctest: +SKIP
        >>> #     "beta": 0.5, "gamma": 1.0}))["carrier_accuracy"]
        """
        p = candidate.params
        tolerance_m, beta, gamma = float(p["tolerance_m"]), float(p["beta"]), float(p["gamma"])
        per_provider: dict[str, float] = {}
        total_compared: dict[str, int] = {}
        for provider, matches in self._fold.items():
            accs, weights = [], []
            for actions, frames, _home in matches:
                acc, n = _match_accuracy(actions, frames, tolerance_m=tolerance_m, beta=beta, gamma=gamma)
                if n > 0 and not np.isnan(acc):
                    accs.append(acc)
                    weights.append(n)
            # Record compared-count for EVERY provider in the fold (0 if none) so signal_sanity sees it.
            total_compared[provider] = int(sum(weights))
            if accs:
                # Within a provider, weight by compared-action count; ACROSS providers, equal weight.
                per_provider[provider] = float(np.average(accs, weights=weights))
        # M1: loudly EXCLUDE providers with ~0 matched carrier events (the old GS=0.0 failure mode),
        # never silently averaged in.
        kept, excluded = signal_sanity({pr: float(n) for pr, n in total_compared.items()}, min_value=1.0)
        self.diagnostics["excluded_providers"] = excluded
        per_provider = {pr: per_provider[pr] for pr in kept if pr in per_provider}
        if not per_provider:
            return {"carrier_accuracy": 0.0}
        metrics: Metrics = {"carrier_accuracy": float(np.mean(list(per_provider.values())))}
        for provider, acc in per_provider.items():
            metrics[f"carrier_accuracy__{provider}"] = acc
            metrics[f"n_compared__{provider}"] = float(total_compared[provider])
        return metrics
