"""Stage-1 carrier-accuracy objective (ruthless Objective; spec §3).

Maximizes the fraction of carrier-actor actions (pass/cross/shot/dribble — actor == ball carrier
by definition) whose inferred ball carrier matches the SPADL actor, averaged with EQUAL WEIGHT per
provider (so match-count imbalance can't dominate). Providers with ~0 matched carrier events are
loudly EXCLUDED (signal_sanity), never silently averaged in.

The accuracy denominator is the set of carrier-actor actions that successfully **link** to a
tracking frame; a linked action with a NaN inferred carrier (the actor fell beyond ``tolerance_m``
of the ball) counts as a MISS, while genuine link failures are excluded. This recall term is what
makes the objective sensitive to ``tolerance_m`` — see ``_match_accuracy`` for the rationale and
the precision-only degeneracy it fixes.

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

# Link tolerance for binding a carrier-actor action to a tracking frame. Mirrors the
# default of ``ball_carrier_at_action`` so the linked set used for the denominator below
# is identical to the one that function links against internally.
_LINK_TOLERANCE_SECONDS = 0.2


def _match_accuracy(actions, frames, *, tolerance_m, beta, gamma) -> tuple[float, int]:
    """Carrier accuracy for one match + the number of carrier-actor actions scored.

    The denominator is the set of carrier-actor actions (pass/cross/shot/dribble — the
    actor IS the ball carrier by definition) that successfully **link** to a tracking
    frame. A linked action whose inferred carrier is NaN — i.e. the actor ended up beyond
    ``tolerance_m`` of the ball, so the model attributed the carrier to nobody — counts as
    a **miss**, not a silent exclusion. Genuine **link** failures (no frame within the link
    tolerance) are excluded, since link success depends only on the fixed link tolerance,
    not on the swept carrier parameters.

    This is what makes the objective sensitive to ``tolerance_m``: an over-tight radius is
    penalized through lost recall (more NaN misses). The earlier formulation averaged only
    over actions where a carrier *was* inferred, which had no recall term — so accuracy rose
    monotonically as the radius shrank and the optimum collapsed onto the search lower bound.
    """
    from silly_kicks.tracking.features import ball_carrier_at_action
    from silly_kicks.tracking.utils import link_actions_to_frames

    if "type_name" in actions.columns:
        mask = actions["type_name"].isin(_CARRIER_ACTION_TYPES)
    else:
        from silly_kicks.spadl.config import actiontypes

        type_ids = {i for i, name in enumerate(actiontypes) if name in _CARRIER_ACTION_TYPES}
        mask = actions["type_id"].isin(type_ids)

    filtered = actions[mask]
    if filtered.empty:
        return float("nan"), 0

    # Linking is independent of the swept (tolerance_m, beta, gamma) — it only depends on the
    # fixed link tolerance. It cleanly separates "could this action be attributed at all"
    # (link success) from "did the carrier model attribute it correctly" (what the swept
    # params control). Restricting the denominator to linked actions is the recall gate.
    pointers, _report = link_actions_to_frames(filtered, frames, tolerance_seconds=_LINK_TOLERANCE_SECONDS)
    linked_ids = pointers.loc[pointers["frame_id"].notna(), "action_id"]
    linked_mask = np.asarray(filtered["action_id"].isin(linked_ids).to_numpy())
    n_linked = int(linked_mask.sum())
    if n_linked == 0:
        return float("nan"), 0

    inferred = ball_carrier_at_action(
        filtered,
        frames,
        tolerance_seconds=_LINK_TOLERANCE_SECONDS,
        tolerance_m=tolerance_m,
        beta=beta,
        gamma=gamma,
    )
    # Compare as strings to avoid Int64/int64/object dtype mismatch (provider-asymmetric ids).
    # A NaN inference stringifies to "nan", which never equals a real actor id, so a
    # tolerance-induced miss correctly scores as not-matched. np.asarray-wrap the .values
    # (ExtensionArray) so numpy ops + pyright are both happy.
    matched = np.asarray(inferred.astype(str).to_numpy()) == np.asarray(filtered["player_id"].astype(str).to_numpy())
    return float(matched[linked_mask].mean()), n_linked


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
