"""Plasmode ATT power (TF-19 sign-off package §5.4).

Generic by design: the GKDV-specific constants live in ``silly_kicks/gkdv/_validate.py``, and the
domain-free ICC power curve lives in ``silly_kicks/_group_metrics.py``. The split is by import
direction -- this module needs the causal estimators, a domain-free statistics module must not.

FIREWALL (spec §5.1). Once Layer 2's design is expressible in code, this module could also RUN it,
producing the H1-vs-H2 answer BEFORE the sign-off meant to authorise it -- from a cycle whose whole
premise is pre-registration. :func:`att_power_curve` therefore takes **no outcome vector at all**: it
accepts an :class:`InjectionSpec` recipe and draws the outcome itself, so an observed outcome is not
merely refused but unrepresentable. A call-count spy on ``estimate_att`` would NOT catch a breach,
because the harness always calls it; the guard has to be provenance, and its RED side is demonstrated
in ``tests/causal/test_power.py``.

Examples
--------
Power to detect a 20 % relative lift on a 15 % base rate::

    import numpy as np
    from silly_kicks.causal.power import InjectionSpec, att_power_curve

    rng = np.random.default_rng(0)
    Z = rng.integers(0, 2, size=400)
    out = att_power_curve(
        Z=Z,
        injection=InjectionSpec(base_rate=0.15, relative_effect=0.20),
        X=rng.normal(size=(400, 3)),
        clusters=np.repeat(np.arange(20), 20),
        sizes=(400,),
        n_replicates=20,
        rng_seed=0,
    )
    out["power_by_size"][400]
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from silly_kicks.causal.matching import estimate_att, fit_propensity

_STAMP = "silly_kicks.causal.power:injected:v1"


@dataclass(frozen=True)
class InjectionSpec:
    """A RECIPE for a known effect -- not a frozen realisation.

    The outcome vector is drawn INSIDE the replicate loop (spec §5.4: "Per replicate: ... inject a
    treatment effect"), so replicates differ in outcome noise rather than being one dataset
    reordered. Passing a recipe rather than a vector is also what makes the FIREWALL structural:
    :func:`att_power_curve` accepts no ``Y``, so an observed outcome cannot be smuggled in.

    Examples
    --------
    A 20 % relative lift on a 15 % base rate is an absolute ATT of 0.03:

    >>> spec = InjectionSpec(base_rate=0.15, relative_effect=0.20)
    >>> round(spec.true_effect, 4)
    0.03

    Drawing is deterministic in the supplied generator, and the treated arm's rate is higher:

    >>> import numpy as np
    >>> Z = np.repeat([0, 1], 5000)
    >>> y = spec.draw(Z, np.random.default_rng(0))
    >>> bool(y[Z == 1].mean() > y[Z == 0].mean())
    True
    """

    base_rate: float
    relative_effect: float
    stamp: str = field(default=_STAMP)

    @property
    def true_effect(self) -> float:
        """The absolute ATT the recipe induces, in outcome-probability units.

        Examples
        --------
        >>> round(InjectionSpec(base_rate=0.15, relative_effect=0.20).true_effect, 4)
        0.03
        >>> InjectionSpec(base_rate=0.15, relative_effect=0.0).true_effect
        0.0
        """
        return float(self.base_rate) * float(self.relative_effect)

    def draw(self, Z, rng) -> np.ndarray:
        """Bernoulli outcomes at ``base_rate``, lifted by ``relative_effect`` x base rate if treated.

        Examples
        --------
        >>> import numpy as np
        >>> Z = np.repeat([0, 1], 4000)
        >>> y = InjectionSpec(base_rate=0.2, relative_effect=0.5).draw(Z, np.random.default_rng(0))
        >>> sorted(np.unique(y).tolist())  # a binary outcome
        [0.0, 1.0]
        >>> bool(y[Z == 1].mean() > y[Z == 0].mean())  # the treated arm is lifted
        True
        """
        Z = np.asarray(Z)
        p = np.full(Z.shape, float(self.base_rate), dtype=float)
        p[Z == 1] = float(self.base_rate) * (1.0 + float(self.relative_effect))
        return (rng.random(Z.shape) < np.clip(p, 0.0, 1.0)).astype(float)


def _require_spec(injection) -> InjectionSpec:
    if not isinstance(injection, InjectionSpec) or getattr(injection, "stamp", None) != _STAMP:
        raise ValueError(
            "att_power_curve was given something that is not an InjectionSpec (spec §5.1 FIREWALL): "
            "computing power on the OBSERVED outcome would answer Layer 2 before sign-off."
        )
    return injection


def _resample_clusters(clusters, ukeys, target_size: int, rng) -> np.ndarray:
    """Resample WHOLE clusters until the target row count is reached -- never individual rows.

    Cluster-preserving for the same reason the ICC null is match-blocked: an i.i.d. row resample
    would inherit none of the clustering the real design carries.
    """
    picked, total = [], 0
    for k in rng.permutation(ukeys):
        members = np.flatnonzero(clusters == k)
        picked.append(members)
        total += members.size
        if total >= target_size:
            break
    return np.concatenate(picked)[:target_size]


def att_power_curve(*, Z, injection, X, clusters, sizes, n_replicates, alpha_z=2.0, rng_seed=0) -> dict:
    """Cluster-resampled plasmode power for an ATT on a binary outcome.

    ``matched_n`` is an OUTPUT, never a dial: matching CONSUMES units and YIELDS a focal count, so
    each replicate records ``(matched_n, detected)`` and power is binned by the requested subsample
    size. Detection reuses the registered ``|ATT| / SE >= 2`` rule so the power target and the gate
    it informs agree by construction.

    Returns
    -------
    dict
        ``power_by_size``, ``matched_n_by_size``, ``n_distinct_outcome_draws`` (the per-replicate
        redraw evidence), ``true_effect``, ``base_rate``, ``n_replicates``.

    Examples
    --------
    The FIREWALL: an outcome vector is not a valid argument at all.

    >>> import numpy as np
    >>> att_power_curve(
    ...     Z=np.array([0, 1]), injection=np.zeros(2), X=np.zeros((2, 1)),
    ...     clusters=np.array([0, 0]), sizes=(2,), n_replicates=1,
    ... )
    Traceback (most recent call last):
        ...
    ValueError: att_power_curve was given something that is not an InjectionSpec...

    A recipe is, and every replicate draws its own outcome:

    >>> rng = np.random.default_rng(0)
    >>> Z = rng.integers(0, 2, size=200)
    >>> out = att_power_curve(
    ...     Z=Z, injection=InjectionSpec(base_rate=0.2, relative_effect=0.5),
    ...     X=rng.normal(size=(200, 2)), clusters=np.repeat(np.arange(10), 20),
    ...     sizes=(200,), n_replicates=5, rng_seed=0,
    ... )
    >>> out["n_distinct_outcome_draws"][200]
    5
    """
    spec = _require_spec(injection)
    rng = np.random.default_rng(rng_seed)
    Z = np.asarray(Z)
    X = np.asarray(X, dtype=float)
    clusters = np.asarray(clusters)
    ukeys = np.unique(clusters)
    power: dict[int, float] = {}
    matched: dict[int, int] = {}
    draws: dict[int, int] = {}
    for size in sizes:
        detected, m_ns, seen = 0, [], set()
        for _ in range(int(n_replicates)):
            idx = _resample_clusters(clusters, ukeys, int(size), rng)
            y = spec.draw(Z[idx], rng)  # FRESH per replicate (spec §5.4)
            seen.add(hash(y.tobytes()))
            ps, _ = fit_propensity(X[idx], Z[idx], seed=int(rng.integers(0, 2**31 - 1)))
            est = estimate_att(y, Z[idx], ps, X[idx])
            # est.n_focal is the MATCHED FOCAL (treated) count -- NOT idx.size, which is the
            # subsample size the resampler was asked for and would echo the input identically.
            m_ns.append(int(est.n_focal))
            if est.se and np.isfinite(est.se) and est.se > 0:
                detected += int(abs(est.estimate) / est.se >= alpha_z)
        power[size] = detected / float(n_replicates)
        matched[size] = int(np.mean(m_ns))
        draws[size] = len(seen)
    return {
        "power_by_size": power,
        "matched_n_by_size": matched,
        "n_distinct_outcome_draws": draws,
        "true_effect": spec.true_effect,
        "base_rate": float(spec.base_rate),
        "n_replicates": int(n_replicates),
    }
