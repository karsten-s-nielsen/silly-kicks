"""Domain-free grouped statistics: one-way ICC + group spread.

Lifted from ``scripts/xtgk_v2_keeper_discrimination.py`` at TF-19 PR-3 so the library --
not a script -- is the single source. **The reason is internal.** An earlier draft justified
the lift by "``gkdv/`` cannot import from ``scripts/``"; that edge never materialised
(``grep -rn "_group_metrics" silly_kicks/gkdv/`` returns nothing, and ``_validate.py``
computes no ICC). The real reason is that these statistics previously lived in an
**unversioned script** which the test suite could not import and the published wheel does
not ship -- ``scripts/`` is outside the ``silly_kicks/`` package. The shipped consumers are
``scripts/xtgk_v2_keeper_discrimination.py`` and the ``tests/gkdv/`` + ``tests/xtgk/``
suites, so the ``scripts/`` dependency is now inverted and the statistics are under test.
Mirrors the ``silly_kicks/_calibration_metrics.py`` precedent, whose docstring likewise
records an internal-consumers-only lift.

PRIVATE (decision D1). It carries no stability promise and **has no downstream consumer**:
the lakehouse confirmed on 2026-07-18 that it neither imports nor plans to import these
statistics -- per-keeper aggregates are dbt models there, and an ICC is a model-validation
statistic they consume as a verdict rather than compute. That is an OBSERVED PATTERN, not a
prediction: their statistical gates already live lakehouse-side in
``src/analytics/xg_calibration.py`` (per-provider discrimination gate, n-aware calibration
test), which is where an ICC would land if they ever wanted one. If that ever changes they
will say so; promoting this to ``silly_kicks/group_metrics.py`` + ``_PUBLIC_MODULE_FILES``
is then a deliberate, requested step -- not something to pre-empt.

TERMINOLOGY (a real cross-repo false-friend): the "discrimination" this module measures is
GROUP-VARIANCE discrimination (ICC -- does the metric separate keepers?). It is NOT the
CLASSIFIER discrimination (ROC-AUC) that the same word denotes ~25 times in the lakehouse
and in our own xG/PSxG gates. Do not conflate them when grepping either repo.

SCOPE: this module holds domain-free grouped statistics -- ICC, spread, and the ICC POWER
SIMULATOR (:func:`icc_power_curve`). The power sim was originally deferred to PR-3b, and that
deferral was **overridden** by the sign-off cycle: TF-19 spec §6.1 registers the power curve as a
PRECONDITION on the ICC gate ("the gate is registered only if detection at the anchor ... is >=
0.8"), so ``gkdv/_validate.py`` shipped ``ICC_ANCHORS`` carrying a docstring that promises "a power
curve is reported at all three" while no code could produce one. The permutation BAND remains
PR-3b and is deliberately absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: A-priori minimum observations per group for a stable within-group term. Single-sourced
#: here because the consuming script both filters on it AND prints it in its report header.
DEFAULT_MIN_N = 20


def _group_centre(values, groups) -> tuple[np.ndarray, np.ndarray, int]:
    """Group-centred values + the group index, computed ONCE per corpus.

    ``bincount``, not a per-group boolean scan: the scan form made the CI smoke a 35 s test, and
    §6.1 mandates a fast one.
    """
    vals = np.asarray(values, dtype=float)
    keys, inv = np.unique(np.asarray(groups), return_inverse=True)
    sums = np.bincount(inv, weights=vals, minlength=len(keys))
    counts = np.bincount(inv, minlength=len(keys))
    return vals - (sums / counts)[inv], inv, len(keys)


def _inject_group_effect(centred, inv, n_keys: int, within_var: float, target_icc: float, rng) -> np.ndarray:
    """Add a group-level effect sized so the between-group variance share equals ``target_icc``.

    Takes the PRE-CENTRED values so the invariant work is hoisted out of the replicate loop.
    """
    if target_icc <= 0.0 or within_var <= 0.0:
        return centred
    between_var = target_icc / (1.0 - target_icc) * within_var
    return centred + rng.normal(0.0, np.sqrt(between_var), size=n_keys)[inv]


def _icc_fast(values: np.ndarray, codes: np.ndarray, n_codes: int) -> float:
    """Numpy-only ICC(1), identical in value to :func:`icc_one_way`.

    A dedicated fast path EXISTS because the permutation loop evaluates this thousands of times and
    ``icc_one_way`` builds a DataFrame plus a ``groupby.apply`` per call. ``icc_one_way`` is shipped,
    consumer-tested and deliberately UNTOUCHED; equivalence is not assumed but gated by
    ``test_fast_icc_matches_the_shipped_icc_exactly``.
    """
    counts = np.bincount(codes, minlength=n_codes)
    sums = np.bincount(codes, weights=values, minlength=n_codes)
    keep = counts >= 2  # a group needs >=2 observations to contribute a within term
    if keep.sum() < 2:
        return float("nan")
    row_keep = keep[codes]
    v = values[row_keep]
    ng = counts[keep].astype(float)
    means = (sums[keep] / counts[keep]).astype(float)
    n, g = len(v), int(keep.sum())
    if g < 2 or n <= g:
        return float("nan")
    grand = float(v.mean())
    ssb = float((ng * (means - grand) ** 2).sum())
    # within-group SS via the per-row group mean -- no per-group Python loop
    remap = np.full(n_codes, -1, dtype=int)
    remap[np.flatnonzero(keep)] = np.arange(g)
    ssw = float(((v - means[remap[codes[row_keep]]]) ** 2).sum())
    msb, msw = ssb / (g - 1), ssw / (n - g)
    n0 = (n - (ng**2).sum() / n) / (g - 1)  # unbalanced correction
    denom = msb + (n0 - 1) * msw
    return float((msb - msw) / denom) if denom != 0 else float("nan")


def _block_permuter(groups, blocks):
    """Precompute the block -> representative-group-CODE map ONCE.

    Invariant across permutations, so hoisting it out of the replicate loop is what keeps the CI
    smoke fast (§6.1 mandates one). Integer codes rather than labels keep the permuted grouping
    directly consumable by :func:`_icc_fast` with no string round-trip.
    """
    codes = np.unique(np.asarray(groups), return_inverse=True)[1]
    blocks = np.asarray(blocks)
    bkeys, block_inv = np.unique(blocks, return_inverse=True)
    first_idx = np.zeros(len(bkeys), dtype=int)
    seen = np.zeros(len(bkeys), dtype=bool)
    for i, b in enumerate(block_inv):
        if not seen[b]:
            seen[b] = True
            first_idx[b] = i
    return codes[first_idx], block_inv


def _permute_groups_by_block(rep_codes, block_inv, rng) -> np.ndarray:
    """Match-block label permutation: shuffle which GROUP attaches to each BLOCK, so within-block
    clustering survives. An i.i.d. shuffle of observations would not."""
    return rng.permutation(rep_codes)[block_inv]


def icc_power_curve(values, groups, blocks, *, anchors, n_replicates, alpha: float = 0.05, rng_seed: int = 0) -> dict:
    """Plasmode power to DETECT a group-level ICC at each anchor (TF-19 spec §6.1, §5.3).

    Real values, real clustering, injected KNOWN effects -- never an i.i.d. simulation, which would
    "inherit none of the clustering and could pass while the real instrument is simultaneously
    underpowered and anti-conservative".

    The input's block structure is load-bearing: if every group sits in exactly one block, the
    block permutation is a pure relabelling of an identical partition and the null equals the
    observed statistic exactly, so nothing is detectable at any anchor.

    Returns
    -------
    dict
        ``power`` (anchor -> detected fraction), ``mean_observed_icc``, ``mean_null_icc``,
        ``mean_observed_icc_at_zero`` (the non-vacuity reference), ``n_replicates``, ``alpha``.
    """
    rng = np.random.default_rng(rng_seed)
    # Everything invariant across replicates is hoisted here (spec §6.1's "fast CI smoke").
    rep_codes, block_inv = _block_permuter(groups, blocks)
    centred, inv, n_keys = _group_centre(values, groups)
    within_var = float(np.var(centred, ddof=0))
    power: dict[float, float] = {}
    mean_icc: dict[float, float] = {}
    mean_null: dict[float, float] = {}
    zero_iccs: list[float] = []
    for anchor in anchors:
        detected, obs, nulls = 0, [], []
        for _ in range(int(n_replicates)):
            injected = _inject_group_effect(centred, inv, n_keys, within_var, float(anchor), rng)
            observed = _icc_fast(injected, inv, n_keys)
            null = np.array(
                [_icc_fast(injected, _permute_groups_by_block(rep_codes, block_inv, rng), n_keys) for _ in range(30)]
            )
            detected += int(observed > float(np.quantile(null, 1.0 - alpha)))
            obs.append(observed)
            nulls.append(float(np.mean(null)))
        power[anchor] = detected / float(n_replicates)
        mean_icc[anchor] = float(np.mean(obs))
        # Reported so the block-structure claim is assertable: an i.i.d. blocking collapses this
        # toward zero while a real one holds it up (measured 20/20 seeds at both anchors).
        mean_null[anchor] = float(np.mean(nulls))
        if float(anchor) == 0.0:
            zero_iccs = obs
    if not zero_iccs:
        zero_iccs = [
            _icc_fast(_inject_group_effect(centred, inv, n_keys, within_var, 0.0, rng), inv, n_keys) for _ in range(10)
        ]
    return {
        "power": power,
        "mean_observed_icc": mean_icc,
        "mean_null_icc": mean_null,
        "mean_observed_icc_at_zero": float(np.mean(zero_iccs)),
        "n_replicates": int(n_replicates),
        "alpha": float(alpha),
    }


def icc_one_way(values: np.ndarray, groups: np.ndarray) -> float:
    """One-way random-effects ICC(1) from OBSERVATION-level values grouped by key: between-group
    variance as a fraction of total. Higher => the metric separates groups. Partitions variance
    from the raw observation-level values (NOT per-group means -- that has no within-group term).

    Examples
    --------
    Two groups, three observations each. What a high ICC means is that the WITHIN-group
    spread is small relative to the gap between groups -- identical group means with noisy
    members score far lower, which is exactly why this reads observation-level values::

        import numpy as np
        from silly_kicks._group_metrics import icc_one_way

        groups = np.array(["A", "A", "A", "B", "B", "B"])
        icc_one_way(np.array([0.10, 0.11, 0.09, 0.50, 0.51, 0.49]), groups)
        # ~0.999 -- groups cleanly separated

        icc_one_way(np.array([0.10, 0.90, -0.70, 0.50, 1.30, -0.30]), groups)
        # ~-0.26 -- SAME group means, but within-group noise swamps the gap.
        # ICC(1) is not bounded below at 0; a negative value reads as "no separation".

    A group with a single observation carries no within-group term, so it is DROPPED rather
    than allowed to inflate the estimate. Fewer than two surviving groups returns NaN --
    "not measurable", never 0.0, which would read as "measured, and flat"::

        icc_one_way(np.array([0.1, 0.2]), np.array(["A", "B"]))  # nan (both singletons)
    """
    df = pd.DataFrame({"v": np.asarray(values, float), "g": np.asarray(groups)}).dropna()
    g_sizes = df.groupby("g")["v"].transform("size")
    df = df[g_sizes >= 2]  # a group needs >=2 observations to contribute a within term
    grp = df.groupby("g")["v"]
    ng, means = grp.count().to_numpy(float), grp.mean().to_numpy(float)
    n, g = len(df), len(ng)
    if g < 2 or n <= g:
        return float("nan")
    grand = df["v"].mean()
    ssb = float((ng * (means - grand) ** 2).sum())
    ssw = float(grp.apply(lambda s: ((s - s.mean()) ** 2).sum()).sum())
    msb, msw = ssb / (g - 1), ssw / (n - g)
    n0 = (n - (ng**2).sum() / n) / (g - 1)  # unbalanced correction
    denom = msb + (n0 - 1) * msw
    return float((msb - msw) / denom) if denom != 0 else float("nan")


def group_spread(values: np.ndarray, keys: np.ndarray, *, min_n: int = DEFAULT_MIN_N) -> dict:
    """ICC (observation-level) + CV (per-group means, secondary/unstable) + per-group mean ranking.

    Renamed from ``keeper_spread`` at lift time: nothing in the body is keeper-specific.
    The returned dict keeps its original keys (``n_keepers``) -- the consuming report renders
    them, and renaming a payload key is a separate, consumer-visible change.

    Examples
    --------
    ``min_n`` filters groups BEFORE anything is computed, so ``n_keepers`` counts the
    SURVIVORS: a thinly-sampled group leaves the ranking entirely rather than contributing
    an unstable mean to it::

        import numpy as np
        from silly_kicks._group_metrics import group_spread

        values = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.5])
        keys = np.array(["A", "A", "A", "B", "B", "B", "C"])

        out = group_spread(values, keys, min_n=3)
        out["n_keepers"]                 # 2 -- C (n=1) never reaches the statistics
        [r[0] for r in out["ranking"]]   # ["B", "A"] -- (key, mean, count), mean-descending

    When fewer than two groups survive the filter the DECLARED SHAPE still comes back -- NaN
    metrics and an empty ranking, not an exception -- so a caller can render "not measurable"
    for a thin cohort without special-casing::

        group_spread(values, keys, min_n=100)
        # {"icc": nan, "cv": nan, "n_keepers": 0, "ranking": []}

    ``cv`` is computed on per-group MEANS and is the secondary, unstable figure; ``icc``
    (observation-level, via :func:`icc_one_way`) is the one to lead with.
    """
    df = pd.DataFrame({"v": np.asarray(values, float), "k": np.asarray(keys)}).dropna()
    cnt = df.groupby("k")["v"].transform("size")
    df = df[cnt >= min_n]
    if df["k"].nunique() < 2:
        return {"icc": float("nan"), "cv": float("nan"), "n_keepers": int(df["k"].nunique()), "ranking": []}
    icc = icc_one_way(df["v"].to_numpy(), df["k"].to_numpy())
    per = df.groupby("k")["v"].agg(["mean", "count"]).sort_values("mean", ascending=False)
    m = per["mean"].to_numpy()
    cv = float(np.std(m) / abs(np.mean(m))) if np.mean(m) != 0 else float("nan")
    ranking = [(str(k), float(r["mean"]), int(r["count"])) for k, r in per.iterrows()]
    return {"icc": icc, "cv": cv, "n_keepers": len(per), "ranking": ranking}
