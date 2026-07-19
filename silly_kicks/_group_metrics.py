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

SCOPE, stated so the next reader does not go looking: the spec §6.1 module concept also
names a permutation band and a power simulator. Those are **PR-3b** and are deliberately
absent here -- this module holds exactly what PR-3 lifted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: A-priori minimum observations per group for a stable within-group term. Single-sourced
#: here because the consuming script both filters on it AND prints it in its report header.
DEFAULT_MIN_N = 20


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
