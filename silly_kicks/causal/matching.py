"""Pure propensity-score matching estimators (ADR-015), numpy + sklearn, no R, no new dep.

1:1 nearest-neighbor matching on the propensity score, WITH REPLACEMENT, ties allowed,
NO caliper (paper-faithful: Cao et al. 2025, arXiv:2505.11841). Variance via the
Abadie-Imbens (2006) matching estimator (Imbens & Rubin 2015, Ch. 19) -- naive/bootstrap
SEs are biased under matching-with-replacement. Determinism via explicit seeds.

See NOTICE for the citation + the state-vs-sender faithfulness caveat; ADR-015 for the two
named approximations (J=1 within-group sigma^2; estimated-PS variance is conservative).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

Target = Literal["att", "atnt"]

# Pre-registered "non-trivial" GK-ablation shift floor + the placebo band percentile (named +
# asserted so they cannot silently drift). Reported, never a CI gate.
GK_ABLATION_MIN_SHIFT = 0.01
PLACEBO_BAND_PERCENTILE = 95.0


@dataclass(frozen=True)
class CausalEstimate:
    estimate: float
    se: float
    balance: pd.DataFrame  # covariate / smd_pre / smd_post
    n_focal: int
    matched: dict[int, int]


def fit_propensity(X: np.ndarray, Z: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Logistic propensity e(x)=P(Z=1|X) on STANDARDIZED covariates (M6: raw confounders are
    multi-scale -- metres/radians/counts -- and lbfgs is not scale-robust). Returns (scores in
    (0,1), coefficients on the standardized scale).

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[0.0], [1.0], [2.0], [3.0]])
    >>> Z = np.array([0, 0, 1, 1])
    >>> ps, _ = fit_propensity(X, Z, seed=0)
    >>> bool(np.all((ps > 0) & (ps < 1)))
    True
    """
    X = np.asarray(X, dtype=float)
    Z = np.asarray(Z, dtype=int)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Xs = (X - mu) / sd
    # Default penalty is l2; pass C only (an explicit penalty="l2" is deprecated in sklearn 1.8+ and
    # floods FutureWarnings -- behaviour is identical to the default l2 path). Near-unregularized (C=1e6).
    clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000, random_state=seed)
    clf.fit(Xs, Z)
    ps = np.clip(clf.predict_proba(Xs)[:, 1], 1e-6, 1 - 1e-6)
    return ps, clf.coef_.ravel()


def propensity_match(ps: np.ndarray, Z: np.ndarray, *, target: Target) -> dict[int, int]:
    """1:1 NN match on the propensity score, with replacement, ties allowed, no caliper.

    ``target="att"``: each TREATED -> nearest CONTROL ({treated_idx: control_idx}).
    ``target="atnt"``: each CONTROL -> nearest TREATED. No unit dropped. NN tie-break is
    index-order deterministic, so no seed is needed (L1).

    Examples
    --------
    >>> import numpy as np
    >>> propensity_match(np.array([0.9, 0.1, 0.88]), np.array([1, 0, 0]), target="att")
    {0: 2}
    """
    ps = np.asarray(ps, dtype=float).reshape(-1, 1)
    Z = np.asarray(Z, dtype=int)
    treated, control = np.where(Z == 1)[0], np.where(Z == 0)[0]
    focal, pool = (treated, control) if target == "att" else (control, treated)
    if len(pool) == 0 or len(focal) == 0:
        return {}
    nn = NearestNeighbors(n_neighbors=1).fit(ps[pool])
    _, idx = nn.kneighbors(ps[focal])
    return {int(focal[i]): int(pool[idx[i, 0]]) for i in range(len(focal))}


def smd_balance(X: np.ndarray, Z: np.ndarray, matched: dict[int, int], *, target: Target) -> pd.DataFrame:
    """Standardized mean differences per covariate, before vs after matching. SMD =
    (mean_focal - mean_comp) / pooled_sd. Post-match uses the matched comparison set (with
    with-replacement multiplicity). Lower |SMD| == better balance.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[0.0], [0.1], [1.0], [1.1]])
    >>> Z = np.array([1, 1, 0, 0])
    >>> m = propensity_match(np.array([0.2, 0.3, 0.25, 0.35]), Z, target="att")
    >>> list(smd_balance(X, Z, m, target="att").columns)
    ['covariate', 'smd_pre', 'smd_post']
    """
    X = np.asarray(X, dtype=float)
    Z = np.asarray(Z, dtype=int)
    focal_val, comp_val = (1, 0) if target == "att" else (0, 1)
    focal = np.where(Z == focal_val)[0]
    comp = np.where(Z == comp_val)[0]
    matched_comp = np.array([matched[int(i)] for i in focal], dtype=int)
    rows = []
    for k in range(X.shape[1]):
        xf, xc_all, xc_m = X[focal, k], X[comp, k], X[matched_comp, k]
        sd = np.sqrt((xf.var(ddof=1) + xc_all.var(ddof=1)) / 2.0) or 1.0
        rows.append(
            {"covariate": k, "smd_pre": (xf.mean() - xc_all.mean()) / sd, "smd_post": (xf.mean() - xc_m.mean()) / sd}
        )
    return pd.DataFrame(rows, columns=["covariate", "smd_pre", "smd_post"])


def _within_group_sigma2(Y: np.ndarray, Z: np.ndarray, ps: np.ndarray) -> np.ndarray:
    """Conditional outcome variance sigma^2(X_i) via the within-treatment-group nearest neighbor
    on the propensity score (J=1): sigma2_i = (1/2)(Y_i - Y_h(i))^2 (ADR-015)."""
    ps = np.asarray(ps, dtype=float).reshape(-1, 1)
    Y = np.asarray(Y, dtype=float)
    sigma2 = np.zeros(len(Y))
    for val in (0, 1):
        idx = np.where(Z == val)[0]
        if len(idx) < 2:
            continue
        nn = NearestNeighbors(n_neighbors=2).fit(ps[idx])
        _, nbr = nn.kneighbors(ps[idx])
        for r, i in enumerate(idx):
            h = idx[nbr[r, 1]]  # column 0 is self
            sigma2[i] = 0.5 * (Y[i] - Y[h]) ** 2
    return sigma2


def abadie_imbens_se(Y, Z, matched: dict[int, int], ps, *, target: Target) -> float:
    """Abadie-Imbens (2006) matching SE for ATT/ATNT, 1:1 with replacement (Imbens & Rubin
    2015, Ch. 19): V = (between + reuse) / N1^2, where between = sum (tau_i - tau_bar)^2 and
    reuse = sum_j K_j(K_j-1) sigma2_j over comparison units used K_j times. Deterministic.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([1.0, 0.0, 2.0, 1.0, 0.5])
    >>> z = np.array([1, 1, 0, 0, 0])
    >>> ps = np.array([0.8, 0.7, 0.75, 0.72, 0.68])
    >>> m = propensity_match(ps, z, target="att")
    >>> se = abadie_imbens_se(y, z, m, ps, target="att")
    >>> bool(np.isfinite(se) and se >= 0)
    True
    """
    Y = np.asarray(Y, dtype=float)
    Z = np.asarray(Z, dtype=int)
    focal = np.array(sorted(matched.keys()), dtype=int)
    comp = np.array([matched[int(i)] for i in focal], dtype=int)
    n_focal = len(focal)
    if n_focal == 0:
        return float("nan")
    tau_i = (Y[focal] - Y[comp]) if target == "att" else (Y[comp] - Y[focal])
    between = float(np.sum((tau_i - tau_i.mean()) ** 2))
    sigma2 = _within_group_sigma2(Y, Z, ps)
    counts = np.bincount(comp, minlength=len(Y))  # K_j
    reuse = float(np.sum(counts * np.maximum(counts - 1, 0) * sigma2))
    return float(np.sqrt(max((between + reuse) / (n_focal**2), 0.0)))


def _estimate(Y, Z, ps, X, *, target: Target) -> CausalEstimate:
    Y = np.asarray(Y, dtype=float)
    Z = np.asarray(Z, dtype=int)
    matched = propensity_match(ps, Z, target=target)
    focal = np.array(sorted(matched.keys()), dtype=int)
    comp = np.array([matched[int(i)] for i in focal], dtype=int)
    if len(focal) == 0:
        return CausalEstimate(
            float("nan"), float("nan"), pd.DataFrame(columns=["covariate", "smd_pre", "smd_post"]), 0, {}
        )
    tau_i = (Y[focal] - Y[comp]) if target == "att" else (Y[comp] - Y[focal])
    se = abadie_imbens_se(Y, Z, matched, ps, target=target)
    balance = smd_balance(X, Z, matched, target=target)
    return CausalEstimate(float(tau_i.mean()), se, balance, len(focal), matched)


def estimate_att(Y, Z, ps, X) -> CausalEstimate:
    """ATT = mean over treated of (Y_treated - Y_matched_control). With-replacement matching;
    Abadie-Imbens SE. Deterministic given ps.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=(400, 1))
    >>> z = (rng.uniform(size=400) < 1 / (1 + np.exp(-x[:, 0]))).astype(int)
    >>> y = x[:, 0] + 0.5 * z
    >>> ps, _ = fit_propensity(x, z, seed=0)
    >>> bool(estimate_att(y, z, ps, x).n_focal == int(z.sum()))
    True
    """
    return _estimate(Y, Z, ps, X, target="att")


def estimate_atnt(Y, Z, ps, X) -> CausalEstimate:
    """ATNT = effect on the untreated = mean over controls of (Y_matched_treated - Y_control).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(1)
    >>> x = rng.normal(size=(300, 1))
    >>> z = (rng.uniform(size=300) < 0.5).astype(int)
    >>> y = x[:, 0] + 0.3 * z
    >>> ps, _ = fit_propensity(x, z, seed=0)
    >>> bool(np.isfinite(estimate_atnt(y, z, ps, x).estimate))
    True
    """
    return _estimate(Y, Z, ps, X, target="atnt")


def _att_with_block(X_base: np.ndarray, extra: np.ndarray, Y, Z, *, seed: int) -> float:
    X = np.hstack([X_base, extra]) if extra.size else X_base
    ps, _ = fit_propensity(X, Z, seed=seed)
    return estimate_att(Y, Z, ps, X).estimate


def _cluster_reassign(X_gk: np.ndarray, cluster_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Whole-cluster REASSIGNMENT of the GK block (P5): draw ``sigma`` = a permutation of the
    unique clusters; destination cluster ``d`` receives source cluster ``sigma(d)``'s rows,
    recycled to ``d``'s size via ``np.resize`` -- each destination cluster gets EXACTLY ONE
    source cluster's values. Under unequal cluster sizes this is reassignment-with-recycling,
    NOT a strict row permutation: the null it draws is the cluster-exchangeable one (the point
    -- a per-destination mapping; concatenating permuted variable-size blocks would straddle
    destination boundaries and drift the null back toward the row-i.i.d. permutation)."""
    X_gk = np.asarray(X_gk, dtype=float)
    ids = np.asarray(cluster_ids)
    # `np.unique` SORTS, and a pooled multi-provider corpus carries mixed id types -- MEASURED:
    # gradientsports `game_id` is int while idsse/skillcorner are str, so the shot arm's 179-match
    # run died with `'<' not supported between instances of 'int' and 'str'` after the corpus pass.
    #
    # The sorted path is KEPT as the primary, and the fallback fires only where it would raise.
    # That is not caution for its own sake: `pd.factorize` orders clusters by FIRST APPEARANCE, so
    # `sigma` maps different sources to different destinations, and `placebo_shift` documents
    # itself as "deterministic given rng_seed". MEASURED over 300 random cluster layouts x 4 seeds,
    # switching unconditionally changed the result in 724/1200 cases -- statistically the same null
    # (a uniform reassignment either way) but not the same NUMBER, which would silently stop every
    # recorded placebo band from reproducing. Sortable ids therefore keep their exact previous
    # value; only the previously-crashing case changes, and it changes from an exception to a
    # result.
    #
    # Hash grouping also keeps `5` and `"5"` DISTINCT, where a stringifying repair would fuse two
    # unrelated matches into one cluster and corrupt the very cluster-exchangeable null this draws.
    try:
        uniq = np.unique(ids)
    except TypeError:  # mixed, unorderable id types (pooled multi-provider corpus)
        import pandas as pd

        codes, labels = pd.factorize(ids, sort=False)
        ids, uniq = codes, np.arange(len(labels))
    sigma = rng.permutation(len(uniq))
    out = np.empty_like(X_gk)
    for d_pos, dest in enumerate(uniq):
        src_rows = X_gk[ids == uniq[sigma[d_pos]]]
        dest_mask = ids == dest
        out[dest_mask] = np.resize(src_rows, (int(dest_mask.sum()), X_gk.shape[1]))
    return out


def placebo_shift(X_base, X_gk, Y, Z, *, n_seeds: int, rng_seed: int, cluster_ids: np.ndarray | None = None) -> dict:
    """Null distribution of the ATT shift from adding a PERMUTED GK block (H3). Permuting
    the rows of X_gk preserves its marginals + within-block correlation and destroys only its
    alignment with (Z, Y) -- isolating "GK carries Z/Y signal" from "GK columns aren't Gaussian".
    The real GK shift is "real" only if it clears band_p95. Deterministic given rng_seed.

    ``cluster_ids=None`` (default) keeps the legacy row-i.i.d. permutation exactly. When
    ``cluster_ids`` is given, each seed reassigns X_gk in WHOLE CLUSTERS via
    ``_cluster_reassign`` (reassignment-with-recycling under unequal sizes, not a strict row
    permutation) -- the null drawn is the cluster-exchangeable one, preserving within-cluster
    dependence that a row permutation would destroy.

    Note (R2-L3): row-permutation also breaks GK<->X_base correlation, so the null is slightly
    conservative vs a pure Z/Y-alignment null (standard for permutation nulls; see ADR-015).

    Returns ``{"shifts": (n_seeds,), "band_p95": float, "base_att": float,
    "permutation_unit": "cluster" | "row"}``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> xb, xg = rng.normal(size=(300, 2)), rng.normal(size=(300, 2))
    >>> z = (rng.uniform(size=300) < 0.5).astype(int)
    >>> y = rng.normal(size=300)
    >>> out = placebo_shift(xb, xg, y, z, n_seeds=5, rng_seed=0)
    >>> out["shifts"].shape, out["permutation_unit"]
    ((5,), 'row')
    """
    X_base = np.asarray(X_base, dtype=float)
    X_gk = np.asarray(X_gk, dtype=float)
    Y, Z = np.asarray(Y, dtype=float), np.asarray(Z, dtype=int)
    n = len(Z)
    base = _att_with_block(X_base, np.empty((n, 0)), Y, Z, seed=rng_seed)
    rng = np.random.default_rng(rng_seed)
    shifts = np.empty(n_seeds)
    for s in range(n_seeds):
        if cluster_ids is None:
            permuted = X_gk[rng.permutation(n)]
        else:
            permuted = _cluster_reassign(X_gk, cluster_ids, rng)
        shifts[s] = _att_with_block(X_base, permuted, Y, Z, seed=rng_seed + 1 + s) - base
    return {
        "shifts": shifts,
        "band_p95": float(np.percentile(np.abs(shifts), PLACEBO_BAND_PERCENTILE)),
        "base_att": float(base),
        "permutation_unit": "cluster" if cluster_ids is not None else "row",
    }
