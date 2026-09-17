"""Pure-numpy crossed random-effects variance components + bootstrap ICC CI.

Scripts-side ANALYSIS tooling for the TF-54b defender-ranking census (spec Sec.8 SPEC-03), NOT a
library API -- ADR-009 keeps rankings and identifiability verdicts consumer-side. ``causal.power``
ships only :func:`att_power_curve` (an ATT estimator) and ``silly_kicks._group_metrics`` ships only a
ONE-WAY ICC (:func:`icc_one_way`, a single grouping factor). Neither separates a defender effect from
a team effect, so both functions here are new work.

WHY A CROSSED MODEL. A per-defender territorial-defense number is *team-conditioned by construction*
(the marginal-removal delta re-partitions vacated space to teammates), so a raw one-way ICC over
defender codes cannot tell a defender-intrinsic signal apart from the team the defender plays for. The
two-way CROSSED random-effects model

    y_k = mu + a_{defender(k)} + b_{team(k)} + eps_k,
    a ~ N(0, s2_defender),  b ~ N(0, s2_team),  eps ~ N(0, s2_resid),

with defender crossed against team, partitions the between-unit variance into the two sources. The
defender-share ICC ``s2_defender / (s2_defender + s2_team + s2_resid)`` is the identifiability
statistic: near zero means the ground a defender patrols is a team property, not a defender one, and a
per-defender ranking is not licensed on the corpus.

ESTIMATOR: Henderson Method III (the fitting-constants method; Searle, Casella & McCulloch,
*Variance Components*, 1992, Ch. 5). It handles the UNBALANCED corpus that a balanced-ANOVA estimator
cannot, and it uses only ordinary least squares reductions in the residual sum of squares plus the
expected-mean-square coefficients, so it needs no iterative mixed-model solver, no statsmodels and no
PyMC -- pure ``numpy`` + ``np.linalg``.

    R(mu)        -- reduction from the mean-only fit
    R(mu, d)     -- reduction from the mean + defender-indicator fit
    R(mu, d, t)  -- reduction from the mean + defender + team fit (the full model)

    SS_d = R(mu, d)    - R(mu)        with df_d = rank[mu, d]    - rank[mu]
    SS_t = R(mu, d, t) - R(mu, d)     with df_t = rank[mu, d, t] - rank[mu, d]
    SSE  = y'y         - R(mu, d, t)  with df_e = n             - rank[mu, d, t]

Each sequential reduction is a quadratic form ``y' P y`` in a projection-difference ``P`` that
annihilates the smaller model's column space (the intercept included), so under the random model

    E[y' P y] = tr(P V),   V = s2_defender Z_d Z_d' + s2_team Z_t Z_t' + s2_resid I,

and the mean term drops out. The expected mean squares are therefore linear in the three components
with coefficients ``tr(P S_c) / df`` (``S_c = Z_c Z_c'``); equating the three observed mean squares to
their expectations gives a 3x3 linear system solved for ``(s2_defender, s2_team, s2_resid)``. For a
balanced design this reduces to the textbook two-way random-effects EMS (``E[MS_d] = s2_resid +
reps*b*s2_defender`` etc.); the general form above stays exact when unbalanced.

SCALABILITY. The projection traces are computed WITHOUT ever forming an ``n x n`` matrix -- a dense
``P`` is 7.6 GiB at 30k rows. Using ``tr(P_X S_c) = tr( pinv(X'X) (X'Z_c)(X'Z_c)' )`` keeps every
matrix ``p x p`` or ``p x k`` (``p = 1 + a + b``), and the reductions come from the same small normal
equations (``R(X) = (X'y)' pinv(X'X) (X'y)``), so the estimator runs on the full corpus without a
dense pitch-sized allocation.

NEGATIVE ESTIMATES. A moment estimator can return a negative variance (a between-source reduction
that falls below its residual expectation under sampling noise, or a truly-null source). Each negative
component is truncated to 0.0 and ``had_negative_estimate`` is set True so a caller can see that the
raw solve went out of range -- never silently.

BOOTSTRAP. :func:`bootstrap_icc_ci` is a nonparametric CLUSTER bootstrap that resamples WHOLE
defenders WITH replacement (a defender drawn twice contributes its rows twice), mirroring the
cluster-preserving STYLE of ``causal.power._resample_clusters`` -- not a call to it, because that
helper subsamples clusters WITHOUT replacement for a power curve, whereas a percentile bootstrap needs
with-replacement resampling to produce a non-degenerate interval. Resampling at the DEFENDER grain (not
the row grain) is the correct unit: the identifiability question is about defenders, and an i.i.d. row
resample would inherit none of the defender clustering the real design carries.

See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

import numpy as np


def _indicator_matrix(codes: np.ndarray) -> np.ndarray:
    """Dense 0/1 indicator (design) matrix, one column per distinct code."""
    _, inv = np.unique(np.asarray(codes), return_inverse=True)
    n, k = len(inv), int(inv.max()) + 1 if len(inv) else 0
    ind = np.zeros((n, k), dtype=float)
    if n:
        ind[np.arange(n), inv] = 1.0
    return ind


def _reduction_and_rank(design: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """OLS reduction ``R(design) = (X'y)' pinv(X'X) (X'y)`` and the design rank.

    Computed from the small ``p x p`` Gram (``p = # design columns``), never an ``n x p`` SVD, so it
    scales to the full corpus. Pseudo-inverse handles the rank deficiency the indicator designs carry
    (the intercept lies in the column span of each indicator block).
    """
    gram = design.T @ design
    xty = design.T @ y
    reduction = float(xty @ (np.linalg.pinv(gram) @ xty))
    rank = int(np.linalg.matrix_rank(gram))
    return reduction, rank


def _trace_proj_source(design: np.ndarray, source_ind: np.ndarray) -> float:
    """``tr( P_design S )`` for ``S = source_ind @ source_ind.T``, computed in the small space.

    ``tr(P_X S) = tr( X (X'X)^- X' S ) = tr( (X'X)^- (X' S X) )`` and ``X' S X = (X' Z)(X' Z)'`` with
    ``Z = source_ind``, so nothing larger than ``p x k`` is ever materialised (no dense ``n x n``).
    """
    m = design.T @ source_ind
    return float(np.trace(np.linalg.pinv(design.T @ design) @ (m @ m.T)))


def crossed_variance_components(y, defender_codes, team_codes) -> dict:
    """Henderson-III variance components for the crossed defender+team random-effects model.

    Fits ``y_k = mu + a_defender(k) + b_team(k) + eps_k`` (defenders crossed with teams, unbalanced)
    and returns the three variance components plus the defender-share ICC.

    Parameters
    ----------
    y : array-like of float
        The outcome, one value per observation.
    defender_codes, team_codes : array-like
        Per-observation grouping codes (any hashable/orderable dtype; only distinctness matters).

    Returns
    -------
    dict
        ``var_defender``, ``var_team``, ``var_resid`` -- the estimated variance components, each
        truncated to 0.0 if the raw moment solve went negative. ``icc_defender`` -- the defender share
        ``var_defender / (var_defender + var_team + var_resid)``; **0.0 when the denominator is 0.0**
        (all three components truncated to zero -- a flat design, reported as "no defender separation"
        rather than NaN, so a downstream ``> threshold`` gate reads it as a clean non-detection).
        ``had_negative_estimate`` -- True iff any raw component was negative before truncation.

    Notes
    -----
    A design too small to admit all three degrees of freedom (``df_d``, ``df_t`` or ``df_e`` non-
    positive) returns ``icc_defender = nan`` (not measurable) with the components it can name.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    z_def = _indicator_matrix(defender_codes)
    z_team = _indicator_matrix(team_codes)

    # col(z_def) already spans the ones vector (its columns sum to 1), so an explicit intercept in the
    # mean+defender / full designs is redundant -- dropping it removes a rank deficiency and one
    # column without changing any reduction.
    design_mu = np.ones((n, 1), dtype=float)
    design_def = z_def
    design_full = np.hstack([z_def, z_team])

    yy = float(y @ y)
    r_mu, rank_mu = _reduction_and_rank(design_mu, y)
    r_def, rank_def = _reduction_and_rank(design_def, y)
    r_full, rank_full = _reduction_and_rank(design_full, y)

    df_d = rank_def - rank_mu
    df_t = rank_full - rank_def
    df_e = n - rank_full

    if df_d <= 0 or df_t <= 0 or df_e <= 0:
        # Too few levels / observations to identify all three sources. This "cannot identify"
        # exit is DISTINCT from the flat-design exit below: here var_resid/icc are NaN (unmeasurable),
        # and had_negative_estimate is False because no raw variance solve ran (nothing was truncated) --
        # a NaN icc reads downstream as "not measurable", never as a fabricated 0.
        return {
            "var_defender": 0.0,
            "var_team": 0.0,
            "var_resid": float("nan"),
            "icc_defender": float("nan"),
            "had_negative_estimate": False,
        }

    ms_d = (r_def - r_mu) / df_d
    ms_t = (r_full - r_def) / df_t
    mse = (yy - r_full) / df_e

    # Expected-mean-square coefficients: E[MS_x] = (coef_def s2_def + coef_team s2_team + df s2_resid)/df.
    # Defender row uses the (mu -> mu,def) projection difference; team row the (mu,def -> full) one;
    # residual row the (full -> I) one, whose source coefficients are ~0 by construction (tr S = n).
    coef_dd = _trace_proj_source(design_def, z_def) - _trace_proj_source(design_mu, z_def)
    coef_dt = _trace_proj_source(design_def, z_team) - _trace_proj_source(design_mu, z_team)
    coef_td = _trace_proj_source(design_full, z_def) - _trace_proj_source(design_def, z_def)
    coef_tt = _trace_proj_source(design_full, z_team) - _trace_proj_source(design_def, z_team)
    coef_ed = float(n) - _trace_proj_source(design_full, z_def)  # tr(S_def) = n
    coef_et = float(n) - _trace_proj_source(design_full, z_team)

    ems = np.array(
        [
            [coef_dd / df_d, coef_dt / df_d, 1.0],
            [coef_td / df_t, coef_tt / df_t, 1.0],
            [coef_ed / df_e, coef_et / df_e, 1.0],
        ]
    )
    observed = np.array([ms_d, ms_t, mse])
    try:
        raw = np.linalg.solve(ems, observed)
    except np.linalg.LinAlgError:  # pragma: no cover - degenerate EMS matrix
        raw, *_ = np.linalg.lstsq(ems, observed, rcond=None)

    had_negative = bool(np.any(raw < 0.0))
    var_def, var_team, var_resid = (float(max(v, 0.0)) for v in raw)

    denom = var_def + var_team + var_resid
    icc = var_def / denom if denom > 0.0 else 0.0

    return {
        "var_defender": var_def,
        "var_team": var_team,
        "var_resid": var_resid,
        "icc_defender": float(icc),
        "had_negative_estimate": had_negative,
    }


def bootstrap_icc_ci(y, defender_codes, team_codes, *, n_boot, alpha, rng_seed) -> dict:
    """Percentile CI for the defender-share ICC via a with-replacement DEFENDER cluster bootstrap.

    Resamples ``k`` whole defenders WITH replacement (``k`` = the number of distinct defenders),
    concatenates their rows into a bootstrap sample, refits :func:`crossed_variance_components`, and
    returns the percentile interval of the resulting ICC draws. Cluster-preserving, mirroring the
    STYLE of ``causal.power._resample_clusters`` (see the module docstring for why this is a re-style,
    not a re-use).

    Parameters
    ----------
    y, defender_codes, team_codes : array-like
        As in :func:`crossed_variance_components`.
    n_boot : int
        Number of bootstrap replicates.
    alpha : float
        Two-sided miscoverage; the interval is the ``[100*alpha/2, 100*(1-alpha/2)]`` percentiles.
    rng_seed : int
        Seed for ``numpy.random.default_rng`` -- deterministic output.

    Returns
    -------
    dict
        ``icc`` -- the point estimate on the full data. ``lo``, ``hi`` -- the percentile CI bounds
        over the finite ICC replicates. (``n_boot_effective`` -- the count of finite replicates that
        entered the percentile, an audit field; a replicate whose resample was too degenerate to
        identify the model contributes a NaN ICC and is dropped.)
    """
    y = np.asarray(y, dtype=float)
    defender_codes = np.asarray(defender_codes)
    team_codes = np.asarray(team_codes)

    point = crossed_variance_components(y, defender_codes, team_codes)["icc_defender"]

    unique_def = np.unique(defender_codes)
    k = len(unique_def)
    # Row index groups per defender, built ONCE (never a per-replicate full-table rescan).
    members = [np.flatnonzero(defender_codes == d) for d in unique_def]

    rng = np.random.default_rng(rng_seed)
    iccs: list[float] = []
    for _ in range(int(n_boot)):
        picked = rng.integers(0, k, size=k)  # WITH replacement -- a real percentile bootstrap
        idx = np.concatenate([members[j] for j in picked])
        val = crossed_variance_components(y[idx], defender_codes[idx], team_codes[idx])["icc_defender"]
        if np.isfinite(val):
            iccs.append(val)

    draws = np.asarray(iccs, dtype=float)
    lo = float(np.percentile(draws, 100.0 * alpha / 2.0))
    hi = float(np.percentile(draws, 100.0 * (1.0 - alpha / 2.0)))

    return {"icc": float(point), "lo": lo, "hi": hi, "n_boot_effective": int(draws.size)}
