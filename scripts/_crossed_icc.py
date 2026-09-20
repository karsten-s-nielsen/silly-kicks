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
matrix ``p x p`` or ``p x k`` (``p = 1 + a + b``). The reductions ``R(X) = y' P y`` come from a
QR-based ``np.linalg.lstsq`` solve (``yhat = X sol = P y``, so ``R(X) = y' yhat``), which never forms
the ``p x p`` Gram at all, so the estimator runs on the full corpus without a dense pitch-sized
allocation.

NUMERICAL ROBUSTNESS. On the real broad corpus a bootstrap resample builds a huge, sparse, ill-
conditioned crossed indicator design; the earlier ``(X'y)' pinv(X'X) (X'y)`` reduction squared the
Gram's condition number and its internal SVD failed to converge (``LinAlgError: SVD did not
converge`` / ``DLASCL parameter 4 illegal value``, the latter a NaN-in-the-matrix symptom). Three
guards make the estimator corpus-safe without changing its output on a well-conditioned design: the
reduction is now the QR-based ``lstsq`` projection above (stable, rank-deficiency-safe, Gram-free);
:func:`crossed_variance_components` DROPS any row whose ``y`` is non-finite before fitting, so a
degenerate metric value never reaches the SVD; and :func:`bootstrap_icc_ci` SKIPS (never aborts on) a
replicate whose resample is too degenerate to fit, computing the percentile CI over the convergent
replicates only and returning the point estimate with a NaN interval + a recorded flag if too few
converge. One bad resample can never kill the corpus pass.

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

from typing import NamedTuple

import numpy as np


class _CrossedFit(NamedTuple):
    """The crossed model's sequential reductions + ranks + Henderson-III EMS coefficients.

    Everything :func:`crossed_variance_components` needs, all computed by ABSORPTION -- no dense
    ``n x p`` design and no ``pinv(X'X)`` at the full ``p = 1 + n_def + n_team``. The six ``coef_*`` are
    the expected-mean-square trace coefficients ``tr(P_X Z_c Z_c')`` differences (see
    :func:`_crossed_reductions`); ``coef_td`` and ``coef_ed`` are identically ``0.0`` because
    ``col(Z_def) subset col([Z_def, Z_team])`` makes ``tr(P_full Z_def Z_def') == tr(P_def Z_def Z_def')
    == n``.
    """

    r_mu: float
    r_def: float
    r_full: float
    rank_mu: int
    rank_def: int
    rank_full: int
    coef_dd: float
    coef_dt: float
    coef_td: float
    coef_tt: float
    coef_ed: float
    coef_et: float


def _is_missing(code: object) -> bool:
    """True for a missing grouping code (``None`` / ``float('nan')`` / ``pd.NA``).

    Canonical ADR-019 codes are strings or numbers, and ``canonical_id_series`` emits ``pd.NA`` for a
    missing id; a genuine string level is never missing. ``pd.NA`` cannot be detected by ``NA != NA``
    (which is ``pd.NA``, not a boolean) nor by ``np.isnan`` (which raises on it), so the one correct
    scalar-missing predicate is ``pandas.isna``. The import is FUNCTION-LOCAL so the module's top-level
    surface stays pure-``numpy`` (this guard fires only on the rare degenerate row); a numpy-only
    fallback covers ``None`` / numeric NaN if pandas is somehow unavailable.
    """
    if code is None:
        return True
    try:
        import pandas as pd

        return bool(pd.isna(code))  # type: ignore[arg-type]  # pd.isna accepts any scalar at runtime
    except ImportError:  # pragma: no cover - pandas is a hard runtime dep
        try:
            return bool(np.isnan(code))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False


def _indicator_matrix(codes: np.ndarray) -> np.ndarray:
    """Dense 0/1 indicator (design) matrix, one column per distinct code."""
    _, inv = np.unique(np.asarray(codes), return_inverse=True)
    n, k = len(inv), int(inv.max()) + 1 if len(inv) else 0
    ind = np.zeros((n, k), dtype=float)
    if n:
        ind[np.arange(n), inv] = 1.0
    return ind


def _reduction_and_rank(design: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """OLS reduction ``R(design) = y' P y`` (P the projector onto col(design)) and the design rank.

    Reference-only now: :func:`crossed_variance_components` gets its reductions/ranks by ABSORPTION
    (:func:`_crossed_reductions`); this dense lstsq form is kept as the byte-identity ORACLE that
    ``test_absorption_reductions_match_lstsq`` gates the absorbed reductions/ranks against -- do not delete.

    Computed via a QR-based least-squares solve, NOT by forming and pseudo-inverting the ``p x p`` Gram
    ``X'X``. The Gram path (``pinv(X'X)``) is O(p^3) and squares the design's condition number, and on
    the real corpus a bootstrap RESAMPLE produces a huge, sparse, ill-conditioned crossed indicator
    design (thousands of defenders) whose Gram is near-singular; that made ``np.linalg.pinv`` fail its
    internal SVD (``LinAlgError: SVD did not converge`` / ``DLASCL parameter 4 illegal value``).

    ``np.linalg.lstsq`` is QR-based (numerically stable), rank-deficiency-safe, and it never forms the
    ``p x p`` Gram, so it scales to the full corpus without the ill-conditioning blow-up. The reduction
    is the same quantity: with ``sol`` the least-norm least-squares solution, ``yhat = design @ sol =
    P y`` (the orthogonal projection of ``y`` onto col(design)), so ``y' yhat = y' P y = R(model)`` --
    equal to the old ``(X'y)' pinv(X'X) (X'y)`` up to numerical precision (verified in the recovery
    test). ``rank`` is the effective rank ``lstsq`` reports (from the same QR/SVD it already computes),
    which is exactly the ``rank[X]`` the sequential-df bookkeeping needs -- identical to the old
    ``matrix_rank(X'X)`` for these indicator designs (``rank(X'X) == rank(X)``).
    """
    sol, _residuals, rank, _sv = np.linalg.lstsq(design, y, rcond=None)
    yhat = design @ sol
    reduction = float(y @ yhat)
    return reduction, int(rank)


def _trace_proj_source(design: np.ndarray, source_ind: np.ndarray) -> float:
    """``tr( P_design S )`` for ``S = source_ind @ source_ind.T``, computed in the small space.

    ``tr(P_X S) = tr( X (X'X)^- X' S ) = tr( (X'X)^- (X' S X) )`` and ``X' S X = (X' Z)(X' Z)'`` with
    ``Z = source_ind``, so nothing larger than ``p x k`` is ever materialised (no dense ``n x n``).

    Reference-only now: :func:`crossed_variance_components` gets these traces by ABSORPTION (via
    ``pinv(W)``, ``n_team``-sized -- see :func:`_crossed_reductions`); this dense-``pinv(X'X)`` form is
    kept as the oracle the coefficient-parity test gates the absorbed path against.
    """
    m = design.T @ source_ind
    return float(np.trace(np.linalg.pinv(design.T @ design) @ (m @ m.T)))


def _bipartite_components(inv_def: np.ndarray, inv_team: np.ndarray, n_def: int, n_team: int) -> int:
    """Connected-component count of the defender<->team bipartite incidence graph (union-find).

    ``rank([Z_def, Z_team]) == n_def + n_team - c`` for 0/1 crossed indicator designs, where ``c`` is the
    number of connected components of the bipartite graph whose edges are the observed ``(defender, team)``
    pairs (each component collapses exactly one otherwise-independent column -- the classic two-way ANOVA
    rank result). This is the analytic ``rank_full`` the absorption path uses in place of an SVD, and it
    is gated equal to ``np.linalg.lstsq``'s reported rank (``test_absorption_reductions_match_lstsq``),
    including a deliberately disconnected two-block design where ``c == 2``.
    """
    parent = list(range(n_def + n_team))  # defenders [0, n_def); teams [n_def, n_def + n_team)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for d, t in zip(np.asarray(inv_def), np.asarray(inv_team), strict=True):
        ra, rb = find(int(d)), find(n_def + int(t))
        if ra != rb:
            parent[ra] = rb
    return len({find(i) for i in range(n_def + n_team)})


def _crossed_reductions(y: np.ndarray, defender_codes: np.ndarray, team_codes: np.ndarray) -> _CrossedFit:
    """The crossed model's sequential OLS reductions + ranks + EMS coefficients, via ABSORPTION.

    Returns a :class:`_CrossedFit`; the reductions/ranks (fields 0-5) are byte-identical (to numeric
    precision, gated ``< 1e-6``) to three :func:`_reduction_and_rank` calls on ``ones`` / ``Z_def`` /
    ``[Z_def, Z_team]`` -- because each reduction ``y' P y`` is the ORTHOGONAL PROJECTION of ``y`` onto
    the column space, which is unique regardless of how it is computed:

    * ``R(mu)  = n * ybar**2``                          (projection onto the ones vector)
    * ``R(def) = sum_d (sum_i in d y_i)**2 / n_d``      (defender group means; ``P_def`` is block means)
    * ``R(full)`` via defender ABSORPTION: ``P_full = P_def + P_{(I-P_def) Z_team}``, so
      ``R(full) = R(def) + v' W^+ v`` with ``W = Z_team'(I-P_def)Z_team`` (``n_team x n_team``) and
      ``v = Z_team'(I-P_def) y`` -- an ``n_team``-sized solve instead of an ``n x (n_def+n_team)`` SVD.

    ``W`` / ``v`` are formed from the defender/team counts and the ``(defender, team)`` cross-count matrix
    ``C`` (``np.add.at``), never a dense ``n``-row design; ``pinv(W)`` handles the rank deficiency when the
    design is disconnected (``c > 1``), matching ``lstsq``'s min-norm projection. Ranks are analytic:
    ``rank_mu = 1``, ``rank_def = n_def`` (every observed defender is a full column), ``rank_full =
    n_def + n_team - c`` (:func:`_bipartite_components`).
    """
    n = len(y)
    if n == 0:
        # Every row dropped (all-NaN y). Ranks 0 -> df_d/df_t/df_e <= 0 -> the caller's "not measurable"
        # NaN exit, matching the old empty-design lstsq path (no division by n / mean-of-empty).
        return _CrossedFit(0.0, 0.0, 0.0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    _ud, inv_d = np.unique(defender_codes, return_inverse=True)
    _ut, inv_t = np.unique(team_codes, return_inverse=True)
    n_def, n_team = len(_ud), len(_ut)
    cnt_d = np.bincount(inv_d, minlength=n_def).astype(float)
    cnt_t = np.bincount(inv_t, minlength=n_team).astype(float)

    ybar = float(y.mean())
    r_mu = float(n * ybar * ybar)

    dsum = np.bincount(inv_d, weights=y, minlength=n_def)
    r_def = float(np.sum(dsum * dsum / cnt_d))

    # Defender absorption: (I - P_def) centres each vector within its defender group.
    dmean_y = dsum / cnt_d
    cross = np.zeros((n_def, n_team), dtype=float)
    np.add.at(cross, (inv_d, inv_t), 1.0)  # C[d, t] = #rows with (defender d, team t)
    tsum = np.bincount(inv_t, weights=y, minlength=n_team)
    # W = Z_t'(I-P_def)Z_t = diag(cnt_t) - sum_d C[d,a]C[d,b]/cnt_d ; v = Z_t'(I-P_def)y = tsum - C'dmean_y
    zt_pd_zt = cross.T @ (cross / cnt_d[:, None])
    w = np.diag(cnt_t) - zt_pd_zt
    v = tsum - (cross.T @ dmean_y)
    w_pinv = np.linalg.pinv(w)  # n_team x n_team -- reused for R(full) AND the EMS traces below
    r_full = r_def + float(v @ (w_pinv @ v))

    c = _bipartite_components(inv_d, inv_t, n_def, n_team)

    # Henderson-III EMS trace coefficients, also by absorption -- tr(P_X Z_c Z_c') via the SAME W/C,
    # never pinv(design_full). Using the orthogonal split P_full = P_def + P_M (M = (I-P_def)Z_team):
    #   tr(P_mu  S_def)  = ||Z_def' 1||^2 / n = sum(cnt_d^2)/n ;  tr(P_mu S_team) = sum(cnt_t^2)/n
    #   tr(P_def S_def)  = n (Z_def spans P_def) ;  tr(P_def S_team) = trace(Z_t'P_def Z_t) = tr(zt_pd_zt)
    #   tr(P_M   S_team) = tr(pinv(W) W^2) ;  M'Z_def = (I-P_def)Z_def = 0  ->  tr(P_M S_def) = 0
    # so tr(P_full S_def) = n and coef_td = coef_ed = 0 EXACTLY (the old pinv path gave ~n +/- 1e-9).
    tr_mu_def = float(np.sum(cnt_d * cnt_d)) / n
    tr_mu_team = float(np.sum(cnt_t * cnt_t)) / n
    tr_def_team = float(np.trace(zt_pd_zt))
    tr_m_team = float(np.trace(w_pinv @ (w @ w)))
    coef_dd = float(n) - tr_mu_def  # tr(P_def S_def) - tr(P_mu S_def)
    coef_dt = tr_def_team - tr_mu_team
    coef_td = 0.0  # tr(P_full S_def) - tr(P_def S_def) = n - n
    coef_tt = tr_m_team  # tr(P_full S_team) - tr(P_def S_team) = tr(P_M S_team)
    coef_ed = 0.0  # n - tr(P_full S_def) = n - n
    coef_et = float(n) - (tr_def_team + tr_m_team)  # n - tr(P_full S_team)
    return _CrossedFit(
        r_mu, r_def, r_full, 1, n_def, n_def + n_team - c, coef_dd, coef_dt, coef_td, coef_tt, coef_ed, coef_et
    )


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

    NaN/Inf guard: any row whose ``y`` (or grouping code) is non-finite/missing is DROPPED before the
    fit. A degenerate metric value must never enter the QR/SVD -- it is the ``DLASCL parameter 4
    illegal value`` root cause -- and dropping it (rather than crashing) keeps a corpus pass alive; if
    every row drops, the empty design falls through to the ``df <= 0`` "not measurable" NaN exit below.
    """
    y = np.asarray(y, dtype=float)
    defender_codes = np.asarray(defender_codes)
    team_codes = np.asarray(team_codes)

    # Drop non-finite y (NaN/Inf) and missing grouping codes before fitting: a degenerate row must
    # never reach the QR/SVD (the "DLASCL parameter 4 illegal value" root cause). Codes are canonical
    # ADR-019 strings (or numeric); a missing code is treated as an undroppable-into-a-level row.
    keep = np.isfinite(y)
    for codes in (defender_codes, team_codes):
        keep = keep & np.array([not _is_missing(c) for c in codes], dtype=bool)
    if not keep.all():
        y = y[keep]
        defender_codes = defender_codes[keep]
        team_codes = team_codes[keep]

    n = len(y)
    yy = float(y @ y)
    # Sequential reductions + ranks + EMS trace coefficients, ALL by absorption (no dense n x p design
    # and no pinv at the full p = 1 + n_def + n_team) -- byte-identical to the lstsq-reduction + pinv-trace
    # path (gated: test_absorption_reductions_match_lstsq + the recovery/null/strong/degenerate suite).
    fit = _crossed_reductions(y, defender_codes, team_codes)

    df_d = fit.rank_def - fit.rank_mu
    df_t = fit.rank_full - fit.rank_def
    df_e = n - fit.rank_full

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

    ms_d = (fit.r_def - fit.r_mu) / df_d
    ms_t = (fit.r_full - fit.r_def) / df_t
    mse = (yy - fit.r_full) / df_e

    # Expected-mean-square coefficients (E[MS_x] = (coef_def s2_def + coef_team s2_team + df s2_resid)/df)
    # from the absorption in _crossed_reductions -- computed via pinv(W) (n_team-sized), NEVER
    # pinv(design_full). Defender row = (mu -> mu,def) projection difference; team row = (mu,def -> full).
    coef_dd, coef_dt = fit.coef_dd, fit.coef_dt
    coef_td, coef_tt = fit.coef_td, fit.coef_tt
    coef_ed, coef_et = fit.coef_ed, fit.coef_et

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

    Reference-only in production now (the census calls the merged :func:`bootstrap_icc_and_power`): kept
    as the byte-identity ORACLE that ``test_bootstrap_icc_and_power_matches_separate_loops`` gates the
    merged one-pass CI against. Do NOT delete -- removing it removes that equivalence reference.

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
        over the CONVERGENT ICC replicates. ``n_boot_effective`` -- the count of convergent replicates
        that entered the percentile (an audit field). ``ci_degenerate`` -- True iff fewer than half the
        requested replicates converged, in which case ``lo``/``hi`` are NaN (a widened / undefined
        interval) rather than a percentile over too thin a sample.

    Robustness
    ----------
    A single bootstrap resample can be too degenerate to fit -- an ill-conditioned crossed design whose
    QR/SVD does not converge, or a design too small to identify all three sources. Each per-replicate
    fit is wrapped so a ``LinAlgError`` / ``ValueError`` SKIPS that replicate rather than aborting the
    whole census, and a NaN ICC (the "not measurable" exit) is dropped too. The percentile CI is
    computed over the convergent replicates only; if too few converge the point estimate is returned
    with a NaN interval and ``ci_degenerate=True``. One bad resample never kills the corpus pass.
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
    n_boot = int(n_boot)
    iccs: list[float] = []
    for _ in range(n_boot):
        if k == 0:
            break
        picked = rng.integers(0, k, size=k)  # WITH replacement -- a real percentile bootstrap
        idx = np.concatenate([members[j] for j in picked])
        try:
            val = crossed_variance_components(y[idx], defender_codes[idx], team_codes[idx])["icc_defender"]
        except (np.linalg.LinAlgError, ValueError):
            # A degenerate / ill-conditioned resample (SVD non-convergence, etc.) is SKIPPED -- never
            # allowed to abort the corpus pass. Convergent replicates still form the interval.
            continue
        if np.isfinite(val):
            iccs.append(val)

    draws = np.asarray(iccs, dtype=float)
    # Too few convergent replicates -> the percentile would be over a sample too thin to trust; return
    # the point estimate with a NaN (widened / undefined) interval and a recorded degeneracy flag.
    if draws.size < n_boot // 2:
        return {
            "icc": float(point),
            "lo": float("nan"),
            "hi": float("nan"),
            "n_boot_effective": int(draws.size),
            "ci_degenerate": True,
        }

    lo = float(np.percentile(draws, 100.0 * alpha / 2.0))
    hi = float(np.percentile(draws, 100.0 * (1.0 - alpha / 2.0)))

    return {
        "icc": float(point),
        "lo": lo,
        "hi": hi,
        "n_boot_effective": int(draws.size),
        "ci_degenerate": False,
    }


def bootstrap_icc_and_power(
    y,
    defender_codes,
    team_codes,
    *,
    n_boot,
    alpha,
    effect_size,
    rng_seed,
    progress_every: int = 50,
) -> dict:
    """CI + power for the defender-share ICC from ONE with-replacement defender-cluster bootstrap pass.

    Merges what were two separate ``n_boot`` loops -- :func:`bootstrap_icc_ci` (the percentile CI) and the
    census-local power proxy -- which drew the SAME resamples from the SAME seed and each refit
    :func:`crossed_variance_components` per replicate. Computing each replicate's ICC ONCE and deriving
    both the interval and the power fraction from it is byte-identical to running the two loops separately
    (same seed -> same ``rng.integers(0, k, size=k)`` sequence -> same per-replicate ICC; a degenerate
    resample is SKIPPED via the same ``try/except``), and halves the crossed-model fits. Gated
    ``test_bootstrap_icc_and_power_matches_separate_loops``.

    Returns ``icc`` (point), ``lo``/``hi`` (percentile CI over convergent replicates), ``power`` (the
    fraction of convergent replicates with ``icc >= effect_size``), ``n_boot_effective``,
    ``ci_degenerate``.

    BLAS is left MULTI-threaded. Absorption made the sequential REDUCTIONS ``n_team``-sized, but the
    three EMS-trace ``pinv(X'X)`` terms are still O(p^3) at the full-corpus design (``p = 1 + n_def +
    n_team`` ~ thousands on the broad open-data corpus), and that ``pinv`` parallelises well -- so the
    per-replicate fit is left to use every core. Do NOT wrap this loop in ``threadpool_limits(1)``: that
    serialises the large ``pinv`` and turns a ~40-min pass into hours (measured at ``p~4863``). Absorbing
    the traces too (via ``pinv(W)``, ``n_team``-sized) is a future refinement that would make the reduce
    O(n + n_team^3); it is deferred as a from-scratch EMS re-derivation, not needed for a one-shot census.
    """
    y = np.asarray(y, dtype=float)
    defender_codes = np.asarray(defender_codes)
    team_codes = np.asarray(team_codes)

    point = crossed_variance_components(y, defender_codes, team_codes)["icc_defender"]

    unique_def = np.unique(defender_codes)
    k = len(unique_def)
    members = [np.flatnonzero(defender_codes == d) for d in unique_def]  # built ONCE (no per-replicate rescan)

    rng = np.random.default_rng(rng_seed)
    n_boot = int(n_boot)
    iccs: list[float] = []
    for i in range(n_boot):
        if k == 0:
            break
        picked = rng.integers(0, k, size=k)  # WITH replacement -- the SAME draw sequence both loops used
        idx = np.concatenate([members[j] for j in picked])
        try:
            val = crossed_variance_components(y[idx], defender_codes[idx], team_codes[idx])["icc_defender"]
        except (np.linalg.LinAlgError, ValueError):
            continue  # a degenerate resample is skipped, never aborts the pass (mirror bootstrap_icc_ci)
        if np.isfinite(val):
            iccs.append(val)
        if progress_every and (i + 1) % progress_every == 0:
            print(f"    bootstrap ICC [{i + 1}/{n_boot}]  convergent={len(iccs)}", flush=True)

    draws = np.asarray(iccs, dtype=float)
    # Power = fraction of CONVERGENT replicates clearing the effect size (denominator excludes
    # non-convergent draws -- never counted as a detection), matching the old census power proxy.
    power = float(np.mean(draws >= effect_size)) if draws.size else float("nan")

    if draws.size < n_boot // 2:
        return {
            "icc": float(point),
            "lo": float("nan"),
            "hi": float("nan"),
            "power": power,
            "n_boot_effective": int(draws.size),
            "ci_degenerate": True,
        }
    return {
        "icc": float(point),
        "lo": float(np.percentile(draws, 100.0 * alpha / 2.0)),
        "hi": float(np.percentile(draws, 100.0 * (1.0 - alpha / 2.0))),
        "power": power,
        "n_boot_effective": int(draws.size),
        "ci_degenerate": False,
    }
