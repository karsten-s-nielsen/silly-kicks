"""Unmeasured-confounding sensitivity for silly_kicks.causal (TF-71, ADR-117; ADR-015 sibling).

E-value (VanderWeele & Ding 2017) + marginal sensitivity model (Zhao-Small-Bhattacharya 2019),
both on a Hajek-IPW ATT, reimplementing the published ZSB algorithm (oracle-validated against the
bootsens/R/ipw.R reference). Pure; deterministic given seed.
The matched ATT (silly_kicks.causal.matching) stays the headline estimate; this module reports a
sibling IPW-ATT, its MSM band, the matched-vs-IPW agreement gap, and a NO-GO when they diverge or
overlap fails. See docs/superpowers/specs/2026-10-11-tf71-causal-sensitivity-design.md.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from silly_kicks.causal.matching import estimate_att, fit_propensity, propensity_match

DEFAULT_GAMMAS = (0.0, 0.22, 0.41, 0.69, 1.10, 1.39, 1.79)  # log-OR; Gamma ~ {1,1.25,1.5,2,3,4,6}
DEFAULT_N_BOOT = 1000
DEFAULT_ALPHA = 0.05
OVERLAP_MIN_ESS_FRACTION = 0.10
OVERLAP_MAX_WEIGHT_SHARE = 0.10


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return np.log(p / (1.0 - p))


def _get_extrema(A, Y, fitted_prob, gamma, *, estimand_all) -> tuple[float, float]:
    """Reimplements ZSB's get.extrema (oracle-validated against bootsens).
    Returns (min, max) of the IPW estimator over the Gamma ball."""
    A = np.asarray(A, dtype=float)
    Y = np.asarray(Y, dtype=float)
    c = 1.0 if estimand_all else 0.0
    eg = np.exp(-_logit(fitted_prob))
    keep = A != 0
    eg, Yk, Ak = eg[keep], Y[keep], A[keep]
    order = np.argsort(-Yk, kind="stable")  # R order(-Y): Y descending
    eg, Yk, Ak = eg[order], Yk[order], Ak[order]
    lo = Ak * (c + np.exp(-gamma) * eg)
    hi = Ak * (c + np.exp(gamma) * eg)

    def _combine(fwd, rev):  # c(0, cumsum(fwd)) + c(rev(cumsum(rev(rev))), 0)
        a = np.concatenate(([0.0], np.cumsum(fwd)))
        b = np.concatenate((np.cumsum(rev[::-1])[::-1], [0.0]))
        return a + b

    with np.errstate(invalid="ignore", divide="ignore"):
        maximum = np.nanmax(_combine(Yk * hi, Yk * lo) / _combine(hi, lo))
        minimum = np.nanmin(_combine(Yk * lo, Yk * hi) / _combine(lo, hi))
    return float(minimum), float(maximum)


def _ols_fit_predict(X, Y, mask) -> np.ndarray:
    """lm(Y ~ X) fit on rows where mask is True, predicted for ALL rows (ipw.R OR step)."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Xd = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xd[mask], Y[mask], rcond=None)
    return Xd @ beta


def _extrema_md(A, X, Y, gamma, *, estimand_all, reg_adjust, seed, fitted_prob=None, y_fitted=None):
    """Reimplements ZSB's extrema.md (oracle-validated against bootsens).
    fitted_prob / y_fitted are the test-only injection seam."""
    A = np.asarray(A, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if fitted_prob is None:
        fitted_prob, _ = fit_propensity(X, A.astype(int), seed=seed)
    if y_fitted is None:
        y_fitted = _ols_fit_predict(X, Y, A == 1) if reg_adjust else np.zeros_like(Y)
    y_fitted = np.asarray(y_fitted, dtype=float)
    lo, hi = _get_extrema(A, Y - y_fitted, fitted_prob, gamma, estimand_all=estimand_all)
    add = float(np.mean(y_fitted)) if estimand_all else float(np.mean(y_fitted[A == 0]))
    return lo + add, hi + add


def _extrema_os_att(A, X, Y, gamma, *, reg_adjust, seed, fitted_prob=None, y_fitted=None):
    """Reimplements ZSB's extrema.os(estimand='att') = mean(Y[A==1]) - rev(extrema.md(1-A,...,'missing'))."""
    A = np.asarray(A, dtype=float)
    Y = np.asarray(Y, dtype=float)
    lo, hi = _extrema_md(
        1.0 - A,
        X,
        Y,
        gamma,
        estimand_all=False,
        reg_adjust=reg_adjust,
        seed=seed,
        fitted_prob=fitted_prob,
        y_fitted=y_fitted,
    )
    treated_mean = float(np.mean(Y[A == 1]))
    return treated_mean - hi, treated_mean - lo  # mean - rev([lo, hi])


def _bootsens_att(A, X, Y, *, gamma, alpha, n_boot, reg_adjust, seed):
    """Reimplements ZSB's bootsens.os (oracle-validated against bootsens): resample n with replacement
    + REFIT propensity inside each resample (ZSB's validity condition; no rematch), percentile CI.
    Deterministic via one default_rng(seed)."""
    A = np.asarray(A, dtype=float)
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n = len(A)
    rng = np.random.default_rng(seed)
    los = np.empty(n_boot)
    his = np.empty(n_boot)
    for b in range(n_boot):
        s = rng.integers(0, n, n)
        los[b], his[b] = _extrema_os_att(A[s], X[s], Y[s], gamma, reg_adjust=reg_adjust, seed=seed)
    return float(np.quantile(los, alpha / 2)), float(np.quantile(his, 1 - alpha / 2))


@dataclass(frozen=True)
class SensitivityResult:
    rr: float | None
    rr_ci: tuple[float, float] | None
    matched_rr: float | None
    e_value_point: float | None
    e_value_ci: float | None
    target: str
    ipw_att: float
    matched_att: float
    agreement_gap: float
    agreement_ok: bool
    overlap_ok: bool
    msm_go: bool
    reg_adjust: bool
    gammas: tuple[float, ...]
    msm_lo: tuple[float, ...]
    msm_hi: tuple[float, ...]
    msm_ci_lo: tuple[float, ...]
    msm_ci_hi: tuple[float, ...]
    gamma_star: float
    n_boot: int
    alpha: float
    seed: int
    outcome_is_binary: bool
    notes: tuple[str, ...]


def _agreement_ok(matched_att: float, ipw_att: float, matched_se: float) -> bool:
    """NO-GO bar, factored out so BOTH branches are unit-testable (gate 6): the matched vs IPW
    ATT point estimates agree within one Abadie-Imbens SE (self-scaling, no arbitrary threshold)."""
    return bool(abs(matched_att - ipw_att) <= matched_se)


def _ess_overlap(Z, X, seed) -> tuple[bool, float]:
    """Hajek-weight overlap guard for the ATT counterfactual. The weights are the ATT control-side
    odds-weights ps/(1-ps) (a PROXY for the Hajek 'missing'-arm weights the ZSB reimplementation forms on 1-A);
    they share the same near-0/1-propensity blow-up, so this ESS/max-share guard is a faithful overlap
    tripwire even though it is not the identical per-replicate weight vector."""
    ps, _ = fit_propensity(X, np.asarray(Z, dtype=int), seed=seed)
    ctrl = np.asarray(Z) == 0
    w = ps[ctrl] / (1.0 - ps[ctrl])  # ATT control weights ~ odds (proxy; see docstring)
    if len(w) == 0 or w.sum() <= 0:
        return False, 0.0
    ess = (w.sum() ** 2) / np.sum(w**2)
    max_share = float(np.max(w) / w.sum())
    n_focal = int((np.asarray(Z) == 1).sum())
    ok = (ess >= OVERLAP_MIN_ESS_FRACTION * n_focal) and (max_share <= OVERLAP_MAX_WEIGHT_SHARE)
    return bool(ok), float(ess)


def marginal_sensitivity(
    Z, X, Y, *, gammas=DEFAULT_GAMMAS, n_boot=DEFAULT_N_BOOT, alpha=DEFAULT_ALPHA, reg_adjust=False, seed
):
    """The ZSB marginal sensitivity model on a Hajek-IPW ATT (E-value fields left ``None``).

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(120, 1))
    >>> Z = (rng.uniform(size=120) < 1 / (1 + np.exp(-X[:, 0]))).astype(int)
    >>> Y = (rng.uniform(size=120) < 0.4).astype(float)
    >>> r = marginal_sensitivity(Z, X, Y, gammas=(0.0, 0.5), n_boot=20, seed=0)
    >>> r.msm_lo[0] <= r.msm_hi[0] and r.e_value_point is None
    True
    """
    Z = np.asarray(Z)
    if not (len(gammas) and gammas[0] == 0.0 and all(gammas[i] <= gammas[i + 1] for i in range(len(gammas) - 1))):
        raise ValueError("gammas must be ascending, non-negative, and start at 0.0")
    A = Z.astype(float)
    lo = []
    hi = []
    ci_lo = []
    ci_hi = []
    for g in gammas:
        el, eh = _extrema_os_att(A, X, Y, g, reg_adjust=reg_adjust, seed=seed)
        cl, ch = _bootsens_att(A, X, Y, gamma=g, alpha=alpha, n_boot=n_boot, reg_adjust=reg_adjust, seed=seed)
        lo.append(el)
        hi.append(eh)
        ci_lo.append(cl)
        ci_hi.append(ch)
    ipw_att = lo[0]  # min == max at gamma = 0
    ps, _ = fit_propensity(X, Z.astype(int), seed=seed)
    ce = estimate_att(Y, Z, ps, X)
    matched_att, matched_se = float(ce.estimate), float(ce.se)
    gap = abs(matched_att - ipw_att)
    agreement_ok = _agreement_ok(matched_att, ipw_att, matched_se)
    overlap_ok, _ess = _ess_overlap(Z, X, seed)
    gstar = next((g for g, cl, ch in zip(gammas, ci_lo, ci_hi, strict=True) if cl <= 0.0 <= ch), float("inf"))
    notes = (
        "MSM on the Hajek-IPW ATT (reimplemented ZSB, oracle-validated against bootsens/ipw.R); "
        "gamma_star is a property of the IPW-ATT, not a certificate for the matched ATT "
        "(point agreement is necessary, not sufficient). "
        "Does NOT establish causality or detect model misspecification.",
    )
    return SensitivityResult(
        rr=None,
        rr_ci=None,
        matched_rr=None,
        e_value_point=None,
        e_value_ci=None,
        target="att",
        ipw_att=ipw_att,
        matched_att=matched_att,
        agreement_gap=gap,
        agreement_ok=agreement_ok,
        overlap_ok=overlap_ok,
        msm_go=(agreement_ok and overlap_ok),
        reg_adjust=reg_adjust,
        gammas=tuple(gammas),
        msm_lo=tuple(lo),
        msm_hi=tuple(hi),
        msm_ci_lo=tuple(ci_lo),
        msm_ci_hi=tuple(ci_hi),
        gamma_star=float(gstar),
        n_boot=n_boot,
        alpha=alpha,
        seed=seed,
        outcome_is_binary=bool(np.all(np.isin(np.unique(Y), (0.0, 1.0)))),
        notes=notes,
    )


def _ev(rr: float) -> float:
    rr = float(rr)
    if rr < 1.0:
        rr = 1.0 / rr
    return rr + np.sqrt(rr * (rr - 1.0))


def _ev_ci_limit(ci: tuple[float, float]) -> float:
    lo, hi = ci
    if lo <= 1.0 <= hi:
        return 1.0
    return _ev(lo if lo > 1.0 else hi)  # limit nearest 1


def _matched_rr(Z, X, Y, *, seed) -> float | None:
    ps, _ = fit_propensity(X, np.asarray(Z, dtype=int), seed=seed)
    matched = propensity_match(ps, np.asarray(Z, dtype=int), target="att")
    if not matched:
        return None
    Y = np.asarray(Y, dtype=float)
    focal = np.array(sorted(matched), dtype=int)
    comp = np.array([matched[int(i)] for i in focal], dtype=int)
    p_focal, p_comp = float(np.mean(Y[focal])), float(np.mean(Y[comp]))
    return p_focal / p_comp if p_comp > 0 else None


def e_value(Z, X, Y, *, target="att", n_boot=DEFAULT_N_BOOT, alpha=DEFAULT_ALPHA, seed):
    """E-value (VanderWeele & Ding) on the Hajek-IPW risk ratio; returns (e_point, e_ci, rr, rr_ci, matched_rr).

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(120, 1))
    >>> Z = (rng.uniform(size=120) < 1 / (1 + np.exp(-X[:, 0]))).astype(int)
    >>> Y = (rng.uniform(size=120) < 0.4).astype(float)
    >>> ep, eci, rr, rr_ci, mrr = e_value(Z, X, Y, n_boot=20, seed=0)
    >>> bool(ep >= 1.0)
    True
    """
    Z = np.asarray(Z)
    Y = np.asarray(Y, dtype=float)
    if not np.all(np.isin(np.unique(Y), (0.0, 1.0))):
        raise ValueError("e_value requires a binary outcome Y in {0, 1}")
    n = len(Z)

    def _rr(zz, xx, yy):
        p1 = float(np.mean(yy[zz == 1]))
        lo, _ = _extrema_md(1.0 - zz.astype(float), xx, yy, 0.0, estimand_all=False, reg_adjust=False, seed=seed)
        return p1 / lo if lo > 0 else np.nan

    Xf = np.asarray(X, dtype=float)
    rr = _rr(Z, Xf, Y)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        s = rng.integers(0, n, n)
        boots[b] = _rr(Z[s], Xf[s], Y[s])
    rr_ci = (float(np.nanquantile(boots, alpha / 2)), float(np.nanquantile(boots, 1 - alpha / 2)))
    return _ev(rr), _ev_ci_limit(rr_ci), float(rr), rr_ci, _matched_rr(Z, X, Y, seed=seed)


def sensitivity_analysis(
    Z, X, Y, *, target="att", gammas=DEFAULT_GAMMAS, n_boot=DEFAULT_N_BOOT, alpha=DEFAULT_ALPHA, reg_adjust=False, seed
):
    """Unmeasured-confounding sensitivity: the ZSB marginal sensitivity model on a Hajek-IPW ATT, plus
    the E-value on the IPW risk ratio when the outcome is binary (the matched RR is a companion).

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(200, 1))
    >>> Z = (rng.uniform(size=200) < 1 / (1 + np.exp(-X[:, 0]))).astype(int)
    >>> Y = (rng.uniform(size=200) < 0.4).astype(float)
    >>> r = sensitivity_analysis(Z, X, Y, gammas=(0.0, 0.5), n_boot=20, seed=0)
    >>> bool(r.gammas[0] == 0.0 and abs(r.msm_lo[0] - r.msm_hi[0]) < 1e-9)
    True
    """
    base = marginal_sensitivity(Z, X, Y, gammas=gammas, n_boot=n_boot, alpha=alpha, reg_adjust=reg_adjust, seed=seed)
    Y = np.asarray(Y, dtype=float)
    if not np.all(np.isin(np.unique(Y), (0.0, 1.0))):
        return base  # non-binary: MSM only, E-value fields stay None
    ep, eci, rr, rr_ci, mrr = e_value(Z, X, Y, target=target, n_boot=n_boot, alpha=alpha, seed=seed)
    return dataclasses.replace(base, rr=rr, rr_ci=rr_ci, matched_rr=mrr, e_value_point=ep, e_value_ci=eci)
