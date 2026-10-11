"""TF-71 unmeasured-confounding sensitivity: the 11 pre-registered gates.

The ipw.R oracle (gate 2) compares the Python ZSB reimplementation against
tests/causal/fixtures/ipw_oracle.json, produced by an INDEPENDENT pinned-Docker R run
(tools/oracle/ipw/) — never the Python reimplementation itself.
"""

from __future__ import annotations

import json
import pathlib
from typing import cast

import numpy as np
import pytest

from silly_kicks.causal import sensitivity as S

_FIX = json.loads((pathlib.Path(__file__).parent / "fixtures" / "ipw_oracle.json").read_text(encoding="utf-8"))


def _case(name):
    return next(c for c in _FIX["cases"] if c["name"] == name)


def _toy(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    Z = (rng.uniform(size=n) < 1 / (1 + np.exp(-X[:, 0]))).astype(int)
    Y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.5 * Z + X[:, 0])))).astype(float)
    return Z, X, Y


_TRUE_LAMBDA = 0.69  # known confounder log-OR -> true Gamma = e^0.69 ~ 2.0; MUST be a gamma grid point


def _planted(seed, *, lam=_TRUE_LAMBDA, n=600):
    """A BINARY unmeasured confounder U (log-OR = lam) drives both Z and Y via X + lam*U; Y has NO Z
    term, so the true unconfounded ATT is 0. U is BINARY so the induced sensitivity Gamma = e^lam is
    DEFINED and finite. The propensity is fit on X only (U omitted) -> confounded."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 1))
    U = (rng.uniform(size=n) < 0.5).astype(float)
    eta = X[:, 0] + lam * U
    p = 1 / (1 + np.exp(-eta))
    Z = (rng.uniform(size=n) < p).astype(int)
    Y = (rng.uniform(size=n) < p).astype(float)
    return Z, X, Y, U


# --- gate 2: ipw.R reference oracle (full ATT path, incl AIPW) ---


def test_get_extrema_matches_ipw_r():
    for name in ("get_extrema_all_g0", "get_extrema_missing_g1"):
        c = _case(name)
        lo, hi = S._get_extrema(
            np.array(c["A"], float),
            np.array(c["Y"], float),
            np.array(c["fitted_prob"], float),
            c["gamma"],
            estimand_all=c["estimand_all"],
        )
        assert (lo, hi) == pytest.approx(tuple(c["expected"]), abs=1e-9)


def test_extrema_os_att_matches_ipw_r_noadjust():
    c = _case("os_att_g1_noadj")
    lo, hi = S._extrema_os_att(
        np.array(c["A"], float),
        np.array(c["X"], float),
        np.array(c["Y"], float),
        c["gamma"],
        reg_adjust=False,
        seed=0,
        fitted_prob=np.array(c["fitted_prob_flip"], float),
    )
    assert (lo, hi) == pytest.approx(tuple(c["expected"]), abs=1e-9)


def test_extrema_os_att_matches_ipw_r_aipw():
    c = _case("os_att_g1_aipw")
    lo, hi = S._extrema_os_att(
        np.array(c["A"], float),
        np.array(c["X"], float),
        np.array(c["Y"], float),
        c["gamma"],
        reg_adjust=True,
        seed=0,
        fitted_prob=np.array(c["fitted_prob_flip"], float),
        y_fitted=np.array(c["y_fitted"], float),
    )
    assert (lo, hi) == pytest.approx(tuple(c["expected"]), abs=1e-9)


# --- gate 3: ZSB percentile bootstrap determinism ---


def test_bootsens_att_deterministic():
    Z, X, Y = _toy()
    a = S._bootsens_att(Z.astype(float), X, Y, gamma=0.5, alpha=0.05, n_boot=50, reg_adjust=False, seed=7)
    b = S._bootsens_att(Z.astype(float), X, Y, gamma=0.5, alpha=0.05, n_boot=50, reg_adjust=False, seed=7)
    assert a == b
    assert a[0] <= a[1]


# --- gate 3 (collapse): Gamma=1 -> point ---


def test_msm_gamma1_collapses_to_point():
    Z, X, Y = _toy()
    r = S.marginal_sensitivity(Z, X, Y, gammas=(0.0, 0.5, 1.0), n_boot=20, seed=1)
    assert r.gammas[0] == 0.0
    assert r.msm_lo[0] == r.ipw_att  # ipw_att IS lo[0] (exact, same assignment)
    # gamma=0 collapse: min == max in exact arithmetic; cumsum summation-order gives ~1e-15 float noise
    assert r.msm_lo[0] == pytest.approx(r.msm_hi[0], abs=1e-9)
    assert all(r.msm_lo[i] <= r.msm_hi[i] for i in range(len(r.gammas)))
    assert r.msm_lo[2] <= r.msm_lo[0] and r.msm_hi[2] >= r.msm_hi[0]
    assert r.e_value_point is None


# --- gate 1: E-value worked examples; gate 8: binary guard; gate 6: NO-GO both sides ---


def test_evalue_worked_examples():
    assert S._ev(3.9) == pytest.approx(7.263, abs=1e-3)
    assert S._ev(2.0) == pytest.approx(3.414, abs=1e-3)
    assert S._ev(1.0) == pytest.approx(1.0, abs=1e-12)
    assert S._ev(0.5) == S._ev(2.0)
    assert S._ev_ci_limit((0.8, 1.3)) == 1.0


def test_evalue_requires_binary_outcome():
    Z, X, _ = _toy()
    with pytest.raises(ValueError):
        S.e_value(Z, X, np.linspace(0, 5, len(Z)), target="att", n_boot=10, alpha=0.05, seed=0)


def test_agreement_predicate_both_sides():
    assert S._agreement_ok(0.10, 0.11, 0.05) is True
    assert S._agreement_ok(0.10, 0.30, 0.05) is False


def test_nogo_integration_all_sides():
    Z, X, Y = _toy()
    r = S.marginal_sensitivity(Z, X, Y, gammas=(0.0, 0.5), n_boot=20, seed=0)
    assert r.overlap_ok is True and r.agreement_ok is True and r.msm_go is True
    rng = np.random.default_rng(3)
    Xp = rng.normal(size=(200, 1)) * 6.0
    Zp = (rng.uniform(size=200) < 1 / (1 + np.exp(-Xp[:, 0]))).astype(int)
    Yp = (rng.uniform(size=200) < 0.3).astype(float)
    rp = S.marginal_sensitivity(Zp, Xp, Yp, gammas=(0.0, 0.5), n_boot=20, seed=0)
    assert rp.overlap_ok is False and rp.msm_go is False


# --- gate 5: fixture precondition; gate 4: coverage; gate 7: determinism; gate 9: AIPW; gate 10: purity ---


def test_planted_confounder_fixture_precondition():
    Z, X, Y, U = _planted(0)
    assert set(np.unique(U)) <= {0.0, 1.0}
    assert 0.2 < Z.mean() < 0.8
    ps, _ = S.fit_propensity(X, Z, seed=0)
    naive = S.estimate_att(Y, Z, ps, X).estimate
    assert abs(naive) > 0.02


def _coverage_rep(seed: int) -> int:
    """One planted-confounder replication: does the MSM CI at the true Gamma cover the true ATT (0)?

    Fully determined by ``seed`` (data + bootstrap), so the result is independent of execution order
    and the pooled coverage count is identical serial or parallel. Module-level + picklable so the
    joblib/loky workers in the coverage gate can call it."""
    Z, X, Y, _ = _planted(seed, lam=_TRUE_LAMBDA)
    r = S.sensitivity_analysis(Z, X, Y, gammas=(0.0, _TRUE_LAMBDA), n_boot=200, seed=seed)
    return int(r.msm_ci_lo[-1] <= 0.0 <= r.msm_ci_hi[-1])


@pytest.mark.slow
def test_planted_confounder_coverage():
    # 500 reps x 200-boot is the pre-registered gate (reps>=500, floor>=0.90, n_boot pinned). Each rep
    # is independent and seed-determined, so we fan the loop across cores: loky pins inner BLAS threads
    # (no oversubscription) and the summed coverage count is byte-identical to the serial loop. The
    # single pooled assertion stays in ONE test (pytest-split shards are separate machines; a pooled
    # statistic cannot span them) — parallelism is the only lever that keeps the invariant intact.
    from joblib import Parallel, delayed

    reps = 500
    results = cast("list[int]", Parallel(n_jobs=-1)(delayed(_coverage_rep)(seed) for seed in range(reps)))
    covered = sum(results)
    assert covered / reps >= 0.90


def test_result_determinism_seed():
    Z, X, Y, _ = _planted(0)
    a = S.sensitivity_analysis(Z, X, Y, gammas=(0.0, 0.5), n_boot=50, seed=1)
    b = S.sensitivity_analysis(Z, X, Y, gammas=(0.0, 0.5), n_boot=50, seed=1)
    c = S.sensitivity_analysis(Z, X, Y, gammas=(0.0, 0.5), n_boot=50, seed=2)
    assert a == b
    assert a.msm_lo == c.msm_lo and a.msm_hi == c.msm_hi and a.ipw_att == c.ipw_att
    assert (a.msm_ci_lo, a.msm_ci_hi) != (c.msm_ci_lo, c.msm_ci_hi)


def test_aipw_reduces_to_hajek_when_or_constant():
    Z, X, Y, _ = _planted(0)
    lo_h, hi_h = S._extrema_os_att(Z.astype(float), X, Y, 0.5, reg_adjust=False, seed=0)
    lo_a, hi_a = S._extrema_os_att(Z.astype(float), X, Y, 0.5, reg_adjust=True, seed=0, y_fitted=np.full(len(Z), 0.3))
    assert (lo_a, hi_a) == pytest.approx((lo_h, hi_h), abs=1e-12)


def test_purity_and_binary_fold():
    Z, X, Y, _ = _planted(0)
    Z0, X0, Y0 = Z.copy(), X.copy(), Y.copy()
    r = S.sensitivity_analysis(Z, X, Y, gammas=(0.0, 0.5), n_boot=20, seed=0)
    assert np.array_equal(Z, Z0) and np.array_equal(X, X0) and np.array_equal(Y, Y0)
    assert r.outcome_is_binary and r.e_value_point is not None and r.matched_rr is not None
