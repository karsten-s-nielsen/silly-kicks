"""Known-truth tests for the crossed variance-components ICC estimator (Task 10).

Every scenario draws a synthetic crossed design with KNOWN variance components at a FIXED seed, so the
assertions are deterministic. The estimator is Henderson Method III (a moment estimator): between-
source components carry sampling error that shrinks with the number of LEVELS, not observations, so
the per-component tolerances below are set to what a design with this many defenders/teams achieves in
one draw (measured), with margin -- they are not arbitrarily tight, and the report records the exact
achieved relative errors.
"""

from __future__ import annotations

import numpy as np
import pytest
from _crossed_icc import (
    _bipartite_components,
    _crossed_reductions,
    _indicator_matrix,
    _reduction_and_rank,
    _trace_proj_source,
    bootstrap_icc_and_power,
    bootstrap_icc_ci,
    crossed_variance_components,
)


def _balanced_crossed_design(seed, n_def, n_team, reps, *, sd_def, sd_team, sd_resid, mu=5.0):
    """A fully-crossed BALANCED design: every (defender, team) cell has ``reps`` observations.

    Balanced so the Henderson-III moment estimator is well-conditioned (the two-way random EMS is
    exact and orthogonal), which is the regime where recovery is testable against known truth.
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, sd_def, n_def)  # defender effects ~ N(0, sd_def)
    b = rng.normal(0.0, sd_team, n_team)  # team effects   ~ N(0, sd_team)
    d_grid, t_grid = np.meshgrid(np.arange(n_def), np.arange(n_team), indexing="ij")
    defender_codes = np.repeat(d_grid.ravel(), reps)
    team_codes = np.repeat(t_grid.ravel(), reps)
    y = mu + a[defender_codes] + b[team_codes] + rng.normal(0.0, sd_resid, size=defender_codes.size)
    return y, defender_codes, team_codes


def test_recovers_known_variance_components_and_icc():
    """A balanced design with known s2_def=4, s2_team=2.25, s2_resid=9 -> ICC = 4/15.25 ~ 0.2623.

    Seed 4 at 80 defenders x 60 teams x 5 reps (24,000 obs) recovers, MEASURED:
    var_defender rel-err 0.050, var_team rel-err 0.064, var_resid rel-err 0.004, ICC err +0.0076.
    Tolerances carry margin above those achieved values for cross-platform BLAS float robustness.
    """
    sd_def, sd_team, sd_resid = 2.0, 1.5, 3.0
    true_def, true_team, true_resid = sd_def**2, sd_team**2, sd_resid**2
    true_icc = true_def / (true_def + true_team + true_resid)

    y, dc, tc = _balanced_crossed_design(4, 80, 60, 5, sd_def=sd_def, sd_team=sd_team, sd_resid=sd_resid)
    out = crossed_variance_components(y, dc, tc)

    assert out["had_negative_estimate"] is False
    # each component within a stated RELATIVE tolerance (measured errs 0.050 / 0.064 / 0.004)
    assert abs(out["var_defender"] - true_def) / true_def < 0.15
    assert abs(out["var_team"] - true_team) / true_team < 0.15
    assert abs(out["var_resid"] - true_resid) / true_resid < 0.05
    # ICC within +/- 0.03 of truth (measured err +0.0076)
    assert abs(out["icc_defender"] - true_icc) < 0.03


def test_defender_null_design_gives_near_zero_icc_with_bootstrap_lo_at_zero():
    """s2_defender = 0: the ICC estimate is ~0 and the bootstrap lower bound does not exceed ~0.

    A defender-null design is the "not identifiable" case the census must not mistake for a ranking.
    The small design (30 x 20 x 4) keeps the 80-replicate bootstrap fast; the directional claim
    (icc ~ 0, lo <= epsilon) is robust to the seed.
    """
    y, dc, tc = _balanced_crossed_design(4, 30, 20, 4, sd_def=0.0, sd_team=1.5, sd_resid=3.0)

    point = crossed_variance_components(y, dc, tc)
    assert point["icc_defender"] < 0.03  # near zero

    ci = bootstrap_icc_ci(y, dc, tc, n_boot=80, alpha=0.10, rng_seed=1)
    assert abs(ci["icc"] - point["icc_defender"]) < 1e-12  # point matches the direct fit
    assert ci["lo"] <= 0.05  # lower bound includes / does not exceed ~0
    assert ci["lo"] <= ci["hi"]


def test_strong_defender_design_gives_bootstrap_lo_above_zero():
    """A large s2_defender: the whole bootstrap interval sits above 0, i.e. the ICC is detectable.

    Same small fast design as the null case (30 x 20 x 4, 80 replicates); the point ICC is ~0.8 and
    the lower bound clears 0 comfortably -- the "defender ranking is identifiable" verdict.
    """
    y, dc, tc = _balanced_crossed_design(4, 30, 20, 4, sd_def=4.0, sd_team=1.0, sd_resid=2.0)

    point = crossed_variance_components(y, dc, tc)
    assert point["icc_defender"] > 0.5

    ci = bootstrap_icc_ci(y, dc, tc, n_boot=80, alpha=0.10, rng_seed=1)
    assert ci["lo"] > 0.0
    assert ci["lo"] <= ci["icc"] <= ci["hi"]


def test_tiny_degenerate_design_truncates_negative_estimate_to_zero():
    """A tiny defender-null design forces a negative RAW variance estimate -> truncated to 0.0 + flag.

    4 defenders x 3 teams x 2 reps (24 obs) with s2_defender = 0: sampling noise pushes the raw
    Henderson-III solve for var_defender negative, so the estimator truncates it to exactly 0.0 and
    sets had_negative_estimate=True. Seed 1 is the committed choice.
    """
    y, dc, tc = _balanced_crossed_design(1, 4, 3, 2, sd_def=0.0, sd_team=3.0, sd_resid=3.0)
    out = crossed_variance_components(y, dc, tc)

    assert out["had_negative_estimate"] is True
    assert out["var_defender"] == 0.0
    # the truncated components stay non-negative and the ICC reads as a clean non-detection
    assert out["var_team"] >= 0.0
    assert out["var_resid"] >= 0.0
    assert out["icc_defender"] == 0.0


def test_icc_denominator_zero_returns_zero_not_nan():
    """A perfectly flat outcome (all components truncate to 0) returns icc_defender = 0.0, not NaN.

    Documents the guarded-denominator choice: a downstream ``icc > threshold`` gate reads 0.0 as
    "measured, no defender separation" rather than tripping on a NaN comparison.
    """
    # constant y across a valid 3-level design: MS_d = MS_t = MSE = 0 -> all components 0.
    y, dc, tc = _balanced_crossed_design(0, 5, 4, 3, sd_def=0.0, sd_team=0.0, sd_resid=0.0)
    out = crossed_variance_components(y, dc, tc)
    assert out["var_defender"] == 0.0
    assert out["icc_defender"] == 0.0
    assert not np.isnan(out["icc_defender"])


# --------------------------------------------------------------------------------------------------
# Numerical-robustness guards (the lstsq reduction + NaN drop + bootstrap skip). Caught on the real
# broad corpus: the old ``pinv(X'X)`` reduction failed its internal SVD ("SVD did not converge" /
# "DLASCL parameter 4 illegal value") on a huge, sparse, ill-conditioned crossed resample, aborting
# the whole census. See scripts/_crossed_icc.py for the fix.
# --------------------------------------------------------------------------------------------------


def _ill_conditioned_design():
    """A rank-deficient crossed design: DUPLICATED indicator columns + single-observation levels.

    Duplicated columns make the Gram ``X'X`` exactly singular (rank-deficient); many one-row defender
    levels make it huge and sparse. This is the shape whose ``pinv(X'X)`` reduction tripped the SVD on
    the corpus. Returned as an explicit ``design`` (with duplicated columns) so the reduction path is
    exercised directly, plus the ``(y, dc, tc)`` a fitter would receive.
    """
    rng = np.random.default_rng(0)
    k = 400  # 400 defenders, each with a single observation -> maximally sparse, rank-deficient
    dc = np.arange(k)
    tc = rng.integers(0, 4, size=k)
    y = rng.normal(size=k)
    return y, dc, tc


def test_reduction_lstsq_matches_pinv_on_well_conditioned_design():
    """The lstsq reduction recovers the old ``(X'y)' pinv(X'X) (X'y)`` value + rank to numeric precision.

    This is the estimate-preserving proof for the reduction change: on a well-conditioned design the
    QR-based projection ``y'(X sol)`` equals the Gram-pseudo-inverse reduction, and the lstsq-reported
    rank equals ``matrix_rank(X'X)``. Measured max |delta| ~ 4e-11 on the reductions below.
    """
    y, dc, tc = _balanced_crossed_design(4, 20, 15, 4, sd_def=2.0, sd_team=1.5, sd_resid=3.0)
    _, inv_d = np.unique(dc, return_inverse=True)
    _, inv_t = np.unique(tc, return_inverse=True)
    z_def = np.zeros((len(y), inv_d.max() + 1))
    z_def[np.arange(len(y)), inv_d] = 1.0
    z_team = np.zeros((len(y), inv_t.max() + 1))
    z_team[np.arange(len(y)), inv_t] = 1.0

    for design in (np.ones((len(y), 1)), z_def, np.hstack([z_def, z_team])):
        red_lstsq, rank_lstsq = _reduction_and_rank(design, y)
        # old Gram-pseudo-inverse reduction + rank, computed inline as the reference
        gram = design.T @ design
        xty = design.T @ y
        red_pinv = float(xty @ (np.linalg.pinv(gram) @ xty))
        rank_pinv = int(np.linalg.matrix_rank(gram))
        assert abs(red_lstsq - red_pinv) < 1e-6
        assert rank_lstsq == rank_pinv


def test_ill_conditioned_design_returns_finite_no_linalgerror():
    """A rank-deficient / near-singular crossed design fits to a finite result, no ``LinAlgError`` escapes.

    The lstsq reduction is rank-deficiency-safe, so the estimator names finite components (or the
    documented NaN "not measurable" ICC) instead of raising the SVD-non-convergence the ``pinv(X'X)``
    path could. The single-observation defender levels here leave ``df_e <= 0`` (n == rank_full), which
    is exactly the documented NaN exit -- the point of the test is that it RETURNS rather than crashing.
    """
    y, dc, tc = _ill_conditioned_design()
    out = crossed_variance_components(y, dc, tc)  # must not raise
    # every named component is finite or the documented NaN; icc is finite-or-NaN, never an exception
    assert np.isfinite(out["var_defender"])
    assert np.isfinite(out["var_team"])
    assert np.isnan(out["icc_defender"]) or np.isfinite(out["icc_defender"])

    # and the reduction path itself is finite + rank-safe on the rank-DEFICIENT (duplicated-col) design
    _, inv_d = np.unique(dc, return_inverse=True)
    z_def = np.zeros((len(y), inv_d.max() + 1))
    z_def[np.arange(len(y)), inv_d] = 1.0
    design_deficient = np.hstack([z_def, z_def[:, :5]])  # duplicated columns -> singular Gram
    red, rank = _reduction_and_rank(design_deficient, y)  # must not raise
    assert np.isfinite(red)
    assert rank == z_def.shape[1]  # duplicates add no rank


def test_nan_in_y_is_dropped_no_crash():
    """A non-finite ``y`` row (NaN / Inf) is dropped before the fit -- no crash, result matches the drop.

    The guard exists so a degenerate metric value never reaches the QR/SVD (the ``DLASCL`` root cause).
    Injecting NaN/Inf into a well-conditioned design must yield the same components as fitting the
    already-cleaned rows.
    """
    y, dc, tc = _balanced_crossed_design(4, 30, 20, 4, sd_def=2.0, sd_team=1.5, sd_resid=3.0)
    y_bad = y.copy()
    y_bad[3] = np.nan
    y_bad[7] = np.inf
    y_bad[11] = -np.inf

    out_bad = crossed_variance_components(y_bad, dc, tc)  # must not raise
    keep = np.isfinite(y_bad)
    out_ref = crossed_variance_components(y_bad[keep], dc[keep], tc[keep])

    assert np.isfinite(out_bad["icc_defender"])
    assert out_bad["var_defender"] == pytest.approx(out_ref["var_defender"])
    assert out_bad["var_team"] == pytest.approx(out_ref["var_team"])
    assert out_bad["var_resid"] == pytest.approx(out_ref["var_resid"])
    assert out_bad["icc_defender"] == pytest.approx(out_ref["icc_defender"])


def test_all_y_nan_falls_through_to_not_measurable_nan():
    """Every ``y`` non-finite -> all rows drop -> the empty design hits the documented NaN exit, no crash."""
    y, dc, tc = _balanced_crossed_design(4, 5, 4, 3, sd_def=2.0, sd_team=1.5, sd_resid=3.0)
    out = crossed_variance_components(np.full_like(y, np.nan), dc, tc)  # must not raise
    assert np.isnan(out["icc_defender"])
    assert out["had_negative_estimate"] is False


def test_bootstrap_skips_degenerate_replicates_without_raising(monkeypatch):
    """Some bootstrap replicates raise/NaN -> the CI is built over the CONVERGENT ones, no crash.

    Monkeypatches the estimator the bootstrap calls so a deterministic subset of replicates raises a
    ``LinAlgError`` (the SVD-non-convergence shape) and another subset returns a NaN ICC; the CI must
    still return finite ``lo``/``hi`` over the convergent replicates and count them in
    ``n_boot_effective``.
    """
    import _crossed_icc as mod

    real = mod.crossed_variance_components
    state = {"i": 0}

    def flaky(y, dc, tc):
        i = state["i"]
        state["i"] += 1
        if i == 0:  # the point estimate on the full data -- keep it real
            return real(y, dc, tc)
        # 2 of every 5 replicates are degenerate (1 raise, 1 NaN); the majority (3/5) converge, so the
        # CI is still formed and ci_degenerate stays False.
        if i % 5 == 1:
            raise np.linalg.LinAlgError("SVD did not converge")  # skipped (raised)
        if i % 5 == 2:
            return {"icc_defender": float("nan")}  # dropped (not finite)
        return {"icc_defender": 0.3 + 0.001 * (i % 5)}  # convergent (a spread, so percentiles differ)

    monkeypatch.setattr(mod, "crossed_variance_components", flaky)
    y, dc, tc = _balanced_crossed_design(4, 30, 20, 4, sd_def=4.0, sd_team=1.0, sd_resid=2.0)
    ci = mod.bootstrap_icc_ci(y, dc, tc, n_boot=100, alpha=0.10, rng_seed=1)

    assert np.isfinite(ci["icc"])
    assert np.isfinite(ci["lo"]) and np.isfinite(ci["hi"])
    assert ci["lo"] <= ci["hi"]
    # 100 replicates, ~3/5 convergent -> comfortably above n_boot//2, CI formed
    assert ci["n_boot_effective"] > 100 // 2
    assert ci["ci_degenerate"] is False


def test_bootstrap_too_few_convergent_returns_point_with_nan_ci():
    """When (almost) every replicate is degenerate, the CI is NaN + ``ci_degenerate`` flag, not a crash."""
    import _crossed_icc as mod

    real = mod.crossed_variance_components
    state = {"i": 0}

    def almost_all_bad(y, dc, tc):
        i = state["i"]
        state["i"] += 1
        if i == 0:
            return real(y, dc, tc)  # point estimate stays real
        raise np.linalg.LinAlgError("SVD did not converge")  # every replicate degenerate

    mod_state = mod.crossed_variance_components
    try:
        mod.crossed_variance_components = almost_all_bad  # type: ignore[assignment]
        y, dc, tc = _balanced_crossed_design(4, 30, 20, 4, sd_def=4.0, sd_team=1.0, sd_resid=2.0)
        ci = mod.bootstrap_icc_ci(y, dc, tc, n_boot=40, alpha=0.10, rng_seed=1)
    finally:
        mod.crossed_variance_components = mod_state  # type: ignore[assignment]

    assert np.isfinite(ci["icc"])  # the point estimate survives
    assert np.isnan(ci["lo"]) and np.isnan(ci["hi"])
    assert ci["n_boot_effective"] == 0
    assert ci["ci_degenerate"] is True


# --------------------------------------------------------------------------------------------------
# Absorption reductions + analytic rank == the lstsq path (perf rewrite byte-identity gate).
# --------------------------------------------------------------------------------------------------
def _lstsq_triplet(y, dc, tc):
    """The three OLS reductions + ranks via the ORIGINAL dense-design lstsq path (the reference)."""
    n = len(y)
    zd, zt = _indicator_matrix(dc), _indicator_matrix(tc)
    r_mu, rk_mu = _reduction_and_rank(np.ones((n, 1), dtype=float), y)
    r_def, rk_def = _reduction_and_rank(zd, y)
    r_full, rk_full = _reduction_and_rank(np.hstack([zd, zt]), y)
    return r_mu, r_def, r_full, rk_mu, rk_def, rk_full


@pytest.mark.parametrize("shape", ["balanced", "unbalanced", "disconnected"])
def test_absorption_reductions_match_lstsq(shape):
    """``_crossed_reductions`` (absorption, no dense SVD) == three ``_reduction_and_rank`` calls.

    The reduction ``y'Py`` is the unique orthogonal projection, so absorption reproduces it to numeric
    precision; the analytic ranks match ``lstsq``'s reported rank -- including a DISCONNECTED design
    where the bipartite defender<->team graph has two components (``c == 2``), the case an ``n_def +
    n_team - 1`` shortcut would get wrong.
    """
    if shape == "balanced":
        y, dc, tc = _balanced_crossed_design(1, 12, 8, 4, sd_def=2.0, sd_team=1.5, sd_resid=3.0)
    elif shape == "unbalanced":
        y, dc, tc = _balanced_crossed_design(2, 10, 6, 3, sd_def=2.0, sd_team=1.5, sd_resid=3.0)
        keep = np.arange(len(y)) % 7 != 0  # drop rows to break the balance
        y, dc, tc = y[keep], dc[keep], tc[keep]
    else:  # two independent (defender-pair, team) blocks -> c == 2
        dc = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        tc = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y = np.random.default_rng(3).normal(size=8)
        assert (
            _bipartite_components(np.unique(dc, return_inverse=True)[1], np.unique(tc, return_inverse=True)[1], 4, 2)
            == 2
        )

    y = np.asarray(y, dtype=float)
    got = _crossed_reductions(y, np.asarray(dc), np.asarray(tc))
    ref = _lstsq_triplet(y, np.asarray(dc), np.asarray(tc))
    assert abs(got[0] - ref[0]) < 1e-6  # R(mu)
    assert abs(got[1] - ref[1]) < 1e-6  # R(def)
    assert abs(got[2] - ref[2]) < 1e-6  # R(full)
    assert (got[3], got[4], got[5]) == (ref[3], ref[4], ref[5])  # ranks exact


def _pinv_ems_coefs(y, dc, tc):
    """The ORIGINAL pinv(design)-based EMS trace coefficients -- oracle for the absorbed coefs."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    zd, zt = _indicator_matrix(dc), _indicator_matrix(tc)
    d_mu, d_def, d_full = np.ones((n, 1), dtype=float), zd, np.hstack([zd, zt])
    return (
        _trace_proj_source(d_def, zd) - _trace_proj_source(d_mu, zd),
        _trace_proj_source(d_def, zt) - _trace_proj_source(d_mu, zt),
        _trace_proj_source(d_full, zd) - _trace_proj_source(d_def, zd),
        _trace_proj_source(d_full, zt) - _trace_proj_source(d_def, zt),
        float(n) - _trace_proj_source(d_full, zd),
        float(n) - _trace_proj_source(d_full, zt),
    )


@pytest.mark.parametrize("shape", ["balanced", "unbalanced"])
def test_absorption_ems_coefficients_match_pinv(shape):
    """The absorbed EMS trace coefficients (via pinv(W), n_team-sized) == the old pinv(design_full) path.

    This is what lets the reduce drop the O(p^3) trace pinv without moving the census ICC: the six
    coefficients feed the EMS solve, so gating them directly (not just the final ICC) catches a coef
    error that a lucky cancellation might hide in the aggregate.
    """
    if shape == "balanced":
        y, dc, tc = _balanced_crossed_design(1, 12, 8, 4, sd_def=2.0, sd_team=1.5, sd_resid=3.0)
    else:
        y, dc, tc = _balanced_crossed_design(2, 10, 6, 3, sd_def=2.0, sd_team=1.5, sd_resid=3.0)
        keep = np.arange(len(y)) % 7 != 0
        y, dc, tc = y[keep], dc[keep], tc[keep]
    fit = _crossed_reductions(np.asarray(y, dtype=float), np.asarray(dc), np.asarray(tc))
    got = (fit.coef_dd, fit.coef_dt, fit.coef_td, fit.coef_tt, fit.coef_ed, fit.coef_et)
    ref = _pinv_ems_coefs(y, dc, tc)
    for g, r, name in zip(got, ref, ["dd", "dt", "td", "tt", "ed", "et"], strict=True):
        assert abs(g - r) < 1e-6, f"coef_{name}: absorbed {g} != pinv {r}"


def _old_power(y, dc, tc, *, n_boot, effect, seed):
    """The census power proxy AS IT WAS (a separate loop, same seed) -- the merge's reference leg."""
    y, dc, tc = np.asarray(y, dtype=float), np.asarray(dc), np.asarray(tc)
    ud = np.unique(dc)
    k = len(ud)
    if k == 0:
        return float("nan")
    members = [np.flatnonzero(dc == d) for d in ud]
    rng = np.random.default_rng(seed)
    det = tot = 0
    for _ in range(n_boot):
        picked = rng.integers(0, k, size=k)
        idx = np.concatenate([members[j] for j in picked])
        val = crossed_variance_components(y[idx], dc[idx], tc[idx])["icc_defender"]
        if np.isfinite(val):
            tot += 1
            if val >= effect:
                det += 1
    return (det / tot) if tot else float("nan")


def test_bootstrap_icc_and_power_matches_separate_loops():
    """The merged one-pass bootstrap == running ``bootstrap_icc_ci`` + the old power loop separately.

    Same seed -> same ``rng.integers`` draw sequence -> same per-replicate ICC, so deriving BOTH the CI
    and the power from one pass is byte-identical to the two former loops (the whole point of the merge).
    """
    y, dc, tc = _balanced_crossed_design(4, 20, 10, 4, sd_def=2.0, sd_team=1.5, sd_resid=3.0)
    ci = bootstrap_icc_ci(y, dc, tc, n_boot=60, alpha=0.10, rng_seed=7)
    pw = _old_power(y, dc, tc, n_boot=60, effect=0.05, seed=7)
    res = bootstrap_icc_and_power(y, dc, tc, n_boot=60, alpha=0.10, effect_size=0.05, rng_seed=7, progress_every=0)
    assert res["icc"] == ci["icc"]
    assert res["lo"] == ci["lo"] and res["hi"] == ci["hi"]
    assert res["n_boot_effective"] == ci["n_boot_effective"]
    assert res["power"] == pw
