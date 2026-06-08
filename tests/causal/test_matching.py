import numpy as np

from silly_kicks._causal import matching as M


def _toy(n=200, seed=0):
    """Confounder x drives both treatment and outcome; known +0.5 effect."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 1))
    p = 1 / (1 + np.exp(-0.8 * x[:, 0]))
    z = (rng.uniform(size=n) < p).astype(int)
    y = 1.0 + 2.0 * x[:, 0] + 0.5 * z + rng.normal(scale=0.1, size=n)
    return x, z, y


# --- Task 1: fit_propensity + propensity_match ---


def test_fit_propensity_in_unit_interval_and_standardizes():
    # Multi-scale columns (metres vs tiny angles): the fit must standardize internally.
    rng = np.random.default_rng(1)
    x = np.column_stack([rng.normal(50, 10, 300), rng.normal(0, 0.01, 300)])
    z = (rng.uniform(size=300) < 0.5).astype(int)
    ps, coefs = M.fit_propensity(x, z, seed=42)
    assert ps.shape == (300,)
    assert np.all((ps > 0) & (ps < 1))
    assert np.isfinite(coefs).all()


def test_fit_propensity_deterministic():
    x, z, _ = _toy()
    a, _ = M.fit_propensity(x, z, seed=42)
    b, _ = M.fit_propensity(x, z, seed=42)
    np.testing.assert_array_equal(a, b)


def test_propensity_match_with_replacement_reuses_controls():
    ps = np.array([0.9, 0.9, 0.9, 0.1, 0.88])
    z = np.array([1, 1, 1, 0, 0])
    matched = M.propensity_match(ps, z, target="att")
    treated = np.where(z == 1)[0]
    assert set(matched.keys()) == set(treated.tolist())
    used = list(matched.values())
    assert all(z[c] == 0 for c in used)
    assert len(used) != len(set(used))  # a control reused (with replacement)


def test_no_caliper_keeps_all_treated():
    x, z, _ = _toy()
    ps, _ = M.fit_propensity(x, z, seed=42)
    matched = M.propensity_match(ps, z, target="att")
    assert set(matched.keys()) == set(np.where(z == 1)[0].tolist())


# --- Task 2: smd_balance + abadie_imbens_se ---


def test_smd_balance_table_shape():
    x, z, _ = _toy(n=400, seed=5)
    ps, _ = M.fit_propensity(x, z, seed=42)
    matched = M.propensity_match(ps, z, target="att")
    bal = M.smd_balance(x, z, matched, target="att")
    assert list(bal.columns) == ["covariate", "smd_pre", "smd_post"]
    assert bal.shape[0] == x.shape[1]
    assert abs(bal["smd_post"].iloc[0]) < abs(bal["smd_pre"].iloc[0])  # matching improves balance


def test_abadie_imbens_se_positive_finite():
    x, z, y = _toy(n=600, seed=11)
    ps, _ = M.fit_propensity(x, z, seed=42)
    matched = M.propensity_match(ps, z, target="att")
    se = M.abadie_imbens_se(y, z, matched, ps, target="att")
    assert se > 0 and np.isfinite(se)


def test_ai_se_differs_from_naive_under_reuse():
    rng = np.random.default_rng(0)
    n_t, n_c = 50, 5  # few controls -> heavy reuse -> AI correction strictly positive
    ps = np.concatenate([np.full(n_t, 0.7), np.linspace(0.65, 0.75, n_c)])
    z = np.concatenate([np.ones(n_t, int), np.zeros(n_c, int)])
    y = np.concatenate([rng.normal(1.0, 1.0, n_t), rng.normal(0.0, 1.0, n_c)])
    matched = M.propensity_match(ps, z, target="att")
    ai = M.abadie_imbens_se(y, z, matched, ps, target="att")
    focal = np.array(sorted(matched.keys()))
    comp = np.array([matched[int(i)] for i in focal])
    naive = (y[focal] - y[comp]).std(ddof=1) / np.sqrt(len(focal))
    assert ai > naive


# --- Task 3: estimate_att / estimate_atnt ---


def test_recovers_known_ate():
    x, z, y = _toy(n=1500, seed=3)
    ps, _ = M.fit_propensity(x, z, seed=42)
    att = M.estimate_att(y, z, ps, x)
    assert abs(att.estimate - 0.5) < 0.15  # true effect 0.5
    assert att.se > 0 and np.isfinite(att.se)
    assert att.n_focal == int(z.sum())


def test_atnt_runs():
    x, z, y = _toy(n=800, seed=7)
    ps, _ = M.fit_propensity(x, z, seed=42)
    atnt = M.estimate_atnt(y, z, ps, x)
    assert np.isfinite(atnt.estimate) and atnt.se > 0


def test_estimate_deterministic():
    x, z, y = _toy(n=500, seed=9)
    ps, _ = M.fit_propensity(x, z, seed=42)
    assert M.estimate_att(y, z, ps, x).estimate == M.estimate_att(y, z, ps, x).estimate


# --- Task 4: placebo_shift ---


def test_placebo_permuted_gk_zero_shift():
    # GK block with NO real association -> permuted-row null centered ~0.
    rng = np.random.default_rng(13)
    n = 1200
    x_base = rng.normal(size=(n, 3))
    x_gk = rng.normal(size=(n, 4))  # independent of (z, y)
    z = (rng.uniform(size=n) < 1 / (1 + np.exp(-0.6 * x_base[:, 0]))).astype(int)
    y = 1.0 + x_base[:, 0] + 0.5 * z + rng.normal(scale=0.1, size=n)
    out = M.placebo_shift(x_base, x_gk, y, z, n_seeds=20, rng_seed=0)
    assert out["shifts"].shape == (20,)
    assert abs(np.median(out["shifts"])) < 0.1
    assert np.isfinite(out["band_p95"])


def test_placebo_deterministic():
    rng = np.random.default_rng(2)
    n = 400
    x_base, x_gk = rng.normal(size=(n, 3)), rng.normal(size=(n, 4))
    z = (rng.uniform(size=n) < 0.5).astype(int)
    y = rng.normal(size=n)
    a = M.placebo_shift(x_base, x_gk, y, z, n_seeds=10, rng_seed=7)["shifts"]
    b = M.placebo_shift(x_base, x_gk, y, z, n_seeds=10, rng_seed=7)["shifts"]
    np.testing.assert_array_equal(a, b)
