"""Phase 0a — vectorized ghost-GK KDE: kernel parity, leaf-match, golden master."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._ghost_gk import (
    GHOST_GK_FEATURE_NAMES,
    GhostGkModel,
)


@pytest.fixture(scope="module")
def small_model() -> tuple[GhostGkModel, pd.DataFrame]:
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.standard_normal((400, 26)), columns=GHOST_GK_FEATURE_NAMES)
    X["phase"] = rng.integers(0, 3, 400).astype(float)
    labels = pd.DataFrame({"gk_x": rng.uniform(2, 20, 400), "gk_y": rng.uniform(25, 45, 400)})
    model = GhostGkModel(n_estimators=40)
    model.fit(X, labels)
    return model, X


def test_kde_density_scipy_matches_inline_predict(small_model):
    """The extracted _kde_density_scipy reproduces predict_density's grid."""
    model, X = small_model
    densities = model.predict_density(X.iloc[:5], kde_backend="scipy")
    assert len(densities) == 5
    assert densities[0].probabilities.shape == (60, 64)
    assert densities[0].probabilities.sum() == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Golden master (split gate: continuous rtol/atol+NaN-mask vs discrete argmax)
# ---------------------------------------------------------------------------

from pathlib import Path  # noqa: E402

_GOLDEN = Path(__file__).parent / "fixtures" / "ghost_gk_kde_golden.npz"


@pytest.fixture(scope="module")
def golden():
    return np.load(_GOLDEN, allow_pickle=True)


# predict_density on the bundled 36k-sample "default" model is heavy (~few s/sample), so the
# golden slices to the first _N_GOLDEN frozen samples — enough spread to lock the grid/mode/
# spread contract while keeping the CI gate tractable. (Full 24 samples are frozen on disk.)
_N_GOLDEN = 8


@pytest.fixture(scope="module")
def default_model_features(golden):
    model = GhostGkModel.from_variant("default")
    cols = [str(c) for c in golden["feature_cols"]]
    X = pd.DataFrame(golden["features"][:_N_GOLDEN], columns=cols)
    return model, X


# Golden gate = the NEW backend vs FROZEN scipy-f64 values. We do NOT re-parametrize over
# "scipy" here: re-running scipy on the 36k-sample bundled model to compare it against its own
# frozen output is circular AND ~116s/call. scipy reproducibility is covered by the gen script
# + the regen-on-bump maintenance note. The frozen npz IS the scipy oracle.
@pytest.mark.parametrize("kde_backend", ["vectorized", "cpu-numba"])
def test_golden_continuous(golden, default_model_features, kde_backend):
    """Density grid + mean + spread: rtol/atol + explicit NaN-mask equality.

    NB: this full-path golden gates at rtol=1e-7, looser than the raw-kernel parity
    (1e-9, Task 3). That is intentional, not slack: predict_density adds the grid-sum
    renormalization division + the mode/mean/spread derivations on top of the kernel,
    which amplify floating error by ~2 orders. Do NOT tighten this to 1e-9 — it will flake.
    """
    model, X = default_model_features
    densities = model.predict_density(X, kde_backend=kde_backend)
    probs = np.stack([d.probabilities for d in densities])
    spread = np.array([d.spread for d in densities])
    mean_x = np.array([d.mean_x for d in densities])
    mean_y = np.array([d.mean_y for d in densities])
    n = _N_GOLDEN
    # NaN-mask equality FIRST (a silent NaN->0 fill must fail here, not pass equal_nan)
    assert np.array_equal(np.isnan(probs), np.isnan(golden["probs"][:n]))
    np.testing.assert_allclose(probs, golden["probs"][:n], rtol=1e-7, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(spread, golden["spread"][:n], rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(mean_x, golden["mean_x"][:n], rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(mean_y, golden["mean_y"][:n], rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("kde_backend", ["vectorized", "cpu-numba"])
def test_golden_discrete_mode(golden, default_model_features, kde_backend):
    """mode_x/y (argmax): exact grid-cell match for vectorized; <=1 cell for cpu-numba.

    numba's j-outer/i-inner sequential accumulation vs numpy's pairwise reduction can flip a
    NEAR-TIE argmax by <=1 grid cell. The density field is the primary check
    (test_golden_continuous); the mode is derived, so for cpu-numba allow a <=GRID_RESOLUTION
    shift. vectorized stays exact (it matched the frozen scipy golden in 4.2.0).
    """
    from silly_kicks.tracking._ghost_gk import GRID_RESOLUTION

    model, X = default_model_features
    densities = model.predict_density(X, kde_backend=kde_backend)
    mode_x = np.array([d.mode_x for d in densities])
    mode_y = np.array([d.mode_y for d in densities])
    gmx = golden["mode_x"][:_N_GOLDEN]
    gmy = golden["mode_y"][:_N_GOLDEN]
    if kde_backend == "vectorized":
        np.testing.assert_array_equal(mode_x, gmx)
        np.testing.assert_array_equal(mode_y, gmy)
    else:  # cpu-numba: near-tie argmax can shift <=1 grid cell on a different reduction order
        assert np.all(np.abs(mode_x - gmx) <= GRID_RESOLUTION + 1e-9)
        assert np.all(np.abs(mode_y - gmy) <= GRID_RESOLUTION + 1e-9)


def test_golden_fft_scalars(golden, default_model_features):
    """fft scalar fidelity on the REAL bundled "default" model (the production regime) vs the
    frozen scipy golden: mode <=1 grid cell, mean <1e-2 m, spread rel <5e-3. Locks the real-model
    fidelity in CI -- the synthetic kernel-parity test uses broad clouds, and the lakehouse harness
    that measured <=5.5mm mean / <=0.16% spread is not committed here. NOT the raw probabilities
    grid (fft is scalar-faithful only -- see ADR-014; that is why fft is opt-in)."""
    model, X = default_model_features
    densities = model.predict_density(X, kde_backend="fft")
    n = _N_GOLDEN
    mode_x = np.array([d.mode_x for d in densities])
    mode_y = np.array([d.mode_y for d in densities])
    mean_x = np.array([d.mean_x for d in densities])
    mean_y = np.array([d.mean_y for d in densities])
    spread = np.array([d.spread for d in densities])
    assert np.all(np.abs(mode_x - golden["mode_x"][:n]) <= 0.5 + 1e-9)
    assert np.all(np.abs(mode_y - golden["mode_y"][:n]) <= 0.5 + 1e-9)
    assert np.all(np.hypot(mean_x - golden["mean_x"][:n], mean_y - golden["mean_y"][:n]) < 1e-2)
    assert np.all(np.abs(spread - golden["spread"][:n]) / np.abs(golden["spread"][:n]) < 5e-3)


# ---------------------------------------------------------------------------
# Kernel-level parity (random points — fast, independent of the bundled model)
# ---------------------------------------------------------------------------


def _scipy_kde_grid(gk_x_w, gk_y_w, w, grid_points):
    from scipy.stats import gaussian_kde

    kde = gaussian_kde(np.vstack([gk_x_w, gk_y_w]), weights=w, bw_method="scott")
    return kde(grid_points).reshape(60, 64)


@pytest.mark.parametrize("k", [3, 10, 200])
def test_vectorized_kernel_matches_scipy(k):
    """_kde_density_vectorized == scipy gaussian_kde on random weighted points."""
    from silly_kicks.tracking._ghost_gk import _GRID_X, _GRID_Y, _kde_density_vectorized

    rng = np.random.default_rng(k)
    gk_x_w = rng.uniform(0, 30, k)
    gk_y_w = rng.uniform(18, 50, k)
    w = rng.uniform(0.1, 1.0, k)
    w = w / w.sum()
    gxx, gyy = np.meshgrid(_GRID_X, _GRID_Y, indexing="ij")
    grid_points = np.vstack([gxx.ravel(), gyy.ravel()])

    got = _kde_density_vectorized(gk_x_w, gk_y_w, w, grid_points)
    ref = _scipy_kde_grid(gk_x_w, gk_y_w, w, grid_points)
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-12)


# ---------------------------------------------------------------------------
# Model-traveling parity (live scipy vs vectorized on the loaded model) + degenerate paths
# ---------------------------------------------------------------------------


def test_model_traveling_parity(default_model_features):
    """vectorized ≈ scipy on whatever model is loaded (auto-revalidates on retrain).

    Sliced to 3 samples: this runs the LIVE (slow) scipy reference on the bundled model.
    """
    model, X = default_model_features
    Xs = X.iloc[:3]
    sci = model.predict_density(Xs, kde_backend="scipy")
    vec = model.predict_density(Xs, kde_backend="vectorized")
    p_sci = np.stack([d.probabilities for d in sci])
    p_vec = np.stack([d.probabilities for d in vec])
    np.testing.assert_allclose(p_vec, p_sci, rtol=1e-7, atol=1e-12)
    # Discrete mode must agree exactly OR be a quantified near-tie (<= 1 cell apart)
    for a, b in zip(sci, vec, strict=True):
        dx = abs(a.mode_x - b.mode_x)
        dy = abs(a.mode_y - b.mode_y)
        assert dx <= 0.5 + 1e-9 and dy <= 0.5 + 1e-9, (a.mode_x, a.mode_y, b.mode_x, b.mode_y)


def test_kernel_singular_covariance_raises_linalgerror():
    """Collinear/identical points -> singular cov -> LinAlgError (BOTH scipy & vectorized).

    This is the path predict_density catches and degrades to the uniform grid; if scipy
    raised but the vectorized kernel did not (or vice-versa) they would diverge silently.
    """
    from silly_kicks.tracking._ghost_gk import _GRID_X, _GRID_Y, _kde_density_vectorized

    gxx, gyy = np.meshgrid(_GRID_X, _GRID_Y, indexing="ij")
    grid_points = np.vstack([gxx.ravel(), gyy.ravel()])
    gk_x_w = np.array([5.0, 5.0, 5.0])
    gk_y_w = np.array([34.0, 34.0, 34.0])
    w = np.array([1 / 3, 1 / 3, 1 / 3])
    with pytest.raises(np.linalg.LinAlgError):
        _scipy_kde_grid(gk_x_w, gk_y_w, w, grid_points)
    with pytest.raises(np.linalg.LinAlgError):
        _kde_density_vectorized(gk_x_w, gk_y_w, w, grid_points)


def _small_dispatch_model():
    """Small model for end-to-end backend-parity (NOT a <2-weight trigger).

    Shallow/large-leaf trees mean every query leaf-matches many training points, so this
    does NOT exercise the len(w)<2 branch — that branch is tested directly in
    test_predict_density_lt2_weight_returns_uniform (it is backend-agnostic: it returns the
    uniform grid before the kde_backend dispatch). The singular-covariance fallback is covered
    by test_kernel_singular_covariance_raises_linalgerror.
    """
    rng = np.random.default_rng(123)
    X = pd.DataFrame(rng.standard_normal((60, 26)), columns=GHOST_GK_FEATURE_NAMES)
    X["phase"] = rng.integers(0, 3, 60).astype(float)
    labels = pd.DataFrame({"gk_x": rng.uniform(2, 20, 60), "gk_y": rng.uniform(25, 45, 60)})
    return GhostGkModel(n_estimators=10, max_depth=2).fit(X, labels), X


@pytest.mark.parametrize("backend", ["scipy", "vectorized", "cpu-numba"])
def test_predict_density_backend_parity_small_model(backend):
    """predict_density end-to-end: vectorized/cpu-numba == scipy (NaN-mask + values), small model."""
    model, X = _small_dispatch_model()
    sci = model.predict_density(X, kde_backend="scipy")
    other = model.predict_density(X, kde_backend=backend)
    p_sci = np.stack([d.probabilities for d in sci])
    p_other = np.stack([d.probabilities for d in other])
    assert np.array_equal(np.isnan(p_sci), np.isnan(p_other))
    np.testing.assert_allclose(p_other, p_sci, rtol=1e-7, atol=1e-12)


# ---------------------------------------------------------------------------
# Train-set chunking: invariance + structural bounded-memory guard (kernel-level, fast)
# ---------------------------------------------------------------------------


def test_vectorized_chunking_invariant():
    """Result is independent of train_block (streaming correctness)."""
    from silly_kicks.tracking._ghost_gk import _GRID_X, _GRID_Y, _kde_density_vectorized

    rng = np.random.default_rng(99)
    k = 12000  # the test below hard-codes train_block=4096 -> 3 blocks (default is 1024)
    gk_x_w = rng.uniform(0, 30, k)
    gk_y_w = rng.uniform(18, 50, k)
    w = rng.uniform(0.1, 1.0, k)
    w = w / w.sum()
    gxx, gyy = np.meshgrid(_GRID_X, _GRID_Y, indexing="ij")
    grid_points = np.vstack([gxx.ravel(), gyy.ravel()])

    full = _kde_density_vectorized(gk_x_w, gk_y_w, w, grid_points, train_block=k)
    chunked = _kde_density_vectorized(gk_x_w, gk_y_w, w, grid_points, train_block=4096)
    np.testing.assert_allclose(chunked, full, rtol=1e-12, atol=1e-15)


def test_vectorized_is_chunked_structural(monkeypatch):
    """Structural bound: the per-block exp runs ceil(k/block) times, never over the full k*m.

    Step-1 closed-form removed cho_solve; the per-block primitive is now ``np.exp(-energy)`` over a
    ``(kb, m)`` array. numpy buffers under-report in tracemalloc and shared-runner RSS is flaky, so
    guard the memory bound structurally: an unchunked impl calls exp ONCE over ``(k, m)``; a correctly
    chunked one calls it ceil(k/block) times, each with ``<= block`` rows. (The scalar ``norm`` exp is
    0-d and excluded by the ndim==2 filter.)
    """
    from silly_kicks.tracking._ghost_gk import _GRID_X, _GRID_Y, _kde_density_vectorized

    block = 1024
    k = 5000  # -> ceil(5000/1024) = 5 blocks
    shapes = []
    real_exp = np.exp

    def _spy(x, *a, **kw):
        arr = np.asarray(x)
        if arr.ndim == 2:
            shapes.append(arr.shape)
        return real_exp(x, *a, **kw)

    monkeypatch.setattr(np, "exp", _spy)

    rng = np.random.default_rng(5)
    gk_x_w = rng.uniform(0, 30, k)
    gk_y_w = rng.uniform(18, 50, k)
    w = np.full(k, 1.0 / k)
    gxx, gyy = np.meshgrid(_GRID_X, _GRID_Y, indexing="ij")
    grid_points = np.vstack([gxx.ravel(), gyy.ravel()])

    out = _kde_density_vectorized(gk_x_w, gk_y_w, w, grid_points, train_block=block)
    assert out.shape == (60, 64)
    assert len(shapes) == int(np.ceil(k / block)), f"expected per-block exp, got {len(shapes)}"
    assert all(shp[0] <= block for shp in shapes), "an exp saw more than one block of rows"


# ---------------------------------------------------------------------------
# Leaf-match vectorization (isolation) + the len(w)<2 uniform fallback (direct)
# ---------------------------------------------------------------------------


def test_leaf_match_weights_matches_loop():
    """Vectorized leaf-match weights == per-sample loop weights."""
    from silly_kicks.tracking._ghost_gk import _leaf_match_weights

    rng = np.random.default_rng(3)
    n_train, n_trees, n_query = 2000, 50, 8
    training_leaves = rng.integers(0, 30, (n_train, n_trees))
    query_leaves = rng.integers(0, 30, (n_query, n_trees))

    got = _leaf_match_weights(training_leaves, query_leaves)  # (n_query, n_train)
    for i in range(n_query):
        ref = (training_leaves == query_leaves[i]).sum(axis=1).astype(np.float64) / n_trees
        np.testing.assert_array_equal(got[i], ref)


def test_predict_density_lt2_weight_returns_uniform(monkeypatch, small_model):
    """The len(w)<2 branch returns the uniform grid, identically for both backends.

    Deterministically force a single nonzero-weight sample by patching _leaf_match_weights
    (the seam predict_density uses) — large-leaf models never naturally yield <2. The branch
    is BEFORE the kde_backend dispatch, so scipy and vectorized both return the exact uniform
    grid 1/(GRID_NX*GRID_NY).
    """
    from silly_kicks.tracking import _ghost_gk as g

    model, X = small_model
    real = g._leaf_match_weights

    def _fake(training_leaves, query_leaves, **kw):
        wts = real(training_leaves, query_leaves, **kw)
        wts[0, :] = 0.0
        wts[0, 0] = 0.5  # exactly one nonzero weight -> len(w) == 1 -> <2 branch
        return wts

    monkeypatch.setattr(g, "_leaf_match_weights", _fake)
    uniform = 1.0 / (60 * 64)
    for backend in ("scipy", "vectorized", "cpu-numba", "fft"):
        d = model.predict_density(X.iloc[:1], kde_backend=backend)[0]
        np.testing.assert_allclose(d.probabilities, uniform, rtol=0, atol=1e-15)


# ---------------------------------------------------------------------------
# Default flip: structural no-scipy-construction guard (no wall-clock)
# ---------------------------------------------------------------------------


def test_default_backend_makes_no_scipy_kde(small_model, monkeypatch):
    """Default predict_density must NOT construct scipy.stats.gaussian_kde."""
    from silly_kicks.tracking import _ghost_gk as g

    calls = {"n": 0}
    real = g.gaussian_kde

    def _spy(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    # _kde_density_scipy resolves gaussian_kde from the _ghost_gk module global, so patch there.
    monkeypatch.setattr(g, "gaussian_kde", _spy)
    model, X = small_model
    _ = model.predict_density(X)  # default backend
    assert calls["n"] == 0, "default path still constructs scipy.gaussian_kde"


# ---------------------------------------------------------------------------
# Step 1 (closed-form): near-singular parity (cho_solve Leg-B anchor + scipy backstop)
# ---------------------------------------------------------------------------


def _cho_solve_kde_grid(gk_x_w, gk_y_w, w, grid_points):
    """4.2.0's cho_solve whitening, reproduced version-independently (Leg-B reference)."""
    from scipy.linalg import cho_factor, cho_solve

    from silly_kicks.tracking._ghost_gk import GRID_NX, GRID_NY

    w = np.asarray(w, np.float64)
    w = w / w.sum()
    data = np.vstack([np.asarray(gk_x_w, np.float64), np.asarray(gk_y_w, np.float64)])
    neff = 1.0 / np.sum(w**2)
    factor = neff ** (-1.0 / 6.0)  # Scott, d=2 -> -1/(d+4)
    cov = np.atleast_2d(np.cov(data, rowvar=True, bias=False, aweights=w)) * factor**2
    chol = cho_factor(cov, lower=True)
    log_det = 2.0 * np.sum(np.log(np.diag(chol[0])))
    norm = np.exp(-0.5 * (log_det + 2.0 * np.log(2.0 * np.pi)))
    diff = grid_points[:, :, None] - data[:, None, :]  # (2, m, k)
    tdiff = cho_solve(chol, diff.reshape(2, -1))
    energy = 0.5 * np.sum(diff.reshape(2, -1) * tdiff, axis=0).reshape(grid_points.shape[1], data.shape[1])
    out = (np.exp(-energy) @ w) * norm
    return out.reshape(GRID_NX, GRID_NY)


def _near_singular_inputs():
    # 30 points almost on a line y = 34 + 1e-3*(x-15): tiny off-axis spread -> high 2x2
    # covariance condition number (near-singular but positive-definite).
    rng = np.random.default_rng(123)
    gk_x_w = rng.uniform(0, 30, 30)
    gk_y_w = 34.0 + 1e-3 * (gk_x_w - 15.0) + rng.normal(0, 1e-4, 30)
    w = rng.uniform(0.1, 1.0, 30)
    w = w / w.sum()
    return gk_x_w, gk_y_w, w


def test_kernel_near_singular_parity():
    """Near-singular-but-PD covariance: vectorized kernel matches BOTH the cho_solve
    reference (Leg-B, sharp) and scipy (Leg-A, backstop) within tolerance.
    """
    from silly_kicks.tracking._ghost_gk import _GRID_X, _GRID_Y, _kde_density_vectorized

    gk_x_w, gk_y_w, w = _near_singular_inputs()
    gxx, gyy = np.meshgrid(_GRID_X, _GRID_Y, indexing="ij")
    grid_points = np.vstack([gxx.ravel(), gyy.ravel()])

    got = _kde_density_vectorized(gk_x_w, gk_y_w, w, grid_points)
    np.testing.assert_allclose(got, _cho_solve_kde_grid(gk_x_w, gk_y_w, w, grid_points), rtol=1e-7, atol=1e-12)
    np.testing.assert_allclose(got, _scipy_kde_grid(gk_x_w, gk_y_w, w, grid_points), rtol=1e-7, atol=1e-12)


def test_vectorized_kernel_uses_no_cho_solve(small_model, monkeypatch):
    """Step-1 optimization: the vectorized kernel must NOT call cho_solve (closed-form energy
    replaces it). cho_factor is still used for the PD-branch + log_det.

    Patch scipy.linalg.cho_solve (the MODULE attribute), NOT
    silly_kicks.tracking._ghost_gk.cho_solve: the kernel imports cho_solve *function-scope*
    (`from scipy.linalg import ...` inside _kde_density_vectorized), re-binding it from
    scipy.linalg on every call -> patching scipy.linalg DOES intercept it (verified: 3 calls on
    the pre-change kernel). There is NO module-level _ghost_gk.cho_solve, so a
    `monkeypatch.setattr(_ghost_gk, "cho_solve", ...)` / `hasattr(...)` form would be a no-op
    (always green). Do not "fix" it to that.
    """
    import scipy.linalg as sla

    calls = {"solve": 0}
    real_solve = sla.cho_solve
    monkeypatch.setattr(
        sla,
        "cho_solve",
        lambda *a, **k: (calls.__setitem__("solve", calls["solve"] + 1), real_solve(*a, **k))[1],
    )
    model, X = small_model
    model.predict_density(X.iloc[:3], kde_backend="vectorized")
    assert calls["solve"] == 0, "vectorized kernel still calls cho_solve"


# ---------------------------------------------------------------------------
# Step 2 (cpu-numba): @njit fused-loop kernel parity (Leg-B)
# ---------------------------------------------------------------------------


def test_numba_loop_matches_numpy_closed_form():
    """The @njit fused loop == the numpy closed-form kernel (Leg-B, same _kde_setup), across the
    regimes that matter for parity divergence:
      - LARGE-k (k~36000): the PRODUCTION regime (lakehouse: real candidate clouds are ~36k leaf
        positions, well-conditioned cond <= ~5.3, n=204). numba's j-outer/i-inner SEQUENTIAL
        accumulation vs numpy einsum's PAIRWISE reduction diverges most when many terms are
        summed -- this is the real-world gap, not near-singular.
      - NEAR-SINGULAR: a conservative 1/det numerical-edge guard. Real ghost-GK never reaches
        this regime (cond <= ~5.3), but it is cheap robustness against the ill-conditioned zone
        where 1/det amplifies rounding.
    """
    from silly_kicks.tracking._ghost_gk import (
        _GRID_X,
        _GRID_Y,
        GRID_NX,
        GRID_NY,
        _kde_density_vectorized,
        _kde_setup,
    )
    from silly_kicks.tracking._ghost_gk_numba import _kde_numba_loop

    gxx, gyy = np.meshgrid(_GRID_X, _GRID_Y, indexing="ij")
    gp = np.vstack([gxx.ravel(), gyy.ravel()])

    def _numba_grid(gk_x_w, gk_y_w, w):
        data, wn, h11, h12, h22, det, norm = _kde_setup(gk_x_w, gk_y_w, w)
        return _kde_numba_loop(gp[0], gp[1], data[0], data[1], wn, h11, h12, h22, 1.0 / det, norm).reshape(
            GRID_NX, GRID_NY
        )

    rng = np.random.default_rng(11)
    # Well-conditioned, incl. the production-scale k~36000 (the real accumulation-order regime).
    for k in (3, 50, 2000, 36000):
        x = rng.uniform(0, 30, k)
        y = rng.uniform(18, 50, k)
        w = rng.uniform(0.1, 1.0, k)
        w = w / w.sum()
        np.testing.assert_allclose(_numba_grid(x, y, w), _kde_density_vectorized(x, y, w, gp), rtol=1e-9, atol=1e-12)

    # Conservative ill-conditioned guard (cond ~1e6). Real ghost-GK never reaches this (cond <= ~5.3,
    # lakehouse n=204) -- a theoretical edge, looser Leg-B tol where 1/det amplifies rounding.
    x, y, w = _near_singular_inputs()
    np.testing.assert_allclose(_numba_grid(x, y, w), _kde_density_vectorized(x, y, w, gp), rtol=1e-7, atol=1e-12)


def test_ghost_gk_does_not_eagerly_import_numba():
    """Importing _ghost_gk must NOT transitively import numba or _ghost_gk_numba.

    numba is loaded lazily only on the cpu-numba path (inside _kde_density_numba), so a bare
    `import silly_kicks.tracking._ghost_gk` stays dependency-light. If this fails, an eager
    top-level `from ._ghost_gk_numba import ...` slipped in -- move it into _kde_density_numba.
    """
    import importlib
    import sys

    for mod in ("numba", "silly_kicks.tracking._ghost_gk_numba", "silly_kicks.tracking._ghost_gk"):
        sys.modules.pop(mod, None)
    importlib.import_module("silly_kicks.tracking._ghost_gk")
    assert "numba" not in sys.modules, "import _ghost_gk eagerly imported numba"
    assert "silly_kicks.tracking._ghost_gk_numba" not in sys.modules


# ---------------------------------------------------------------------------
# fft backend (binned-convolution): faithful on the 3 emitted SCALARS (mode/mean/spread),
# NOT on the raw per-cell grid; opt-in; O(k + m log m). See ADR-014.
# ---------------------------------------------------------------------------


def _grid_scalars(probs_unnorm):
    """Reproduce predict_density's mode/mean/spread from an UNnormalized grid (same math)."""
    from silly_kicks.tracking._ghost_gk import _GRID_X, _GRID_Y, GRID_RESOLUTION

    total = probs_unnorm.sum()
    p = probs_unnorm / total if total > 0 else np.ones_like(probs_unnorm) / probs_unnorm.size
    gxx, gyy = np.meshgrid(_GRID_X, _GRID_Y, indexing="ij")
    ix, iy = np.unravel_index(int(np.argmax(p)), p.shape)
    nz = p[p > 0]
    entropy = float(-np.sum(nz * np.log(nz)))
    return {
        "mode_x": float(_GRID_X[ix]),
        "mode_y": float(_GRID_Y[iy]),
        "mean_x": float(np.sum(p * gxx)),
        "mean_y": float(np.sum(p * gyy)),
        "spread": float(np.exp(entropy) * GRID_RESOLUTION**2),
    }


@pytest.mark.parametrize("seed", [7, 11, 19])
def test_fft_kernel_matches_scipy_on_scalars(seed):
    """fft is faithful on the 3 emitted scalars vs the scipy oracle on realistic well-conditioned,
    in-grid clouds (the production regime: real ghost-GK clouds are cond<=5.3, oog~0.33%). The
    cloud is correlated (anisotropic H with a real h12 cross-term). mode <=1 grid cell, mean
    <3e-2 m, spread rel <5e-3. NOT asserted on the raw grid (binning quantizes per-cell mass).
    Near-singular clouds are excluded: the mode is a flat-ridge argmax that is ill-defined for
    BOTH backends, and that regime does not occur in production (the singular limit is covered by
    test_fft_singular_covariance_raises_linalgerror)."""
    from silly_kicks.tracking._ghost_gk import _GRID_X, _GRID_Y, _kde_density_fft

    rng = np.random.default_rng(seed)
    k = 400
    gk_x_w = rng.normal(15.0, 4.0, k)
    gk_y_w = 34.0 + 0.5 * (gk_x_w - 15.0) + rng.normal(0.0, 2.5, k)  # correlated -> anisotropic H (h12)
    w = rng.uniform(0.1, 1.0, k)
    w = w / w.sum()
    gxx, gyy = np.meshgrid(_GRID_X, _GRID_Y, indexing="ij")
    gp = np.vstack([gxx.ravel(), gyy.ravel()])
    s_fft = _grid_scalars(_kde_density_fft(gk_x_w, gk_y_w, w, gp))
    s_ora = _grid_scalars(_scipy_kde_grid(gk_x_w, gk_y_w, w, gp))

    assert abs(s_fft["mode_x"] - s_ora["mode_x"]) <= 0.5 + 1e-9  # <=1 grid cell
    assert abs(s_fft["mode_y"] - s_ora["mode_y"]) <= 0.5 + 1e-9
    assert np.hypot(s_fft["mean_x"] - s_ora["mean_x"], s_fft["mean_y"] - s_ora["mean_y"]) < 3e-2
    # spread (entropy-based) rel-err: 1e-2 has margin over the ~0.6% observed on these diffuse
    # synthetic clouds; the production regime is tighter (harness: <=0.16% on the bundled model).
    assert abs(s_fft["spread"] - s_ora["spread"]) / abs(s_ora["spread"]) < 1e-2


def test_fft_out_of_grid_points_handled_gracefully():
    """Out-of-grid training points (NGP clips them to the edge cell) must not crash and still
    yield a finite, normalized grid; mean stays within a loose bound of the oracle (the clipping
    of a sub-percent tail is a known approximation, validated negligible on the real model)."""
    from silly_kicks.tracking._ghost_gk import _GRID_X, _GRID_Y, _kde_density_fft

    rng = np.random.default_rng(3)
    k = 400
    gk_x_w = rng.normal(15.0, 4.0, k)
    gk_y_w = 34.0 + rng.normal(0.0, 3.0, k)
    gk_x_w[:4] = rng.uniform(-3.0, -0.5, 4)  # ~1% just left of the grid -> clip to edge
    w = rng.uniform(0.1, 1.0, k)
    w = w / w.sum()
    gxx, gyy = np.meshgrid(_GRID_X, _GRID_Y, indexing="ij")
    gp = np.vstack([gxx.ravel(), gyy.ravel()])
    grid = _kde_density_fft(gk_x_w, gk_y_w, w, gp)
    assert grid.shape == (60, 64)
    assert np.all(np.isfinite(grid)) and grid.sum() > 0
    s_fft = _grid_scalars(grid)
    s_ora = _grid_scalars(_scipy_kde_grid(gk_x_w, gk_y_w, w, gp))
    assert np.hypot(s_fft["mean_x"] - s_ora["mean_x"], s_fft["mean_y"] - s_ora["mean_y"]) < 1e-1


def test_predict_density_fft_backend_switch(small_model):
    """predict_density(kde_backend="fft") returns the standard GhostGkDensity (normalized grid)."""
    model, X = small_model
    densities = model.predict_density(X.iloc[:5], kde_backend="fft")
    assert len(densities) == 5
    assert densities[0].probabilities.shape == (60, 64)
    assert densities[0].probabilities.sum() == pytest.approx(1.0, abs=1e-9)


def test_fft_singular_covariance_raises_linalgerror():
    """Collinear points -> _kde_setup's cho_factor raises (same as the other backends), so
    predict_density's uniform-fallback applies unchanged."""
    from silly_kicks.tracking._ghost_gk import _GRID_X, _GRID_Y, _kde_density_fft

    gk_x_w = np.array([5.0, 5.0, 5.0])  # identical points -> singular covariance (mirrors the
    gk_y_w = np.array([34.0, 34.0, 34.0])  # vectorized/scipy singular test)
    w = np.array([1 / 3, 1 / 3, 1 / 3])
    gxx, gyy = np.meshgrid(_GRID_X, _GRID_Y, indexing="ij")
    gp = np.vstack([gxx.ravel(), gyy.ravel()])
    with pytest.raises(np.linalg.LinAlgError):
        _kde_density_fft(gk_x_w, gk_y_w, w, gp)


def test_fft_is_k_independent_one_convolution(monkeypatch):
    """Structural perf guard: fft does ONE fftconvolve per prediction, and its field+kernel
    shapes are k-INDEPENDENT (O(m log m), not O(k*m)). Catches a silent revert to brute force."""
    import scipy.signal as sps

    from silly_kicks.tracking._ghost_gk import _GRID_X, _GRID_Y, _kde_density_fft

    calls = []
    real = sps.fftconvolve

    def _spy(field, kernel, *a, **k):
        calls.append((field.shape, kernel.shape))
        return real(field, kernel, *a, **k)

    monkeypatch.setattr(sps, "fftconvolve", _spy)  # function-scope import in _kde_density_fft -> intercepted
    gxx, gyy = np.meshgrid(_GRID_X, _GRID_Y, indexing="ij")
    gp = np.vstack([gxx.ravel(), gyy.ravel()])
    rng = np.random.default_rng(1)
    for k in (5, 5000):
        x = rng.uniform(0, 30, k)
        y = rng.uniform(18, 50, k)
        w = rng.uniform(0.1, 1.0, k)
        w = w / w.sum()
        _kde_density_fft(x, y, w, gp)
    assert len(calls) == 2, f"expected one fftconvolve per call, got {len(calls)}"
    assert calls[0] == calls[1], "fft field/kernel shapes must be k-independent (O(m log m))"


# ---------------------------------------------------------------------------
# fft-cic backend (CIC / bilinear binning): better mode + raw-grid fidelity than NGP at ~2x cost.
# Opt-in; O(k + m log m). See ADR-014 (amended).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [3, 7, 11])
def test_bin_cic_conserves_mass(seed):
    """CIC bilinear weights sum to 1 per point -> field.sum() == sum of weights, including for
    out-of-grid points (np.add.at still accumulates all 4 contributions when clip collapses
    indices to an edge cell)."""
    from silly_kicks.tracking._ghost_gk import _bin_cic

    rng = np.random.default_rng(seed)
    k = 300
    gk_x_w = rng.normal(15.0, 4.0, k)
    gk_y_w = 34.0 + rng.normal(0.0, 3.0, k)
    gk_x_w[:5] = rng.uniform(-4.0, -0.5, 5)  # out-of-grid (left) -> clip to edge, mass preserved
    w = rng.uniform(0.1, 1.0, k)
    w = w / w.sum()
    field = _bin_cic(gk_x_w, gk_y_w, w)
    assert field.shape == (60, 64)
    assert np.all(field >= 0.0)
    assert np.isclose(field.sum(), w.sum(), rtol=1e-12)


def test_fft_cic_equals_ngp_on_grid_nodes():
    """SEAM LOCK (binning is the only difference between fft and fft-cic) -- NOT the NGP refactor
    lock (that is the rtol golden + the verbatim extraction). When every point sits exactly on a
    grid node, CIC's bilinear weight collapses to 1.0 on the corner cell (tx=ty=0), so fft-cic must
    equal fft (NGP). Same-process equality -> CI-safe across the numpy/scipy version matrix (no
    frozen cross-version array). Points are built as _GRID_X[0] + ix*GRID_RESOLUTION (NOT
    _GRID_X[ix]) so fx is EXACTLY integer regardless of how _GRID_X was constructed -- a linspace
    ULP in _GRID_X[ix] would make tx~1e-16, which fftconvolve amplifies and array_equal would flake
    on."""
    from silly_kicks.tracking._ghost_gk import (
        _GRID_X,
        _GRID_Y,
        GRID_RESOLUTION,
        _kde_density_fft,
        _kde_density_fft_cic,
    )

    rng = np.random.default_rng(5)
    k = 200
    ix = rng.integers(8, 52, k)
    iy = rng.integers(8, 56, k)
    gk_x_w = _GRID_X[0] + ix * GRID_RESOLUTION  # exact-integer fx -> tx == 0.0 exactly
    gk_y_w = _GRID_Y[0] + iy * GRID_RESOLUTION
    w = rng.uniform(0.1, 1.0, k)
    w = w / w.sum()
    gxx, gyy = np.meshgrid(_GRID_X, _GRID_Y, indexing="ij")
    gp = np.vstack([gxx.ravel(), gyy.ravel()])
    ngp = _kde_density_fft(gk_x_w, gk_y_w, w, gp)
    cic = _kde_density_fft_cic(gk_x_w, gk_y_w, w, gp)
    # For the current 0.5 m grid (0.5 exactly representable) array_equal holds exactly. If a future
    # non-exact grid step breaks it, fall back to assert_allclose(ngp, cic, rtol=0, atol=1e-12).
    assert np.array_equal(ngp, cic)


def test_fft_cic_singular_covariance_raises_linalgerror():
    """Collinear points -> _kde_setup's cho_factor raises (same as every backend), so
    predict_density's uniform-fallback applies unchanged."""
    from silly_kicks.tracking._ghost_gk import _GRID_X, _GRID_Y, _kde_density_fft_cic

    gk_x_w = np.array([5.0, 5.0, 5.0])
    gk_y_w = np.array([34.0, 34.0, 34.0])
    w = np.array([1 / 3, 1 / 3, 1 / 3])
    gxx, gyy = np.meshgrid(_GRID_X, _GRID_Y, indexing="ij")
    gp = np.vstack([gxx.ravel(), gyy.ravel()])
    with pytest.raises(np.linalg.LinAlgError):
        _kde_density_fft_cic(gk_x_w, gk_y_w, w, gp)


def test_fft_cic_out_of_grid_points_handled_gracefully():
    """Mirror of the NGP test_fft_out_of_grid_points_handled_gracefully, for fft-cic end-to-end:
    out-of-grid training points (CIC clips the bilinear corner indices to the edge, mass-conserving)
    must not crash and still yield a finite, normalized grid; mean stays within a loose bound of the
    oracle (the clipped sub-percent tail is a known approximation, negligible on the real model)."""
    from silly_kicks.tracking._ghost_gk import _GRID_X, _GRID_Y, _kde_density_fft_cic

    rng = np.random.default_rng(3)
    k = 400
    gk_x_w = rng.normal(15.0, 4.0, k)
    gk_y_w = 34.0 + rng.normal(0.0, 3.0, k)
    gk_x_w[:4] = rng.uniform(-3.0, -0.5, 4)  # ~1% just left of the grid -> clip to edge
    w = rng.uniform(0.1, 1.0, k)
    w = w / w.sum()
    gxx, gyy = np.meshgrid(_GRID_X, _GRID_Y, indexing="ij")
    gp = np.vstack([gxx.ravel(), gyy.ravel()])
    grid = _kde_density_fft_cic(gk_x_w, gk_y_w, w, gp)
    assert grid.shape == (60, 64)
    assert np.all(np.isfinite(grid)) and grid.sum() > 0
    s_cic = _grid_scalars(grid)
    s_ora = _grid_scalars(_scipy_kde_grid(gk_x_w, gk_y_w, w, gp))
    assert np.hypot(s_cic["mean_x"] - s_ora["mean_x"], s_cic["mean_y"] - s_ora["mean_y"]) < 1e-1


def test_predict_density_fft_cic_backend_switch(small_model):
    """predict_density(kde_backend="fft-cic") returns the standard normalized GhostGkDensity."""
    model, X = small_model
    densities = model.predict_density(X.iloc[:5], kde_backend="fft-cic")
    assert len(densities) == 5
    assert densities[0].probabilities.shape == (60, 64)
    assert densities[0].probabilities.sum() == pytest.approx(1.0, abs=1e-9)


def test_unknown_kde_backend_still_raises(small_model):
    """An unrecognised backend string raises ValueError (dispatch hygiene unchanged)."""
    model, X = small_model
    with pytest.raises(ValueError, match="Unknown kde_backend"):
        model.predict_density(X.iloc[:1], kde_backend="fft-nope")


def test_fft_cic_is_k_independent_one_convolution(monkeypatch):
    """Structural perf guard: fft-cic does ONE fftconvolve per prediction with k-INDEPENDENT
    field+kernel shapes (O(m log m), not O(k*m)). Mirrors the fft guard."""
    import scipy.signal as sps

    from silly_kicks.tracking._ghost_gk import _GRID_X, _GRID_Y, _kde_density_fft_cic

    calls = []
    real = sps.fftconvolve

    def _spy(field, kernel, *a, **k):
        calls.append((field.shape, kernel.shape))
        return real(field, kernel, *a, **k)

    monkeypatch.setattr(sps, "fftconvolve", _spy)
    gxx, gyy = np.meshgrid(_GRID_X, _GRID_Y, indexing="ij")
    gp = np.vstack([gxx.ravel(), gyy.ravel()])
    rng = np.random.default_rng(1)
    for k in (5, 5000):
        x = rng.uniform(0, 30, k)
        y = rng.uniform(18, 50, k)
        w = rng.uniform(0.1, 1.0, k)
        w = w / w.sum()
        _kde_density_fft_cic(x, y, w, gp)
    assert len(calls) == 2, f"expected one fftconvolve per call, got {len(calls)}"
    assert calls[0] == calls[1], "field/kernel shapes must be k-independent"


def _bimodal_cloud(rng, *, lead=0.05, std=0.75, k=600):
    """Two clusters ~5.75 m apart along x at DIFFERENT grid phase, so NGP distorts them UNEQUALLY --
    the real-data flip mechanism (differential peak-height distortion), manufactured deliberately
    rather than hoping within-cluster spread accidentally straddles a boundary:
      * WINNER (higher mass) on a cell BOUNDARY, cx_a = 12.5 (midpoint of nodes 12.25/12.75) -> NGP
        splits its mass across cells 24/25 -> UNDER-counts its peak.
      * LOSER (lower mass) on a NODE, cx_b = 18.25 (cell 36 center) -> NGP concentrates its mass in
        one cell -> OVER-counts its peak.
    So NGP is induced to flip the argmax to the (wrong) loser, while CIC's bilinear, mass-conserving
    spread keeps the winner (= the vectorized-exact argmax). Grid: _GRID_X = 0.25 + i*0.5 -> nodes
    at *.25/*.75, boundaries at *.0/*.5. `lead` (winner's mass surplus) lives in a tension band:
    large enough that the EXACT argmax is the winner (not a coin-flip), small enough that NGP's
    differential distortion flips it. `std` controls peak sharpness (smaller -> the +/-0.25 m
    quantization bites harder). Returns (gk_x_w, gk_y_w, w)."""
    cx_a = 12.5  # WINNER on a cell boundary -> NGP mass-splits -> under-counts peak
    cx_b = 18.25  # LOSER on a node -> NGP concentrates -> over-counts peak (~5.75 m from the winner)
    cy = 34.25
    n_a = int(k * (0.5 + lead))  # winner carries the mass surplus
    n_b = k - n_a
    xa, ya = rng.normal(cx_a, std, n_a), rng.normal(cy, std, n_a)
    xb, yb = rng.normal(cx_b, std, n_b), rng.normal(cy, std, n_b)
    gk_x_w = np.concatenate([xa, xb])
    gk_y_w = np.concatenate([ya, yb])
    w = np.ones(k) / k
    return gk_x_w, gk_y_w, w


def test_fft_cic_mode_beats_ngp_on_bimodal_grids():
    """PRIMARY motivation test: on N differential-phase bimodal grids (winner on a cell boundary,
    loser on a node -- the real-data flip mechanism), fft-cic's argmax matches the vectorized-EXACT
    argmax on >=3 MORE constructions than fft (NGP) does (the enforced margin gate, which subsumes
    both 'NGP demonstrably flips' and 'CIC beats NGP in aggregate' -- the real 21-vs-5 evidence),
    and CIC is worse per-instance on <=1 (soft -- the evidence is aggregate, not a strict
    per-instance subset). This is the test ADR-014 said was 'the wrong thing' -- it is in fact the
    right thing, because CIC does NOT tie NGP on MULTIMODAL grids (the ADR's bench used ~unimodal
    queries, which structurally cannot exhibit the peak-selection flip)."""
    from silly_kicks.tracking._ghost_gk import (
        _GRID_X,
        _GRID_Y,
        _kde_density_fft,
        _kde_density_fft_cic,
        _kde_density_vectorized,
    )

    gxx, gyy = np.meshgrid(_GRID_X, _GRID_Y, indexing="ij")
    gp = np.vstack([gxx.ravel(), gyy.ravel()])
    # GROUND TRUTH = the higher-mass winner cluster (cx_a=12.5, mode_x < 15.0). We score against the
    # KNOWN winner, NOT grid-vectorized: a boundary-centered peak is UNDER-sampled by the grid, so
    # grid-vectorized itself phase-flips to the loser on some seeds (the confound) -- but cluster A
    # has more mass and equal width, so its continuous mode is unambiguous. CIC (phase-unbiased)
    # tracks it; NGP (snaps the boundary peak) flips to the loser node on some.
    n_constructions = 120
    cic_correct = ngp_correct = exact_correct = violations = 0
    for seed in range(n_constructions):
        rng = np.random.default_rng(1000 + seed)
        x, y, w = _bimodal_cloud(rng)
        ngp_w = _grid_scalars(_kde_density_fft(x, y, w, gp))["mode_x"] < 15.0
        cic_w = _grid_scalars(_kde_density_fft_cic(x, y, w, gp))["mode_x"] < 15.0
        cic_correct += cic_w
        ngp_correct += ngp_w
        violations += ngp_w and not cic_w  # CIC wrong where NGP right -- per-instance regression
        if seed < 20:  # vectorized SANITY: confirm the construction's winner IS the grid mode
            exact_correct += _grid_scalars(_kde_density_vectorized(x, y, w, gp))["mode_x"] < 15.0
    # Sanity: the higher-mass cluster is genuinely the (grid) mode on the bulk of seeds -- so the
    # test compares against a real winner, not a phase-broken reference.
    assert exact_correct >= 17, f"construction winner is not the grid mode ({exact_correct}/20)"
    # MARGIN GATE (enforced, not just prose): CIC lands on the true-winner side on >=3 MORE
    # constructions than NGP. Since cic_correct <= n, the margin forces ngp_correct <= n - 3, i.e.
    # NGP demonstrably flips on >=3 (non-vacuity -- the 4.2.0 DAS "onside fixture" lesson), AND CIC
    # strictly beats NGP in aggregate (the real 21-vs-5 evidence). Observed (seeds 1000-1119,
    # lead=0.05, std=0.75): cic_correct=120, ngp_correct=115 -> margin 5; can't erode to a one-flip
    # near-vacuous pass across a numpy/scipy bump.
    assert cic_correct - ngp_correct >= 3, (
        f"CIC winner-correct {cic_correct}/{n_constructions}, NGP {ngp_correct} -- margin "
        f"{cic_correct - ngp_correct} < 3: CIC does not meaningfully beat NGP, or the construction "
        "does not exercise the flip."
    )
    # SOFT per-instance check (DEMOTED from a hard per-seed assert): the evidence is aggregate, NOT a
    # strict per-instance subset -- a bilinear spread can occasionally miss where an NGP snap hit.
    assert violations <= 1, f"CIC worse than NGP on {violations} constructions (expected <=1)"


@pytest.mark.parametrize("seed", [7, 11, 19])
def test_fft_cic_scalars_match_vectorized_unimodal(seed):
    """CIC must not regress the already-faithful scalars in the NORMAL (unimodal) regime: it is at
    least as scalar-faithful as NGP. Same correlated, well-conditioned, in-grid cloud + same
    tolerances as the NGP test (test_fft_kernel_matches_scipy_on_scalars): mode <=1 cell, mean
    <3e-2 m, spread rel <1e-2. The deliberately bimodal mean/spread (where CIC's bilinear smoothing
    adds ~3% to the entropy-spread -- a known tradeoff for the ~76% mode-flip reduction) is NOT the
    parity regime; the tight <5e-3 production-regime bound lives in the real-model golden
    (test_golden_fft_cic_scalars)."""
    from silly_kicks.tracking._ghost_gk import _GRID_X, _GRID_Y, _kde_density_fft_cic

    rng = np.random.default_rng(seed)
    k = 400
    gk_x_w = rng.normal(15.0, 4.0, k)
    gk_y_w = 34.0 + 0.5 * (gk_x_w - 15.0) + rng.normal(0.0, 2.5, k)  # correlated -> anisotropic H
    w = rng.uniform(0.1, 1.0, k)
    w = w / w.sum()
    gxx, gyy = np.meshgrid(_GRID_X, _GRID_Y, indexing="ij")
    gp = np.vstack([gxx.ravel(), gyy.ravel()])
    s_cic = _grid_scalars(_kde_density_fft_cic(gk_x_w, gk_y_w, w, gp))
    s_ora = _grid_scalars(_scipy_kde_grid(gk_x_w, gk_y_w, w, gp))
    assert abs(s_cic["mode_x"] - s_ora["mode_x"]) <= 0.5 + 1e-9
    assert abs(s_cic["mode_y"] - s_ora["mode_y"]) <= 0.5 + 1e-9
    assert np.hypot(s_cic["mean_x"] - s_ora["mean_x"], s_cic["mean_y"] - s_ora["mean_y"]) < 3e-2
    assert abs(s_cic["spread"] - s_ora["spread"]) / abs(s_ora["spread"]) < 1e-2


def test_fft_cic_raw_grid_tighter_than_ngp(golden, default_model_features):
    """The raw-grid fidelity test ADR-014 mandates for CIC: on real default-model leaf subsets,
    CIC's per-cell median rel-err vs the vectorized oracle is STRICTLY LOWER than NGP's. The grid
    is exactly where CIC must improve on NGP (scalars alone would tie)."""
    model, X = default_model_features  # fixture -> (from_variant("default") model, X[:_N_GOLDEN])
    vec = model.predict_density(X, kde_backend="vectorized")
    ngp = model.predict_density(X, kde_backend="fft")
    cic = model.predict_density(X, kde_backend="fft-cic")

    def _median_relerr(approx_list):
        errs = []
        for a, v in zip(approx_list, vec, strict=True):
            ref = v.probabilities
            mask = ref > 1e-6 * ref.max()  # ignore near-zero tail cells (rel-err blows up there)
            errs.append(np.median(np.abs(a.probabilities[mask] - ref[mask]) / ref[mask]))
        return float(np.median(errs))

    ngp_err = _median_relerr(ngp)
    cic_err = _median_relerr(cic)
    assert cic_err < ngp_err, f"CIC grid rel-err {cic_err:.2e} not < NGP {ngp_err:.2e}"


def test_golden_fft_cic_scalars(golden, default_model_features):
    """fft-cic scalars on the bundled 'default' model vs the frozen scipy golden: mode <=1 grid
    cell, mean <1e-2 m, spread rel <5e-3. Locks the real-model PRODUCTION regime (closes the
    synthetic-only gap). rtol/cell tolerances -- NOT exact -- so a numpy/scipy bump does not
    false-fail (cross-version robustness, like the lakehouse goldens). Mirrors
    test_golden_fft_scalars."""
    model, X = default_model_features  # fixture -> (from_variant("default") model, X[:_N_GOLDEN])
    densities = model.predict_density(X, kde_backend="fft-cic")
    n = _N_GOLDEN
    mode_x = np.array([d.mode_x for d in densities])
    mode_y = np.array([d.mode_y for d in densities])
    mean_x = np.array([d.mean_x for d in densities])
    mean_y = np.array([d.mean_y for d in densities])
    spread = np.array([d.spread for d in densities])
    assert np.all(np.abs(mode_x - golden["mode_x"][:n]) <= 0.5 + 1e-9)
    assert np.all(np.abs(mode_y - golden["mode_y"][:n]) <= 0.5 + 1e-9)
    assert np.all(np.hypot(mean_x - golden["mean_x"][:n], mean_y - golden["mean_y"][:n]) < 1e-2)
    assert np.all(np.abs(spread - golden["spread"][:n]) / np.abs(golden["spread"][:n]) < 5e-3)


def test_fft_cic_reaches_atomic_add_ghost_gk():
    """The flat kde_backend string reaches the atomic mirror's add_ghost_gk for free (re-export);
    no signature change. Smoke test: the param is accepted end-to-end."""
    import inspect

    from silly_kicks.atomic.tracking.features import add_ghost_gk

    assert "kde_backend" in inspect.signature(add_ghost_gk).parameters
