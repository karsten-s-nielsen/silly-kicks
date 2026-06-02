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
    for backend in ("scipy", "vectorized", "cpu-numba"):
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
