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
@pytest.mark.parametrize("kde_backend", ["vectorized"])
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


@pytest.mark.parametrize("kde_backend", ["vectorized"])
def test_golden_discrete_mode(golden, default_model_features, kde_backend):
    """mode_x/y (argmax): exact grid-cell match (NOT rtol)."""
    model, X = default_model_features
    densities = model.predict_density(X, kde_backend=kde_backend)
    mode_x = np.array([d.mode_x for d in densities])
    mode_y = np.array([d.mode_y for d in densities])
    np.testing.assert_array_equal(mode_x, golden["mode_x"][:_N_GOLDEN])
    np.testing.assert_array_equal(mode_y, golden["mode_y"][:_N_GOLDEN])


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


@pytest.mark.parametrize("backend", ["scipy", "vectorized"])
def test_predict_density_backend_parity_small_model(backend):
    """predict_density end-to-end: vectorized == scipy (NaN-mask + values) on a small model."""
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
    """Structural bound: cho_solve runs per-block, never on the full k*m at once.

    numpy buffers under-report in tracemalloc and shared-runner RSS is flaky, so guard the
    memory bound structurally: an unchunked impl calls cho_solve ONCE with kb=k; a correctly
    chunked one calls it ceil(k/block) times, each with shape[1] <= block*m.
    """
    import scipy.linalg as sla

    from silly_kicks.tracking._ghost_gk import _GRID_X, _GRID_Y, _kde_density_vectorized

    m = 60 * 64
    block = 1024
    k = 5000  # -> ceil(5000/1024) = 5 blocks
    calls = []
    real = sla.cho_solve

    def _spy(c_and_lower, b, **kw):
        calls.append(b.shape)
        return real(c_and_lower, b, **kw)

    monkeypatch.setattr(sla, "cho_solve", _spy)

    rng = np.random.default_rng(5)
    gk_x_w = rng.uniform(0, 30, k)
    gk_y_w = rng.uniform(18, 50, k)
    w = np.full(k, 1.0 / k)
    gxx, gyy = np.meshgrid(_GRID_X, _GRID_Y, indexing="ij")
    grid_points = np.vstack([gxx.ravel(), gyy.ravel()])

    out = _kde_density_vectorized(gk_x_w, gk_y_w, w, grid_points, train_block=block)
    assert out.shape == (60, 64)
    assert len(calls) == int(np.ceil(k / block)), f"expected per-block cho_solve, got {len(calls)}"
    assert all(shp[1] <= block * m for shp in calls), "a cho_solve saw more than one block of work"


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
    for backend in ("scipy", "vectorized"):
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
