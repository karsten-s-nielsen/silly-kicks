"""Bit-identity + dispatch gates for the numba-accelerated ghost-GK leaf traversal.

The two numpy leaf-traversal ports (:func:`_vectorized_leaf_values`,
:func:`_vectorized_leaf_indices`) dispatch to a serial ``@njit`` adapter pair
(:func:`_leaf_values_numba`, :func:`_leaf_indices_numba`) when the ``[numba]`` extra is
installed, with an explicit ``_FlatTrees`` value object. The contract is BIT-IDENTICAL output
(``np.array_equal``), incl. the ASYMMETRIC failure mode: the VALUES path RAISES on a
>depth-cap tree (it reads ``value`` = garbage), the INDICES path does NOT (it never reads
``value``; it returns the non-converged LOCAL index). ``SILLY_KICKS_GHOST_FORCE_NUMPY=1``
forces the numpy fallback so both adapters run on every CI leg.

Discrimination proofs (each gate can fail; verified locally, perturbations NOT committed):
- ``test_leaf_kernels_are_bit_identical_to_numpy_incl_nan``: flip the ``go_left`` inequality
  in ``_leaf_values_numba`` -> the ``np.array_equal`` assertion fails.
- ``test_over_depth_cap_failure_mode_is_ASYMMETRIC``: remove the values-kernel convergence
  guard -> the "values raises" leg fails.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._ghost_gk import GHOST_GK_FEATURE_NAMES, GhostGkModel


def _fit(n_estimators=30, n=400):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((n, 26)), columns=GHOST_GK_FEATURE_NAMES)
    X["phase"] = rng.integers(0, 3, n).astype(float)
    labels = pd.DataFrame({"gk_x": rng.uniform(2, 20, n), "gk_y": rng.uniform(25, 45, n)})
    m = GhostGkModel(n_estimators=n_estimators)
    m.fit(X, labels)
    return m, X


def test_flatten_trees_preserves_local_indices_and_dtypes():
    from silly_kicks.tracking._ghost_gk import _flatten_trees

    m, _ = _fit()
    assert m._tree_nodes is not None
    flat = _flatten_trees(m._tree_nodes)
    # offsets: one per tree + 1, monotonic, last == total nodes
    assert flat.offsets[0] == 0
    assert flat.offsets[-1] == sum(len(t) for t in m._tree_nodes)
    assert np.all(np.diff(flat.offsets) >= 0)
    # dtypes pinned
    assert flat.left.dtype == np.int64 and flat.val.dtype == np.float64
    # LOCAL left/right: a flattened tree's slice equals the source tree's RAW left/right
    for i, t in enumerate(m._tree_nodes):
        lo, hi = flat.offsets[i], flat.offsets[i + 1]
        assert np.array_equal(flat.left[lo:hi], t["left"].astype(np.int64)), (
            "left must stay per-tree LOCAL (not offset-shifted)"
        )
        assert np.array_equal(flat.right[lo:hi], t["right"].astype(np.int64))


# The @njit kernels below cannot even be imported without the [numba] extra
# (_ghost_gk_numba raises ImportError at module load); guard the rest of the module.
pytest.importorskip("numba")


def test_leaf_kernels_are_bit_identical_to_numpy_incl_nan():
    from silly_kicks.tracking._ghost_gk import _flatten_trees, _vectorized_leaf_indices, _vectorized_leaf_values
    from silly_kicks.tracking._ghost_gk_numba import _leaf_indices_numba, _leaf_values_numba

    m, X = _fit()
    feature_names = m._feature_names()  # capture before the assert so narrowing survives
    assert m._tree_nodes is not None
    Xa = X.copy()
    Xa.iloc[0, :3] = np.nan  # exercise BOTH missing_go_to_left branches
    Xn = Xa[feature_names].to_numpy(np.float64)
    flat = _flatten_trees(m._tree_nodes)

    ref_v = _vectorized_leaf_values(m._tree_nodes, Xn)
    got_v = _leaf_values_numba(flat.left, flat.right, flat.feat, flat.thr, flat.miss, flat.val, flat.offsets, Xn)
    assert np.array_equal(ref_v, got_v), "values kernel must be BIT-identical (incl NaN rows)"

    ref_i = _vectorized_leaf_indices(m._tree_nodes, Xn)
    got_i = _leaf_indices_numba(flat.left, flat.right, flat.feat, flat.thr, flat.miss, flat.val, flat.offsets, Xn)
    assert np.array_equal(ref_i, got_i), "indices kernel must return the LOCAL current index, bit-identical"


def _one_tree(nodes):  # helper: build _FlatTrees for a single hand-built node array
    from silly_kicks.tracking._ghost_gk import _flatten_trees

    return _flatten_trees([nodes])


_NODE_DTYPE = np.dtype(
    [
        ("left", "i8"),
        ("right", "i8"),
        ("feature_idx", "i8"),
        ("num_threshold", "f8"),
        ("missing_go_to_left", "i8"),
        ("value", "f8"),
    ]
)


def test_single_node_leaf_root_tree():
    from silly_kicks.tracking._ghost_gk import _vectorized_leaf_values
    from silly_kicks.tracking._ghost_gk_numba import _leaf_values_numba

    nodes = np.array([(0, 0, 0, 0.0, 0, 3.5)], dtype=_NODE_DTYPE)  # root IS a leaf (left==0)
    f = _one_tree(nodes)
    X = np.zeros((4, 26))
    assert np.array_equal(
        _vectorized_leaf_values([nodes], X),
        _leaf_values_numba(f.left, f.right, f.feat, f.thr, f.miss, f.val, f.offsets, X),
    )


def test_over_depth_cap_failure_mode_is_ASYMMETRIC():
    """values: BOTH raise; indices: NEITHER raises and both return the same non-converged index."""
    from silly_kicks.tracking._ghost_gk import _vectorized_leaf_indices, _vectorized_leaf_values
    from silly_kicks.tracking._ghost_gk_numba import _leaf_indices_numba, _leaf_values_numba

    # A degenerate chain 0->1->...->199 (never a leaf within the 100-step cap; terminal leaf at 199):
    # each node i<199 has left=i+1, so the walk reaches local node 100 and stops non-converged.
    n = 200
    nodes = np.zeros(n, dtype=_NODE_DTYPE)
    for i in range(n - 1):
        nodes[i] = (i + 1, i + 1, 0, 1e18, 1, 0.0)  # always go_left to i+1 (feat 0 <= 1e18)
    nodes[n - 1] = (0, 0, 0, 0.0, 0, 1.0)  # terminal leaf, but unreachable within 100 steps
    f = _one_tree(nodes)
    X = np.zeros((2, 26))

    with pytest.raises(RuntimeError, match="did not converge"):
        _vectorized_leaf_values([nodes], X)
    with pytest.raises(RuntimeError, match="did not converge"):
        _leaf_values_numba(f.left, f.right, f.feat, f.thr, f.miss, f.val, f.offsets, X)

    # indices: NEITHER raises; both return the SAME non-converged LOCAL index.
    ri = _vectorized_leaf_indices([nodes], X)
    gi = _leaf_indices_numba(f.left, f.right, f.feat, f.thr, f.miss, f.val, f.offsets, X)
    assert np.array_equal(ri, gi), "indices path must not raise and must return the identical non-converged index"


def test_dispatch_both_paths_identical(monkeypatch):
    from silly_kicks.tracking import _ghost_gk as G

    m, X = _fit()
    assert m._tree_nodes is not None
    Xn = X.to_numpy(np.float64)
    flat = G._flatten_trees(m._tree_nodes)

    monkeypatch.setenv("SILLY_KICKS_GHOST_FORCE_NUMPY", "1")
    numpy_path = G._vectorized_leaf_values(m._tree_nodes, Xn, flat=flat)
    monkeypatch.delenv("SILLY_KICKS_GHOST_FORCE_NUMPY", raising=False)
    numba_path = G._vectorized_leaf_values(m._tree_nodes, Xn, flat=flat)
    assert np.array_equal(numpy_path, numba_path)


def test_dispatch_both_paths_identical_indices(monkeypatch):
    from silly_kicks.tracking import _ghost_gk as G

    m, X = _fit()
    assert m._tree_nodes is not None
    Xn = X.to_numpy(np.float64)
    flat = G._flatten_trees(m._tree_nodes)

    monkeypatch.setenv("SILLY_KICKS_GHOST_FORCE_NUMPY", "1")
    numpy_path = G._vectorized_leaf_indices(m._tree_nodes, Xn, flat=flat)
    monkeypatch.delenv("SILLY_KICKS_GHOST_FORCE_NUMPY", raising=False)
    numba_path = G._vectorized_leaf_indices(m._tree_nodes, Xn, flat=flat)
    assert np.array_equal(numpy_path, numba_path)


def test_model_caches_flat_trees_for_both_x_and_y():
    m, _ = _fit()
    assert m._flat_trees is not None and m._flat_trees_y is not None
    assert m._tree_nodes is not None and m._tree_nodes_y is not None
    assert m._flat_trees.offsets[-1] == sum(len(t) for t in m._tree_nodes)
    assert m._flat_trees_y.offsets[-1] == sum(len(t) for t in m._tree_nodes_y)


def test_predict_mean_numba_equals_numpy_bit_identical(monkeypatch):
    m, X = _fit()
    monkeypatch.setenv("SILLY_KICKS_GHOST_FORCE_NUMPY", "1")
    ref = m.predict_mean(X)
    monkeypatch.delenv("SILLY_KICKS_GHOST_FORCE_NUMPY", raising=False)
    got = m.predict_mean(X)
    assert np.array_equal(ref, got), "predict_mean must be bit-identical numba-vs-numpy"


def test_predict_mean_uses_the_numba_kernel_on_the_default_path(monkeypatch):
    """Regression guard: a silent revert to the numpy path (e.g. dropping flat=) is caught."""
    from silly_kicks.tracking import _ghost_gk as G

    m, X = _fit()  # fit() triggers the lazy numba bind, so G._leaf_values_numba is the real kernel
    calls = {"n": 0}
    real = G._leaf_values_numba

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(G, "_leaf_values_numba", counting)
    m.predict_mean(X.iloc[:4])
    assert calls["n"] == 2, "predict_mean must dispatch to the numba kernel (x-tree + y-tree) when numba is available"


def test_flat_trees_built_once_not_reflattened_per_prediction(monkeypatch):
    """The cache is built at fit(), NOT re-flattened per predict_mean call (spec §5.6)."""
    from silly_kicks.tracking import _ghost_gk as G

    m, X = _fit()  # the two _flatten_trees calls happen HERE (fit)
    calls = {"n": 0}
    real = G._flatten_trees

    def spy(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(G, "_flatten_trees", spy)
    m.predict_mean(X.iloc[:4])
    m.predict_mean(X.iloc[:4])
    assert calls["n"] == 0, "predict_mean must reuse the cached _FlatTrees, never re-flatten per call"


def test_flat_trees_rebuilt_after_load():
    """load() rebuilds the derived cache (never serialized) so the numba path serves post-load (spec §5.6)."""
    m = GhostGkModel.from_variant("default")  # goes through load()
    assert m._flat_trees is not None and m._flat_trees_y is not None
    assert m._tree_nodes is not None and m._tree_nodes_y is not None
    assert m._flat_trees.offsets[-1] == sum(len(t) for t in m._tree_nodes)
    assert m._flat_trees_y.offsets[-1] == sum(len(t) for t in m._tree_nodes_y)


def test_flat_trees_are_a_fresh_object_per_model():
    """No cross-model leakage: each fit builds its own _FlatTrees (spec §5.6)."""
    m1, _ = _fit()
    m2, _ = _fit()
    assert m1._flat_trees is not m2._flat_trees
    assert m1._flat_trees_y is not m2._flat_trees_y


def test_kernels_do_not_mutate_inputs():
    """Purity: the kernels only read the node arrays + X and write a fresh out (spec §5.6)."""
    from silly_kicks.tracking._ghost_gk import _flatten_trees
    from silly_kicks.tracking._ghost_gk_numba import _leaf_indices_numba, _leaf_values_numba

    m, X = _fit()
    assert m._tree_nodes is not None
    Xn = X.to_numpy(np.float64)
    flat = _flatten_trees(m._tree_nodes)
    xn_before = Xn.copy()
    # Snapshot ALL SIX flat arrays + offsets, not a subset -- a kernel that mutated any of the
    # inputs it only reads (right/feat/thr/miss/offsets, not just left/val) must be caught.
    before = {name: getattr(flat, name).copy() for name in flat._fields}

    _leaf_values_numba(flat.left, flat.right, flat.feat, flat.thr, flat.miss, flat.val, flat.offsets, Xn)
    _leaf_indices_numba(flat.left, flat.right, flat.feat, flat.thr, flat.miss, flat.val, flat.offsets, Xn)

    assert np.array_equal(Xn, xn_before), "kernels must not mutate X"
    for name in flat._fields:
        assert np.array_equal(getattr(flat, name), before[name]), f"kernels must not mutate the node array ({name})"


def test_boundary_tie_pins_the_le_semantics_numba_equals_numpy():
    """A feature value EXACTLY equal to a split threshold: <= sends LEFT; a <-flip sends RIGHT.

    Random continuous data never hits fv == thr, so it cannot separate <= from < (measured: the
    <=->< mutation passes the bit-identity gate). This hand-built tie row makes the boundary the
    ONLY thing under test: it pins numba == numpy AND the <= semantics (the LEFT leaf value),
    so a <=-> kernel flip fails here even though it would slip past the random-data gate.
    """
    from silly_kicks.tracking._ghost_gk import _vectorized_leaf_indices, _vectorized_leaf_values
    from silly_kicks.tracking._ghost_gk_numba import _leaf_indices_numba, _leaf_values_numba

    # root splits feature 0 at threshold 5.0; node1 = LEFT leaf (10.0), node2 = RIGHT leaf (20.0).
    nodes = np.array(
        [(1, 2, 0, 5.0, 0, 0.0), (0, 0, 0, 0.0, 0, 10.0), (0, 0, 0, 0.0, 0, 20.0)],
        dtype=_NODE_DTYPE,
    )
    f = _one_tree(nodes)
    X = np.zeros((1, 26))
    X[0, 0] = 5.0  # EXACT tie with the threshold -> <= goes LEFT (value 10.0), < would go RIGHT (20.0)

    ref_v = _vectorized_leaf_values([nodes], X)
    got_v = _leaf_values_numba(f.left, f.right, f.feat, f.thr, f.miss, f.val, f.offsets, X)
    assert np.array_equal(got_v, ref_v), "values kernel must match numpy on an exact fv==thr tie"
    assert got_v[0] == 10.0, "fv == thr must route LEFT (<=), not RIGHT (a <-flip would give 20.0)"

    ref_i = _vectorized_leaf_indices([nodes], X)
    got_i = _leaf_indices_numba(f.left, f.right, f.feat, f.thr, f.miss, f.val, f.offsets, X)
    assert np.array_equal(got_i, ref_i)
    assert got_i[0, 0] == 1, "fv == thr must reach the LEFT child leaf (local index 1)"


def test_predict_density_uses_the_indices_numba_kernel_on_the_default_path(monkeypatch):
    """Indices-side companion to the values dispatch guard: a silent drop of flat= in the query
    leaf-match (a perf regression bit-identity output cannot detect) is caught here (spec §5.5)."""
    from silly_kicks.tracking import _ghost_gk as G

    m, X = _fit()  # fit() triggers the lazy numba bind AND builds _flat_trees
    calls = {"n": 0}
    real = G._leaf_indices_numba

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(G, "_leaf_indices_numba", counting)
    m.predict_density(X.iloc[:2])
    assert calls["n"] == 1, "predict_density must dispatch the query leaf-match to the numba indices kernel"
