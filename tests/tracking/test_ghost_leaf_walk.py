"""Spec r3 D16: the flat active-set ghost leaf walk is bit-identical to the reference numpy body.

The HF training env has no numba, so every committed ghost artifact ran the reference numpy body
(`tests/tracking/_ghost_leaf_reference.py`, kept verbatim). On HF that body took 1,012 s of a 1,075 s
`ghost_outfield` cv fold on one core (canary job `6ae4a0dfc78863f9c62e806e`); the flat walk took 52 s of the
same predict (canary job `6ae4ab410e0338d1bb7cb791`). The flat walk advances only the rows not yet at a leaf, in
row chunks, and sums each row's leaf values tree by tree in tree order: the same float additions in the same
order, so it must be `np.array_equal` to the reference at every chunking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from silly_kicks.tracking import _ghost_gk as G
from tests.tracking._ghost_leaf_reference import reference_leaf_indices, reference_leaf_values

_CHUNKS = (777, 2_048, 32_768)  # 777 and 2,048 cut the 5,000 test rows unevenly; 32,768 is one chunk


@pytest.fixture(scope="module")
def forest():
    """Real HistGB node arrays (depth 8, trained with NaNs so both missing-value branches occur) and test rows."""
    rng = np.random.default_rng(7)
    x_train = rng.normal(size=(4000, 6))
    x_train[rng.random(x_train.shape) < 0.05] = np.nan
    y = np.nansum(x_train[:, :3], axis=1) + rng.normal(size=4000)
    # "auto" leaves early stopping OFF below sklearn's 10,000-row threshold, so all 40 trees are grown
    reg = HistGradientBoostingRegressor(max_iter=40, max_depth=8, early_stopping="auto", random_state=0)
    reg.fit(x_train, y)
    nodes = [t[0].nodes.copy() for t in reg._predictors]
    x = rng.normal(size=(5000, 6))
    x[rng.random(x.shape) < 0.05] = np.nan
    x[0, :] = np.nan  # an all-missing row walks only the missing-value branches
    return nodes, x


@pytest.fixture
def numpy_walk(monkeypatch):
    """Force the numpy walk where numba is installed too (the HF training env has no numba)."""
    monkeypatch.setenv("SILLY_KICKS_GHOST_FORCE_NUMPY", "1")


@pytest.mark.parametrize("chunk", _CHUNKS)
def test_flat_leaf_values_are_bit_identical_to_the_reference_body(forest, numpy_walk, monkeypatch, chunk):
    nodes, x = forest
    monkeypatch.setattr(G, "_LEAF_WALK_CHUNK", chunk)
    got = G._vectorized_leaf_values(nodes, x, flat=G._flatten_trees(nodes))
    assert np.array_equal(got, reference_leaf_values(nodes, x))


@pytest.mark.parametrize("chunk", _CHUNKS)
def test_flat_leaf_indices_are_bit_identical_to_the_reference_body(forest, numpy_walk, monkeypatch, chunk):
    nodes, x = forest
    monkeypatch.setattr(G, "_LEAF_WALK_CHUNK", chunk)
    got = G._vectorized_leaf_indices(nodes, x, flat=G._flatten_trees(nodes))
    ref = reference_leaf_indices(nodes, x)
    assert got.dtype == ref.dtype and np.array_equal(got, ref)


def test_the_walk_flattens_the_trees_itself_when_no_flat_is_given(forest, numpy_walk):
    nodes, x = forest
    assert np.array_equal(G._vectorized_leaf_values(nodes, x), reference_leaf_values(nodes, x))
    assert np.array_equal(G._vectorized_leaf_indices(nodes, x), reference_leaf_indices(nodes, x))


def test_the_walk_of_zero_rows_is_empty(forest, numpy_walk):
    nodes, _ = forest
    empty = np.zeros((0, 6))
    assert G._vectorized_leaf_values(nodes, empty).shape == (0,)
    assert G._vectorized_leaf_indices(nodes, empty).shape == (0, len(nodes))


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


def _chain_tree(n: int = 200) -> np.ndarray:
    """A chain 0 -> 1 -> ... -> n-1 that no walk can finish within the 100-step depth cap."""
    nodes = np.zeros(n, dtype=_NODE_DTYPE)
    for i in range(n - 1):
        nodes[i] = (i + 1, i + 1, 0, 1e18, 1, 0.0)
    nodes[n - 1] = (0, 0, 0, 0.0, 0, 1.0)
    return nodes


def test_the_flat_walk_keeps_the_asymmetric_depth_cap_contract(numpy_walk, monkeypatch):
    """VALUES raises past the depth cap (it would read an internal node's value); INDICES does not raise and
    returns the reference's non-converged LOCAL index (ADR-076), across chunk boundaries."""
    monkeypatch.setattr(G, "_LEAF_WALK_CHUNK", 2)
    nodes = [_chain_tree()]
    x = np.zeros((5, 6))
    with pytest.raises(RuntimeError, match="did not converge"):
        G._vectorized_leaf_values(nodes, x)
    assert np.array_equal(G._vectorized_leaf_indices(nodes, x), reference_leaf_indices(nodes, x))


def test_ghost_gk_predict_mean_and_training_leaves_match_the_reference_body(numpy_walk):
    """End to end through the model: the served mean and the stored training leaves equal what the reference body
    gives on the same trees (the committed bundles were fitted with the reference body)."""
    rng = np.random.default_rng(0)
    n = 600
    x = pd.DataFrame(rng.standard_normal((n, len(G.GHOST_GK_FEATURE_NAMES))), columns=G.GHOST_GK_FEATURE_NAMES)
    x["phase"] = rng.integers(0, 3, n).astype(float)
    labels = pd.DataFrame({"gk_x": rng.uniform(2, 20, n), "gk_y": rng.uniform(25, 45, n)})
    model = G.GhostGkModel(n_estimators=25).fit(x, labels)
    assert model._tree_nodes is not None and model._tree_nodes_y is not None
    xn = x[model._feature_names()].to_numpy(np.float64)
    assert np.array_equal(model._training_leaves, reference_leaf_indices(model._tree_nodes, xn))
    want = np.column_stack(
        [
            model._baseline_x + reference_leaf_values(model._tree_nodes, xn),
            model._baseline_y + reference_leaf_values(model._tree_nodes_y, xn),
        ]
    )
    assert np.array_equal(model.predict_mean(x), want)


def test_ghost_outfield_predict_mean_matches_the_reference_body(numpy_walk):
    from silly_kicks.tracking._ghost_outfield import GhostOutfieldModel

    model = GhostOutfieldModel(n_estimators=25, max_depth=6)
    rng = np.random.default_rng(1)
    names = model._feature_names()
    data = pd.DataFrame(rng.standard_normal((800, len(names))), columns=names)
    data["target_x"] = rng.uniform(0, 105, 800)
    data["target_y"] = rng.uniform(0, 68, 800)
    model._fit_extracted(data)
    assert model._tree_nodes is not None and model._tree_nodes_y is not None
    xn = data[names].to_numpy(np.float64)
    want = np.column_stack(
        [
            model._baseline_x + reference_leaf_values(model._tree_nodes, xn),
            model._baseline_y + reference_leaf_values(model._tree_nodes_y, xn),
        ]
    )
    assert np.array_equal(model.predict_mean(data), want)
