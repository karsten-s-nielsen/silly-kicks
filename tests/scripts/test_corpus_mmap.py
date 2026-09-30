"""Task 5 guard: the shared design-matrix mmap preserves columns/dtypes/bytes and is zero-copy."""

import gc

import numpy as np
import pandas as pd

from scripts._corpus_mmap import load_design_matrix, persist_design_matrix


def _memmap_backed(df: pd.DataFrame) -> bool:
    """True if the frame's values chain back to a numpy memmap (shared across processes)."""
    b = df.to_numpy()
    while b is not None:
        if isinstance(b, np.memmap):
            return True
        b = getattr(b, "base", None)
    return False


def test_roundtrip_preserves_columns_dtypes_and_bytes(tmp_path):
    X = pd.DataFrame(
        np.random.default_rng(0).standard_normal((500, 3)).astype(np.float32),
        columns=["a_speed", "b_angle", "c_dist"],
    )
    y = (X["a_speed"] > 0).to_numpy().astype(np.int8)
    groups = np.arange(500) % 5
    p = tmp_path / "dm"
    persist_design_matrix(X, y, groups, p)
    Xr, yr, gr = load_design_matrix(p)
    try:
        pd.testing.assert_frame_equal(Xr, X, check_exact=True)  # names + dtypes + bytes
        assert np.array_equal(yr, y) and np.array_equal(gr, groups)
        assert _memmap_backed(Xr)  # uniform dtype -> zero-copy shared mmap
    finally:
        del Xr  # release the memmap so Windows can delete the temp file
        gc.collect()


def test_string_groups_roundtrip_pickle_free(tmp_path):
    X = pd.DataFrame(np.zeros((6, 2), dtype=np.float64), columns=["r", "theta"])
    y = np.array([0, 1, 0, 1, 0, 1], dtype=int)
    groups = np.array(["m1", "m1", "m2", "m2", "m3", "m3"], dtype=object)  # str game ids
    p = tmp_path / "dm"
    persist_design_matrix(X, y, groups, p)
    _, _, gr = load_design_matrix(p)
    assert np.array_equal(gr.astype(str), groups.astype(str))


def test_mixed_dtype_falls_back_to_exact_copy(tmp_path):
    X = pd.DataFrame(
        {
            "r": np.array([1.5, 2.5, 3.5], dtype=np.float64),
            "count": np.array([1, 2, 3], dtype=np.int32),  # mixed -> copy-fallback path
        }
    )
    y = np.array([0, 1, 0], dtype=int)
    groups = np.array([0, 0, 1])
    p = tmp_path / "dm"
    persist_design_matrix(X, y, groups, p)
    Xr, _, _ = load_design_matrix(p)
    pd.testing.assert_frame_equal(Xr, X, check_exact=True)  # dtypes preserved exactly
    assert not _memmap_backed(Xr)  # copy-fallback, not shared
