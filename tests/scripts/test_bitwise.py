"""The shared bit-identity helper (spec 2026-11-08 tracking-batch §5.1).

Any two NaNs are equal; ``-0.0`` and ``0.0`` differ; dtype and index must be equal; a failure names the first
differing element.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from scripts._bitwise import assert_bits_equal, bits_equal


def test_bits_equal_rejects_signed_zero():
    assert not bits_equal(np.array([0.0]), np.array([-0.0]))
    with pytest.raises(AssertionError):
        assert_bits_equal(np.array([0.0]), np.array([-0.0]))


def test_bits_equal_any_nan_payload_equal():
    a = np.array([np.nan])
    b = np.frombuffer(np.uint64(0x7FF8000000000001).tobytes(), dtype=np.float64)
    assert_bits_equal(a, b)


def test_bits_equal_checks_dtype_and_index():
    with pytest.raises(AssertionError):
        assert_bits_equal(pd.Series([1.0], index=[0]), pd.Series([1.0], index=[1]))
    with pytest.raises(AssertionError):
        assert_bits_equal(np.array([1.0], np.float32), np.array([1.0], np.float64))


def test_bits_equal_rejects_one_ulp():
    a = np.array([1.0, 2.0])
    b = a.copy()
    b[1] = np.nextafter(2.0, 3.0)
    with pytest.raises(AssertionError, match=r"\[1\]"):
        assert_bits_equal(a, b)


def test_scalar_float():
    assert bits_equal(np.nan, float("nan"))
    assert not bits_equal(0.0, -0.0)
    assert bits_equal(np.float64(1.5), 1.5)


def test_frame_names_first_differing_column_and_row():
    a = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    b = a.copy()
    b.loc[1, "y"] = 4.5
    with pytest.raises(AssertionError, match=r"'y'.*\[1\]"):
        assert_bits_equal(a, b)


def test_frame_column_order_and_dtype():
    a = pd.DataFrame({"x": [1.0], "y": [2.0]})
    with pytest.raises(AssertionError):
        assert_bits_equal(a, a[["y", "x"]])
    with pytest.raises(AssertionError):
        assert_bits_equal(a, a.astype({"x": np.float32}))


def test_extension_dtypes_with_na():
    a = pd.Series(pd.array([1, pd.NA, 3], dtype="Int64"))
    assert_bits_equal(a, a.copy())
    with pytest.raises(AssertionError):
        assert_bits_equal(a, pd.Series(pd.array([1, 2, 3], dtype="Int64")))
    c = pd.Series(["a", None, "b"], dtype="category")
    assert_bits_equal(c, c.copy())
    with pytest.raises(AssertionError):
        assert_bits_equal(c, pd.Series(["a", None, "a"], dtype=c.dtype))


def test_object_values_with_none_and_mixed_types():
    a = pd.Series([7, "7", None], dtype=object)
    assert_bits_equal(a, a.copy())
    with pytest.raises(AssertionError):
        assert_bits_equal(a, pd.Series([7, 7, None], dtype=object))


def test_dataclass_and_sequences():
    @dataclasses.dataclass(frozen=True)
    class G:
        edges: np.ndarray
        faces: list

    g = G(np.array([[0, 1]]), [frozenset({0, 1, 2})])
    assert_bits_equal([g, (1.0, 2.0)], [G(np.array([[0, 1]]), [frozenset({0, 1, 2})]), (1.0, 2.0)])
    with pytest.raises(AssertionError, match="faces"):
        assert_bits_equal(g, G(np.array([[0, 1]]), [frozenset({0, 1, 3})]))
    with pytest.raises(AssertionError):
        assert_bits_equal([1.0], [1.0, 2.0])
