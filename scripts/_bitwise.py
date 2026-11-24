"""Bit-identity comparison, the single helper for every bit-identity claim (spec 2026-11-08 tracking-batch §5.1).

``np.array_equal`` and ``pd.testing.assert_frame_equal`` are too weak for a bit-identity claim: both treat ``-0.0``
as ``0.0``, and ``assert_frame_equal`` compares floats with a tolerance by default. This helper compares the bytes.

Rules:

- Two floats are equal when their bits are equal, or when both are NaN (any payload). So ``-0.0 != 0.0``.
- The dtype, the shape, the index, and the column labels must be equal.
- A pandas extension array compares its NA mask, then its non-NA values. A categorical compares its categories and
  its codes.
- An object value compares with ``==`` after an NA check, and its type must be equal (``7`` differs from ``"7"``).
- A dataclass compares field by field. A list or tuple compares element by element. A ``frozenset`` compares with
  ``==``.

``assert_bits_equal`` raises ``AssertionError`` that names the first differing element.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["assert_bits_equal", "bits_equal"]


class _MismatchError(Exception):
    """Internal: carries the path of the first differing element."""


def _fail(path: str, msg: str) -> None:
    raise _MismatchError(f"{path or '<root>'}: {msg}")


def _scalar_na(v: Any) -> bool:
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _cmp_ndarray(a: np.ndarray, b: np.ndarray, path: str) -> None:
    if a.dtype != b.dtype:
        _fail(path, f"dtype {a.dtype} != {b.dtype}")
    if a.shape != b.shape:
        _fail(path, f"shape {a.shape} != {b.shape}")
    kind = a.dtype.kind
    if kind in "fc":
        width = a.dtype.itemsize if kind == "f" else a.dtype.itemsize // 2
        view = {2: np.uint16, 4: np.uint32, 8: np.uint64}[width]
        av = np.ascontiguousarray(a).view(view)
        bv = np.ascontiguousarray(b).view(view)
        if kind == "c":
            av, bv = av.reshape(*a.shape, 2), bv.reshape(*b.shape, 2)
            same = (av == bv).all(axis=-1) | (np.isnan(a) & np.isnan(b))
        else:
            same = (av == bv) | (np.isnan(a) & np.isnan(b))
    elif kind == "O":
        flat_a, flat_b = a.ravel(), b.ravel()
        same_flat = np.empty(flat_a.shape, dtype=bool)
        for i, (x, y) in enumerate(zip(flat_a, flat_b, strict=True)):
            same_flat[i] = _object_equal(x, y)
        same = same_flat.reshape(a.shape)
    elif kind in "mM":
        same = a.view(np.int64) == b.view(np.int64)
    else:
        same = a == b
    if not np.all(same):
        idx = np.unravel_index(int(np.flatnonzero(~np.asarray(same).ravel())[0]), a.shape)
        pos = idx[0] if len(idx) == 1 else tuple(int(i) for i in idx)
        _fail(f"{path}[{pos}]", f"{a[idx]!r} != {b[idx]!r}")


def _object_equal(x: Any, y: Any) -> bool:
    if type(x) is not type(y):
        return False
    if isinstance(x, float) and isinstance(y, float):
        return bits_equal(x, y)
    if _scalar_na(x) or _scalar_na(y):
        return _scalar_na(x) and _scalar_na(y)
    return bool(x == y)


def _cmp_array_like(a: pd.Series | pd.Index, b: pd.Series | pd.Index, path: str) -> None:
    if a.dtype != b.dtype:
        _fail(path, f"dtype {a.dtype} != {b.dtype}")
    if len(a) != len(b):
        _fail(path, f"length {len(a)} != {len(b)}")
    if isinstance(a.dtype, pd.CategoricalDtype):
        ca, cb = pd.Categorical(a), pd.Categorical(b)
        _cmp_array_like(pd.Index(ca.categories), pd.Index(cb.categories), f"{path}.categories")
        _cmp_ndarray(np.asarray(ca.codes), np.asarray(cb.codes), f"{path}.codes")
        return
    if isinstance(a.dtype, pd.api.extensions.ExtensionDtype):
        ma, mb = np.asarray(pd.isna(a.array)), np.asarray(pd.isna(b.array))
        _cmp_ndarray(ma, mb, f"{path}.na_mask")
        va = np.asarray(a.array[~ma].to_numpy(), dtype=object)
        vb = np.asarray(b.array[~mb].to_numpy(), dtype=object)
        _cmp_ndarray(va, vb, f"{path}.values")
        return
    _cmp_ndarray(a.to_numpy(), b.to_numpy(), path)


def _cmp(a: Any, b: Any, path: str) -> None:
    if type(a) is not type(b) and not (_is_float(a) and _is_float(b)):
        _fail(path, f"type {type(a).__name__} != {type(b).__name__}")
    if isinstance(a, pd.DataFrame):
        _cmp_array_like(a.columns, b.columns, f"{path}.columns")
        _cmp_array_like(a.index, b.index, f"{path}.index")
        for j, col in enumerate(a.columns):
            _cmp_array_like(a.iloc[:, j], b.iloc[:, j], f"{path}[{col!r}]")
    elif isinstance(a, pd.Series):
        _cmp_array_like(a.index, b.index, f"{path}.index")
        if a.name != b.name:
            _fail(path, f"name {a.name!r} != {b.name!r}")
        _cmp_array_like(a, b, path)
    elif isinstance(a, pd.Index):
        _cmp_array_like(a, b, path)
    elif isinstance(a, np.ndarray):
        _cmp_ndarray(a, b, path)
    elif _is_float(a):
        if not bits_equal(a, b):
            _fail(path, f"{a!r} != {b!r}")
    elif dataclasses.is_dataclass(a) and not isinstance(a, type):
        for f in dataclasses.fields(a):
            _cmp(getattr(a, f.name), getattr(b, f.name), f"{path}.{f.name}")
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            _fail(path, f"length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _cmp(x, y, f"{path}[{i}]")
    elif isinstance(a, dict):
        if list(a) != list(b):
            _fail(path, f"keys {list(a)!r} != {list(b)!r}")
        for k in a:
            _cmp(a[k], b[k], f"{path}[{k!r}]")
    elif not _object_equal(a, b):
        _fail(path, f"{a!r} != {b!r}")


def _is_float(v: Any) -> bool:
    return isinstance(v, (float, np.floating))


def assert_bits_equal(a: Any, b: Any) -> None:
    """Raise ``AssertionError`` naming the first differing element unless ``a`` and ``b`` are bit-identical."""
    try:
        _cmp(a, b, "")
    except _MismatchError as exc:
        raise AssertionError(f"not bit-identical at {exc}") from None


def bits_equal(a: Any, b: Any) -> bool:
    """True when ``a`` and ``b`` are bit-identical by the module rules."""
    if _is_float(a) and _is_float(b):
        # A Python float and an np.float64 are the same 64 bits; any other width difference is a dtype difference.
        if np.asarray(a).dtype != np.asarray(b).dtype:
            return False
        x, y = np.asarray(a), np.asarray(b)
        view = {2: np.uint16, 4: np.uint32, 8: np.uint64}[x.dtype.itemsize]
        return bool((np.isnan(x) and np.isnan(y)) or x.view(view) == y.view(view))
    try:
        assert_bits_equal(a, b)
    except AssertionError:
        return False
    return True
