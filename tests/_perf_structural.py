"""Shared helper for STRUCTURAL performance guards.

Replaces flaky wall-clock ``assert mean_ms < budget`` perf tests with deterministic
call-count spies on each function's dominant expensive primitive. A real performance
regression (a broken pitch-control cache, a per-row/per-player recompute, a de-vectorised
hot path, per-action re-linking) changes *how many times* that primitive runs — which we
assert exactly. Wall-clock ceilings flaked on shared CI runners (e.g. ``compute_team_shape``
6.2ms > 5ms, ``compute_gk_influence`` 10.4ms > 10ms); a call-count invariant never does.

This is the same pattern the repo already uses in
``test_ghost_gk_kde_vectorized.py::test_vectorized_is_chunked_structural`` /
``::test_fft_is_k_independent_one_convolution``.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd


def call_counter(monkeypatch: Any, module: Any, name: str) -> dict[str, int]:
    """Patch ``module.name`` with a pass-through spy; return ``{"n": <call count>}``.

    The spy forwards args/kwargs and the return value unchanged, so behaviour is identical
    — only the invocation count is observed. ``monkeypatch`` restores the original on teardown.

    Patch the symbol *as the function under test resolves it*: a name imported via
    ``from .x import f`` is rebound into the importing module's namespace, so patch *that*
    module's attribute (not the definition site).
    """
    calls = {"n": 0}
    real = getattr(module, name)

    def _spy(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(module, name, _spy)
    return calls


def row_iteration_counter(monkeypatch: Any) -> dict[str, int]:
    """Count pandas row-wise iteration calls — the de-vectorisation signature.

    ``DataFrame.apply(axis=1)`` / ``.iterrows()`` / ``.itertuples()`` are how a vectorised
    converter regresses into an O(n) Python row loop (the blow-up the throughput wall-clock
    budgets in ``test_benchmark.py`` proxied). A vectorised ``np.select`` converter calls none of
    them. Column-wise ``apply(axis=0)`` is allowed (it is not row iteration). Returns ``{"n": ...}``.
    """
    import pandas as pd

    calls = {"n": 0}
    real_apply = pd.DataFrame.apply
    real_iterrows = pd.DataFrame.iterrows
    real_itertuples = pd.DataFrame.itertuples

    def _apply(self: Any, func: Any, *args: Any, **kwargs: Any) -> Any:
        axis = kwargs.get("axis", args[0] if args else 0)
        if axis in (1, "columns"):
            calls["n"] += 1
        return real_apply(self, func, *args, **kwargs)

    def _iterrows(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        return real_iterrows(self, *args, **kwargs)

    def _itertuples(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        return real_itertuples(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "apply", _apply)
    monkeypatch.setattr(pd.DataFrame, "iterrows", _iterrows)
    monkeypatch.setattr(pd.DataFrame, "itertuples", _itertuples)
    return calls


def assert_subquadratic_growth(
    measure_work: Any,
    *,
    sizes: tuple[int, ...] = (256, 1024, 4096),
    max_exponent: float = 1.5,
    work_floor: int = 1,
    degenerate_ok: bool = False,
    label: str = "",
) -> float | None:
    """Assert a primitive's deterministic work-count grows sub-quadratically (scale-guard harness).

    ``measure_work(n) -> int`` builds a size-n input, runs the primitive with a work counter
    installed, and returns the observed integer op-count. Asserts the extreme-pair growth exponent
    ``log(work_hi/work_lo)/log(size_hi/size_lo) <= max_exponent``. Requires ``work[max] >=
    work_floor`` unless ``degenerate_ok``. Returns the exponent on pass (or None on the degenerate
    path). Reference exponents at (256,1024,4096): linear 1.0, n*log n 1.16, n^1.5 1.50, quadratic
    2.0 -- a quadratic-ish detector by design. Integer counts => exact => never flakes.
    """
    if len(sizes) < 2:
        raise ValueError("assert_subquadratic_growth needs >= 2 sizes")
    counts = [int(measure_work(n)) for n in sizes]
    lo, hi = counts[0], counts[-1]
    n_lo, n_hi = sizes[0], sizes[-1]
    if hi < work_floor or hi == 0:  # a 0-count is ALWAYS degenerate (avoids math.log(0))
        if degenerate_ok:
            return None
        raise AssertionError(
            f"{label or 'assert_subquadratic_growth'}: work_floor not met "
            f"(work[{n_hi}]={hi} < {work_floor}) -- the counter never fired, so this is a "
            f"mis-wired guard, not a passing one. Counts {dict(zip(sizes, counts, strict=True))}. Pass "
            f"degenerate_ok=True with a reason for a genuinely zero-work primitive."
        )
    exponent = math.log(hi / max(lo, 1)) / math.log(n_hi / n_lo)
    assert exponent <= max_exponent, (
        f"{label or 'assert_subquadratic_growth'}: growth exponent {exponent:.3f} > "
        f"{max_exponent} -- super-linear scaling. Counts {dict(zip(sizes, counts, strict=True))}."
    )
    return exponent


def _is_boolean_key(key: Any) -> bool:
    """True only for a boolean mask/array row filter -- NOT a label/column or int-array select."""
    if isinstance(key, pd.Series):
        return key.dtype == bool
    if isinstance(key, np.ndarray):
        return key.dtype == bool
    if isinstance(key, list) and key:
        return all(isinstance(x, (bool, np.bool_)) for x in key)
    return False


@contextmanager
def rows_scanned_counter() -> Any:
    """Count rows touched by boolean-mask ``__getitem__`` / ``.loc[mask]``, ``.groupby`` construction,
    and axis-0 ``.take`` -- the rescan proxy. Installed only for the with-block (restored in finally).
    """
    from pandas.core.indexing import _LocIndexer

    counts = {"n": 0}
    depth = {"mask": 0}  # re-entrancy guard: pandas routes df[mask] through an internal .take
    real_getitem, real_take, real_groupby = (
        pd.DataFrame.__getitem__,
        pd.DataFrame.take,
        pd.DataFrame.groupby,
    )
    real_loc = _LocIndexer.__getitem__  # type: ignore[attr-defined]

    def _getitem(self: Any, key: Any) -> Any:
        if _is_boolean_key(key):
            counts["n"] += len(self)
            depth["mask"] += 1
            try:
                return real_getitem(self, key)
            finally:
                depth["mask"] -= 1
        return real_getitem(self, key)

    def _take(self: Any, indices: Any, *a: Any, **k: Any) -> Any:
        axis = k.get("axis", a[0] if a else 0)
        # Count ONLY a direct ROW take (axis 0, NOT inside a mask-getitem): skips the re-entrant
        # mask-take double-count and every axis=1 column-selection take. group_rows .get()->take is
        # a direct axis-0 take -> still counts.
        if depth["mask"] == 0 and axis in (0, "index"):
            counts["n"] += len(indices)
        return real_take(self, indices, *a, **k)

    def _groupby(self: Any, *a: Any, **k: Any) -> Any:
        counts["n"] += len(self)
        return real_groupby(self, *a, **k)

    def _loc(self: Any, key: Any) -> Any:
        k0 = key[0] if isinstance(key, tuple) else key
        if _is_boolean_key(k0):
            counts["n"] += len(self.obj)
            depth["mask"] += 1
            try:
                return real_loc(self, key)
            finally:
                depth["mask"] -= 1
        return real_loc(self, key)

    pd.DataFrame.__getitem__ = _getitem
    pd.DataFrame.take = _take  # type: ignore[assignment]
    pd.DataFrame.groupby = _groupby
    _LocIndexer.__getitem__ = _loc  # type: ignore[assignment]
    try:
        yield counts
    finally:
        pd.DataFrame.__getitem__ = real_getitem
        pd.DataFrame.take = real_take  # type: ignore[assignment]
        pd.DataFrame.groupby = real_groupby
        _LocIndexer.__getitem__ = real_loc  # type: ignore[assignment]
