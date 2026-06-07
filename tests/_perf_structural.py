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

from typing import Any


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
