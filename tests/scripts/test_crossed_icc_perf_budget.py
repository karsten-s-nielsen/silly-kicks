"""Structural perf guard for the crossed-ICC reduce (OPT-IMPL-04).

The census reduce ran ~7h on the broad corpus because ``crossed_variance_components`` did a
``pinv(X'X)`` at the FULL design width ``p = 1 + n_def + n_team`` (~4863) -- an O(p^3) per replicate,
x400 x2 bootstrap loops. Absorption routes EVERY pinv through the ``n_team``-sized ``W`` (the reduction
and all six EMS trace coefficients), so the largest pinv is ``n_team x n_team`` regardless of ``n_def``.

These are DETERMINISTIC structural guards (a pinv call-SIZE spy + an operation-count growth exponent),
NOT wall-clock budgets -- the repo bans ``assert ms < budget`` perf tests (CLAUDE.md: perf regressions
are guarded structurally). They END the "found by hand each round" loop: a reintroduced dense pinv(p)
fails here, in CI, instead of on an 8-hour DGX pass. Both are RED on the pre-absorption path (max pinv
dim = p; pinv-work grows as n_def^3).
"""

from __future__ import annotations

import numpy as np
from _crossed_icc import crossed_variance_components

from tests._perf_structural import assert_subquadratic_growth
from tests.scripts.test_crossed_icc import _balanced_crossed_design

_TRUE_PINV = np.linalg.pinv  # captured before any monkeypatch, so the spy never double-wraps


def test_no_pinv_at_full_design_width(monkeypatch):
    """Every ``pinv`` inside a fit is at most ``n_team x n_team`` -- never the full design width ``p``.

    n_def (60) >> n_team (5): the old dense ``pinv(design_full.T @ design_full)`` was ``(1+n_def+n_team)``
    wide (66); absorption keeps every pinv <= n_team (5). This is the invariant the whole ~920x rests on.
    """
    n_team = 5
    seen: dict[str, int] = {"max_dim": 0}

    def spy(a, *args, **kwargs):
        arr = np.asarray(a)
        if arr.ndim == 2:
            seen["max_dim"] = max(seen["max_dim"], arr.shape[0])
        return _TRUE_PINV(a, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "pinv", spy)
    y, dc, tc = _balanced_crossed_design(0, 60, n_team, 3, sd_def=2.0, sd_team=1.5, sd_resid=3.0)
    crossed_variance_components(y, dc, tc)
    assert seen["max_dim"] > 0, "fit did no pinv -- guard is vacuous (design must be identifiable)"
    assert seen["max_dim"] <= n_team, f"pinv at width {seen['max_dim']} > n_team {n_team}: dense pinv(p) reintroduced"


def test_pinv_work_is_flat_in_n_def(monkeypatch):
    """pinv operation-count (sum of ``dim^3``) is ~flat as ``n_def`` grows -- cost is O(n + n_team^3).

    n_team fixed: absorption's only pinv is the ``n_team``-sized W, so pinv-work is CONSTANT in n_def
    (growth exponent ~0). The old dense ``pinv(p)`` grew as ``n_def^3`` (exponent ~3) -- this fails then.
    """
    n_team = 4

    def measure(n_def: int) -> int:
        work = {"ops": 0}

        def spy(a, *args, **kwargs):
            arr = np.asarray(a)
            if arr.ndim == 2:
                work["ops"] += int(arr.shape[0]) ** 3
            return _TRUE_PINV(a, *args, **kwargs)

        monkeypatch.setattr(np.linalg, "pinv", spy)
        y, dc, tc = _balanced_crossed_design(0, n_def, n_team, 2, sd_def=2.0, sd_team=1.5, sd_resid=3.0)
        crossed_variance_components(y, dc, tc)
        return work["ops"]

    assert_subquadratic_growth(measure, sizes=(50, 100, 200), max_exponent=1.5)
