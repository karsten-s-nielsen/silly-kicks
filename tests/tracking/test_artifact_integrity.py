"""Unit tests for the trained-artifact integrity predicates (ADR-126)."""

from __future__ import annotations


def test_base_rate_plausible_derived_bound():
    from silly_kicks.tracking._artifact_integrity import base_rate_plausible, physical_upper_bound

    kw = dict(n_events=21, horizon_s=1.0, fps=10.0, n_rows=8394)  # skillcorner-1925299
    # physical max (tol=1) = 21*10/8394 = 0.0250; derived bound (tol=1.5) = 0.0375
    assert abs(physical_upper_bound(**kw, tol=1.5) - 0.0375) < 1e-3
    assert base_rate_plausible(0.0154, **kw) is True  # correct xShot (below physical max)
    assert base_rate_plausible(0.2220, **kw) is False  # the 13x defect
    assert base_rate_plausible(0.0600, **kw) is False  # a ~2.4x inflation still fails
    assert base_rate_plausible(0.0450, **kw) is False  # in (0.0375, 0.05): derived catches, old 0.05 passed
    assert base_rate_plausible(0.0001, **kw) is False  # strictly below the lo=0.0005 floor


def test_physical_upper_bound_degenerate_rows():
    from silly_kicks.tracking._artifact_integrity import physical_upper_bound

    assert physical_upper_bound(n_events=10, horizon_s=1.0, fps=10.0, n_rows=0) == float("inf")


def test_base_rate_plausible_nan_safe():
    from silly_kicks.tracking._artifact_integrity import base_rate_plausible

    assert base_rate_plausible(float("nan"), n_events=21, horizon_s=1.0, fps=10.0, n_rows=8394) is False
