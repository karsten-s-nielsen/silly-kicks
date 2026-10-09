"""TF-58 Task 14: CoordinationReport merge, conservation, warning category, drop precedence."""

from __future__ import annotations

import dataclasses

import pytest

from silly_kicks.coordination._report import (
    CoordinationCoverageWarning,
    CoordinationReport,
    first_drop_reason,
)
from silly_kicks.tracking._warnings import (
    IgnoredSurfaceInputsWarning,
    OrientationUnresolvedWarning,
    SyntheticEPVWarning,
)


def _report(**overrides) -> CoordinationReport:
    base = {
        "params": None,
        "provider": "sportec",
        "window_source": "period",
        "detection_source": "fully_observed",
        "stoppage_source": "ball_state",
        "native_hz": 25.0,
        "effective_hz": 10.0,
        "rate_capped": True,
        "n_windows_in": 2,
        "n_windows_scored": 2,
    }
    base.update(overrides)
    return CoordinationReport(**base)


def test_warning_category_is_distinct():
    others = (SyntheticEPVWarning, IgnoredSurfaceInputsWarning, OrientationUnresolvedWarning)
    for other in others:
        assert not issubclass(CoordinationCoverageWarning, other)
        assert not issubclass(other, CoordinationCoverageWarning)
    assert issubclass(CoordinationCoverageWarning, UserWarning)


def test_merge_sums_and_rejects_provenance_mismatch():
    a = _report(
        n_windows_in=2,
        n_windows_scored=1,
        windows_dropped={"too_short": 1},
        rows_by_source={"pair": {"scored": 1}},
        surrogate_rows_by_source={"pair": {"computed": 1}},
        dead_seconds=5.0,
    )
    b = _report(
        n_windows_in=3,
        n_windows_scored=2,
        windows_dropped={"too_short": 1},
        rows_by_source={"pair": {"scored": 2}},
        surrogate_rows_by_source={"pair": {"computed": 2}},
        dead_seconds=2.5,
    )
    merged = a.merge(b)
    assert merged.n_windows_in == 5
    assert merged.n_windows_scored == 3
    assert merged.windows_dropped == {"too_short": 2}
    assert merged.rows_by_source == {"pair": {"scored": 3}}
    assert merged.dead_seconds == 7.5
    with pytest.raises(ValueError, match="provider differs"):
        a.merge(dataclasses.replace(b, provider="idsse"))


def test_conservation_errors_detects_a_planted_leak():
    clean = _report(n_windows_in=3, n_windows_scored=2, windows_dropped={"too_short": 1})
    assert clean.conservation_errors() == []
    leak = _report(n_windows_in=3, n_windows_scored=2, windows_dropped={"too_short": 5})
    assert any("window conservation" in e for e in leak.conservation_errors())
    surrogate_leak = _report(rows_by_source={"pair": {"scored": 3}}, surrogate_rows_by_source={"pair": {"computed": 1}})
    assert any("surrogate conservation" in e for e in surrogate_leak.conservation_errors())


def test_drop_reason_precedence():
    assert first_drop_reason({"too_short", "insufficient_players", "degenerate_constant"}) == "insufficient_players"
    assert first_drop_reason({"goal_end_unresolved", "too_short"}) == "goal_end_unresolved"
    assert first_drop_reason({"entropy_undefined"}) == "entropy_undefined"
    assert first_drop_reason(set()) is None
