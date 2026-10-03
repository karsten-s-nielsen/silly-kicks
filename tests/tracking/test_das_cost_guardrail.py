"""ADR-105 Task 7: an advisory DAS cost estimate + a one-time, opt-out DasCostWarning before a large
all-frame run. Pure + reported-not-gated -- DAS VALUES are unchanged (no behaviour change)."""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from silly_kicks.tracking import DasCostWarning, estimate_das_cost
from silly_kicks.tracking._das import _DAS_COST_WARN_SECONDS, _DAS_SECONDS_PER_FRAME, _maybe_warn_das_cost

#: distinct-frame count whose serial estimate just crosses the seconds budget (§6.14).
_FRAMES_OVER_BUDGET = int(_DAS_COST_WARN_SECONDS / _DAS_SECONDS_PER_FRAME) + 1


def _frames(n_distinct: int) -> pd.DataFrame:
    return pd.DataFrame({"game_id": 1, "period_id": 1, "frame_id": range(n_distinct)})


def _pin_numba(monkeypatch) -> None:
    """These tests assume the numba constant: pin the engine probe so they are independent of the CI leg."""
    import silly_kicks.tracking._das_engine as eng

    monkeypatch.setattr(eng, "_numba_available", lambda: True)


def test_estimate_das_cost_is_pure_and_scales_with_distinct_frames(monkeypatch):
    _pin_numba(monkeypatch)
    assert estimate_das_cost(_frames(10)) == pytest.approx(10 * _DAS_SECONDS_PER_FRAME)
    assert estimate_das_cost(_frames(1000)) == pytest.approx(1000 * _DAS_SECONDS_PER_FRAME)


def test_estimate_das_cost_is_thread_aware(monkeypatch):
    """n_threads>1 selects the prange kernel -> a strictly lower estimate (§6.14)."""
    _pin_numba(monkeypatch)
    assert estimate_das_cost(_frames(1000), n_threads=4) < estimate_das_cost(_frames(1000))


def test_das_cost_warning_fires_above_threshold():
    with pytest.warns(DasCostWarning):
        _maybe_warn_das_cost(_frames(_FRAMES_OVER_BUDGET), warn_cost=True)


def test_das_cost_warning_opts_out_and_stays_silent_below_threshold():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DasCostWarning)  # any warn would raise
        _maybe_warn_das_cost(_frames(_FRAMES_OVER_BUDGET), warn_cost=False)  # opt-out -> silent
        _maybe_warn_das_cost(_frames(10), warn_cost=True)  # below threshold -> silent


def test_estimate_das_cost_uses_the_engine_that_will_run(monkeypatch):
    import silly_kicks.tracking._das as das_mod
    import silly_kicks.tracking._das_engine as eng

    frames = _frames(10)
    monkeypatch.setattr(eng, "_numba_available", lambda: True)
    assert estimate_das_cost(frames) == pytest.approx(10 * das_mod._DAS_SECONDS_PER_FRAME)
    monkeypatch.setattr(eng, "_numba_available", lambda: False)
    assert estimate_das_cost(frames) == pytest.approx(10 * das_mod._DAS_SECONDS_PER_FRAME_NUMPY)
