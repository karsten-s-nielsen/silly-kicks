"""ADR-105 Task 7: an advisory DAS cost estimate + a one-time, opt-out DasCostWarning before a large
all-frame run. Pure + reported-not-gated -- DAS VALUES are unchanged (no behaviour change)."""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from silly_kicks.tracking import DasCostWarning, estimate_das_cost
from silly_kicks.tracking._das import _DAS_COST_WARN_FRAMES, _DAS_SECONDS_PER_FRAME, _maybe_warn_das_cost


def _frames(n_distinct: int) -> pd.DataFrame:
    return pd.DataFrame({"game_id": 1, "period_id": 1, "frame_id": range(n_distinct)})


def test_estimate_das_cost_is_pure_and_scales_with_distinct_frames():
    assert estimate_das_cost(_frames(10)) == pytest.approx(10 * _DAS_SECONDS_PER_FRAME)
    assert estimate_das_cost(_frames(1000)) == pytest.approx(1000 * _DAS_SECONDS_PER_FRAME)


def test_das_cost_warning_fires_above_threshold():
    with pytest.warns(DasCostWarning):
        _maybe_warn_das_cost(_frames(_DAS_COST_WARN_FRAMES), warn_cost=True)


def test_das_cost_warning_opts_out_and_stays_silent_below_threshold():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DasCostWarning)  # any warn would raise
        _maybe_warn_das_cost(_frames(_DAS_COST_WARN_FRAMES), warn_cost=False)  # opt-out -> silent
        _maybe_warn_das_cost(_frames(10), warn_cost=True)  # below threshold -> silent
