"""Runtime guard: ``_finalize_output`` raises on non-chronological ``action_id``.

The chronological-``action_id`` invariant (spec 2026-08-20) is enforced at the converter
choke point ``_finalize_output`` (all converters + ``convert_to_atomic`` pass through it).
Unlike the warn-default ``SILLY_KICKS_ASSERT_INVARIANTS`` orientation checks, this one RAISES
by default (spec §3c): a non-chronological ``action_id`` is a hard downstream crash or silent
corruption, so failing fast at the boundary is correct. NaN ``time_seconds`` rows cannot be
ordered and are not violations; empty frames pass trivially.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.spadl.utils import _assert_chronological_action_id


def _actions(*, action_id, period_id, time_seconds, game_id) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": np.asarray(game_id, dtype="int64"),
            "period_id": np.asarray(period_id, dtype="int64"),
            "action_id": np.asarray(action_id, dtype="int64"),
            "time_seconds": np.asarray(time_seconds, dtype="float64"),
        }
    )


def test_guard_raises_on_non_chronological_action_id():
    df = _actions(action_id=[0, 1], period_id=[1, 1], time_seconds=[5.0, 2.0], game_id=[1, 1])
    with pytest.raises(ValueError, match=r"non-decreasing|chronolog"):
        _assert_chronological_action_id(df)


def test_guard_passes_chronological_empty_and_nan_time():
    _assert_chronological_action_id(
        _actions(action_id=[0, 1], period_id=[1, 1], time_seconds=[2.0, 5.0], game_id=[1, 1])
    )
    _assert_chronological_action_id(_actions(action_id=[], period_id=[], time_seconds=[], game_id=[]))
    _assert_chronological_action_id(
        _actions(action_id=[0, 1], period_id=[1, 1], time_seconds=[float("nan"), float("nan")], game_id=[1, 1])
    )


def test_guard_is_per_group_not_global():
    # Two periods: within EACH period action_id-order is chronological, but the second period's
    # times are lower than the first's. This is legal (period is the grouping key) -- must NOT raise.
    df = _actions(
        action_id=[0, 1, 2, 3],
        period_id=[1, 1, 2, 2],
        time_seconds=[2600.0, 2700.0, 5.0, 10.0],
        game_id=[1, 1, 1, 1],
    )
    _assert_chronological_action_id(df)


def test_guard_ignores_nan_time_rows_between_finite_rows():
    # A NaN-time row (e.g. an unimputed foul) is excluded from the ordering check; the finite
    # rows around it stay non-decreasing, so no raise.
    df = _actions(
        action_id=[0, 1, 2],
        period_id=[1, 1, 1],
        time_seconds=[10.0, float("nan"), 20.0],
        game_id=[1, 1, 1],
    )
    _assert_chronological_action_id(df)


def test_guard_noop_when_required_columns_absent():
    # Non-action frames (e.g. a schema without time_seconds) are outside scope -- no raise.
    df = pd.DataFrame({"game_id": [1, 1], "action_id": [0, 1], "x": [1.0, 2.0]})
    _assert_chronological_action_id(df)
