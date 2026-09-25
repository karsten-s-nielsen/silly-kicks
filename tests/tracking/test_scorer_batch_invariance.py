"""ADR-105 Task 4 (VKS-SPEC-03): each PC-consuming scorer's output is BYTE-IDENTICAL across chunk sizes
(``batch_size in {None, 1, ...}``) -- the merge prerequisite for flipping the default to a bounded value.
rest_defense's invariance is in tests/restdefense/test_danger_batch_parity.py; this covers off_ball."""

from __future__ import annotations

import pandas as pd
import pytest

from silly_kicks.tracking import detect_off_ball_runs, value_off_ball_runs
from tests.tracking.test_run_values_value import _actions, _const_xt, _frames


@pytest.mark.parametrize("batch_size", [None, 1, 2, 7])
def test_value_off_ball_runs_invariant_to_batch_size(batch_size):
    actions, frames = _actions(), _frames()
    runs = detect_off_ball_runs(actions, frames)
    xt = _const_xt()
    ref = value_off_ball_runs(runs, actions, frames, xt, batch_size=None)
    got = value_off_ball_runs(runs, actions, frames, xt, batch_size=batch_size)
    pd.testing.assert_frame_equal(
        got.reset_index(drop=True), ref.reset_index(drop=True), check_dtype=False, check_categorical=False
    )
