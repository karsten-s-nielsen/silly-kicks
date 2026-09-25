"""ADR-105 Task 2: rest_defense Layer-2 routing through the batched pitch-control kernel is
BYTE-IDENTICAL to the pre-batch per-sample path, and invariant to the ``batch_size`` chunk size."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from restdefense._fixtures import make_fitted_xt, make_rest_defense_fixture

from silly_kicks.restdefense import compute_rest_defense

_GOLDEN = Path(__file__).parent / "golden_rest_defense_layer2.parquet"


def test_rest_defense_byte_identical_to_pre_batch_golden():
    # The golden was captured on the pre-Task-2 per-sample path (all 4 pitch-control legs direct). The
    # default is now batch_size=64 (chunked + batched surf_a/gk_influence + threat legs); it must match.
    actions, frames = make_rest_defense_fixture()
    samples, _ = compute_rest_defense(actions, frames, xt=make_fitted_xt())
    golden = pd.read_parquet(_GOLDEN)
    assert list(samples.columns) == list(golden.columns)
    pd.testing.assert_frame_equal(
        samples.reset_index(drop=True), golden.reset_index(drop=True), check_dtype=False, check_categorical=False
    )


@pytest.mark.parametrize("batch_size", [None, 1, 2, 3])
def test_rest_defense_output_invariant_to_batch_size(batch_size):
    actions, frames = make_rest_defense_fixture()
    xt = make_fitted_xt()
    ref, _ = compute_rest_defense(actions, frames, xt=xt, batch_size=None)
    got, _ = compute_rest_defense(actions, frames, xt=xt, batch_size=batch_size)
    pd.testing.assert_frame_equal(
        got.reset_index(drop=True), ref.reset_index(drop=True), check_dtype=False, check_categorical=False
    )
