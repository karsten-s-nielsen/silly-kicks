"""Physical invariant: sync_score values in [0, 1] and min <= mean.

Parametrised over the 4 supported tracking providers via the per-period-dense
synthesizer + link_actions_to_frames.

Memory: feedback_invariant_testing (PR-S24 dedicated-files structural fix).
"""

from __future__ import annotations

import pytest

from silly_kicks.tracking.utils import link_actions_to_frames, sync_score
from tests.tracking._provider_inputs import (
    load_provider_frames,
    synthesize_actions_per_period_dense,
)


@pytest.mark.parametrize("provider", ["sportec", "metrica", "skillcorner", "pff"])
def test_sync_score_bounds_and_monotonicity(provider):
    frames = load_provider_frames(provider)
    actions = synthesize_actions_per_period_dense(frames)
    pointers, _report = link_actions_to_frames(actions, frames, tolerance_seconds=0.5)
    if pointers["link_quality_score"].dropna().empty:
        pytest.skip(f"{provider}: linker found no quality scores in fixture")
    df = sync_score(pointers, high_quality_threshold=0.85)
    for col in df.columns:
        valid = df[col].dropna()
        assert (valid >= 0.0).all(), f"{provider}: {col} has values below 0"
        assert (valid <= 1.0).all(), f"{provider}: {col} has values above 1"
    valid_pair = df.dropna(subset=["sync_score_min", "sync_score_mean"])
    assert (valid_pair["sync_score_min"] <= valid_pair["sync_score_mean"] + 1e-9).all(), (
        f"{provider}: sync_score_min > sync_score_mean for some action"
    )
