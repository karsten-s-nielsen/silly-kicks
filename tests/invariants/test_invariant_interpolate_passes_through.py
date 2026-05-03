"""Physical invariant: interpolate_frames preserves observed (non-NaN) values
exactly. Only NaN cells inside short gaps are filled.

Parametrised over the 4 supported tracking providers via ``_provider_inputs``.

Memory: feedback_invariant_testing (PR-S24 dedicated-files structural fix).
"""

from __future__ import annotations

import pandas as pd
import pytest

from silly_kicks.tracking.preprocess import PreprocessConfig, interpolate_frames
from tests.tracking._provider_inputs import load_provider_frames


@pytest.mark.parametrize("provider", ["sportec", "metrica", "skillcorner", "pff"])
def test_observed_values_unchanged(provider):
    frames = load_provider_frames(provider)
    cfg = PreprocessConfig.for_provider(provider)
    out = interpolate_frames(frames, config=cfg)
    raw_valid = frames["x"].notna() & frames["y"].notna()
    pd.testing.assert_series_equal(
        out.loc[raw_valid, "x"].reset_index(drop=True),
        frames.loc[raw_valid, "x"].reset_index(drop=True),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        out.loc[raw_valid, "y"].reset_index(drop=True),
        frames.loc[raw_valid, "y"].reset_index(drop=True),
        check_names=False,
    )


@pytest.mark.parametrize("provider", ["sportec", "metrica", "skillcorner", "pff"])
def test_idempotence(provider):
    frames = load_provider_frames(provider)
    cfg = PreprocessConfig.for_provider(provider)
    once = interpolate_frames(frames, config=cfg)
    twice = interpolate_frames(once, config=cfg)
    pd.testing.assert_series_equal(once["x"], twice["x"])
