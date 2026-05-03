"""Physical invariant: smooth_frames(smooth_frames(x)) == smooth_frames(x).

Idempotent under the same config. Parametrised over the 4 supported
tracking providers via ``tests/tracking/_provider_inputs.py`` slim parquets.

Memory: feedback_invariant_testing (PR-S24 dedicated-files structural fix).
"""

from __future__ import annotations

import pandas as pd
import pytest

from silly_kicks.tracking.preprocess import PreprocessConfig, smooth_frames
from tests.tracking._provider_inputs import load_provider_frames


@pytest.mark.parametrize("provider", ["sportec", "metrica", "skillcorner", "pff"])
def test_idempotence(provider):
    frames = load_provider_frames(provider)
    cfg = PreprocessConfig.for_provider(provider)
    once = smooth_frames(frames, config=cfg)
    twice = smooth_frames(once, config=cfg)
    pd.testing.assert_series_equal(once["x_smoothed"], twice["x_smoothed"])
    pd.testing.assert_series_equal(once["y_smoothed"], twice["y_smoothed"])
