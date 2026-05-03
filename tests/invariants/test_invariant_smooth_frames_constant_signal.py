"""Physical invariant: smoothing a constant signal returns the constant.

Polynomial of degree 0 is preserved by Savitzky-Golay smoothing of any
window-length / poly-order >= 0. Parametrised over the 4 supported tracking
providers (using their per-provider configs).

Memory: feedback_invariant_testing (PR-S24 dedicated-files structural fix).
"""

from __future__ import annotations

import numpy as np
import pytest

from silly_kicks.tracking.preprocess import PreprocessConfig, smooth_frames
from tests.tracking._provider_inputs import load_provider_frames


@pytest.mark.parametrize("provider", ["sportec", "metrica", "skillcorner", "pff"])
def test_constant_signal_passes_through(provider):
    frames = load_provider_frames(provider).copy()
    frames["x"] = 50.0
    frames["y"] = 30.0
    cfg = PreprocessConfig.for_provider(provider)
    out = smooth_frames(frames, config=cfg)
    valid = out["x_smoothed"].notna() & out["y_smoothed"].notna()
    assert np.allclose(out.loc[valid, "x_smoothed"], 50.0, atol=1e-6), (
        f"{provider}: constant x=50 not preserved by smoothing"
    )
    assert np.allclose(out.loc[valid, "y_smoothed"], 30.0, atol=1e-6), (
        f"{provider}: constant y=30 not preserved by smoothing"
    )
