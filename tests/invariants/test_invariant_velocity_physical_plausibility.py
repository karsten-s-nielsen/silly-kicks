"""Physical invariant: derived speed magnitudes must be physically plausible.

For real football tracking data, >= 99.9% of player frames must have
``speed <= 12.0 m/s`` (max sprint speed of human players). Parametrised over
the 4 supported tracking providers via ``_provider_inputs`` slim parquets.

Memory: feedback_invariant_testing (PR-S24 dedicated-files structural fix).
"""

from __future__ import annotations

import pytest

from silly_kicks.tracking.preprocess import (
    PreprocessConfig,
    derive_velocities,
    smooth_frames,
)
from tests.tracking._provider_inputs import load_provider_frames


@pytest.mark.parametrize("provider", ["sportec", "metrica", "skillcorner", "pff"])
def test_speed_within_human_bounds(provider):
    frames = load_provider_frames(provider)
    cfg = PreprocessConfig.for_provider(provider)
    smoothed = smooth_frames(frames, config=cfg)
    out = derive_velocities(smoothed, config=cfg)
    player_rows = out[~out["is_ball"].astype(bool)]  # explicit cast: feedback_python314_pandas_gotchas
    valid_speed = player_rows["speed"].dropna()
    if len(valid_speed) == 0:
        pytest.skip(f"{provider}: derived speed has no valid rows")
    fraction_under_12mps = float((valid_speed <= 12.0).mean())
    assert fraction_under_12mps >= 0.999, (
        f"{provider}: only {fraction_under_12mps:.4f} of player frames have speed <= 12 m/s "
        f"(expected >= 0.999). Outliers may indicate smoothing-window misconfiguration."
    )
