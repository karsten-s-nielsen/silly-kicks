"""TF-65 §11 / SPEC-05: consumers stay NaN-tolerant on guard-NaN'd ball velocity.

The ~6 % of ball frames the plausibility guard NaNs (out-of-pitch / big-gap) must not crash the
velocity consumers (ghost-GK ball features, elastic-sync). ADR-003 nan_safe_enrichment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from silly_kicks.tracking.features import add_elastic_sync
from silly_kicks.tracking.preprocess import PreprocessConfig, derive_velocities, smooth_frames

_SLICE = Path(__file__).resolve().parents[2] / "datasets" / "elastic_sync" / "j03wmx_slice"


def _preprocessed_with_nan_ball_velocity():
    frames = pd.read_parquet(_SLICE / "frames.parquet")
    cfg = PreprocessConfig.default()
    out = derive_velocities(smooth_frames(frames, config=cfg), config=cfg)
    ball = out["is_ball"].astype(bool)
    out.loc[ball, ["vx", "vy", "speed"]] = np.nan  # simulate the guard output on ball rows
    return out


def test_elastic_sync_nan_tolerant():
    frames = _preprocessed_with_nan_ball_velocity()
    actions = pd.read_parquet(_SLICE / "actions.parquet")
    out = add_elastic_sync(actions, frames=frames)  # must not raise
    assert out is not None and len(out) == len(actions)
