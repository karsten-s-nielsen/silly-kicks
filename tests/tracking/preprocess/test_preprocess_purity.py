"""TF-65: the preprocess functions are PURE (no caller-input mutation).

``PURITY_ENTRIES`` (tests/test_add_star_purity.py) is add_*-exact (ADR-056 meta-gate), so the
preprocess functions -- which are not ``add_*`` -- cannot register there; this dedicated gate covers
them. Mirrors the _assert_pure contract: snapshot the inputs, invoke, assert the originals are
unchanged (and the output is a new object).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.tracking.preprocess import (
    PreprocessConfig,
    derive_velocities,
    interpolate_frames,
    smooth_frames,
)


def _raw_frames(hz=25.0, n=60):
    return pd.DataFrame(
        {
            "game_id": "g",
            "period_id": 1,
            "frame_id": np.arange(n),
            "time_seconds": np.arange(n) / hz,
            "frame_rate": hz,
            "player_id": "p1",
            "is_ball": False,
            "x": 50.0 + 0.1 * np.arange(n),
            "y": 30.0 + 0.02 * np.arange(n),
        }
    )


def _assert_pure(fn, frames):
    snap = frames.copy(deep=True)
    out = fn(frames)
    assert snap.equals(frames), f"{fn.__name__} MUTATED the caller frames in place"
    assert out is not frames, f"{fn.__name__} returned the SAME object as the input"
    return out


def test_smooth_frames_is_pure():
    _assert_pure(lambda f: smooth_frames(f, config=PreprocessConfig.default()), _raw_frames())


def test_derive_velocities_is_pure():
    smoothed = smooth_frames(_raw_frames(), config=PreprocessConfig.default())
    _assert_pure(lambda f: derive_velocities(f, config=PreprocessConfig.default()), smoothed)


def test_interpolate_frames_is_pure():
    f = _raw_frames()
    f.loc[f["frame_id"].eq(30), "x"] = np.nan  # a NaN-position run to interpolate
    _assert_pure(lambda g: interpolate_frames(g, config=PreprocessConfig.default()), f)
