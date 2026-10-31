"""Finding G (TF-46 Part 2 spec section 27.15; DL design 2026-10-26; ADR-069 / ADR-125 amended): Gradient Sports is
broadcast-derived tracking -- detection-aware (G1), estimated in z (G3), not dead-ball observed (G4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._provider_visibility import dead_ball_observed, detected_mask
from silly_kicks.tracking.gradientsports import VISIBILITY_NATIVE_VALUES, convert_to_frames

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "datasets" / "tracking" / "gradientsports"
TINY = pd.read_parquet(FIXTURE_DIR / "tiny.parquet")


def _with_flag(raw: pd.DataFrame) -> pd.DataFrame:
    """TINY with the raw GS flag as `visibility_native` (the fixture's player flags; the ball VISIBLE)."""
    out = raw.copy()
    flag = out["visibility"].astype(object)
    out["visibility_native"] = [v if isinstance(v, str) else "VISIBLE" for v in flag]
    return out


def _convert(raw: pd.DataFrame) -> pd.DataFrame:
    frames, _ = convert_to_frames(raw, home_team_id=100, home_team_start_left=True, output_convention="absolute_frame")
    return frames


def test_the_raw_flag_becomes_the_boolean_visibility():
    raw = _with_flag(TINY)
    frames = _convert(raw)
    want = raw["visibility_native"].map(VISIBILITY_NATIVE_VALUES)
    assert frames["visibility"].tolist() == want.tolist()  # row order is kept by the adapter
    assert set(frames["visibility"].tolist()) == {True, False}  # non-vacuity: the fixture mixes both values
    # a detection consumer reads it: ESTIMATED rows are undetected (the mutant -- GS fully observed -- read all True)
    mask = detected_mask(frames["visibility"], provider="gradientsports")
    np.testing.assert_array_equal(mask, want.to_numpy(dtype=bool))


@pytest.mark.parametrize("bad", ["visible", "UNKNOWN", None])
def test_the_flag_vocabulary_is_closed(bad):
    raw = _with_flag(TINY)
    raw.loc[raw.index[0], "visibility_native"] = bad
    with pytest.raises(ValueError, match="visibility_native"):
        _convert(raw)


def test_without_the_flag_the_detection_consumers_fail_loud():
    frames = _convert(TINY.drop(columns=["visibility"]))
    assert frames["visibility"].isna().all()
    with pytest.raises(ValueError, match="visibility_native"):
        detected_mask(frames["visibility"], provider="gradientsports")


def test_gradientsports_z_is_estimated_and_not_dead_ball_observed():
    frames = _convert(_with_flag(TINY))
    assert set(frames["z_source"].astype(str)) == {"estimated"}  # G3: static per match, never per row
    assert dead_ball_observed("gradientsports") is False  # G4
