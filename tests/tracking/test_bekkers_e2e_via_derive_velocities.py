"""Spec section 8.9: full pipeline test raw -> derive_velocities -> pressure."""

from __future__ import annotations

import pytest

from silly_kicks.tracking.features import pressure_on_actor
from silly_kicks.tracking.preprocess import derive_velocities, smooth_frames

from ._provider_inputs import SLIM_DIR, load_provider_frames, synthesize_actions


def test_bekkers_requires_velocities_then_passes_after_derive() -> None:
    parquet = SLIM_DIR / "sportec_slim.parquet"
    if not parquet.exists():
        pytest.skip(f"{parquet} missing")
    frames_raw = load_provider_frames("sportec")
    actions = synthesize_actions(frames_raw)
    if "vx" in frames_raw.columns:
        frames_raw = frames_raw.drop(columns=["vx", "vy"])

    with pytest.raises(ValueError, match="missing velocity columns"):
        pressure_on_actor(actions, frames_raw, method="bekkers_pi")

    frames = derive_velocities(smooth_frames(frames_raw))
    result = pressure_on_actor(actions, frames, method="bekkers_pi")
    if result.dropna().empty:
        pytest.skip("bekkers all NaN on this slim slice (ball-row coverage)")
    assert ((result.dropna() >= 0) & (result.dropna() <= 1)).all()
