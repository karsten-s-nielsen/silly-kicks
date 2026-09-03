from __future__ import annotations

from silly_kicks.shot_stopping import ShotStoppingParams


def test_default_and_flag():
    assert ShotStoppingParams().shootout_period_id == 5
    assert ShotStoppingParams.default().is_default() is True
    assert ShotStoppingParams.default(force_universal=True).is_default() is False
    assert ShotStoppingParams().is_default() is False  # hand-built != .default()


def test_for_provider_returns_base_for_unlisted():
    assert ShotStoppingParams.for_provider("statsbomb") == ShotStoppingParams()
