"""Dead-ball provider taxonomy (TF-58 D20, Task 5)."""

from __future__ import annotations

import pytest

from silly_kicks.tracking._provider_visibility import (
    _DEAD_BALL_OBSERVED_PROVIDERS,
    dead_ball_observed,
    validate_provider,
)


def test_observed_members():
    assert _DEAD_BALL_OBSERVED_PROVIDERS == frozenset({"sportec", "idsse", "gradientsports"})
    for p in _DEAD_BALL_OBSERVED_PROVIDERS:
        assert dead_ball_observed(p) is True


def test_skillcorner_and_metrica_unobserved():
    assert dead_ball_observed("skillcorner") is False
    assert dead_ball_observed("metrica") is False


def test_unclassified_provider_raises():
    with pytest.raises(ValueError, match="_DETECTION_AWARE_PROVIDERS"):
        dead_ball_observed("wyscout")


def test_observed_is_subset_of_classified():
    for p in _DEAD_BALL_OBSERVED_PROVIDERS:
        validate_provider(p)  # must not raise


def test_snapshot_is_not_classified():
    # Freeze frames are refused at the coordination edge (spec 7.3); the taxonomy never treats them as observed.
    with pytest.raises(ValueError):
        dead_ball_observed("snapshot")
