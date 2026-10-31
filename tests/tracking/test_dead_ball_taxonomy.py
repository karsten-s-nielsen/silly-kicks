"""Dead-ball provider taxonomy (TF-58 D20, Task 5)."""

from __future__ import annotations

import pytest

from silly_kicks.tracking._provider_visibility import (
    _DEAD_BALL_OBSERVED_PROVIDERS,
    dead_ball_observed,
    validate_provider,
)


def test_observed_members():
    # ADR-069 amended 2026-10-26 (TF-46 Part 2 section 27.15 G4): Gradient Sports left the set (its loader writes a
    # constant "alive"; the schema has no null value)
    assert _DEAD_BALL_OBSERVED_PROVIDERS == frozenset({"sportec", "idsse"})
    for p in _DEAD_BALL_OBSERVED_PROVIDERS:
        assert dead_ball_observed(p) is True


def test_skillcorner_metrica_and_gradientsports_unobserved():
    assert dead_ball_observed("skillcorner") is False
    assert dead_ball_observed("metrica") is False
    assert dead_ball_observed("gradientsports") is False


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
