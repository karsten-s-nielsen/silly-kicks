"""RestDefenseParams (TF-60, ADR-080) -- frozen, for_provider, flag-based is_default."""

import dataclasses

from silly_kicks.restdefense import RestDefenseParams


def test_defaults():
    p = RestDefenseParams()
    assert p.n_rearguard == 4 and p.min_ball_advance_m == 52.5
    assert p.zone_depth_m is None and p.danger_field_weight is False and p.possession_stride == 1


def test_is_default_is_flag_based_not_value_equality():
    assert RestDefenseParams.default().is_default() is True
    assert RestDefenseParams().is_default() is False  # same field values, different provenance
    assert RestDefenseParams() == RestDefenseParams.default()  # __eq__ ignores the provenance flag


def test_default_force_universal_disables_the_flag():
    assert RestDefenseParams.default(force_universal=True).is_default() is False
    assert RestDefenseParams.default(force_universal=False).is_default() is True


def test_for_provider_returns_base_for_unlisted():
    assert RestDefenseParams.for_provider("skillcorner") == RestDefenseParams()


def test_frozen():
    p = RestDefenseParams()
    try:
        p.n_rearguard = 5  # type: ignore[misc]
        raise AssertionError("expected FrozenInstanceError")
    except dataclasses.FrozenInstanceError:
        pass
