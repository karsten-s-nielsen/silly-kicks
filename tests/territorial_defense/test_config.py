"""TF-54b TerritorialDefenseParams: frozen, spearman-only, empty for_provider (ADR-009)."""

import dataclasses

import pytest

from silly_kicks.territorial_defense import TerritorialDefenseParams


def test_defaults():
    p = TerritorialDefenseParams()
    assert p.defensive_action_type_ids == (9, 10, 18)
    assert p.pitch_control_method == "spearman"
    assert p.min_defenders_after_removal == 1
    assert p.arm_b_rule == "nearest_to_target"
    # L8: the Arm-B own-half hull cut is configurable (default = midfield), matching the sibling
    # ``territory`` package's ``own_half_max_x`` rather than a hardcoded ``fl / 2``.
    assert p.own_half_max_x == 52.5


@pytest.mark.parametrize("method", ["voronoi", "fernandez_bornn"])
def test_gk_blind_method_rejected_at_construction(method):
    with pytest.raises(ValueError, match="GK-blind"):
        TerritorialDefenseParams(pitch_control_method=method)


def test_unknown_arm_b_rule_rejected_at_construction():
    # "receiver_lane" is a RESERVED follow-on rule, not yet accepted (spec §11.4 / Decision 4).
    with pytest.raises(ValueError, match="arm_b_rule"):
        TerritorialDefenseParams(arm_b_rule="receiver_lane")


def test_for_provider_empty_returns_base():
    # ADR-009: the override map ships empty, so every provider resolves to the base config.
    for prov in ("statsbomb", "skillcorner", "gradientsports"):
        assert TerritorialDefenseParams.for_provider(prov) == TerritorialDefenseParams()


def test_frozen():
    p = TerritorialDefenseParams()
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.lambda_gk = 9.0  # type: ignore[misc]
