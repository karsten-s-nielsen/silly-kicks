"""TerritoryParams -- frozen defaults + flag semantics + empty per-provider override (ADR-009)."""

from __future__ import annotations

from silly_kicks.territory import TerritoryParams


def test_default_and_flag():
    assert TerritoryParams.default().is_default() is True
    assert TerritoryParams.default(force_universal=True).is_default() is False
    # A hand-built config is NOT a factory default even with identical fields.
    assert TerritoryParams().is_default() is False


def test_for_provider_returns_base_for_unlisted():
    # The override map ships EMPTY (ADR-009), so every provider resolves to the base config.
    assert TerritoryParams.for_provider("statsbomb") == TerritoryParams()
    assert TerritoryParams.for_provider("gradientsports") == TerritoryParams()


def test_defaults_match_spec():
    p = TerritoryParams()
    assert p.trim_fraction == 0.70
    assert p.forward_threshold_m == 0.0
    assert p.defensive_action_types == ("tackle", "interception", "clearance")
    assert p.own_half_max_x == 52.5
