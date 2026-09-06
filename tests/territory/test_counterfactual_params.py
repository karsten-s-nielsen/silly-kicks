"""CounterfactualParams -- frozen defaults + flag semantics + empty per-provider override (ADR-009).

Mirrors tests/territory/test_config.py for TerritoryParams (SPEC-04 Task 6).
"""

from __future__ import annotations

from silly_kicks.territory import CounterfactualParams


def test_default_and_flag():
    assert CounterfactualParams.default().is_default() is True
    assert CounterfactualParams.default(force_universal=True).is_default() is False
    # A hand-built config is NOT a factory default even with identical fields.
    assert CounterfactualParams().is_default() is False


def test_for_provider_returns_base_for_unlisted():
    # The override map ships EMPTY (ADR-009), so every provider resolves to the base config.
    assert CounterfactualParams.for_provider("statsbomb") == CounterfactualParams()
    assert CounterfactualParams.for_provider("gradientsports") == CounterfactualParams()


def test_defaults_match_spec():
    p = CounterfactualParams()
    assert p.direction_cone_degrees == 45.0
    assert p.min_transition_support == 1e-6
