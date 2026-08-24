"""Task 7 (H3): CoverShadowParams.for_provider -- per-provider sigma/lambda, additive."""

from __future__ import annotations

from silly_kicks.tracking import _cover_shadows as cs
from silly_kicks.tracking._cover_shadows import CoverShadowParams


def test_for_provider_returns_incumbent_when_map_empty():
    inc = CoverShadowParams()
    for prov in ("gradientsports", "skillcorner", "sportec", "unknown"):
        p = CoverShadowParams.for_provider(prov)
        assert p.sigma == inc.sigma == 0.20
        assert p.lambda_ctrl == inc.lambda_ctrl == 4.3


def test_default_params_byte_identical():
    assert CoverShadowParams().sigma == 0.20 and CoverShadowParams().lambda_ctrl == 4.3


def test_map_is_the_only_mutation_point_and_is_provider_scoped(monkeypatch):
    monkeypatch.setitem(cs._PROVIDER_COVER_SHADOW_PARAMS, "gradientsports", {"sigma": 0.30, "lambda_ctrl": 5.0})
    assert CoverShadowParams.for_provider("gradientsports").sigma == 0.30  # GS re-tune applied
    assert CoverShadowParams.for_provider("gradientsports").lambda_ctrl == 5.0
    assert CoverShadowParams.for_provider("skillcorner").sigma == 0.20  # other providers untouched (H3)
    assert CoverShadowParams().sigma == 0.20  # the global default is unchanged
