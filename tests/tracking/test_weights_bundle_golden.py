"""Golden test: the bundled default weights load AND their chirality fingerprint re-verifies on
THIS platform (TF-19 PR-2).

This is the cross-environment gate. `from_variant("default")` runs the STRICT (non-override)
chirality path, so a passing load means the fingerprint recorded at training time reproduces on
the current platform + xgboost version within tolerance. It is what caught the xgboost-3.x
bracketed-`base_score` mis-serve (the defensive guard in `_xshot_occurrence` normalizes it); if
this test ever fails with a chirality mismatch, a trained/serve inconsistency has been introduced
— do NOT loosen the tolerance to force it green.
"""

import importlib

import pytest


@pytest.mark.parametrize(
    "mod,cls_name",
    [
        ("silly_kicks.tracking._xshot_occurrence", "XShotOccurrenceModel"),
        ("silly_kicks.tracking._xcross_attempt", "XCrossAttemptModel"),
        ("silly_kicks.tracking._ghost_gk", "GhostGkModel"),
    ],
)
def test_bundled_default_loads_and_chirality_reverifies(mod, cls_name):
    m = importlib.import_module(mod)
    # clear the memo cache so the load + strict chirality re-verification actually runs
    if hasattr(m, "_VARIANT_CACHE"):
        m._VARIANT_CACHE.clear()
    model = getattr(m, cls_name).from_variant("default")
    assert model is not None
    assert model._booster is not None if cls_name != "GhostGkModel" else True
