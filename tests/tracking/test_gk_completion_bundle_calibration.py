"""Every BUNDLED GkCompletionModel variant's recorded metrics must clear its own gate.

The sibling rho model has had this since 4.44.0
(`tests/xtgk/test_retention_bundle_calibration.py`); the completion model did not, and 4.73.0 is the
first release to retrain it since the gate constants were introduced. The SHA256SUMS tests verify a
bundle has not been *altered*; this verifies it *meets the bar*, turning "bundle only if it passes /
no lowered bar" into a CI-enforced invariant that guards every future re-bundle rather than a
property someone re-checked by eye at retrain time.

**The bar is the canonical constants imported from the trainer, never the fields recorded in
`metrics.json`** — otherwise a hand-loosened artifact self-certifies. The recorded thresholds are
then asserted EQUAL to the canonical ones, which is the tamper check.

The two variants are gated on DIFFERENT quantities, deliberately, because their trainers compute
different things: `default` publishes a pooled native-origin AUC with a bootstrap CI, `skillcorner`
publishes per-sub-domain AUC/ECE/slope against a GK-pass floor. Asserting one shape over both would
force a lowest-common-denominator gate that checks less than either trainer already decides on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.train_gk_completion import _ECE_TOL, _GKPASS_AUC_FLOOR, _N_NATIVE_FLOOR, _SLOPE_TOL

_WEIGHTS = Path(__file__).resolve().parents[2] / "silly_kicks" / "tracking" / "_gk_completion_weights"


def _metrics(variant: str) -> dict:
    path = _WEIGHTS / variant / "metrics.json"
    assert path.exists(), f"no bundled {variant} metrics.json at {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def test_bundled_variants_are_discovered():
    """Non-vacuity: if the discovery ever finds nothing, both tests below pass by doing nothing."""
    found = sorted(d.name for d in _WEIGHTS.iterdir() if d.is_dir() and (d / "metrics.json").exists())
    assert found == ["default", "skillcorner"], f"bundled variant set changed: {found}"


def test_bundled_default_clears_its_native_origin_gate():
    """`default`'s green gate: pooled native-origin AUC with its CI lower bound above chance."""
    m = _metrics("default")
    assert m["n_native"] >= _N_NATIVE_FLOOR, f"n_native {m['n_native']} < {_N_NATIVE_FLOOR}"
    assert m["native_auc_ci95"][0] > 0.5, f"AUC CI lower bound {m['native_auc_ci95'][0]} <= chance"
    assert m["native_brier"] < m["base_rate_brier"], (
        f"Brier {m['native_brier']} not better than the base rate {m['base_rate_brier']}"
    )


def test_bundled_skillcorner_clears_its_gkpass_gate():
    """`skillcorner`'s decision gate: the GK-pass sub-domain, which is the domain it is served on.

    Goal-kicks are deliberately NOT gated on AUC here — they route to `base_rate` precisely because
    their AUC is at chance, so asserting a floor on them would contradict the shipped serve mode.
    """
    m = _metrics("skillcorner")
    gk_pass = m["subdomains"]["gk_pass"]
    assert gk_pass["sc_auc"] >= _GKPASS_AUC_FLOOR, f"GK-pass AUC {gk_pass['sc_auc']} < {_GKPASS_AUC_FLOOR}"
    assert gk_pass["sc_ece"] <= _ECE_TOL, f"GK-pass ECE {gk_pass['sc_ece']} > {_ECE_TOL}"
    assert abs(gk_pass["sc_slope"] - 1.0) <= _SLOPE_TOL, f"GK-pass slope {gk_pass['sc_slope']}"
    assert m["bundled"] is True and m["decision"] == "bundle_skillcorner"


@pytest.mark.parametrize(
    ("variant", "field", "canonical"),
    [
        ("skillcorner", "gkpass_auc_floor", _GKPASS_AUC_FLOOR),
        ("skillcorner", "ece_tol", _ECE_TOL),
        # slope_tol is RECORDED in metrics.json and was omitted here, so a loosened value in
        # that one field could self-certify while the other two looked guarded.
        ("skillcorner", "slope_tol", _SLOPE_TOL),
    ],
)
def test_recorded_thresholds_match_the_canonical_constants(variant, field, canonical):
    """Tamper check: a metrics.json that records a loosened bar must not be able to self-certify."""
    m = _metrics(variant)
    assert m[field] == canonical, f"{variant}: recorded {field}={m[field]!r} != canonical {canonical!r}"
