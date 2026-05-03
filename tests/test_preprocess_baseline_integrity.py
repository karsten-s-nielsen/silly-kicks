"""Asserts the generated _PROVIDER_DEFAULTS matches the JSON baseline within rel_tol=1e-6.

This is a deterministic invariant: scripts/regenerate_provider_defaults.py
generates the Python from the JSON. Any drift means regen has not been run.

Failure-message contract: the assertion message MUST include the regen
command so future readers know how to fix.

Memory: feedback_codegen_for_data_to_code_integrity (PR-S24 S1 fix).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_JSON = REPO_ROOT / "tests" / "fixtures" / "baselines" / "preprocess_baseline.json"
REGEN_HINT = (
    "Run `uv run python scripts/regenerate_provider_defaults.py` to regenerate _provider_defaults_generated.py."
)


@pytest.fixture(scope="module")
def baseline_json() -> dict:
    return json.loads(BASELINE_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def provider_defaults():
    """Use the public getter, NOT a private submodule import (PR-S24 N5 fix)."""
    from silly_kicks.tracking.preprocess import get_provider_defaults

    return get_provider_defaults()


@pytest.mark.parametrize("provider", ["sportec", "pff", "metrica", "skillcorner"])
def test_integer_fields_exact_match(baseline_json, provider_defaults, provider):
    block = baseline_json[provider]
    cfg = provider_defaults[provider]
    derived = block["_derived_defaults"]
    assert cfg.sg_poly_order == derived["sg_poly_order"], (
        f"{provider}: sg_poly_order mismatch -- "
        f"_PROVIDER_DEFAULTS={cfg.sg_poly_order} vs JSON={derived['sg_poly_order']}. {REGEN_HINT}"
    )


@pytest.mark.parametrize("provider", ["sportec", "pff", "metrica", "skillcorner"])
@pytest.mark.parametrize(
    "field,attr",
    [
        ("sg_window_seconds", "sg_window_seconds"),
        ("ema_alpha", "ema_alpha"),
        ("max_gap_seconds", "max_gap_seconds"),
    ],
)
def test_derived_default_floats_match(baseline_json, provider_defaults, provider, field, attr):
    expected = baseline_json[provider]["_derived_defaults"][field]
    actual = getattr(provider_defaults[provider], attr)
    assert math.isclose(actual, expected, rel_tol=1e-6, abs_tol=0.0), (
        f"{provider}.{attr}: code={actual} vs JSON.{field}={expected}. {REGEN_HINT}"
    )


@pytest.mark.parametrize("provider", ["sportec", "pff", "metrica", "skillcorner"])
def test_link_quality_high_threshold_match(baseline_json, provider_defaults, provider):
    expected = baseline_json[provider]["link_quality_high_threshold"]
    actual = provider_defaults[provider].link_quality_high_threshold
    assert math.isclose(actual, expected, rel_tol=1e-6, abs_tol=0.0), REGEN_HINT


def test_provenance_block_present(baseline_json):
    assert "_provenance" in baseline_json
    for k in ("generated_by", "generated_at", "silly_kicks_version", "providers_probed"):
        assert k in baseline_json["_provenance"]


def test_sweep_log_exists():
    sweep_log = REPO_ROOT / "tests" / "fixtures" / "baselines" / "preprocess_sweep_log.json"
    assert sweep_log.exists(), "preprocess_sweep_log.json must accompany the baseline JSON (spec section 5.1 #3)"
    payload = json.loads(sweep_log.read_text(encoding="utf-8"))
    assert "_provenance" in payload
    assert "by_provider" in payload
    for p in ("sportec", "pff", "metrica", "skillcorner"):
        assert p in payload["by_provider"], f"sweep_log missing provider block: {p}"
