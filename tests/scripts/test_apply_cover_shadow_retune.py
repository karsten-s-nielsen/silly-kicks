"""Task 12: the three-conjunct, per-provider cover-shadow sigma/lambda apply gate."""

from __future__ import annotations

import ast
import inspect
import sys
from typing import Any

import pytest

from scripts.apply_cover_shadow_retune import decide_apply

# dict[str, Any]: heterogeneous (noise_ok is bool, the rest float), so `**_OK` unpacking into
# decide_apply's typed params is only well-formed for the type checker under Any.
_OK: dict[str, Any] = dict(
    coverage=0.50, receiver_margin=0.10, ablation_share=0.20, noise_ok=True, candidate_sigma=0.30, candidate_lambda=5.0
)


def test_clears_all_conjuncts_applies():
    o = decide_apply(**_OK)
    assert o.outcome == "applied" and o.sigma == 0.30 and o.lambda_ctrl == 5.0


def test_low_coverage_is_unvalidatable():
    assert decide_apply(**{**_OK, "coverage": 0.10}).outcome == "null:unvalidatable"


def test_low_receiver_margin_is_unvalidatable():
    assert decide_apply(**{**_OK, "receiver_margin": 0.01}).outcome == "null:unvalidatable"


def test_high_bias_share_is_biased():
    assert decide_apply(**{**_OK, "ablation_share": 0.60}).outcome == "null:biased"


def test_noise_floor_not_cleared_is_within_noise():
    assert decide_apply(**{**_OK, "noise_ok": False}).outcome == "null:within-noise"


def test_null_paths_leave_the_library_byte_identical():
    from silly_kicks.tracking import _cover_shadows as cs
    from silly_kicks.tracking._cover_shadows import CoverShadowParams

    before = dict(cs._PROVIDER_COVER_SHADOW_PARAMS)
    for kw in ({"coverage": 0.1}, {"ablation_share": 0.6}, {"noise_ok": False}):
        assert decide_apply(**{**_OK, **kw}).outcome.startswith("null:")
    assert dict(cs._PROVIDER_COVER_SHADOW_PARAMS) == before  # decide_apply never mutates the map
    assert CoverShadowParams().sigma == 0.20 and CoverShadowParams().lambda_ctrl == 4.3


def test_nan_inputs_route_to_null_not_spurious_apply():
    """HIGH (review): a degenerate corpus -> coverage 0/0 = NaN, or an all-NaN sweep -> NaN candidate
    params / ablation share. A `NaN < 0.30` reject comparison is False, so without a finite guard these
    would slip through as `applied`. Each must route to the safe null instead."""
    nan = float("nan")
    assert decide_apply(**{**_OK, "coverage": nan}).outcome == "null:unvalidatable"
    assert decide_apply(**{**_OK, "receiver_margin": nan}).outcome == "null:unvalidatable"
    assert decide_apply(**{**_OK, "ablation_share": nan}).outcome == "null:biased"
    assert decide_apply(**{**_OK, "candidate_sigma": nan}).outcome == "null:unvalidatable"
    assert decide_apply(**{**_OK, "candidate_lambda": nan}).outcome == "null:unvalidatable"


def test_gate_references_pinned_thresholds_and_has_no_inline_literals():
    """R5 (strengthened): the gate reads the named constants AND contains no inline numeric threshold
    literal that could shadow the pinned bar, so the bar cannot be moved silently."""
    tree = ast.parse(inspect.getsource(decide_apply))
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert {"MIN_COVERAGE", "MIN_RECEIVER_MARGIN", "MAX_BIAS_SHARE"} <= attrs
    literals = [
        n.value
        for cmp in ast.walk(tree)
        if isinstance(cmp, ast.Compare)
        for n in [cmp.left, *cmp.comparators]
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool)
    ]
    assert not literals, f"inline numeric threshold literal(s) in decide_apply: {literals}"


def test_help_exits_zero():
    from scripts import apply_cover_shadow_retune as A

    with pytest.raises(SystemExit) as exc:
        old = sys.argv
        sys.argv = ["apply_cover_shadow_retune.py", "--help"]
        try:
            A.main()
        finally:
            sys.argv = old
    assert exc.value.code == 0
