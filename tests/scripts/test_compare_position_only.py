"""Unit test for the position-only comparability delta (Task 8 / spec D6)."""

from __future__ import annotations

import pytest

import scripts.compare_position_only_variants as cmp


def test_compute_skill_delta_numeric():
    d = cmp.compute_skill_delta({"pr_auc": 0.80, "brier": 0.12}, {"pr_auc": 0.74, "brier": 0.15}, ["pr_auc", "brier"])
    assert d["pr_auc"]["delta"] == pytest.approx(0.06)  # velocity - position_only
    assert d["brier"]["delta"] == pytest.approx(-0.03)
    assert d["pr_auc"]["velocity"] == 0.80
    assert d["pr_auc"]["position_only"] == 0.74


def test_compute_skill_delta_missing_key_is_none_not_zero():
    # An unmeasured metric yields delta=None -- REPORTED, never fabricated as 0.
    d = cmp.compute_skill_delta({"pr_auc": 0.8}, {}, ["pr_auc", "brier"])
    assert d["pr_auc"]["delta"] is None  # absent on position_only side
    assert d["brier"]["delta"] is None  # absent on both


def test_compute_skill_delta_ignores_bool():
    # A bool acceptance flag must not be treated as a numeric metric (0/1 subtraction).
    d = cmp.compute_skill_delta({"passed": True}, {"passed": False}, ["passed"])
    assert d["passed"]["delta"] is None
