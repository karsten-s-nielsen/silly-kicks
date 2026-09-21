"""TF-63 bundled-weights gates (skip until the commit-2 weights land).

These run against the SHIPPED artifact ``silly_kicks/win_probability/weights/``. Until the corpus fit
produces it (commit-2), they skip -- exactly like the sibling bundled-model tests. Once the bundle
exists they enforce: fail-closed load, ``certify_coherence`` on the shipped weights, calibration
(``ece <= 0.10`` AND ``|reliability_slope - 1| <= 0.25``), and the expected-goals gate (the fitted
model's expected total goals match the corpus rate within the recorded tolerance -- TF63-PLAN-02).
"""

from __future__ import annotations

import json
import pathlib

import pytest

import silly_kicks.win_probability as wp
from silly_kicks.win_probability import WinProbabilityModel, WinProbabilityParams
from silly_kicks.win_probability._chain import expected_total_goals

_WEIGHTS = pathlib.Path(wp.__file__).resolve().parent / "weights"
_HAS_WEIGHTS = (_WEIGHTS / "model.json").exists()

pytestmark = pytest.mark.skipif(
    not _HAS_WEIGHTS, reason="no bundled win_probability weights yet (commit-2, owner-run corpus)"
)


def _metrics() -> dict:
    return json.loads((_WEIGHTS / "metrics.json").read_text(encoding="utf-8"))


def test_bundled_loads_fail_closed():
    # SHA256 + feature-contract probe; raises on tamper/skew.
    m = WinProbabilityModel.bundled()
    assert m._fitted and m._beta is not None


def test_bundled_certifies_coherence():
    # the fail-closed bundle-time gate must pass on the SHIPPED weights (TF63-SPEC-09).
    WinProbabilityModel.bundled().certify_coherence()


def test_bundled_calibration_within_gates():
    meta = _metrics()
    p = WinProbabilityParams.default()
    assert meta["ece"] <= p.ece_max, f"ECE {meta['ece']} > {p.ece_max}"
    assert abs(meta["reliability_slope"] - 1.0) <= p.slope_tol, (
        f"|slope-1| {abs(meta['reliability_slope'] - 1.0)} > {p.slope_tol}"
    )


def test_bundled_expected_goals_matches_corpus_rate():
    # TF63-PLAN-02 (corpus half): the FITTED bundled model's expected total goals (0-0, full match) must
    # match the empirical corpus goals/match within the recorded tolerance -- the mis-scaled-hazard guard.
    meta = _metrics()
    m = WinProbabilityModel.bundled()
    p = WinProbabilityParams.default()
    n = p.regulation_minutes // p.interval_minutes

    def hz_home(d, mm):
        return m._hazard(score_diff=d, minutes_remaining=mm, base_strength=0.0, home=True, man_advantage=0)

    def hz_away(d, mm):
        return m._hazard(score_diff=-d, minutes_remaining=mm, base_strength=0.0, home=False, man_advantage=0)

    eg = expected_total_goals(hz_home, hz_away, n_intervals=n, K=p.lattice_pad)
    assert abs(eg - meta["corpus_goals_per_match"]) < meta["expected_goals_tol"], (
        f"model expected goals {eg:.3f} vs corpus {meta['corpus_goals_per_match']:.3f} "
        f"(tol {meta['expected_goals_tol']})"
    )
