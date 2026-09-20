import sys

import numpy as np

from silly_kicks.win_probability import WinProbabilityModel, WinProbabilityParams


def _inject(m: WinProbabilityModel, beta, intercept):
    m._beta = np.asarray(beta, dtype="float64")
    m._intercept = float(intercept)
    m._isotonic = None
    m._fitted = True
    return m


def test_hazard_serves_pure_numpy_no_sklearn(monkeypatch):
    m = _inject(WinProbabilityModel(params=WinProbabilityParams.default()), [0.1, -0.01, 0.2, 0.05, 0.1], -3.0)
    monkeypatch.setitem(sys.modules, "sklearn", None)  # forbid sklearn at serve
    ph = m._hazard(score_diff=0, minutes_remaining=45.0, base_strength=0.0, home=True, man_advantage=0)
    assert 0.0 < ph < 1.0


def test_fit_then_serve_roundtrip(sample_two_match_actions, sample_games):
    m = WinProbabilityModel(params=WinProbabilityParams.default()).fit(sample_two_match_actions, games=sample_games)
    w, draw, loss = m.predict_outcome(
        score_diff=0, minutes_remaining=90.0, base_strength=0.0, home=True, man_advantage=0
    )
    assert abs(w + draw + loss - 1.0) < 1e-9


def test_kickoff_anchor_symmetric_when_no_home_effect():
    # §9.1 KICKOFF ANCHOR (TF63-PLAN-01): even strength + zero home coeff -> P(win)==P(loss) at (0-0).
    m = _inject(WinProbabilityModel(params=WinProbabilityParams.default()), [0.3, -0.005, 0.2, 0.0, 0.1], -3.0)
    w, _draw, loss = m.predict_outcome(
        score_diff=0, minutes_remaining=90.0, base_strength=0.0, home=True, man_advantage=0
    )
    assert abs(w - loss) < 1e-9


def test_kickoff_anchor_home_advantage_sign():
    m = _inject(WinProbabilityModel(params=WinProbabilityParams.default()), [0.3, -0.005, 0.2, 0.5, 0.1], -3.0)
    wh, _, _ = m.predict_outcome(score_diff=0, minutes_remaining=90.0, base_strength=0.0, home=True, man_advantage=0)
    wa, _, _ = m.predict_outcome(score_diff=0, minutes_remaining=90.0, base_strength=0.0, home=False, man_advantage=0)
    assert wh > wa
