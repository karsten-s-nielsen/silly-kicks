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


def test_interval_training_leaves_the_shootout_out():
    # A match decided on penalties trains as the draw it was: the shoot-out rows add no goal and no minute.
    import pandas as pd

    from silly_kicks.win_probability import WinProbabilityParams
    from silly_kicks.win_probability._model import _build_interval_training
    from tests.win_probability.conftest import SHOT, SUCC, _match

    m = _match("g1", 600, 1800)  # 1-1
    so = pd.DataFrame(
        {
            "game_id": ["g1"] * 2,
            "action_id": [50, 51],
            "period_id": [5, 5],
            "team_id": ["A", "B"],
            "time_seconds": [5, 10],
            "type_id": [SHOT, SHOT],
            "result_id": [SUCC, SUCC],
        }
    )
    params = WinProbabilityParams.default()
    games = pd.DataFrame({"game_id": ["g1"], "home_team_id": ["A"]})
    base = _build_interval_training(m, games, None, params)
    with_so = _build_interval_training(pd.concat([m, so], ignore_index=True), games, None, params)
    for a, b in zip(base, with_so, strict=True):
        np.testing.assert_array_equal(a, b)


def test_interval_training_reads_a_missing_result_as_no_goal():
    # A shot with a missing result (a nullable result_id) is no goal and no red card, as in derive_match_states.
    import pandas as pd

    from silly_kicks.win_probability import WinProbabilityParams
    from silly_kicks.win_probability._model import _build_interval_training
    from tests.win_probability.conftest import _match

    m = _match("g1", 600, 1800)
    games = pd.DataFrame({"game_id": ["g1"], "home_team_id": ["A"]})
    na = m.copy()
    na["result_id"] = na["result_id"].astype("Int64")
    na.loc[na.index[2], "result_id"] = pd.NA  # B's failed shot
    params = WinProbabilityParams.default()
    for a, b in zip(
        _build_interval_training(m, games, None, params), _build_interval_training(na, games, None, params), strict=True
    ):
        np.testing.assert_array_equal(a, b)


def test_interval_training_reads_a_foul_with_a_missing_result_as_no_card():
    import pandas as pd

    from silly_kicks.spadl import config as C
    from silly_kicks.win_probability import WinProbabilityParams
    from silly_kicks.win_probability._model import _build_interval_training
    from tests.win_probability.conftest import _match

    m = _match("g1", 600, 1800)
    foul = m.iloc[[0]].assign(action_id=99, type_id=C.actiontype_id["foul"], result_id=C.result_id["fail"])
    with_foul = pd.concat([m, foul], ignore_index=True)
    na = with_foul.copy()
    na["result_id"] = na["result_id"].astype("Int64")
    na.loc[na["action_id"] == 99, "result_id"] = pd.NA
    games = pd.DataFrame({"game_id": ["g1"], "home_team_id": ["A"]})
    params = WinProbabilityParams.default()
    for a, b in zip(
        _build_interval_training(with_foul, games, None, params),
        _build_interval_training(na, games, None, params),
        strict=True,
    ):
        np.testing.assert_array_equal(a, b)
