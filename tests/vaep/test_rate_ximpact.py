"""TF-63 VAEP.rate_ximpact -- VAEP_adjusted x goal-leverage (reuses tests/vaep/conftest fixtures)."""

import numpy as np
import pandas as pd
import pytest

from silly_kicks.win_probability import WinProbabilityModel, goal_leverage


@pytest.fixture
def games(game):
    return pd.DataFrame([{"game_id": game["game_id"], "home_team_id": game["home_team_id"]}])


@pytest.fixture
def fitted_wpm(actions, games):
    return WinProbabilityModel().fit(actions, games=games)


def test_ximpact_equals_adjusted_times_leverage(fitted_vaep, fitted_xs, fitted_wpm, game, actions, games):
    adjusted = fitted_vaep.rate_adjusted(game, actions, fitted_xs)
    lev = goal_leverage(actions, model=fitted_wpm, games=games)
    xi = fitted_vaep.rate_ximpact(game, actions, fitted_xs, win_prob_model=fitted_wpm, games=games)
    expected = adjusted["vaep_value"].to_numpy() * lev.to_numpy()
    np.testing.assert_allclose(xi.to_numpy(), expected, equal_nan=True, atol=1e-12)
    assert xi.name == "ximpact"


def test_raises_on_hybrid_vaep(fitted_hybrid, fitted_xs, fitted_wpm, game, actions, games):
    with pytest.raises(ValueError, match=r"result-bearing|no-op|STANDARD"):
        fitted_hybrid.rate_ximpact(game, actions, fitted_xs, win_prob_model=fitted_wpm, games=games)


def test_nan_propagates(fitted_vaep, fitted_xs, fitted_wpm, game, actions, games):
    a = actions.copy()
    a.loc[a.index[0], "start_x"] = np.nan  # -> xSuccess NaN on row 0 -> adjusted NaN -> xImpact NaN
    xi = fitted_vaep.rate_ximpact(game, a, fitted_xs, win_prob_model=fitted_wpm, games=games)
    assert np.isnan(xi.iloc[0])


def test_leverage_reweights_non_vacuous(fitted_vaep, fitted_xs, fitted_wpm, game, actions, games):
    # non-vacuity: leverage actually re-weights (xImpact is not just VAEP_adjusted) and it varies.
    adjusted = fitted_vaep.rate_adjusted(game, actions, fitted_xs)
    lev = goal_leverage(actions, model=fitted_wpm, games=games)
    xi = fitted_vaep.rate_ximpact(game, actions, fitted_xs, win_prob_model=fitted_wpm, games=games)
    assert lev.to_numpy().std() > 0.0  # leverage varies across game states
    assert not np.allclose(xi.dropna().to_numpy(), adjusted["vaep_value"].reindex(xi.dropna().index).to_numpy())


def test_games_derived_from_game_when_omitted(fitted_vaep, fitted_xs, fitted_wpm, game, actions, games):
    xi_explicit = fitted_vaep.rate_ximpact(game, actions, fitted_xs, win_prob_model=fitted_wpm, games=games)
    xi_derived = fitted_vaep.rate_ximpact(game, actions, fitted_xs, win_prob_model=fitted_wpm)
    np.testing.assert_allclose(xi_explicit.to_numpy(), xi_derived.to_numpy(), equal_nan=True, atol=1e-12)


def test_purity_no_mutation(fitted_vaep, fitted_xs, fitted_wpm, game, actions, games):
    snap = actions.copy(deep=True)
    fitted_vaep.rate_ximpact(game, actions, fitted_xs, win_prob_model=fitted_wpm, games=games)
    pd.testing.assert_frame_equal(actions, snap)
