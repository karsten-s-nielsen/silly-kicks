"""Fixtures for TF-61 VAEP_adjusted tests: a tiny synthetic 2-team game + a fitted VAEP / HybridVAEP
and a fitted XSuccessModel."""

import numpy as np
import pandas as pd
import pytest

import silly_kicks.spadl.config as cfg
from silly_kicks.vaep import VAEP
from silly_kicks.vaep.hybrid import HybridVAEP


@pytest.fixture
def actions():
    rng = np.random.default_rng(7)
    n = 48
    teams = np.where(np.arange(n) % 2 == 0, 100, 200)  # strict alternation -> both scores & concedes vary
    type_ids = np.full(n, cfg.actiontype_id["pass"])
    result_ids = np.full(n, cfg.result_id["success"])
    for i in (10, 25, 40):  # scoring shots -> label variance
        type_ids[i] = cfg.actiontype_id["shot"]
        result_ids[i] = cfg.result_id["success"]
    type_ids[30] = cfg.actiontype_id["bad_touch"]  # own goal -> concedes variance
    result_ids[30] = cfg.result_id["owngoal"]
    return pd.DataFrame(
        dict(
            game_id=1,
            period_id=1,
            action_id=np.arange(n),
            team_id=teams,
            player_id=(np.arange(n) % 22) + 1,
            type_id=type_ids,
            result_id=result_ids,
            bodypart_id=cfg.bodypart_id["foot"],
            start_x=rng.uniform(0, 105, n),
            start_y=rng.uniform(0, 68, n),
            end_x=rng.uniform(0, 105, n),
            end_y=rng.uniform(0, 68, n),
            time_seconds=np.arange(n, dtype=float) * 2.0,
        )
    )


@pytest.fixture
def game():
    return pd.Series(dict(game_id=1, home_team_id=100))


@pytest.fixture
def fitted_vaep(game, actions):
    v = VAEP()
    X = v.compute_features(game, actions)
    y = v.compute_labels(game, actions)
    v.fit(X, y, random_state=0)
    return v


@pytest.fixture
def fitted_hybrid(game, actions):
    v = HybridVAEP()
    X = v.compute_features(game, actions)
    y = v.compute_labels(game, actions)
    v.fit(X, y, random_state=0)
    return v


@pytest.fixture
def fitted_xs():
    from silly_kicks.xsuccess import XSuccessModel
    from tests.xsuccess.test_model_fit_predict import _corpus

    return XSuccessModel().fit(_corpus())
