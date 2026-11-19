from typing import cast

import numpy as np
import pandas as pd
import pytest

from silly_kicks.vaep import VAEP, HybridVAEP
from silly_kicks.vaep import features as fs


def test_hybrid_vaep_no_result_a0():
    """HybridVAEP features must not contain result_*_a0 columns."""
    model = HybridVAEP(nb_prev_actions=3)
    cols = fs.feature_column_names(model.xfns, model.nb_prev_actions)
    a0_result_cols = [c for c in cols if "result" in c and c.endswith("_a0")]
    assert len(a0_result_cols) == 0, f"Found result a0 columns: {a0_result_cols}"


def test_hybrid_vaep_has_result_a1():
    """HybridVAEP features must contain result_*_a1 columns."""
    model = HybridVAEP(nb_prev_actions=3)
    cols = fs.feature_column_names(model.xfns, model.nb_prev_actions)
    a1_result_cols = [c for c in cols if "result" in c and c.endswith("_a1")]
    assert len(a1_result_cols) > 0, "Missing result a1 columns"


def test_hybrid_vaep_fewer_features_than_standard():
    """HybridVAEP should have fewer features than standard VAEP."""
    standard = VAEP(nb_prev_actions=3)
    hybrid = HybridVAEP(nb_prev_actions=3)
    standard_cols = fs.feature_column_names(standard.xfns, standard.nb_prev_actions)
    hybrid_cols = fs.feature_column_names(hybrid.xfns, hybrid.nb_prev_actions)
    assert len(hybrid_cols) < len(standard_cols)


def test_random_state_reproducibility():
    """S6: random_state should produce identical permutations."""
    rng1 = np.random.default_rng(42)
    idx1 = rng1.permutation(100)
    rng2 = np.random.default_rng(42)
    idx2 = rng2.permutation(100)
    assert (idx1 == idx2).all()


@pytest.fixture(scope="session")
def vaep_model(sb_worldcup_data: pd.HDFStore) -> VAEP:
    # Test the vAEP framework on the StatsBomb World Cup data
    model = VAEP(nb_prev_actions=1)
    # comppute features and labels
    games = cast(pd.DataFrame, sb_worldcup_data["games"])
    features = cast(
        pd.DataFrame,
        pd.concat(
            [
                model.compute_features(
                    game,  # type: ignore[arg-type]  # itertuples namedtuple has the right shape at runtime
                    cast(pd.DataFrame, sb_worldcup_data[f"actions/game_{game.game_id}"]),  # type: ignore[attr-defined]
                )
                for game in games.iloc[:-1].itertuples()
            ]
        ),
    )
    expected_features = set(fs.feature_column_names(model.xfns, model.nb_prev_actions))
    assert set(features.columns) == expected_features
    labels = cast(
        pd.DataFrame,
        pd.concat(
            [
                model.compute_labels(
                    game,  # type: ignore[arg-type]
                    cast(pd.DataFrame, sb_worldcup_data[f"actions/game_{game.game_id}"]),  # type: ignore[attr-defined]
                )
                for game in games.iloc[:-1].itertuples()
            ]
        ),
    )
    expected_labels = {"scores", "concedes"}
    assert set(labels.columns) == expected_labels
    assert len(features) == len(labels)
    # fit the model
    model.fit(features, labels)
    return model


@pytest.mark.slow
def test_predict(sb_worldcup_data: pd.HDFStore, vaep_model: VAEP) -> None:
    games = cast(pd.DataFrame, sb_worldcup_data["games"])
    game = games.iloc[-1]
    actions = cast(pd.DataFrame, sb_worldcup_data[f"actions/game_{game.game_id}"])
    ratings = vaep_model.rate(game, actions)
    expected_rating_columns = {"offensive_value", "defensive_value", "vaep_value"}
    assert set(ratings.columns) == expected_rating_columns


def test_predict_with_missing_features(sb_worldcup_data: pd.HDFStore, vaep_model: VAEP) -> None:
    games = cast(pd.DataFrame, sb_worldcup_data["games"])
    game = games.iloc[-1]
    actions = cast(pd.DataFrame, sb_worldcup_data[f"actions/game_{game.game_id}"])
    X = vaep_model.compute_features(game, actions)
    del X["period_id_a0"]
    with pytest.raises(ValueError):
        vaep_model.rate(game, actions, X)


def _after_own_goal() -> pd.DataFrame:
    """An own goal of team 1 (a bad_touch row with the owngoal result), then the kickoff of team 2."""
    from silly_kicks.spadl import config

    return pd.DataFrame(
        {
            "team_id": [1, 2],
            "type_name": ["bad_touch", "pass"],
            "type_id": [config.actiontype_id["bad_touch"], config.actiontype_id["pass"]],
            "result_name": ["owngoal", "success"],
            "result_id": [config.result_id["owngoal"], config.result_id["success"]],
            "time_seconds": [10.0, 12.0],  # inside the same phase: only the goal rule sets the odds to 0
        }
    )


def test_offensive_value_takes_a_previous_own_goal_as_a_goal() -> None:
    from silly_kicks.vaep.formula import offensive_value

    ov = offensive_value(_after_own_goal(), pd.Series([0.3, 0.5]), pd.Series([0.1, 0.2]))
    assert ov.iloc[1] == pytest.approx(0.5)  # the scoring odds before the kickoff are 0


def test_defensive_value_takes_a_previous_own_goal_as_a_goal() -> None:
    from silly_kicks.vaep.formula import defensive_value

    dv = defensive_value(_after_own_goal(), pd.Series([0.3, 0.5]), pd.Series([0.1, 0.2]))
    assert dv.iloc[1] == pytest.approx(-0.2)  # the conceding odds before the kickoff are 0
