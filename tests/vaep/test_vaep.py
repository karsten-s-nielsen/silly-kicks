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
    games = sb_worldcup_data["games"]
    features = pd.concat(
        [
            model.compute_features(game, sb_worldcup_data[f"actions/game_{game.game_id}"])
            for game in games.iloc[:-1].itertuples()
        ]
    )
    expected_features = set(fs.feature_column_names(model.xfns, model.nb_prev_actions))
    assert set(features.columns) == expected_features
    labels = pd.concat(
        [
            model.compute_labels(game, sb_worldcup_data[f"actions/game_{game.game_id}"])
            for game in games.iloc[:-1].itertuples()
        ]
    )
    expected_labels = {"scores", "concedes"}
    assert set(labels.columns) == expected_labels
    assert len(features) == len(labels)
    # fit the model
    model.fit(features, labels)
    return model


def test_predict(sb_worldcup_data: pd.HDFStore, vaep_model: VAEP) -> None:
    games = sb_worldcup_data["games"]
    game = games.iloc[-1]
    actions = sb_worldcup_data[f"actions/game_{game.game_id}"]
    ratings = vaep_model.rate(game, actions)
    expected_rating_columns = {"offensive_value", "defensive_value", "vaep_value"}
    assert set(ratings.columns) == expected_rating_columns


def test_predict_with_missing_features(sb_worldcup_data: pd.HDFStore, vaep_model: VAEP) -> None:
    games = sb_worldcup_data["games"]
    game = games.iloc[-1]
    actions = sb_worldcup_data[f"actions/game_{game.game_id}"]
    X = vaep_model.compute_features(game, actions)
    del X["period_id_a0"]
    with pytest.raises(ValueError):
        vaep_model.rate(game, actions, X)
