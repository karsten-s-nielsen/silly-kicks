from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest
from pytest_mock import MockerFixture
from sklearn.exceptions import NotFittedError

import silly_kicks.spadl as spadl
import silly_kicks.xthreat as xt
from silly_kicks.spadl.config import field_length, field_width
from silly_kicks.xthreat import _grid
from silly_kicks.xthreat._params import GridSpec
from silly_kicks.xthreat._transitions import singh_transition_matrix


class TestGridCount:
    """Tests for counting the number of actions occuring in each grid cell.

    Grid cells ares represented by 2D pitch coordinates. The (0,0) coordinate
    corresponds to the bottom left corner of the pitch. The 2D coordinates are
    mapped to a flat index. For a 2x2 grid, these flat indices are:
        0 1
        2 3
    """

    N = 2
    M = 2

    def test_get_cell_indexes(self) -> None:
        """It should map pitch coordinates to a 2D cell index."""
        x = pd.Series([0, field_length / 2 - 1, field_length])
        y = pd.Series([0, field_width / 2 + 1, field_width])
        xi, yi = _grid._get_cell_indexes(x, y, self.N, self.M)
        pd.testing.assert_series_equal(xi, pd.Series([0, 0, 1]))
        pd.testing.assert_series_equal(yi, pd.Series([0, 1, 1]))

    def test_get_cell_indexes_out_of_bounds(self) -> None:
        """It should map out-of-bounds coordinates to the nearest cell index."""
        x = pd.Series([-10, field_length + 10])
        y = pd.Series([-10, field_width + 10])
        xi, yi = _grid._get_cell_indexes(x, y, self.N, self.M)
        pd.testing.assert_series_equal(xi, pd.Series([0, 1]))
        pd.testing.assert_series_equal(yi, pd.Series([0, 1]))

    def test_get_flat_indexes(self) -> None:
        """It should map pitch coordinates to a flat index."""
        x = pd.Series([0, field_length / 2 - 1, field_length / 2 + 1, field_length])
        y = pd.Series([0, field_width / 2 + 1, field_width / 2 - 1, field_width])
        idx = _grid._get_flat_indexes(x, y, self.N, self.M)
        pd.testing.assert_series_equal(idx, pd.Series([2, 0, 3, 1]))

    def test_count(self) -> None:
        """It should return the number of occurences in each grid cell."""
        x = pd.Series([0, field_length / 2 - 1, field_length, field_length + 10])
        y = pd.Series([0, field_width / 2 + 1, field_width, field_width + 10])
        cnt = _grid._count(x, y, self.N, self.M)
        np.testing.assert_array_equal(cnt, [[1, 2], [1, 0]])


def test_get_move_actions(spadl_actions: pd.DataFrame) -> None:
    """It should filter passes, dribbles and crosses."""
    move_actions = _grid._get_move_actions(spadl_actions)
    assert move_actions.type_id.isin(
        [
            spadl.config.actiontypes.index("pass"),
            spadl.config.actiontypes.index("dribble"),
            spadl.config.actiontypes.index("cross"),
        ]
    ).all()


def test_get_successful_move_actions(spadl_actions: pd.DataFrame) -> None:
    """It should filter successful passes, dribbles and crosses."""
    move_actions = _grid._get_successful_move_actions(spadl_actions)
    assert move_actions.type_id.isin(
        [
            spadl.config.actiontypes.index("pass"),
            spadl.config.actiontypes.index("dribble"),
            spadl.config.actiontypes.index("cross"),
        ]
    ).all()
    assert (move_actions.result_id == spadl.config.results.index("success")).all()


def test_action_prob(spadl_actions: pd.DataFrame) -> None:
    """It should return the proportion of shots and moves for each cell."""
    shot_prob, move_prob = _grid._action_prob(spadl_actions, 10, 5)
    assert shot_prob.shape == (5, 10)
    assert move_prob.shape == (5, 10)
    assert np.any(shot_prob > 0)
    assert np.any(move_prob > 0)
    assert np.all(((move_prob + shot_prob) == 1) | ((move_prob + shot_prob) == 0))


def test_scoring_prob(spadl_actions: pd.DataFrame) -> None:
    """It should return the proportion of successful shots for each cell."""
    shots = spadl_actions.type_id == spadl.config.actiontypes.index("shot")
    goals = shots & (spadl_actions.result_id == spadl.config.results.index("success"))
    scoring_prob = _grid._scoring_prob(spadl_actions, 1, 1)
    assert scoring_prob.shape == (1, 1)
    assert sum(goals) / sum(shots) == scoring_prob[0]


def test_move_transition_matrix() -> None:
    """It should return the move transition matrix."""
    pass_id = spadl.config.actiontypes.index("pass")
    success_id = spadl.config.results.index("success")
    spadl_actions = pd.DataFrame(
        [
            {
                "game_id": 1,
                "original_event_id": "a",
                "action_id": 1,
                "period_id": 1,
                "time_seconds": 1.0,
                "team_id": 1,
                "player_id": 1,
                "start_x": 10.0,
                "end_x": 10.0,
                "start_y": 10.0,
                "end_y": 10.0,
                "bodypart_id": 1,
                "type_id": pass_id,
                "result_id": success_id,
            },
            {
                "game_id": 1,
                "original_event_id": "a",
                "action_id": 2,
                "period_id": 1,
                "time_seconds": 1.2,
                "team_id": 1,
                "player_id": 1,
                "start_x": 10.0,
                "end_x": 10.0,
                "start_y": 10.0,
                "end_y": 10.0,
                "bodypart_id": 1,
                "type_id": pass_id,
                "result_id": success_id,
            },
        ]
    )
    move_mat = singh_transition_matrix(spadl_actions, GridSpec(n_zones_x=2, n_zones_y=2))
    assert np.sum(move_mat) == 1
    assert move_mat.shape == (4, 4)
    # (10, 10) is mapped to flat index 2 in a 2x2 grid
    assert move_mat[2, 2] == 1


class TestNaNCoordinates:
    """NaN coordinates in move actions must not crash _get_cell_indexes.

    Real-world providers (Metrica, Sportec/IDSSE) produce NaN coordinates
    on certain action types. These must be silently skipped during fitting
    and rating, not raise IntCastingNaNError.
    """

    @pytest.fixture()
    def actions_with_nan(self) -> pd.DataFrame:
        pass_id = spadl.config.actiontypes.index("pass")
        shot_id = spadl.config.actiontypes.index("shot")
        success_id = spadl.config.results.index("success")
        fail_id = spadl.config.results.index("fail")
        return pd.DataFrame(
            {
                "game_id": [1] * 5,
                "period_id": [1] * 5,
                "action_id": list(range(5)),
                "time_seconds": [0.0, 1.0, 2.0, 3.0, 4.0],
                "team_id": [1, 1, 1, 1, 2],
                "player_id": [101, 102, 103, 104, 201],
                "start_x": [10.0, None, 50.0, 80.0, 90.0],
                "start_y": [34.0, None, 34.0, 34.0, 34.0],
                "end_x": [20.0, 60.0, 70.0, None, 100.0],
                "end_y": [34.0, 34.0, 34.0, None, 34.0],
                "type_id": [pass_id, pass_id, pass_id, pass_id, shot_id],
                "result_id": [success_id, success_id, success_id, fail_id, success_id],
                "bodypart_id": [0] * 5,
            }
        )

    def test_fit_with_nan_coordinates(self, actions_with_nan: pd.DataFrame) -> None:
        """fit() must not crash when move actions have NaN coordinates."""
        model = xt.ExpectedThreat(l=4, w=3)
        model.fit(actions_with_nan)
        assert model.transition_matrix is not None
        assert np.isfinite(model.xT).all()

    def test_rate_with_nan_coordinates(self, actions_with_nan: pd.DataFrame) -> None:
        """rate() must assign NaN to move actions with NaN coordinates."""
        model = xt.ExpectedThreat(l=4, w=3)
        model.fit(actions_with_nan)
        ratings = model.rate(actions_with_nan)
        assert len(ratings) == len(actions_with_nan)
        # Action 0 (pass, success, valid coords) should be rated
        assert np.isfinite(ratings[0])
        # Action 1 (pass, success, NaN start) should be NaN
        assert np.isnan(ratings[1])
        # Action 2 (pass, success, valid coords) should be rated
        assert np.isfinite(ratings[2])
        # Action 3 (pass, fail) and action 4 (shot) are not successful moves → NaN
        assert np.isnan(ratings[3])
        assert np.isnan(ratings[4])


def test_xt_model_init() -> None:
    """It should initialize all instance variables."""
    xTModel = xt.ExpectedThreat(l=8, w=6, eps=1e-3)
    assert xTModel.l == 8
    assert xTModel.w == 6
    assert xTModel.eps == 1e-3
    assert np.sum(xTModel.xT) == 0
    assert xTModel.scoring_prob_matrix is None
    assert xTModel.scoring_prob_matrix is None
    assert xTModel.shot_prob_matrix is None
    assert xTModel.move_prob_matrix is None
    assert xTModel.transition_matrix is None
    assert len(xTModel.heatmaps) == 0


def test_xt_model_fit(spadl_actions: pd.DataFrame) -> None:
    """It should update all instance variables."""
    xTModel = xt.ExpectedThreat()
    xTModel.fit(spadl_actions)
    assert xTModel.scoring_prob_matrix is not None
    assert xTModel.shot_prob_matrix is not None
    assert xTModel.move_prob_matrix is not None
    assert xTModel.transition_matrix is not None
    assert len(xTModel.heatmaps) > 0
    assert np.sum(xTModel.xT) > 0


def test_xt_model_rate_not_fitted(spadl_actions: pd.DataFrame) -> None:
    """It should raise a NotFittedError."""
    xTModel = xt.ExpectedThreat()
    with pytest.raises(NotFittedError):
        xTModel.rate(spadl_actions)


def test_xt_model_rate(spadl_actions: pd.DataFrame) -> None:
    """It should rate all successful move actions and assign all other actions NaN."""
    xTModel = xt.ExpectedThreat()
    xTModel.fit(spadl_actions)
    successful_move_actions_idx = _grid._get_successful_move_actions(spadl_actions).index
    ratings = xTModel.rate(spadl_actions)
    assert ratings.shape == (len(spadl_actions),)
    assert np.all(~np.isnan(ratings[successful_move_actions_idx]))
    assert np.all(np.isnan(np.delete(ratings, successful_move_actions_idx)))


def test_interpolate_xt_grid_no_scipy(mocker: MockerFixture) -> None:
    """It should raise an ImportError if scipy is not installed."""
    mocker.patch("silly_kicks.xthreat._model.RectBivariateSpline", None)
    xTModel = xt.ExpectedThreat()
    with pytest.raises(ImportError, match=r"Interpolation requires scipy to be installed\."):
        xTModel.interpolator()


@pytest.fixture(scope="session")
def xt_model(sb_worldcup_data: pd.HDFStore) -> xt.ExpectedThreat:
    """Test the xT framework on the StatsBomb World Cup data."""
    # 1. Load a set of actions to train the model on
    df_games = cast(pd.DataFrame, sb_worldcup_data["games"]).set_index("game_id")
    # 2. Convert direction of play
    actions_ltr = cast(
        pd.DataFrame,
        pd.concat(
            [
                spadl.play_left_to_right(
                    cast(pd.DataFrame, sb_worldcup_data[f"actions/game_{game_id}"]),
                    game.home_team_id,
                )
                for game_id, game in df_games.iterrows()
            ]
        ),
    )
    # 3. Train xT model
    xTModel = xt.ExpectedThreat(l=16, w=12)
    xTModel.fit(actions_ltr)
    return xTModel


def test_predict(sb_worldcup_data: pd.HDFStore, xt_model: xt.ExpectedThreat) -> None:
    games = cast(pd.DataFrame, sb_worldcup_data["games"])
    game = games.iloc[-1]
    actions = cast(pd.DataFrame, sb_worldcup_data[f"actions/game_{game.game_id}"])
    ratings = xt_model.rate(actions)
    assert ratings.dtype is np.dtype(np.float64)
    assert len(ratings) == len(actions)


def test_predict_with_interpolation(sb_worldcup_data: pd.HDFStore, xt_model: xt.ExpectedThreat) -> None:
    games = cast(pd.DataFrame, sb_worldcup_data["games"])
    game = games.iloc[-1]
    actions = cast(pd.DataFrame, sb_worldcup_data[f"actions/game_{game.game_id}"])
    ratings = xt_model.rate(actions, use_interpolation=True)
    assert ratings.dtype is np.dtype(np.float64)
    assert len(ratings) == len(actions)


def test_singh_path_byte_identical_to_legacy(spadl_actions: pd.DataFrame) -> None:
    """Default ExpectedThreat (Singh, 16x12) must reproduce the pre-refactor output exactly."""
    import tests.xthreat_legacy_reference as legacy

    new = xt.ExpectedThreat().fit(spadl_actions)
    old = legacy.ExpectedThreat().fit(spadl_actions)
    np.testing.assert_array_equal(new.xT, old.xT)
    np.testing.assert_array_equal(new.transition_matrix, old.transition_matrix)
    np.testing.assert_array_equal(new.scoring_prob_matrix, old.scoring_prob_matrix)
    np.testing.assert_array_equal(new.shot_prob_matrix, old.shot_prob_matrix)
    np.testing.assert_array_equal(new.move_prob_matrix, old.move_prob_matrix)


def test_singh_path_byte_identical_on_worldcup(sb_worldcup_data: pd.HDFStore) -> None:
    """Same, on a real multi-match corpus, including rate() output."""
    import tests.xthreat_legacy_reference as legacy
    from tests._xthreat_helpers import _worldcup_ltr

    actions = _worldcup_ltr(sb_worldcup_data)
    new = xt.ExpectedThreat(l=16, w=12).fit(actions)
    old = legacy.ExpectedThreat(l=16, w=12).fit(actions)
    np.testing.assert_array_equal(new.xT, old.xT)
    np.testing.assert_array_equal(new.transition_matrix, old.transition_matrix)
    last = cast(pd.DataFrame, sb_worldcup_data["games"]).iloc[-1]
    acts = cast(pd.DataFrame, sb_worldcup_data[f"actions/game_{last.game_id}"])
    np.testing.assert_array_equal(new.rate(acts), old.rate(acts))
