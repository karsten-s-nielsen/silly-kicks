import pandas as pd
from pandas import testing as tm

import silly_kicks.spadl as spadl
import silly_kicks.spadl as spadlcfg
import silly_kicks.spadl.utils as spu
from silly_kicks.vaep import features as fs

xfns = [
    fs.actiontype,
    fs.actiontype_onehot,
    fs.result,
    fs.result_onehot,
    fs.actiontype_result_onehot,
    fs.bodypart,
    fs.bodypart_detailed,
    fs.bodypart_onehot,
    fs.bodypart_detailed_onehot,
    fs.time,
    fs.startlocation,
    fs.endlocation,
    fs.startpolar,
    fs.endpolar,
    fs.movement,
    fs.team,
    fs.time_delta,
    fs.space_delta,
    fs.goalscore,
]


def test_same_index(spadl_actions: pd.DataFrame) -> None:
    """The feature generators should not change the index of the input dataframe."""
    spadl_actions = spadl_actions.set_index(spadl_actions.index + 10)
    game_actions_with_names = spadlcfg.add_names(spadl_actions)
    gamestates = fs.gamestates(game_actions_with_names, 3)
    gamestates = fs.play_left_to_right(gamestates, 782)
    for fn in xfns:
        features = fn(gamestates)
        tm.assert_index_equal(features.index, spadl_actions.index)


def test_actiontype(spadl_actions: pd.DataFrame) -> None:
    gamestates = fs.gamestates(spadl_actions)
    ltr_gamestates = fs.play_left_to_right(gamestates, 782)
    out = fs.actiontype(ltr_gamestates)
    assert out.shape == (len(spadl_actions), 3)


def test_actiontype_onehot(spadl_actions: pd.DataFrame) -> None:
    gamestates = fs.gamestates(spadl_actions)
    ltr_gamestates = fs.play_left_to_right(gamestates, 782)
    out = fs.actiontype_onehot(ltr_gamestates)
    assert out.shape == (len(spadl_actions), len(spadl.config.actiontypes) * 3)


def test_result(spadl_actions: pd.DataFrame) -> None:
    gamestates = fs.gamestates(spadl_actions)
    ltr_gamestates = fs.play_left_to_right(gamestates, 782)
    out = fs.result(ltr_gamestates)
    assert out.shape == (len(spadl_actions), 3)


def test_result_onehot(spadl_actions: pd.DataFrame) -> None:
    gamestates = fs.gamestates(spadl_actions)
    ltr_gamestates = fs.play_left_to_right(gamestates, 782)
    out = fs.result_onehot(ltr_gamestates)
    assert out.shape == (len(spadl_actions), len(spadl.config.results) * 3)


def test_actiontype_result_onehot(spadl_actions: pd.DataFrame) -> None:
    gamestates = fs.gamestates(spadl_actions)
    ltr_gamestates = fs.play_left_to_right(gamestates, 782)
    out = fs.actiontype_result_onehot(ltr_gamestates)
    assert out.shape == (
        len(spadl_actions),
        len(spadl.config.actiontypes) * len(spadl.config.results) * 3,
    )


def test_bodypart(spadl_actions: pd.DataFrame) -> None:
    gamestates = fs.gamestates(spadl_actions)
    ltr_gamestates = fs.play_left_to_right(gamestates, 782)
    out = fs.bodypart(ltr_gamestates)
    assert out.shape == (len(spadl_actions), 3)


def test_bodypart_onehot(spadl_actions: pd.DataFrame) -> None:
    gamestates = fs.gamestates(spadl_actions)
    ltr_gamestates = fs.play_left_to_right(gamestates, 782)
    out = fs.bodypart_onehot(ltr_gamestates)
    assert out.shape == (len(spadl_actions), 4 * 3)


def test_time(spadl_actions: pd.DataFrame) -> None:
    gamestates = fs.gamestates(spadl_actions)
    out = fs.time(gamestates)
    assert out.shape == (len(spadl_actions), 9)
    assert out.loc[0, "period_id_a0"] == 1
    assert out.loc[0, "time_seconds_a0"] == 0.533
    assert out.loc[0, "time_seconds_overall_a0"] == 0.533
    assert out.loc[200, "period_id_a0"] == 2
    assert out.loc[200, "time_seconds_a0"] == 0.671
    assert out.loc[200, "time_seconds_overall_a0"] == 0.671 + 45 * 60


def test_player_possession_time(spadl_actions: pd.DataFrame) -> None:
    gamestates = fs.gamestates(spadl_actions)
    out = fs.player_possession_time(gamestates)
    assert out.shape == (len(spadl_actions), len(gamestates))
    assert "player_possession_time_a0" in out.columns
    assert out.loc[0, "player_possession_time_a0"] == 0.0
    assert out.loc[1, "player_possession_time_a0"] == 0.0
    assert out.loc[2, "player_possession_time_a0"] == 0.881


def test_time_delta(spadl_actions: pd.DataFrame) -> None:
    gamestates = fs.gamestates(spadl_actions)
    out = fs.time_delta(gamestates)
    assert out.shape == (len(spadl_actions), 2)
    # Start of H1
    print(out)
    assert out.loc[0, "time_delta_1"] == 0.0
    assert out.loc[0, "time_delta_2"] == 0.0
    assert out.loc[1, "time_delta_1"] == 0.719
    assert out.loc[1, "time_delta_2"] == 0.719
    assert out.loc[2, "time_delta_1"] == 0.881
    assert out.loc[2, "time_delta_2"] == 1.6
    # Start of H2
    assert out.loc[200, "time_delta_1"] == 0.0
    assert out.loc[200, "time_delta_2"] == 0.0
    assert out.loc[201, "time_delta_1"] == 1.32
    assert out.loc[201, "time_delta_2"] == 1.32


def test_gamestates_empty_dataframe() -> None:
    """Bug #507: gamestates should not crash on empty input."""
    empty_actions = pd.DataFrame(
        columns=[
            "game_id",
            "period_id",
            "action_id",
            "time_seconds",
            "team_id",
            "player_id",
            "start_x",
            "start_y",
            "end_x",
            "end_y",
            "type_id",
            "result_id",
            "bodypart_id",
            "type_name",
            "result_name",
            "bodypart_name",
        ]
    )
    result = fs.gamestates(empty_actions, nb_prev_actions=3)
    assert len(result) == 3
    for gs in result:
        assert len(gs) == 0


def test_result_onehot_prev_only(spadl_actions: pd.DataFrame) -> None:
    spadl_actions = spu.add_names(spadl_actions)
    gamestates = fs.gamestates(spadl_actions, nb_prev_actions=3)
    gamestates = fs.play_left_to_right(gamestates, spadl_actions.iloc[0].team_id)
    result = fs.result_onehot_prev_only(gamestates)
    a0_cols = [c for c in result.columns if c.endswith("_a0")]
    a1_cols = [c for c in result.columns if c.endswith("_a1")]
    assert len(a0_cols) == 0, f"Found a0 columns: {a0_cols}"
    assert len(a1_cols) > 0


def test_actiontype_result_onehot_prev_only(spadl_actions: pd.DataFrame) -> None:
    spadl_actions = spu.add_names(spadl_actions)
    gamestates = fs.gamestates(spadl_actions, nb_prev_actions=3)
    gamestates = fs.play_left_to_right(gamestates, spadl_actions.iloc[0].team_id)
    result = fs.actiontype_result_onehot_prev_only(gamestates)
    a0_cols = [c for c in result.columns if c.endswith("_a0")]
    a1_cols = [c for c in result.columns if c.endswith("_a1")]
    assert len(a0_cols) == 0
    assert len(a1_cols) > 0


def test_cross_zone(spadl_actions: pd.DataFrame):
    spadl_actions = spu.add_names(spadl_actions)
    gamestates = fs.gamestates(spadl_actions, nb_prev_actions=2)
    gamestates = fs.play_left_to_right(gamestates, spadl_actions.iloc[0].team_id)
    result = fs.cross_zone(gamestates)
    assert result.shape[0] == len(spadl_actions)
    # Should have 4 zone columns per gamestate
    zone_cols = [c for c in result.columns if "cross_zone" in c]
    assert len(zone_cols) >= 4


def test_assist_type(spadl_actions: pd.DataFrame):
    spadl_actions = spu.add_names(spadl_actions)
    gamestates = fs.gamestates(spadl_actions, nb_prev_actions=3)
    gamestates = fs.play_left_to_right(gamestates, spadl_actions.iloc[0].team_id)
    result = fs.assist_type(gamestates)
    assert result.shape[0] == len(spadl_actions)
    assist_cols = [c for c in result.columns if "assist_" in c]
    assert len(assist_cols) == 6


def test_actiontype_result_onehot_vectorized(spadl_actions: pd.DataFrame) -> None:
    spadl_actions = spu.add_names(spadl_actions)
    gamestates = fs.gamestates(spadl_actions, nb_prev_actions=2)
    gamestates = fs.play_left_to_right(gamestates, spadl_actions.iloc[0].team_id)
    result = fs.actiontype_result_onehot(gamestates)
    assert any("actiontype_pass_result_success" in c for c in result.columns)
    assert result.shape[0] == len(spadl_actions)


def _goal_rows(rows: list[tuple[int, str, str, int]]) -> pd.DataFrame:
    """Minimal actions from (team_id, type_name, result_name, period_id), one second apart."""
    return pd.DataFrame(
        {
            "team_id": [r[0] for r in rows],
            "type_name": [r[1] for r in rows],
            "type_id": [spadl.config.actiontype_id[r[1]] for r in rows],
            "result_name": [r[2] for r in rows],
            "result_id": [spadl.config.result_id[r[2]] for r in rows],
            "period_id": [r[3] for r in rows],
            "time_seconds": [float(i) for i in range(len(rows))],
        }
    )


def test_goalscore_counts_an_own_goal_for_the_other_team() -> None:
    # Converters emit an own goal as a bad_touch row with the owngoal result (ADR-018).
    actions = _goal_rows([(1, "bad_touch", "owngoal", 1), (2, "pass", "success", 1), (1, "pass", "success", 1)])
    out = fs.goalscore([actions])
    assert out["goalscore_team"].tolist() == [0, 1, 0]
    assert out["goalscore_opponent"].tolist() == [0, 0, 1]


def test_goalscore_counts_no_goal_of_the_shootout() -> None:
    # Period 5 is the penalty shoot-out: each of its rows holds the score at the end of period 4.
    actions = _goal_rows([(1, "shot", "success", 1), (2, "shot_penalty", "success", 5), (1, "pass", "success", 5)])
    out = fs.goalscore([actions])
    assert out["goalscore_team"].tolist() == [0, 0, 1]
    assert out["goalscore_opponent"].tolist() == [0, 1, 0]
