import pandas as pd
import silly_kicks.atomic.spadl as spadlcfg
import silly_kicks.atomic.spadl.config as atomicspadl
from pandas import testing as tm
from silly_kicks.atomic.vaep import features as fs

xfns = [
    fs.actiontype,
    fs.actiontype_onehot,
    fs.bodypart,
    fs.bodypart_detailed,
    fs.bodypart_onehot,
    fs.bodypart_detailed_onehot,
    fs.team,
    fs.time,
    fs.time_delta,
    fs.location,
    fs.polar,
    fs.movement_polar,
    fs.direction,
    fs.goalscore,
]


def test_actiontype_includes_atomic_types() -> None:
    """Bug #950: actiontype must handle atomic-only action types."""
    from silly_kicks.atomic.vaep.features import actiontype

    receival_id = atomicspadl.actiontype_id["receival"]
    actions = pd.DataFrame({
        "game_id": [1],
        "period_id": [1],
        "action_id": [0],
        "type_id": [receival_id],
    })
    result = actiontype(actions)
    # @simple wraps the column name as "actiontype_a0"
    assert "actiontype_a0" in result.columns
    assert result["actiontype_a0"].iloc[0] == "receival"


def test_same_index(atomic_spadl_actions: pd.DataFrame) -> None:
    """The feature generators should not change the index of the input dataframe."""
    atomic_spadl_actions.index += 10
    game_actions_with_names = spadlcfg.add_names(atomic_spadl_actions)
    gamestates = fs.gamestates(game_actions_with_names, 3)
    gamestates = fs.play_left_to_right(gamestates, 782)
    for fn in xfns:
        features = fn(gamestates)
        tm.assert_index_equal(features.index, atomic_spadl_actions.index)
