import pandas as pd

from silly_kicks.gk_decision import SkillCornerGIOptionSet
from silly_kicks.gk_decision._columns import OPTION_ROW_COLUMNS


def _parsed():
    # possession by keeper 99 (in keeper_ids); one by outfielder 50 (excluded)
    base = dict(game_id="555", period_id=1, target_x=0.0, target_y=0.0, opponents_bypassed=1.0)
    rows = [
        dict(
            decision_id="p1", possessor_id=99, team_id=7, target_player_id=11, is_chosen=False, completion=0.98, **base
        ),
        dict(
            decision_id="p1", possessor_id=99, team_id=7, target_player_id=12, is_chosen=True, completion=0.66, **base
        ),
        dict(
            decision_id="p1", possessor_id=99, team_id=7, target_player_id=13, is_chosen=False, completion=0.99, **base
        ),
        dict(decision_id="p9", possessor_id=50, team_id=7, target_player_id=11, is_chosen=True, completion=0.9, **base),
    ]
    return pd.DataFrame(rows)


def test_gi_optionset_uniform_schema_and_gk_filter():
    os_ = SkillCornerGIOptionSet(_parsed(), keeper_ids=[99])
    rows = os_.option_rows()
    assert list(rows.columns) == list(OPTION_ROW_COLUMNS)
    assert set(rows["decision_id"]) == {"p1"}  # outfielder possession p9 excluded
    assert (rows["option_set_source"] == "native").all()
    assert rows["is_chosen"].sum() == 1
