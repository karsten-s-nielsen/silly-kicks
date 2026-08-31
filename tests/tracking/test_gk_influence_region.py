"""compute_gk_influence region restriction (TF-60 PR2): additive, default byte-identical."""

from silly_kicks.tracking._gk_influence import compute_gk_influence
from tests.tracking.test_compute_threat_pc import HOME_GOAL_MAP, _fitted_xt, _frame


def _reach(region=None):
    # Home (team 1) keeper defends x=0; team 2 attacks toward x=0. gk_player_id resolved from _frame().
    frame = _frame()
    gk_id = frame[(frame["team_id"] == 1) & frame["is_goalkeeper"].astype(bool)]["player_id"].iloc[0]
    return compute_gk_influence(
        frame,
        attacking_team_id=2,
        gk_player_id=gk_id,
        xt=_fitted_xt(),
        goal_map=HOME_GOAL_MAP,
        region=region,
    ).reachable_area_m2


def test_region_none_is_whole_pitch():
    whole = _reach(region=None)
    also_whole = _reach(region=(0.0, 105.0, 0.0, 68.0))
    assert whole == also_whole


def test_region_restriction_never_exceeds_whole_pitch():
    whole = _reach(region=None)
    near_goal = _reach(region=(0.0, 20.0, 0.0, 68.0))
    assert 0.0 <= near_goal <= whole + 1e-9


def test_disjoint_region_is_zero():
    # A keeper defending x=0 has no reachable cells in the far attacking third.
    assert _reach(region=(90.0, 105.0, 0.0, 68.0)) == 0.0
