import pandas as pd

from silly_kicks.tracking.defensive_credit._params import DefensiveCreditParams
from silly_kicks.tracking.defensive_credit._resolution import resolve_responsible_defenders
from tests.tracking._defensive_credit_fixtures import frame_with_defender, one_action


def _ctx(defender_x):
    actions = one_action(start_x=95.0, start_y=34.0)
    frames = frame_with_defender(defender_x=defender_x, defender_y=34.0)
    return actions, frames


def test_defender_within_threshold_is_returned():
    actions, frames = _ctx(defender_x=96.0)  # 1.0 m from anchor (95,34), inside outside-box radius 4.5
    res = resolve_responsible_defenders(
        actions,
        frames,
        anchor_x=95.0,
        anchor_y=34.0,
        acting_team_id=10,
        mode="nearest",
        params=DefensiveCreditParams(),
    )
    assert list(res["player_id"]) == [900]
    assert list(res["team_id"]) == [20]


def test_defender_outside_threshold_returns_empty():
    actions, frames = _ctx(defender_x=90.0)  # 5.0 m from (95,34) > 4.5
    res = resolve_responsible_defenders(
        actions,
        frames,
        anchor_x=95.0,
        anchor_y=34.0,
        acting_team_id=10,
        mode="nearest",
        params=DefensiveCreditParams(),
    )
    assert res.empty


def test_all_within_beyond_nearest_drops_the_closest():
    actions = one_action(start_x=95.0, start_y=34.0)
    # two defenders within 4.5 m: one at 1 m, one at 2 m
    frames = frame_with_defender(defender_x=96.0)  # 1 m -> player 900 (row 0)
    # add a second defender row by copying the defender row (row 0)
    extra = frames.iloc[[0]].copy()
    extra["player_id"] = 901
    extra["x"] = 97.0  # 2 m
    frames = pd.concat([frames, extra], ignore_index=True)
    res = resolve_responsible_defenders(
        actions,
        frames,
        anchor_x=95.0,
        anchor_y=34.0,
        acting_team_id=10,
        mode="all_within_beyond_nearest",
        params=DefensiveCreditParams(),
    )
    assert set(res["player_id"]) == {901}  # nearest (900) dropped
