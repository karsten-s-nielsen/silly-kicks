"""Direct unit tests for the shared adapter orientation tail (TF-23b, ADR-035)."""

import pandas as pd
import pytest

from silly_kicks.tracking import direction


def _raw(period, team, player, isgk, isball, x, y):
    return {
        "game_id": "g1",
        "period_id": period,
        "frame_id": period * 10,
        "is_ball": isball,
        "is_goalkeeper": isgk,
        "team_id": team,
        "player_id": player,
        "x": x,
        "y": y,
    }


def _p1_frame():
    # home GK deep at low x (=20); home attacks right under home_team_start_left=True.
    return pd.DataFrame(
        [
            _raw(1, "H", "hgk", True, False, 20.0, 34.0),
            _raw(1, "A", "agk", True, False, 85.0, 34.0),
            _raw(1, None, None, False, True, 50.0, 34.0),
        ]
    )


def test_finalize_correct_flag_labels_and_noop():
    out = direction.finalize_orientation(
        _p1_frame(),
        home_team_id="H",
        home_team_start_left=True,
        home_team_start_left_extratime=None,
        source="test",
    )
    hgk = out[(out.period_id == 1) & (out.player_id == "hgk")].iloc[0]
    assert hgk.x == pytest.approx(20.0)  # correct flag => no flip, no backstop
    assert hgk.team_attacking_direction == "ltr"  # period-gated label


def test_finalize_does_not_mutate_input():
    df = _p1_frame()
    before = df.copy(deep=True)
    direction.finalize_orientation(
        df,
        home_team_id="H",
        home_team_start_left=True,
        home_team_start_left_extratime=None,
        source="test",
    )
    pd.testing.assert_frame_equal(df, before)  # copy-at-entry: input untouched


def test_finalize_wrong_et_flag_self_corrects():
    df = pd.DataFrame(
        [
            _raw(3, "H", "hgk", True, False, 20.0, 34.0),  # raw: home GK deep at low x
            _raw(3, "A", "agk", True, False, 85.0, 34.0),
        ]
    )
    # extratime=False flips P3 (home GK -> x=85); the geometric backstop restores it to low x.
    out = direction.finalize_orientation(
        df,
        home_team_id="H",
        home_team_start_left=True,
        home_team_start_left_extratime=False,
        source="test",
    )
    assert out[(out.period_id == 3) & (out.player_id == "hgk")].iloc[0].x == pytest.approx(20.0)
