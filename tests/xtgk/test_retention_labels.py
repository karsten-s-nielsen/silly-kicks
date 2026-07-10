import numpy as np
import pandas as pd

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.xtgk._retention_labels import retains

PASS = spadlconfig.actiontype_id["pass"]
SHOT = spadlconfig.actiontype_id["shot"]
GOALKICK = spadlconfig.actiontype_id["goalkick"]
SUCCESS = spadlconfig.result_id["success"]
FAIL = spadlconfig.result_id["fail"]


def _row(aid, t, team, typ, res, pid):
    return dict(
        game_id=1,
        period_id=1,
        action_id=aid,
        time_seconds=t,
        team_id=team,
        player_id=1,
        type_id=typ,
        result_id=res,
        possession_id=pid,
        start_x=5.0,
        start_y=34.0,
        end_x=20.0,
        end_y=34.0,
    )


def test_retained_when_team_keeps_ball_through_window():
    # window 1.5s is fully covered by the 2s of data -> observed retention -> 1.0 (not NaN)
    a = pd.DataFrame(
        [
            _row(0, 0.0, 10, GOALKICK, SUCCESS, 0),
            _row(1, 1.0, 10, PASS, SUCCESS, 0),
            _row(2, 2.0, 10, PASS, SUCCESS, 0),
        ]
    )
    out = retains(a, window_seconds=1.5)
    assert out.iloc[0] == 1.0


def test_lost_when_opponent_takes_over_in_window():
    a = pd.DataFrame(
        [
            _row(0, 0.0, 10, GOALKICK, SUCCESS, 0),
            _row(1, 1.0, 20, PASS, SUCCESS, 1),  # opponent possession
        ]
    )
    out = retains(a, window_seconds=10.0)
    assert out.iloc[0] == 0.0


def test_retained_when_team_shoots_in_window():
    a = pd.DataFrame(
        [
            _row(0, 0.0, 10, GOALKICK, SUCCESS, 0),
            _row(1, 1.0, 10, SHOT, FAIL, 0),
        ]
    )
    out = retains(a, window_seconds=10.0)  # decisive shot -> 1.0 regardless of truncation
    assert out.iloc[0] == 1.0


def test_truncated_window_with_no_decisive_event_is_nan():
    # a lone goal-kick near a period end: the 10s window is truncated to 0s of observable data and
    # nothing decisive happens -> we did NOT observe retention -> NaN (excluded from training).
    a = pd.DataFrame([_row(0, 2699.0, 10, GOALKICK, SUCCESS, 0)])
    out = retains(a, window_seconds=10.0)
    assert np.isnan(out.iloc[0])
