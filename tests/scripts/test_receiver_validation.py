"""Task 4: failure-mode classification + trajectory-weak-labelled failed-pass validation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts._receiver_validation import (  # type: ignore
    _R,
    _T,
    R1_CAVEAT,
    classify_failure_mode,
    receiver_failed_pass_accuracy,
    trajectory_weak_labels,
)
from silly_kicks.tracking._receiver import ReceiverModel

_ACT_COLS = [
    "action_id",
    "game_id",
    "period_id",
    "time_seconds",
    "team_id",
    "player_id",
    "type_id",
    "result_id",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
]


def _actions() -> pd.DataFrame:
    P, TI = _T["pass"], _T["throw_in"]
    F, S = _R["fail"], _R["success"]
    rows = [
        (1, 1, 1, 10.0, 1, 9, P, F, 50, 34, 62, 34),  # intercepted failed pass (travel 12)
        (2, 1, 1, 11.0, 2, 20, P, S, 62, 34, 40, 34),  # opponent recovery -> aid1 = intercepted
        (3, 1, 1, 20.0, 1, 9, P, S, 30, 30, 45, 40),  # completed pass -> other
        (4, 1, 1, 30.0, 1, 9, P, F, 30, 30, 31, 30),  # foot-blocked failed pass (travel 1)
        (5, 1, 1, 31.0, 2, 21, P, S, 31, 30, 50, 30),  # opponent recovery -> aid4 intercepted, NOT covered
        (6, 1, 1, 40.0, 1, 9, P, F, 40, 60, 40, 68),  # out failed pass
        (7, 1, 1, 41.0, 2, 22, TI, S, 40, 68, 45, 60),  # throw_in -> aid6 = out
    ]
    return pd.DataFrame(rows, columns=_ACT_COLS)


def _frame_100() -> pd.DataFrame:
    rows = [
        (True, pd.NA, pd.NA, False, 50.0, 34.0, 10.0, 0.0),
        (False, 9, 1, False, 50.0, 34.0, 0.0, 0.0),
        (False, 10, 1, False, 75.0, 34.0, 0.0, 0.0),  # on the release ray -> the weak label
        (False, 11, 1, False, 55.0, 50.0, 0.0, 0.0),  # open, off-ray
        (False, 20, 2, False, 62.0, 34.0, 0.0, 0.0),  # interceptor
        (False, 30, 2, True, 100.0, 34.0, 0.0, 0.0),
    ]
    df = pd.DataFrame(rows, columns=["is_ball", "player_id", "team_id", "is_goalkeeper", "x", "y", "vx", "vy"])
    df["game_id"], df["period_id"], df["frame_id"] = 1, 1, 100
    return df.astype({"player_id": "Int64", "team_id": "Int64"})


def _links() -> pd.DataFrame:
    return pd.DataFrame({"action_id": [1], "frame_id": [100]})


def _fitted_model() -> ReceiverModel:
    intended = pd.DataFrame({"ball_dist": [15.0] * 12, "lane_pressure": [0.0] * 12, "space": [12.0] * 12})
    others = pd.DataFrame({"ball_dist": [15.0] * 12, "lane_pressure": [1.5] * 12, "space": [4.0] * 12})
    X = pd.concat([intended, others], ignore_index=True)
    return ReceiverModel("public").fit(X, np.array([1] * 12 + [0] * 12))


def test_classify_failure_mode():
    m = dict(zip(_actions()["action_id"], classify_failure_mode(_actions()), strict=False))
    assert m[1] == "intercepted"  # next action is an opponent open-play recovery
    assert m[4] == "intercepted"
    assert m[6] == "out"  # next action is a throw_in
    assert m[3] == "other"  # a completed pass is never a failure mode


def test_classify_skips_non_action_noise_row_for_the_next_touch():
    """M1: the next TOUCH, not the raw next row. Here the immediate next row is an opponent non_action
    (GS noise, not a touch), but the next real touch is the SAME team -> 'other', not a false
    'intercepted'. A raw shift(-1) would tag it intercepted off the opponent-labelled noise row."""
    P, NA_, F, S = _T["pass"], _T["non_action"], _R["fail"], _R["success"]
    rows = [
        (1, 1, 1, 10.0, 1, 9, P, F, 50, 34, 60, 34),  # failed pass, team 1
        (2, 1, 1, 10.5, 2, 20, NA_, S, 60, 34, 60, 34),  # opponent NON_ACTION noise (skipped)
        (3, 1, 1, 11.0, 1, 9, P, S, 58, 34, 70, 34),  # same-team next TOUCH -> not intercepted
    ]
    assert classify_failure_mode(pd.DataFrame(rows, columns=_ACT_COLS)).loc[1] == "other"


def test_classify_does_not_cross_period_boundary():
    """M2: a failed pass that is its period's LAST touch must not borrow the next period's first action.
    aid1 is period-1's last touch; period 2 opens with the opponent -> 'other', not a boundary-crossing
    false 'intercepted'."""
    P, F, S = _T["pass"], _R["fail"], _R["success"]
    rows = [
        (1, 1, 1, 40.0, 1, 9, P, F, 50, 34, 60, 34),  # failed pass, LAST touch of period 1
        (2, 1, 2, 1.0, 2, 20, P, S, 52, 34, 40, 34),  # period-2 kickoff by the opponent
    ]
    assert classify_failure_mode(pd.DataFrame(rows, columns=_ACT_COLS)).loc[1] == "other"


def test_trajectory_weak_labels_cover_the_clear_case_only():
    labels = trajectory_weak_labels(_actions(), _frame_100(), links=_links())
    by = labels.set_index("action_id")
    assert bool(by.loc[1, "covered"]) is True and by.loc[1, "weak_receiver_id"] == "10"
    assert bool(by.loc[4, "covered"]) is False  # foot-blocked: travel < min


def test_receiver_failed_pass_accuracy_reports_upper_bound():
    acc = receiver_failed_pass_accuracy(_fitted_model(), _actions(), _frame_100(), links=_links())
    assert acc["n_intercepted"] == 2 and acc["n_covered"] == 1 and acc["coverage"] == 0.5
    assert acc["n_scored"] == 1
    assert acc["top1_proxy"] == 1.0  # the geometric proxy picks the on-ray teammate = the weak label
    assert acc["r1_caveat"] == R1_CAVEAT
