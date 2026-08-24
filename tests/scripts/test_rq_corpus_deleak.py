"""Task 8: the de-leaked, failure-mode-conditional failed-pass target in extract_played_passes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts._receiver_validation import _R, _T
from scripts._rq_corpus import extract_played_passes
from silly_kicks.tracking._receiver import ReceiverModel

_COLS = [
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
    P, S, F = _T["pass"], _R["success"], _R["fail"]
    rows = [
        (1, 1, 1, 10.0, 1, 9, P, F, 50, 34, 62, 34),  # intercepted failed pass
        (2, 1, 1, 11.0, 2, 20, P, S, 62, 34, 40, 34),  # opponent recovery -> aid1 = intercepted
    ]
    return pd.DataFrame(rows, columns=_COLS)


def _frame() -> pd.DataFrame:
    rows = [
        (True, pd.NA, pd.NA, False, 50.0, 34.0, 10.0, 0.0),
        (False, 9, 1, False, 50.0, 34.0, 0.0, 0.0),
        (False, 10, 1, False, 75.0, 34.0, 0.0, 0.0),
        (False, 11, 1, False, 55.0, 50.0, 0.0, 0.0),
        (False, 20, 2, False, 62.0, 34.0, 0.0, 0.0),
        (False, 30, 2, True, 100.0, 34.0, 0.0, 0.0),
    ]
    df = pd.DataFrame(rows, columns=["is_ball", "player_id", "team_id", "is_goalkeeper", "x", "y", "vx", "vy"])
    df["game_id"], df["period_id"], df["frame_id"] = 1, 1, 100
    df["team_attacking_direction"] = "ltr"
    return df.astype({"player_id": "Int64", "team_id": "Int64"})


def _links() -> pd.DataFrame:
    return pd.DataFrame({"action_id": [1, 2], "frame_id": [100, 100]})


def _model() -> ReceiverModel:
    intended = pd.DataFrame({"ball_dist": [15.0] * 12, "lane_pressure": [0.0] * 12, "space": [12.0] * 12})
    others = pd.DataFrame({"ball_dist": [15.0] * 12, "lane_pressure": [1.5] * 12, "space": [4.0] * 12})
    X = pd.concat([intended, others], ignore_index=True)
    return ReceiverModel("public").fit(X, np.array([1] * 12 + [0] * 12))


def test_model_none_keeps_the_leaked_end_xy():
    out = extract_played_passes(_actions(), _frame(), links=_links(), model=None)
    a1 = out[out["action_id"] == 1].iloc[0]
    assert bool(a1["is_fail"]) and a1["target_source"] == "end_xy"  # 4.87.0 behaviour preserved
    # L2: assert the COORDS too, not just the source -- team 1 attacks ltr so end_xy is unreflected;
    # a reflection regression in the model=None path would pass a source-only check.
    assert (a1["target_x"], a1["target_y"]) == (62.0, 34.0)


def test_model_deleaks_intercepted_to_intended_receiver():
    out = extract_played_passes(_actions(), _frame(), links=_links(), model=_model())
    a1 = out[out["action_id"] == 1].iloc[0]
    assert bool(a1["is_fail"]) and a1["target_source"] == "intended_receiver"
    assert (a1["target_x"], a1["target_y"]) != (62.0, 34.0)  # NOT the leaked interception point
