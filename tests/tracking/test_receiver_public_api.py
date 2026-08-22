"""Task 5: public receiver surface -- resolve_intended_receiver / intended_receiver_positions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.tracking._receiver import (
    ReceiverModel,
    intended_receiver_positions,
    resolve_intended_receiver,
)

_ACT_COLS = ["action_id", "team_id", "player_id", "start_x", "start_y", "end_x", "end_y"]


def _actions() -> pd.DataFrame:
    return pd.DataFrame([(1, 1, 9, 50.0, 34.0, 62.0, 34.0)], columns=_ACT_COLS)


def _frame() -> pd.DataFrame:
    rows = [
        (True, pd.NA, pd.NA, False, 50.0, 34.0, 10.0, 0.0),
        (False, 9, 1, False, 50.0, 34.0, 0.0, 0.0),
        (False, 10, 1, False, 75.0, 34.0, 0.0, 0.0),  # on the ball-velocity ray
        (False, 11, 1, False, 55.0, 50.0, 0.0, 0.0),  # open, off-ray
        (False, 20, 2, False, 62.0, 34.0, 0.0, 0.0),
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


def test_resolve_intended_receiver_proxy_vs_model():
    proxy = resolve_intended_receiver(_actions(), _frame(), model=None, links=_links())
    assert proxy.loc[1] == "10"  # geometric proxy: on-ray teammate
    model = resolve_intended_receiver(_actions(), _frame(), model=_fitted_model(), links=_links())
    assert model.loc[1] == "11"  # model prefers the open teammate


def test_intended_receiver_positions_carries_source_and_coords():
    pos = intended_receiver_positions(_actions(), _frame(), model=None, links=_links()).set_index("action_id")
    assert pos.loc[1, "source"] == "geometric_proxy"
    assert (pos.loc[1, "x"], pos.loc[1, "y"]) == (75.0, 34.0)  # teammate 10's frame position


def test_resolve_dtype_safe_action_id_mismatched_links():
    """M3 (review, ADR-019): a caller-supplied ``links`` whose action_id dtype differs from ``actions``
    (str vs int64) must still resolve -- not silently return an all-NA receiver column (the id-dtype
    miss). Without canonicalizing both sides of the join, ``frame_of.get(1)`` misses the ``"1"`` key."""
    links_str = pd.DataFrame({"action_id": ["1"], "frame_id": [100]})  # string action_id vs int actions
    got = resolve_intended_receiver(_actions(), _frame(), model=_fitted_model(), links=links_str)
    assert got.loc[1] == "11"  # resolves despite the str-vs-int action_id dtype mismatch


def test_proxy_serve_ball_less_frame_is_na_but_missing_columns_raises():
    """Q5 serve path: the geometric proxy (model=None) on a per-frame ball gap -> pd.NA for that action,
    never a crash; a velocity-less frame SET (missing vx/vy columns) still raises loud."""
    import pytest

    fr = _frame()  # carries the ball row + vx/vy
    fr_no_ball = fr[~fr["is_ball"].to_numpy(dtype=bool)].copy()
    got = resolve_intended_receiver(_actions(), fr_no_ball, model=None, links=_links())
    assert got.loc[1] is pd.NA  # per-frame ball gap -> NA
    with pytest.raises(KeyError):
        resolve_intended_receiver(_actions(), fr.drop(columns=["vx", "vy"]), model=None, links=_links())


def test_public_surface_is_pure():
    actions, frame = _actions(), _frame()
    a0, f0 = actions.copy(deep=True), frame.copy(deep=True)
    resolve_intended_receiver(actions, frame, model=_fitted_model(), links=_links())
    intended_receiver_positions(actions, frame, model=None, links=_links())
    pd.testing.assert_frame_equal(actions, a0)
    pd.testing.assert_frame_equal(frame, f0)
