"""Task 6b (M-A): velocity ablation on COMPLETED passes + deployment gate on the FAILED subset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.train_receiver_model import (
    _DEPLOY_SHARD_COLUMNS,
    _pooled_deployment,
    deployment_decision,
    deployment_gate,
    velocity_ablation_completed,
)
from silly_kicks.tracking._receiver import ReceiverModel

_ATT, _DEF = 1, 2


def _owner_rows() -> pd.DataFrame:
    """Velocity-separable, positions uninformative: only closing_speed distinguishes the receiver."""
    rows = []
    for g in (1, 2):
        for a in range(4):
            aid = g * 10 + a
            for cid, is_rx in [(1, 1), (2, 0), (3, 0)]:
                rows.append(
                    {
                        "ball_dist": 15.0,
                        "lane_pressure": 0.5,
                        "space": 8.0,  # identical -> uninformative
                        "release_dir_align": 0.9 if is_rx else 0.1,
                        "closing_speed": 3.0 if is_rx else 0.0,
                        "candidate_id": str(cid),
                        "label": is_rx,
                        "game_id": g,
                        "action_id": aid,
                        "n_candidates": 3,
                    }
                )
    return pd.DataFrame(rows)


def test_velocity_ablation_on_completed_passes():
    abl = velocity_ablation_completed(_owner_rows())
    # velocity separates the receiver where positions cannot -> the velocity variant wins
    assert abl["top1_positions_velocity"] >= abl["top1_positions"]
    assert abl["velocity_delta"] >= 0.0
    assert "COMPLETED" in abl["caveat"]


def _frame() -> pd.DataFrame:
    rows = [
        (True, pd.NA, pd.NA, False, 50.0, 34.0, 10.0, 0.0),
        (False, 9, _ATT, False, 50.0, 34.0, 0.0, 0.0),
        (False, 10, _ATT, False, 75.0, 34.0, 0.0, 0.0),
        (False, 11, _ATT, False, 55.0, 50.0, 0.0, 0.0),
        (False, 20, _DEF, False, 62.0, 34.0, 0.0, 0.0),
        (False, 30, _DEF, True, 100.0, 34.0, 0.0, 0.0),
    ]
    df = pd.DataFrame(rows, columns=["is_ball", "player_id", "team_id", "is_goalkeeper", "x", "y", "vx", "vy"])
    df["game_id"], df["period_id"], df["frame_id"] = 1, 1, 100
    return df.astype({"player_id": "Int64", "team_id": "Int64"})


def _actions() -> pd.DataFrame:
    from scripts._receiver_validation import _R, _T

    cols = [
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
    rows = [
        (1, 1, 1, 10.0, 1, 9, _T["pass"], _R["fail"], 50, 34, 62, 34),  # intercepted
        (2, 1, 1, 11.0, 2, 20, _T["pass"], _R["success"], 62, 34, 40, 34),
    ]
    return pd.DataFrame(rows, columns=cols)


def _model() -> ReceiverModel:
    intended = pd.DataFrame({"ball_dist": [15.0] * 12, "lane_pressure": [0.0] * 12, "space": [12.0] * 12})
    others = pd.DataFrame({"ball_dist": [15.0] * 12, "lane_pressure": [1.5] * 12, "space": [4.0] * 12})
    X = pd.concat([intended, others], ignore_index=True)
    return ReceiverModel("public").fit(X, np.array([1] * 12 + [0] * 12))


def test_deployment_gate_non_decisive_on_equal_models():
    m = _model()
    links = pd.DataFrame({"action_id": [1], "frame_id": [100]})
    gate = deployment_gate(m, m, _actions(), _frame(), links=links)  # same model both legs
    assert gate["decisive"] is False  # zero margin cannot clear MIN_RECEIVER_MARGIN
    assert gate["margin"] == 0.0
    assert "unmeasurable on the easy tail" in gate["r1_caveat"]  # R1 scoping present


def test_deployment_decision_pooled_arithmetic():
    decisive = deployment_decision({"n_scored": 100, "top1": 0.40}, {"n_scored": 100, "top1": 0.55}, min_margin=0.05)
    assert decisive["margin"] == pytest.approx(0.15) and decisive["decisive"] is True
    thin = deployment_decision({"n_scored": 100, "top1": 0.52}, {"n_scored": 100, "top1": 0.54}, min_margin=0.05)
    assert thin["decisive"] is False  # +0.02 < 0.05
    # one side unscored (the easy tail found nothing for a model) -> NaN margin, non-decisive
    empty = deployment_decision({"n_scored": 0, "top1": float("nan")}, {"n_scored": 100, "top1": 0.5})
    assert empty["decisive"] is False and np.isnan(empty["margin"])


def test_pooled_deployment_sums_counts_then_decides():
    counts = pd.DataFrame(
        [
            {
                "match_id": "1",
                "pub_n_scored": 10,
                "pub_hits": 4,
                "own_n_scored": 10,
                "own_hits": 6,
                "n_covered": 10,
                "n_intercepted": 20,
            },
            {
                "match_id": "2",
                "pub_n_scored": 10,
                "pub_hits": 5,
                "own_n_scored": 10,
                "own_hits": 7,
                "n_covered": 5,
                "n_intercepted": 10,
            },
        ],
        columns=_DEPLOY_SHARD_COLUMNS,
    )
    d = _pooled_deployment(counts)  # pool THEN decide -- not a mean of per-match margins
    assert d["public_top1"] == pytest.approx(9 / 20) and d["owner_top1"] == pytest.approx(13 / 20)
    assert d["n_scored"] == 20 and d["coverage"] == pytest.approx(15 / 30)
