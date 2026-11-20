"""The readers of a shot row's time skip a row with ``is_time_unmeasured`` (action-time repair spec section 5.4).

A derived SkillCorner penalty row with ``"placeholder"`` or ``"order_only"`` time has no measured kick: its time is not
the shot's moment, so no reader takes the frames at it, labels a frame from it, or opens a window at it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig

_T = spadlconfig.actiontype_id
_R = spadlconfig.result_id
_UNMEASURED = ["placeholder", "order_only"]


# ---- the xShot label --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("src", "label"), [("placeholder", 0), ("order_only", 0), ("kick", 1), ("tracking", 1)])
def test_xshot_label_skips_an_unmeasured_shot(src, label):
    from silly_kicks.tracking._xshot_occurrence import build_xshot_labels

    fidx = pd.DataFrame({"game_id": [1], "period_id": [1], "time_seconds": [10.0], "team_in_possession": ["A"]})
    shots = pd.DataFrame({"game_id": [1], "period_id": [1], "team_id": ["A"], "time_seconds": [10.5],
                          "type_id": [_T["shot_penalty"]], "time_source": [src]})  # fmt: skip
    assert build_xshot_labels(fidx, shots).tolist() == [label]


# ---- the pre-shot keeper position and angle (one kernel seam, standard and atomic) -------------------------------


class _Ctx:
    def __init__(self, actions):
        self.actions = actions
        self.defending_gk_rows = pd.DataFrame({"action_id": actions["action_id"], "x": 103.0, "y": 34.0})


def _shot_rows(src: str) -> pd.DataFrame:
    return pd.DataFrame({"action_id": [0], "type_id": [_T["shot_penalty"]], "start_x": [94.0], "start_y": [34.0],
                         "time_source": [src]})  # fmt: skip


@pytest.mark.parametrize("src", _UNMEASURED)
def test_pre_shot_gk_position_is_nan_at_an_unmeasured_shot(src):
    from silly_kicks.tracking import _kernels

    a = _shot_rows(src)
    out = _kernels._pre_shot_gk_position(
        a["start_x"],
        a["start_y"],
        _Ctx(a),  # type: ignore[arg-type]  # a duck-typed context
        shot_type_ids=frozenset({_T["shot_penalty"]}),
    )
    assert out.isna().all().all()


@pytest.mark.parametrize("src", _UNMEASURED)
def test_pre_shot_gk_angle_is_nan_at_an_unmeasured_shot(src):
    from silly_kicks.tracking import _kernels

    a = _shot_rows(src)
    out = _kernels._pre_shot_gk_angle(
        a["start_x"],
        a["start_y"],
        _Ctx(a),  # type: ignore[arg-type]  # a duck-typed context
        shot_type_ids=frozenset({_T["shot_penalty"]}),
    )
    assert out.isna().all().all()


def test_pre_shot_gk_position_at_a_kick_shot_is_computed():
    from silly_kicks.tracking import _kernels

    a = _shot_rows("kick")
    out = _kernels._pre_shot_gk_position(
        a["start_x"],
        a["start_y"],
        _Ctx(a),  # type: ignore[arg-type]  # a duck-typed context
        shot_type_ids=frozenset({_T["shot_penalty"]}),
    )
    assert out["pre_shot_gk_x"].iloc[0] == pytest.approx(103.0)


# ---- the shot goal-mouth ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(("src", "is_na"), [("placeholder", True), ("order_only", True), ("tracking", False)])
def test_shot_goalmouth_is_nan_at_an_unmeasured_shot(src, is_na):
    from silly_kicks.tracking._shot_goalmouth import compute_shot_goalmouth
    from tests.tracking.test_shot_goalmouth import make_match

    actions, frames = make_match()
    out = compute_shot_goalmouth(actions.assign(time_source=src), frames)
    assert pd.isna(out["shot_crossing_y"].iloc[0]) is is_na
    if is_na:
        assert pd.isna(out["shot_crossing_source"].iloc[0])


# ---- the causal treatment and outcome windows ---------------------------------------------------------------------


def _causal_actions(src: str) -> pd.DataFrame:
    return pd.DataFrame({"game_id": [1, 1], "period_id": [1, 1], "team_id": [5, 5], "time_seconds": [10.0, 12.0],
                         "type_id": [_T["shot_penalty"], _T["shot"]], "result_id": [_R["success"], _R["fail"]],
                         "start_x": [94.0, 90.0], "start_y": [34.0, 30.0],
                         "time_source": [src, "tracking"]})  # fmt: skip


@pytest.mark.parametrize(("src", "n"), [("placeholder", 1), ("order_only", 1), ("kick", 2)])
def test_causal_treatment_skips_an_unmeasured_shot(src, n):
    from silly_kicks.causal.opportunities import _team_period_action_times

    ts = _team_period_action_times(_causal_actions(src), 1, 1, 5, ("shot", "shot_penalty"))
    assert len(ts) == n


@pytest.mark.parametrize(("src", "n"), [("placeholder", 1), ("order_only", 1), ("kick", 2)])
def test_causal_outcome_skips_an_unmeasured_shot(src, n):
    import dataclasses

    from silly_kicks.causal.opportunities import _outcome_distances, shot_arm_config

    cfg = dataclasses.replace(shot_arm_config({}), outcome_result_ids=None)
    d = _outcome_distances(_causal_actions(src), 1, 1, 5, 9.0, cfg)
    assert len(d) == n


# ---- TF-49 packing: a reception row that is an unmeasured shot -------------------------------------------------


@pytest.mark.parametrize(("src", "is_na"), [("placeholder", True), ("order_only", True), ("kick", False)])
def test_packing_secured_is_na_at_an_unmeasured_shot_reception(src, is_na):
    from silly_kicks.tracking._packing import secured_reception

    a = pd.DataFrame({"game_id": 1, "period_id": 1, "action_id": [0, 1], "time_seconds": [10.0, 11.0],
                      "team_id": [1, 1], "player_id": [11, 12], "type_id": [_T["pass"], _T["shot_penalty"]],
                      "result_id": [_R["success"], _R["success"]], "bodypart_id": 0, "start_x": [60.0, 94.0],
                      "start_y": 34.0, "end_x": [94.0, 105.0], "end_y": 34.0,
                      "time_source": ["tracking", src]})  # fmt: skip
    out = secured_reception(a, pd.Series([30.0, np.nan]), pd.Series([1, pd.NA], dtype="Int64"))
    assert pd.isna(out.iloc[0]) is is_na
