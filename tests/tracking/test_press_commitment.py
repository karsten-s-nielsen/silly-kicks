"""Pressure-commitment cue primitive (TF-51 v2 Item 5, spec section 6 / N7)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._press_commitment import (
    PRESS_COMMITMENT_SOURCE_VALUES,
    PressCommitmentParams,
    compute_press_commitment,
)
from tests.tracking._defensive_credit_fixtures import one_action


def _pc_frames(vx_by_time, *, actor_xy=(95.0, 34.0), def_xy=(94.0, 34.0), speed_source="derived"):
    """Frames over the window: a stationary actor (team 10, id 5), a moving defender (team 20, id 20)
    whose per-frame vx is given, and the ball. Only vx/vy feed the closing-speed slope."""
    rows = []
    for t, vx in vx_by_time:
        fid = round(t * 10)
        common = dict(
            game_id="g1", period_id=1, frame_id=fid, time_seconds=float(t), vy=0.0,
            is_goalkeeper=False, source_provider="test", speed_source=speed_source,
        )  # fmt: skip
        rows.append(
            {
                **common,
                "team_id": 20,
                "player_id": 20,
                "x": def_xy[0],
                "y": def_xy[1],
                "vx": float(vx),
                "is_ball": False,
            }
        )
        rows.append(
            {**common, "team_id": 10, "player_id": 5, "x": actor_xy[0], "y": actor_xy[1], "vx": 0.0, "is_ball": False}
        )
        rows.append(
            {**common, "team_id": np.nan, "player_id": np.nan, "x": 90.0, "y": 34.0, "vx": 0.0, "is_ball": True}
        )
    return pd.DataFrame(rows)


def _pc_action(action_time=10.0):
    return one_action(
        type_name="pass", result_name="success", team_id=10, player_id=5,
        start_x=95.0, start_y=34.0, time_seconds=action_time, action_id=1,
    )  # fmt: skip


def _row(a, frames):
    return compute_press_commitment(a, frames).iloc[0]


def test_source_vocab_is_closed():
    assert set(PRESS_COMMITMENT_SOURCE_VALUES) == {
        "computed", "no_pressing_defender", "velocity_unavailable",
        "window_too_short", "degenerate_axis", "unlinked",
    }  # fmt: skip


def test_committing_press_is_positive():
    # defender accelerating toward the actor (vx 1->3, axis=+x) -> positive slope
    r = _row(_pc_action(), _pc_frames([(9.6, 1.0), (9.8, 2.0), (10.0, 3.0)]))
    assert r["press_commitment_source"] == "computed"
    assert r["press_commitment"] > 0
    assert r["press_commitment_closing_speed"] == pytest.approx(3.0)  # v_close at the action frame


def test_containing_press_is_negative():
    # defender braking (vx 3->1) -> negative slope
    r = _row(_pc_action(), _pc_frames([(9.6, 3.0), (9.8, 2.0), (10.0, 1.0)]))
    assert r["press_commitment_source"] == "computed"
    assert r["press_commitment"] < 0


def test_no_pressing_defender_when_far():
    r = _row(_pc_action(), _pc_frames([(9.6, 1.0), (9.8, 2.0), (10.0, 3.0)], def_xy=(85.0, 34.0)))  # 10 m
    assert r["press_commitment_source"] == "no_pressing_defender"
    assert np.isnan(r["press_commitment"])


def test_window_too_short_when_single_frame():
    r = _row(_pc_action(), _pc_frames([(10.0, 3.0)]))  # one window frame -> < 2 points
    assert r["press_commitment_source"] == "window_too_short"
    assert np.isnan(r["press_commitment"])


def test_degenerate_axis_when_defender_on_top_of_actor():
    r = _row(_pc_action(), _pc_frames([(9.6, 1.0), (9.8, 2.0), (10.0, 3.0)], def_xy=(95.0, 34.2)))  # 0.2 m < 0.5
    assert r["press_commitment_source"] == "degenerate_axis"
    assert np.isnan(r["press_commitment"])


def test_velocity_unavailable_when_all_rows_marked():
    frames = _pc_frames([(9.6, 1.0), (9.8, 2.0), (10.0, 3.0)], speed_source="unavailable")
    r = _row(_pc_action(), frames)
    assert r["press_commitment_source"] == "velocity_unavailable"
    assert np.isnan(r["press_commitment"])


def test_missing_vx_vy_raises_loud():
    frames = _pc_frames([(9.6, 1.0), (9.8, 2.0), (10.0, 3.0)]).drop(columns=["vx", "vy"])
    with pytest.raises(ValueError, match="vx/vy"):
        compute_press_commitment(_pc_action(), frames)


def test_unlinked_action_is_nan():
    # action in a period with no frames -> unlinked
    a = _pc_action()
    a["period_id"] = 2
    r = _row(a, _pc_frames([(9.6, 1.0), (9.8, 2.0), (10.0, 3.0)]))
    assert r["press_commitment_source"] == "unlinked"
    assert np.isnan(r["press_commitment"])


def test_params_validate_positive():
    with pytest.raises(ValueError):
        PressCommitmentParams(press_max_distance_m=-1.0)
