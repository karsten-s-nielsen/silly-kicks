"""TF-49 _packing_at_actions kernel -- GEOMETRY-ONLY batch loop (review major 7 split).

Receiver / secured / require_secured tests live in test_add_packing.py -- the
kernel never touches them (packing_xfns calls it on shifted gamestate slots
where next-row relationships are meaningless).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking._kernels import _packing_at_actions
from tests.tracking._goal_map_helpers import goal_map_like_home_team_id
from tests.tracking.test_defensive_line import _make_frame_rows

_T = spadlconfig.actiontype_id
_R = spadlconfig.result_id

_COLS = ["packing_made", "packing_net", "packing_goal_threat", "line_x"]


def _acts(rows):
    base = {
        "game_id": 1,
        "period_id": 1,
        "time_seconds": 1.0,
        "team_id": 1,
        "player_id": 50,
        "start_x": 40.0,
        "start_y": 34.0,
        "end_x": 70.0,
        "end_y": 34.0,
        "type_id": _T["pass"],
        "result_id": _R["success"],
    }
    recs = []
    for i, r in enumerate(rows):
        d = dict(base)
        d.update(r)
        d["action_id"] = i + 1
        recs.append(d)
    return pd.DataFrame(recs)


def _frame(**kw):
    # away outfielders at x = 50, 60, 30, 80 act as defenders for a HOME action
    return _make_frame_rows(
        home_outfield_xs=[20.0, 22.0, 24.0, 26.0],
        home_outfield_ys=[20.0, 30.0, 40.0, 50.0],
        away_outfield_xs=[50.0, 60.0, 30.0, 80.0],
        away_outfield_ys=[34.0, 20.0, 34.0, 34.0],
        **kw,
    )


def _run(acts, frames=None):
    """_packing_at_actions with the map the removed home_team_id=1 implied (ADR-051 D3).

    goal_map_like_home_team_id rather than resolve_defended_goals: every test below is
    about the kernel's domain/NaN/alignment behaviour and the orientation is scaffolding, so
    deriving the map from the fixture would silently re-decide the convention from wherever the
    fixture parked its keeper. See tests/tracking/_goal_map_helpers.py for the two spellings.
    """
    frames = _frame() if frames is None else frames
    return _packing_at_actions(acts, frames, goal_map=goal_map_like_home_team_id(frames, 1))


def test_columns_and_completed_pass_computes():
    out = _run(_acts([{}]))
    assert list(out.columns) == _COLS
    assert out["packing_made"].iloc[0] == 2  # away 50, 60 in (40, 70]
    assert out["packing_net"].iloc[0] == pytest.approx(2.0)  # forward
    assert out["packing_goal_threat"].iloc[0] == 2  # back-4 of 4 outfielders
    assert out["line_x"].iloc[0] == pytest.approx(60.0)


def test_completion_gate_failed_action_all_nan():
    out = _run(_acts([{"result_id": _R["fail"]}]))
    assert out.iloc[0].isna().all()


def test_off_domain_type_all_nan():
    out = _run(_acts([{"type_id": _T["tackle"]}]))
    assert out.iloc[0].isna().all()


def test_dribble_with_real_end_is_numeric():
    out = _run(_acts([{"type_id": _T["dribble"]}]))
    assert out["packing_made"].iloc[0] == 2


def test_degenerate_dribble_nan_but_degenerate_pass_zero():
    """Spec s5.6: dribble start==end is placeholder-indistinguishable -> NaN;
    a pass-class start==end is recorded data -> honest geometric 0."""
    acts = _acts(
        [
            {"type_id": _T["dribble"], "start_x": 55.0, "start_y": 30.0, "end_x": 55.0, "end_y": 30.0},
            {"type_id": _T["pass"], "start_x": 55.0, "start_y": 30.0, "end_x": 55.0, "end_y": 30.0},
        ]
    )
    out = _run(acts)
    assert out.iloc[0].isna().all()
    assert out["packing_made"].iloc[1] == 0.0
    assert out["packing_net"].iloc[1] == pytest.approx(0.0)
    assert np.isnan(out["line_x"].iloc[1])  # nothing bypassed -> no line


def test_nan_actor_team_all_nan():
    out = _run(_acts([{"team_id": np.nan}]))
    assert out.iloc[0].isna().all()


def test_non_finite_coords_all_nan():
    out = _run(_acts([{"end_x": np.nan}]))
    assert out.iloc[0].isna().all()


def test_unlinked_action_all_nan():
    out = _run(_acts([{}]), _frame(time_seconds=500.0))
    assert out.iloc[0].isna().all()


def test_duplicate_action_id_slot_does_not_raise():
    """Shifted VAEP boundary slots repeat action_id (ADR-020 class)."""
    acts = _acts([{}, {"type_id": _T["tackle"]}])
    acts["action_id"] = [1, 1]
    out = _run(acts)
    assert len(out) == 2
    assert out["packing_made"].iloc[0] == 2
    assert out.iloc[1].isna().all()


def test_index_alignment_preserved():
    acts = _acts([{}, {}])
    acts.index = pd.Index([30, 40])
    out = _run(acts)
    assert list(out.index) == [30, 40]
