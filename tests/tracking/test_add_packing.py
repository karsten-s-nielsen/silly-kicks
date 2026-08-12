"""TF-49 add_packing aggregator assembly + packing_xfns (review major 7: the
receiver / secured / require_secured logic lives HERE, not in the kernel)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking import PackingParams, add_packing, packing_xfns
from tests.tracking.test_defensive_line import _make_frame_rows

_T = spadlconfig.actiontype_id
_R = spadlconfig.result_id

_PACKING_COLS = [
    "packing_made",
    "packing_net",
    "packing_goal_threat",
    "packing_receiver_player_id",
    "packing_secured",
]


def _frames():
    """Away defenders at x = 50, 60, 30, 80; one frame per in-domain action time."""
    parts = [
        _make_frame_rows(
            home_outfield_xs=[20.0, 22.0, 24.0, 26.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0],
            away_outfield_xs=[50.0, 60.0, 30.0, 80.0],
            away_outfield_ys=[34.0, 20.0, 34.0, 34.0],
            frame_id=k,
            time_seconds=t,
        )
        for k, t in [(1, 10.0), (2, 30.0), (3, 50.0), (4, 60.0), (5, 90.0)]
    ]
    return pd.concat(parts, ignore_index=True)


def _actions():
    """Event stream with one packing sequence per scenario (single game/period).

    Row 0  pass 40->70 @10   linked (frame 1): made 2, line_x 60; secured True
    Row 1  pass @10.5        reception of row 0
    Row 2  pass @11.5 x65    inside window, ahead of the line
    Row 3  pass @14   x66    beyond window -> window observed
    Row 4  pass 40->70 @30   linked (frame 2): made 2; secured False (bounce)
    Row 5  pass @31   x58    reception of row 4
    Row 6  pass @32   x45    behind line 60 inside window -> False
    Row 7  pass 40->45 @50   linked (frame 3): made 0 -> secured <NA>
    Row 8  pass @51          reception of row 7
    Row 9  dribble 40->70 @60 linked (frame 4): made 2; receiver+secured <NA>
    Row 10 pass FAIL @70     off-domain -> all NaN incl. receiver
    Row 11 pass @71          tail (receiver of nothing relevant)
    Row 12 pass 40->70 @90   linked (frame 5): made 2; truncated window -> secured <NA>
    Row 13 pass @91   x65    reception of row 12, then data ends
    """
    base = {
        "game_id": 1,
        "period_id": 1,
        "team_id": 1,
        "start_y": 34.0,
        "end_y": 34.0,
        "type_id": _T["pass"],
        "result_id": _R["success"],
    }
    rows = [
        {"time_seconds": 10.0, "start_x": 40.0, "end_x": 70.0, "player_id": 50},
        {"time_seconds": 10.5, "start_x": 70.0, "end_x": 71.0, "player_id": 51},
        {"time_seconds": 11.5, "start_x": 65.0, "end_x": 66.0, "player_id": 52},
        {"time_seconds": 14.0, "start_x": 66.0, "end_x": 67.0, "player_id": 53},
        {"time_seconds": 30.0, "start_x": 40.0, "end_x": 70.0, "player_id": 50},
        {"time_seconds": 31.0, "start_x": 58.0, "end_x": 59.0, "player_id": 51},
        {"time_seconds": 32.0, "start_x": 45.0, "end_x": 46.0, "player_id": 52},
        {"time_seconds": 50.0, "start_x": 40.0, "end_x": 45.0, "player_id": 50},
        {"time_seconds": 51.0, "start_x": 45.0, "end_x": 46.0, "player_id": 51},
        {"time_seconds": 60.0, "start_x": 40.0, "end_x": 70.0, "player_id": 54, "type_id": _T["dribble"]},
        {"time_seconds": 70.0, "start_x": 40.0, "end_x": 70.0, "player_id": 50, "result_id": _R["fail"]},
        {"time_seconds": 71.0, "start_x": 60.0, "end_x": 61.0, "player_id": 51},
        {"time_seconds": 90.0, "start_x": 40.0, "end_x": 70.0, "player_id": 50},
        {"time_seconds": 91.0, "start_x": 65.0, "end_x": 66.0, "player_id": 55},
    ]
    recs = []
    for i, r in enumerate(rows):
        d = dict(base)
        d.update(r)
        d["action_id"] = i
        recs.append(d)
    return pd.DataFrame(recs)


def test_columns_present_and_line_x_dropped():
    out = add_packing(_actions(), _frames())
    for col in _PACKING_COLS:
        assert col in out.columns
    assert "line_x" not in out.columns


def test_dtypes_contract():
    out = add_packing(_actions(), _frames())
    assert out["packing_made"].dtype == "Int64"
    assert out["packing_goal_threat"].dtype == "Int64"
    assert out["packing_net"].dtype == np.float64
    assert out["packing_secured"].dtype == "boolean"
    assert out["packing_receiver_player_id"].dtype == "Int64"  # int64 source passthrough


def test_geometry_and_receiver_values():
    out = add_packing(_actions(), _frames())
    assert out["packing_made"].iloc[0] == 2
    assert out["packing_receiver_player_id"].iloc[0] == 51
    assert out["packing_receiver_player_id"].iloc[4] == 51
    # unlinked in-domain rows: geometry NaN but receiver still resolves (event-only)
    assert pd.isna(out["packing_made"].iloc[1])
    assert out["packing_receiver_player_id"].iloc[1] == 52


def test_secured_tri_state():
    out = add_packing(_actions(), _frames())
    sec = out["packing_secured"]
    assert sec.iloc[0] == True  # noqa: E712 -- window observed past the line
    assert sec.iloc[4] == False  # noqa: E712 -- bounce pass behind the line
    assert pd.isna(sec.iloc[7])  # packing_made == 0 -> undefined
    assert pd.isna(sec.iloc[12])  # truncated window


def test_dribble_receiver_and_secured_na_but_counts_numeric():
    """Spec s3: the receiver seam is packing-agnostic; the dribble mask is assembly."""
    out = add_packing(_actions(), _frames())
    assert out["packing_made"].iloc[9] == 2
    assert pd.isna(out["packing_receiver_player_id"].iloc[9])
    assert pd.isna(out["packing_secured"].iloc[9])


def test_off_domain_receiver_masked():
    """A failed pass has a same-team next touch, but off-domain rows get <NA>."""
    out = add_packing(_actions(), _frames())
    assert pd.isna(out["packing_receiver_player_id"].iloc[10])
    assert pd.isna(out["packing_made"].iloc[10])


def test_require_secured_gates_receiver_bearing_only():
    """Review F3: secured False -> 0 counts; secured <NA> -> NaN counts; dribbles
    and packing_made == 0 rows untouched."""
    out = add_packing(_actions(), _frames(), params=PackingParams(require_secured=True))
    assert out["packing_made"].iloc[0] == 2  # secured True -> kept
    assert out["packing_made"].iloc[4] == 0  # secured False -> zeroed
    assert out["packing_net"].iloc[4] == pytest.approx(0.0)
    assert out["packing_goal_threat"].iloc[4] == 0
    assert out["packing_made"].iloc[7] == 0  # made 0 stays 0 (nothing to un-secure)
    assert out["packing_made"].iloc[9] == 2  # dribble untouched
    assert pd.isna(out["packing_made"].iloc[12])  # secured <NA> -> NaN counts


def test_provenance_idempotent_no_suffix_duplicates():
    a, f = _actions(), _frames()
    once = add_packing(a, f)
    twice = add_packing(once, f)
    assert not [c for c in twice.columns if c.endswith(("_x", "_y")) and "packing" in c]
    assert "frame_id" in twice.columns


def test_returns_new_frame_and_input_unmutated():
    a, f = _actions(), _frames()
    snap = a.copy(deep=True)
    out = add_packing(a, f)
    assert out is not a
    pd.testing.assert_frame_equal(a, snap)


def test_packing_xfns_rejects_require_secured():
    with pytest.raises(ValueError, match="require_secured"):
        packing_xfns(params=PackingParams(require_secured=True))


def test_packing_xfns_emits_nine_numeric_columns():
    xfns = packing_xfns()
    assert len(xfns) == 1
    a, f = _actions(), _frames()
    states = [a, a, a]
    out = xfns[0](states, f)
    assert out.shape == (len(a), 9)
    assert [c for c in out.columns if c.startswith("packing_made")] == [
        "packing_made_a0",
        "packing_made_a1",
        "packing_made_a2",
    ]
    assert out["packing_made_a0"].iloc[0] == 2


def test_packing_xfns_frames_none_all_nan():
    xfns = packing_xfns()
    a = _actions()
    out = xfns[0]([a, a, a], None)
    assert out.shape == (len(a), 9)
    assert out.isna().all().all()
