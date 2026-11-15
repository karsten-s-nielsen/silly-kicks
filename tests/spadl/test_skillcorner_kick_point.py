"""SkillCorner native rows sit at the kick, not the reception (spec 2026-11-13-cross-definition r8 §6.6 A).

A `player_possession` row runs from the reception (`time_start`, `x_start`/`y_start`) to the on-ball action at its
end (`time_end`, `x_end`/`y_end`). The SPADL row is that action, so it takes the end. Rows that happen at the
possession start (start-type interceptions and recoveries, keeper saves) stay at the start.

Coordinates are centred metres on a 105 x 68 pitch, so SPADL x = x + 52.5 and y = y + 34.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl.skillcorner import convert_to_actions

_META = json.loads((Path(__file__).parent.parent / "datasets" / "skillcorner" / "match_metadata.json").read_text())
_ID = spadlconfig.actiontype_id


def _clock(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


def _pp(rows: list[dict]) -> pd.DataFrame:
    """player_possession rows; each dict sets ts/te (seconds) and the fields it needs."""
    recs = []
    for i, r in enumerate(rows):
        te = r.get("te", r["ts"])
        recs.append(
            {
                "event_type": "player_possession",
                "event_id": f"{i}_0",
                "period": r.get("period", 1),
                "time_start": _clock(r["ts"]),
                "time_end": None if te is None else _clock(te),
                "team_id": r.get("team", "team_home"),
                "player_id": r.get("player", f"p{i}"),
                "x_start": r["xs"],
                "y_start": r["ys"],
                "x_end": r.get("xe", r["xs"]),
                "y_end": r.get("ye", r["ys"]),
                "start_type": r.get("start_type", "pass_reception"),
                "end_type": r.get("end_type", "pass"),
                "game_interruption_before": r.get("gi_before"),
                "game_interruption_after": r.get("gi_after"),
                "is_header": False,
                "hand_pass": False,
                "player_targeted_x_reception": r.get("tx"),
                "player_targeted_y_reception": r.get("ty"),
                "player_targeted_third_pass": None,
                "player_targeted_channel_pass": None,
            }
        )
    return pd.DataFrame(recs)


def _convert(rows: list[dict]) -> pd.DataFrame:
    actions, _ = convert_to_actions(_pp(rows), _META)
    return actions


def _native(actions: pd.DataFrame, event_id: str) -> pd.Series:
    rows = actions[(actions["original_event_id"] == event_id) & (actions["action_provenance"] == "native")]
    assert len(rows) == 1, rows
    return rows.iloc[0]


def test_native_row_at_possession_end():
    a = _convert(
        [
            {"ts": 10.0, "te": 12.5, "xs": 0.0, "ys": 0.0, "xe": 10.0, "ye": 5.0, "tx": 20.0, "ty": 5.0},
            {"ts": 13.0, "te": 13.0, "xs": 20.0, "ys": 5.0, "end_type": "possession_loss"},
        ]
    )
    row = _native(a, "0_0")
    assert row["time_seconds"] == pytest.approx(12.5)
    assert (row["start_x"], row["start_y"]) == pytest.approx((62.5, 39.0))


def test_start_type_rows_stay_at_start():
    a = _convert(
        [
            {"ts": 10.0, "te": 11.0, "xs": 0.0, "ys": 0.0, "xe": 5.0, "ye": 0.0, "tx": 15.0, "ty": 0.0},
            {
                "ts": 12.0,
                "te": 14.0,
                "xs": -10.0,
                "ys": 3.0,
                "xe": -2.0,
                "ye": 3.0,
                "team": "team_away",
                "start_type": "pass_interception",
            },
        ]
    )
    derived = a[(a["action_provenance"] == "derived") & a["type_id"].isin([_ID["interception"], _ID["tackle"]])]
    assert len(derived) == 1
    d = derived.iloc[0]
    assert d["time_seconds"] == pytest.approx(12.0 - 0.01)
    assert (d["start_x"], d["start_y"]) == pytest.approx((42.5, 37.0))
    assert _native(a, "1_0")["time_seconds"] == pytest.approx(14.0)


def test_keeper_save_stays_at_the_next_possession_start():
    # Review B CD-SB-24: the save is inferred from the possession AFTER the shot and sits at that possession's
    # reception (-0.01 s), not at its end. time_end differs from time_start, so a save fed the end time fails.
    a = _convert(
        [
            {"ts": 10.0, "te": 12.0, "xs": 30.0, "ys": 0.0, "xe": 40.0, "ye": 1.0, "end_type": "shot"},
            {
                "ts": 20.0,
                "te": 23.0,
                "xs": -48.0,
                "ys": 1.0,
                "xe": -30.0,
                "ye": 5.0,
                "team": "team_away",
                "start_type": "recovery",
                "end_type": "possession_loss",
            },
        ]
    )
    saves = a[a["type_id"] == _ID["keeper_save"]]
    assert len(saves) == 1, a[["type_id", "time_seconds", "start_x"]]
    s = saves.iloc[0]
    assert s["time_seconds"] == pytest.approx(20.0 - 0.01)
    assert (s["start_x"], s["start_y"]) == pytest.approx((4.5, 35.0))


def test_missing_time_end_is_nan_not_time_start():
    rows = [
        {"ts": 10.0, "te": None, "xs": 0.0, "ys": 0.0, "xe": 4.0, "ye": 2.0, "tx": 15.0, "ty": 0.0},
        {"ts": 13.0, "te": 14.0, "xs": 15.0, "ys": 0.0, "xe": 16.0, "ye": 0.0, "end_type": "possession_loss"},
    ]
    with pytest.warns(UserWarning, match=r"\b1\b.*time_end"):
        actions, _ = convert_to_actions(_pp(rows), _META)
    row = _native(actions, "0_0")
    assert np.isnan(row["time_seconds"])
    assert (row["start_x"], row["start_y"]) == pytest.approx((56.5, 36.0))


def test_nan_end_row_keeps_its_reception_order():
    # Review B CD-SB-25 SC3: the possessions are ordered by reception time, not by the (here NaN) action time. Row 1
    # has no time_end; sorted by time_seconds it would move to the period's end, and row 0 would end at row 2's
    # reception (30, 6) instead of row 1's (18, 6).
    rows = [
        {"ts": 10.0, "te": 11.0, "xs": 0.0, "ys": 0.0, "xe": 3.0, "ye": 0.0},
        {"ts": 13.0, "te": None, "xs": 18.0, "ys": 6.0, "xe": 26.0, "ye": 6.0, "player": "p2"},
        {"ts": 16.0, "te": 17.0, "xs": 30.0, "ys": 6.0, "end_type": "possession_loss", "player": "p3"},
    ]
    with pytest.warns(UserWarning, match="time_end"):
        actions, _ = convert_to_actions(_pp(rows), _META)
    assert (_native(actions, "0_0")["end_x"], _native(actions, "0_0")["end_y"]) == pytest.approx((70.5, 40.0))


def test_no_warning_when_every_time_end_is_present():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _convert([{"ts": 10.0, "te": 11.0, "xs": 0.0, "ys": 0.0, "end_type": "possession_loss"}])


def test_carry_becomes_a_dribble():
    a = _convert(
        [
            {"ts": 10.0, "te": 11.0, "xs": 0.0, "ys": 0.0, "tx": 20.0, "ty": 0.0, "player": "p1"},
            {
                "ts": 12.0,
                "te": 14.0,
                "xs": 20.0,
                "ys": 0.0,
                "xe": 26.0,
                "ye": 0.0,
                "tx": 35.0,
                "ty": 0.0,
                "player": "p2",
            },
            {"ts": 15.0, "te": 15.0, "xs": 35.0, "ys": 0.0, "end_type": "possession_loss", "player": "p3"},
        ]
    )
    dribbles = a[a["type_id"] == _ID["dribble"]]
    carry = dribbles[(dribbles["start_x"].round(3) == 72.5) & (dribbles["end_x"].round(3) == 78.5)]
    assert len(carry) == 1, dribbles[["start_x", "end_x", "time_seconds"]]


@pytest.mark.parametrize(
    ("end_type", "type_name"),
    [("shot", "shot"), ("foul_suffered", "foul"), ("possession_loss", "non_action"), ("unknown", "non_action")],
)
def test_non_pass_rows_end_at_their_start(end_type, type_name):
    a = _convert([{"ts": 10.0, "te": 12.0, "xs": 30.0, "ys": 2.0, "xe": 40.0, "ye": 1.0, "end_type": end_type}])
    row = _native(a, "0_0")
    assert row["type_id"] == _ID[type_name]
    assert (row["start_x"], row["start_y"]) == pytest.approx((92.5, 35.0))
    assert (row["end_x"], row["end_y"]) == pytest.approx((row["start_x"], row["start_y"]))


@pytest.mark.parametrize(("end_type", "type_name"), [("pass", "pass"), ("clearance", "clearance")])
def test_untargeted_pass_and_clearance_end_at_the_next_reception(end_type, type_name):
    # Cross spec 6.6 A (review B CD-SB-23): with no targeted receiver the ball's destination is the next possession's
    # RECEPTION (18, 6), not its kick (26, 6); the receiver's carry from reception to kick is a synthetic dribble.
    a = _convert(
        [
            {"ts": 10.0, "te": 11.0, "xs": 0.0, "ys": 0.0, "xe": 3.0, "ye": 0.0, "end_type": end_type},
            {"ts": 13.0, "te": 14.0, "xs": 18.0, "ys": 6.0, "xe": 26.0, "ye": 6.0, "player": "p2"},
            {"ts": 15.0, "te": 15.0, "xs": 30.0, "ys": 6.0, "end_type": "possession_loss", "player": "p3"},
        ]
    )
    row = _native(a, "0_0")
    assert row["type_id"] == _ID[type_name]
    assert (row["start_x"], row["start_y"]) == pytest.approx((55.5, 34.0))
    assert (row["end_x"], row["end_y"]) == pytest.approx((70.5, 40.0))
    dribbles = a[a["type_id"] == _ID["dribble"]]
    carry = dribbles[(dribbles["start_x"].round(3) == 70.5) & (dribbles["end_x"].round(3) == 78.5)]
    assert len(carry) == 1, dribbles[["start_x", "end_x", "time_seconds"]]


def test_targeted_pass_keeps_its_targeted_reception():
    # Both sides of the r9 rule: a targeted reception (20, 0) wins over the next possession's reception (18, 6).
    a = _convert(
        [
            {"ts": 10.0, "te": 11.0, "xs": 0.0, "ys": 0.0, "xe": 3.0, "ye": 0.0, "tx": 20.0, "ty": 0.0},
            {"ts": 13.0, "te": 14.0, "xs": 18.0, "ys": 6.0, "xe": 26.0, "ye": 6.0, "end_type": "possession_loss"},
        ]
    )
    assert (_native(a, "0_0")["end_x"], _native(a, "0_0")["end_y"]) == pytest.approx((72.5, 34.0))


def test_next_reception_never_crosses_a_period():
    # Review A CD-IMPL-13: the period's last possession has no next reception IN ITS PERIOD; the next period's
    # reception (-30, -20) is not its destination. Its end stays its start (72.5, 40); a period-blind shift would
    # give (22.5, 14).
    a = _convert(
        [
            {"ts": 10.0, "te": 11.0, "xs": 0.0, "ys": 0.0, "xe": 20.0, "ye": 6.0},
            {"period": 2, "ts": 2710.0, "te": 2712.0, "xs": -30.0, "ys": -20.0, "end_type": "possession_loss"},
        ]
    )
    row = _native(a, "0_0")
    assert (row["start_x"], row["start_y"]) == pytest.approx((72.5, 40.0))
    assert (row["end_x"], row["end_y"]) == pytest.approx((72.5, 40.0))


def test_last_possession_of_a_period_falls_back_to_the_derived_end():
    # No next reception in the period: the end equals the start, and _derive_end_coordinates leaves it there
    # (no next action in the period either).
    a = _convert([{"ts": 10.0, "te": 11.0, "xs": 0.0, "ys": 0.0, "xe": 3.0, "ye": 0.0}])
    row = _native(a, "0_0")
    assert (row["end_x"], row["end_y"]) == pytest.approx((row["start_x"], row["start_y"]))


def test_period_relative_after_rebase():
    a = _convert([{"period": 2, "ts": 2710.0, "te": 2712.0, "xs": 0.0, "ys": 0.0, "end_type": "possession_loss"}])
    assert _native(a, "0_0")["time_seconds"] == pytest.approx(12.0)


@pytest.mark.parametrize(
    ("reception_y", "kick_y", "is_cross"),
    [(-30.0, 0.0, False), (0.0, -30.0, True)],
    ids=["reception_wide_kick_central", "reception_central_kick_wide"],
)
def test_kick_point_decides_the_cross_zone(reception_y, kick_y, is_cross):
    a = _convert(
        [
            {"ts": 10.0, "te": 12.0, "xs": 30.0, "ys": reception_y, "xe": 30.0, "ye": kick_y, "player": "p1"},
            {"ts": 13.0, "te": 13.0, "xs": 45.0, "ys": 0.0, "end_type": "shot", "player": "p2"},
        ]
    )
    assert bool(_native(a, "0_0")["type_id"] == _ID["cross"]) is is_cross
