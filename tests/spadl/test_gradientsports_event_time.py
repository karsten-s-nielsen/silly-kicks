"""Gradient Sports: ``time_seconds`` is the possession event's own time, period-relative (ADR-133).

``time_seconds = event_time - kickoff start_time`` of the row's period (ADR-017: period-relative). The kickoff rows
are read before the converter's exclusion filter drops them. The input ``startGameClock`` (``time_seconds``) no
longer reaches the output, so the Option D foul imputation is gone: every event carries ``eventTime``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl import gradientsports as gs_mod
from silly_kicks.spadl.utils import _assert_chronological_action_id

_COLS = sorted(gs_mod.EXPECTED_INPUT_COLUMNS | {"time_seconds", "event_time", "start_time"})
_KICK = {1: "FIRSTKICKOFF", 2: "SECONDKICKOFF", 3: "THIRDKICKOFF", 4: "FOURTHKICKOFF"}
_KICK_T = {1: 100.0, 2: 3100.0, 3: 6100.0, 4: 7100.0}


def _row(eid, period, event_time, start_time=None, *, ge="OTB", pe: str | None = "PA", clock=None, **extra):
    r = dict.fromkeys(_COLS)
    r.update(
        {
            "game_id": 10502,
            "event_id": eid,
            "possession_event_id": eid if pe else None,
            "period_id": period,
            "time_seconds": clock,
            "start_time": event_time if start_time is None else start_time,
            "event_time": event_time,
            "team_id": 100 if ge not in _KICK.values() else None,
            "player_id": 1 if ge not in _KICK.values() else None,
            "game_event_type": ge,
            "possession_event_type": pe,
            "set_piece_type": "O",
            "ball_x": 0.0,
            "ball_y": 0.0,
            "body_type": "R",
            "pass_outcome_type": "C" if pe == "PA" else None,
        }
    )
    r.update(extra)
    return r


def _kick(period, t=None):
    t = _KICK_T[period] if t is None else t
    return _row(-period, period, t, ge=_KICK[period], pe=None)


def _frame(rows):
    df = pd.DataFrame(rows, columns=_COLS)
    for col in ("possession_event_id", "player_id", "team_id"):
        df[col] = df[col].astype("Int64")
    for col in ("game_id", "event_id", "period_id"):
        df[col] = df[col].astype("int64")
    for col in ("time_seconds", "start_time", "event_time", "ball_x", "ball_y"):
        df[col] = df[col].astype("float64")
    return df


def _convert(rows, **kw):
    actions, _ = gs_mod.convert_to_actions(
        _frame(rows), home_team_id=100, home_team_start_left=True, home_team_start_left_extratime=True, **kw
    )
    return actions


def _time_of(actions, eid):
    return float(actions.loc[actions["original_event_id"] == eid, "time_seconds"].iloc[0])


@pytest.mark.parametrize("period", [1, 2, 3, 4])
def test_time_is_event_time_minus_kickoff(period):
    a = _convert([_kick(period), _row(1, period, _KICK_T[period] + 12.3, clock=2700.0 * (period - 1))])
    assert _time_of(a, 1) == pytest.approx(12.3)  # period-relative, not the match clock


@pytest.mark.parametrize("period", [1, 2, 3, 4])
def test_missing_kickoff_raises(period):
    with pytest.raises(ValueError, match=rf"kickoff.*period {period}"):
        _convert([_row(1, period, _KICK_T[period] + 5.0)])


def test_shootout_period_5_anchors_at_first_event():
    rows = [
        _kick(1),
        _row(1, 1, 150.0),
        _row(50, 5, 9000.0, pe="SH", set_piece_type="P", shot_outcome_type="G"),
        _row(51, 5, 9050.0, pe="SH", set_piece_type="P", shot_outcome_type="S"),
    ]
    a = _convert(rows)
    assert _time_of(a, 50) == pytest.approx(0.0)
    assert _time_of(a, 51) == pytest.approx(50.0)


def _end(period, t):
    return _row(-10 - period, period, t, ge="END", pe=None)


def _penalty(eid, period, t, outcome="G"):
    return _row(eid, period, t, pe="SH", set_piece_type="P", shot_outcome_type=outcome)


def test_rows_after_the_period_4_end_are_the_shootout():
    # The feed stores the shoot-out in period 4, after the END game event of period 4.
    rows = [_kick(4), _row(1, 4, 7200.0), _end(4, 7300.0), _penalty(50, 4, 9000.0), _penalty(51, 4, 9050.0, "S")]
    a = _convert(rows)
    period = a.set_index("original_event_id")["period_id"]
    assert (period[1], period[50], period[51]) == (4, 5, 5)
    assert _time_of(a, 50) == pytest.approx(0.0)  # the shoot-out clock counts from its first event
    assert _time_of(a, 51) == pytest.approx(50.0)


def _shootout_rows(x=10.0, y=5.0, x2=None, y2=None):
    """The period-4 END, then one kick of each team; the feed repeats one stopped ball position on each row."""
    second = _penalty(51, 4, 9050.0, "S")
    second.update(team_id=200, ball_x=x if x2 is None else x2, ball_y=y if y2 is None else y2)
    first = _penalty(50, 4, 9000.0)
    first.update(ball_x=x, ball_y=y)
    return [_kick(4), _row(1, 4, 7200.0), _end(4, 7300.0), first, second]


def test_shootout_rows_have_no_location():
    # The stopped ball position on a shoot-out row is not the kick's location and cannot tell the goal end.
    a = _convert(_shootout_rows())
    so = a[a["original_event_id"].isin([50, 51])]
    assert so[["start_x", "start_y", "end_x", "end_y"]].isna().all().all()


def test_more_than_one_shootout_position_warns():
    with pytest.warns(UserWarning, match="shoot-out"):
        a = _convert(_shootout_rows(x2=-10.0, y2=-5.0))
    so = a[a["original_event_id"].isin([50, 51])]
    assert so[["start_x", "start_y", "end_x", "end_y"]].isna().all().all()


def test_a_penalty_before_the_period_4_end_stays_in_period_4():
    rows = [_kick(4), _penalty(40, 4, 7200.0), _end(4, 7300.0)]
    rows[1].update(ball_x=40.0, ball_y=0.5)
    a = _convert(rows)
    pen = a[a["original_event_id"] == 40].iloc[0]
    assert int(pen["period_id"]) == 4
    assert _time_of(a, 40) == pytest.approx(100.0)
    assert pd.notna(pen["start_x"]) and pd.notna(pen["start_y"])  # an in-match penalty keeps its location


def test_no_period_4_end_moves_no_row():
    # Without the END marker the converter does not guess where the shoot-out starts.
    a = _convert([_kick(4), _penalty(50, 4, 9000.0)])
    assert int(a.loc[a["original_event_id"] == 50, "period_id"].iloc[0]) == 4
    assert _time_of(a, 50) == pytest.approx(1900.0)
    assert pd.notna(a.loc[a["original_event_id"] == 50, "start_x"].iloc[0])


def test_rows_after_the_period_4_end_without_a_penalty_stay():
    # On-ball rows can follow an END with no shoot-out; only a penalty kick after the END marks one.
    rows = [_kick(4), _row(1, 4, 7200.0), _end(4, 7300.0), _row(2, 4, 7310.0)]
    rows[3].update(ball_x=5.0, ball_y=2.0)
    a = _convert(rows)
    row = a[a["original_event_id"] == 2].iloc[0]
    assert int(row["period_id"]) == 4
    assert pd.notna(row["start_x"]) and pd.notna(row["start_y"])


def test_an_input_period_5_keeps_its_period_and_time_but_no_location():
    # A caller's own period 5 is the shoot-out too: its rows get no location (no direction from team identity).
    first = _penalty(50, 5, 9000.0)
    first.update(ball_x=10.0, ball_y=5.0)
    a = _convert([_kick(1), _row(1, 1, 150.0), first, _penalty(51, 5, 9050.0, "S")])
    so = a[a["original_event_id"].isin([50, 51])]
    assert so["period_id"].tolist() == [5, 5]
    assert _time_of(a, 50) == pytest.approx(0.0) and _time_of(a, 51) == pytest.approx(50.0)
    assert so[["start_x", "start_y", "end_x", "end_y"]].isna().all().all()


def test_period_without_actions_needs_no_anchor():
    marker = _row(99, 0, 50.0, ge="G", pe=None)  # a period-0 game marker, excluded by the converter
    a = _convert([marker, _kick(1), _row(1, 1, 110.0)])
    assert list(a["original_event_id"]) == [1]


def test_inversion_pair_sorts_by_event_time():
    # A starts earlier but its possession event comes later; the old key (start clock, start_time) put A first.
    rows = [
        _kick(1),
        _row(1, 1, event_time=118.0, start_time=110.0, clock=10.0),
        _row(2, 1, event_time=114.0, start_time=112.0, clock=12.0),
    ]
    a = _convert(rows)
    assert list(a["original_event_id"]) == [2, 1]
    assert list(a["action_id"]) == [0, 1]
    _assert_chronological_action_id(a)


def test_full_tie_sorts_by_event_id():
    # Equal time, start_time and possession_event_id leave only event_id to decide; the input
    # lists 7 before 5, so a tiebreak without event_id keeps the input order.
    rows = [
        _kick(1),
        _row(7, 1, 120.0, start_time=115.0, possession_event_id=3),
        _row(5, 1, 120.0, start_time=115.0, possession_event_id=3),
    ]
    assert list(_convert(rows)["original_event_id"]) == [5, 7]


def test_tiebreak_precedence_start_time_then_possession_event_then_event_id():
    # ADR-065: the order is a pure function of content, and each key carries chronology. At equal event time the
    # earlier on-ball sequence (start_time) goes first; at equal start_time the possession event decides (the
    # records of one game event share its event_id). Each case disagrees with event_id order.
    by_start = [_kick(1), _row(5, 1, 120.0, start_time=116.0), _row(9, 1, 120.0, start_time=112.0)]
    assert list(_convert(by_start)["original_event_id"]) == [9, 5]
    by_possession = [
        _kick(1),
        _row(4, 1, 130.0, start_time=125.0, possession_event_id=8),
        _row(6, 1, 130.0, start_time=125.0, possession_event_id=2),
    ]
    assert list(_convert(by_possession)["original_event_id"]) == [6, 4]


def test_duplicate_kickoff_anchors_at_the_earliest():
    # Two FIRSTKICKOFF events; the period clock starts at the earlier one.
    rows = [_kick(1, 100.0), _row(-9, 1, 105.0, ge="FIRSTKICKOFF", pe=None), _row(1, 1, 110.0)]
    assert _time_of(_convert(rows), 1) == pytest.approx(10.0)


def test_foul_without_start_clock_gets_event_time():
    rows = [
        _kick(1),
        _row(1, 1, 160.0, clock=60.0),
        _row(2, 1, 165.0, ge="FOUL", pe="FO", clock=None, foul_type="I", final_foul_outcome_type="Y"),
    ]
    a = _convert(rows)
    foul = a[a["type_id"] == spadlconfig.actiontype_id["foul"]]
    assert foul["time_seconds"].tolist() == pytest.approx([65.0])


def test_nan_event_time_raises():
    with pytest.raises(ValueError, match="event_time"):
        _convert([_kick(1), _row(1, 1, np.nan, start_time=150.0)])


def test_input_time_seconds_is_ignored():
    rows = [_kick(1), _row(1, 1, 130.0, clock=30.0), _row(2, 1, 140.0, clock=40.0)]
    other = [_kick(1), _row(1, 1, 130.0, clock=999.0), _row(2, 1, 140.0, clock=-5.0)]
    pd.testing.assert_frame_equal(_convert(rows), _convert(other))


@pytest.mark.parametrize(("delta", "raises"), [(0.0, False), (-0.01, True)], ids=["at_kickoff", "before_kickoff"])
def test_event_before_kickoff_raises(delta, raises):
    rows = [_kick(1), _row(1, 1, _KICK_T[1] + delta)]
    if raises:
        with pytest.raises(ValueError, match="before its period's kickoff"):
            _convert(rows)
    else:
        assert _time_of(_convert(rows), 1) == pytest.approx(0.0)


def test_order_insensitive_under_shuffle():
    rows = [
        _kick(1),
        _kick(2),
        _row(1, 1, 118.0, start_time=110.0),
        _row(2, 1, 114.0, start_time=112.0),
        _row(3, 1, 114.0, start_time=112.0),  # an exact tie on both keys: possession_event_id decides
        _row(4, 2, 3105.0),
    ]
    base = _convert(rows)
    rng = np.random.default_rng(0)
    for _ in range(5):
        shuffled = [rows[i] for i in rng.permutation(len(rows))]
        pd.testing.assert_frame_equal(_convert(shuffled), base)
