"""W8: DFL event period, time and actor derivation (spec 2026-11-08 tracking-batch section 4.8; plan Task 14).

The test names follow the spec's acceptance list. The rules are the parser's (``parse_dfl_events``), single-sourced
in ``silly_kicks.providers.sportec._event_derivation`` and exposed through public adapters for any DFL consumer.
"""

from __future__ import annotations

import sys
import warnings
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from scripts._bitwise import assert_bits_equal
from silly_kicks.providers.sportec import (
    DFL_EVENT_TYPE_PREFIX,
    DFL_NESTED_PREFIX,
    DflEventTimeWarning,
    DflGameSectionWarning,
    DflPeriodReport,
    MatchInfo,
    derive_dfl_event_periods,
    dfl_actor_columns,
    dfl_event_column,
    resolve_dfl_event_actors,
)
from silly_kicks.providers.sportec._event_derivation import _dfl_actor_rule

_DAY = "2024-03-09T"


def _t(clock: str, tz: str = "+00:00") -> str:
    return f"{_DAY}{clock}{tz}"


def _derive(rows: Sequence[tuple[str, str | None, str | None]]):
    """``rows``: (event_type, game_section, event_time) in document order."""
    types = pd.Series([r[0] for r in rows], dtype=object)
    sections = pd.Series([r[1] for r in rows], dtype=object)
    times = pd.Series([r[2] for r in rows], dtype=object)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out, report = derive_dfl_event_periods(times, types, sections)
    return out, report, [w.category for w in caught]


def _ko(section: str, clock: str):
    return ("KickOff", section, _t(clock))


def _ev(clock: str, tz: str = "+00:00"):
    return ("Play", None, _t(clock, tz))


# ---- W8.1 the period rule ---------------------------------------------------------------------------------------


def test_latest_kickoff_rule_nonmonotone():
    rows = [_ko("firstHalf", "15:00:00"), _ko("secondHalf", "16:00:00"), _ko("extraTimeFirstHalf", "15:30:00")]
    rows += [_ev("15:45:00"), _ev("16:10:00"), _ev("15:10:00")]
    out, report, _w = _derive(rows)
    assert out["period"].tolist()[3:] == [3, 2, 1]  # the latest kickoff <= the event, not the largest period
    assert out["timestamp_seconds"].tolist()[3] == 900.0
    assert list(report.kickoff_times) == [1, 2, 3]


def test_equal_kickoff_times_first_wins():
    out, _report, _w = _derive([_ko("firstHalf", "15:00:00"), _ko("secondHalf", "15:00:00"), _ev("15:10:00")])
    assert out["period"].tolist() == [1, 1, 1]


def test_two_pass_first_half_event_after_second_kickoff():
    rows = [_ko("firstHalf", "15:00:00"), _ev("15:20:00"), _ko("secondHalf", "16:00:00"), _ev("15:40:00")]
    out, _report, _w = _derive(rows)
    assert out["period"].tolist() == [1, 1, 2, 1]
    assert out["timestamp_seconds"].iloc[3] == 2400.0


def test_pre_kickoff_counted_not_warned():
    out, report, caught = _derive([_ev("14:50:00"), _ko("firstHalf", "15:00:00"), _ev("15:01:00")])
    assert pd.isna(out["period"].iloc[0]) and np.isnan(out["timestamp_seconds"].iloc[0])
    assert report.n_pre_kickoff == 1 and report.n_unparseable_time == 0
    assert DflEventTimeWarning not in caught


def test_unparseable_and_incomparable_warned_once():
    rows = [_ko("firstHalf", "15:00:00"), ("Play", None, "garbage"), ("Play", None, ""), ("Play", None, None)]
    rows += [("Play", None, f"{_DAY}15:10:00"), ("Play", None, f"{_DAY}15:11:00")]  # naive vs an aware kickoff
    out, report, caught = _derive(rows)
    assert out["period"].isna().tolist() == [False, True, True, True, True, True]
    assert report.n_unparseable_time == 3 and report.n_incomparable_time == 2 and report.n_pre_kickoff == 0
    assert caught.count(DflEventTimeWarning) == 1


def test_unrecognized_section_warns_and_reports():
    out, report, caught = _derive([("KickOff", "overtimeBonus", _t("15:00:00")), _ko("firstHalf", "15:00:00")])
    assert report.unrecognized_sections == ("overtimeBonus",)
    assert caught.count(DflGameSectionWarning) == 1
    assert out["period"].tolist() == [1, 1]  # a row's period comes from its time, whatever its own section


def test_sectionless_kickoff_after_period_start_is_a_restart():
    """A KickOff with no GameSection after a period started is a restart after a goal (all 18 such KickOffs of the
    7 DFL corpus matches): counted, never warned (owner ruling 2026-11-12), and never a period start."""
    rows = [_ko("firstHalf", "15:00:00"), _ev("15:10:00"), ("KickOff", None, _t("15:11:00")), _ev("15:12:00")]
    rows += [_ko("secondHalf", "16:00:00"), ("KickOff", "", _t("16:20:00")), _ev("16:21:00")]
    out, report, caught = _derive(rows)
    assert DflGameSectionWarning not in caught
    assert report.n_restart_kickoffs == 2 and report.unrecognized_sections == ()
    assert out["period"].tolist() == [1, 1, 1, 1, 2, 2, 2]
    assert out["timestamp_seconds"].tolist()[5] == 1200.0


def test_sectionless_kickoff_before_any_period_start_warns():
    """No period has started, so a section-less KickOff can only be a period start without its section."""
    rows = [("KickOff", None, _t("15:00:00")), _ko("firstHalf", "15:00:05"), _ko("secondHalf", "16:00:00")]
    _out, report, caught = _derive(rows)
    assert report.unrecognized_sections == ("",) and report.n_restart_kickoffs == 0
    assert caught.count(DflGameSectionWarning) == 1


def test_restarts_without_a_second_half_kickoff_warn():
    """Restarts but no recognised secondHalf KickOff: one section-less KickOff may be the second-half start."""
    rows = [_ko("firstHalf", "15:00:00"), ("KickOff", None, _t("15:30:00")), ("KickOff", None, _t("16:00:00"))]
    rows.append(_ev("16:10:00"))
    out, report, caught = _derive(rows)
    assert report.n_restart_kickoffs == 2 and report.unrecognized_sections == ()
    assert caught.count(DflGameSectionWarning) == 1
    assert out["period"].tolist() == [1, 1, 1, 1]  # the values stay the parser's: warned, not changed


def test_unparseable_kickoff_skipped_next_wins():
    rows = [("KickOff", "firstHalf", "not-a-time"), _ko("firstHalf", "15:00:00"), _ko("firstHalf", "15:05:00")]
    rows.append(_ev("15:02:00"))
    out, report, _w = _derive(rows)
    assert report.kickoff_times == {1: datetime(2024, 3, 9, 15, 0, tzinfo=timezone.utc)}
    assert report.skipped_kickoffs == (1,)  # the later parseable KickOff of period 1 was ignored
    assert out["timestamp_seconds"].iloc[3] == 120.0


def test_round_half_way_python_semantics():
    start = datetime(2024, 3, 9, 15, 0, tzinfo=timezone.utc)
    event = start + timedelta(microseconds=150)  # 0.00015 s: a half-way case at 4 decimals
    out, _report, _w = _derive([("KickOff", "firstHalf", start.isoformat()), ("Play", None, event.isoformat())])
    assert out["timestamp_seconds"].iloc[1] == round((event - start).total_seconds(), 4)


@pytest.mark.parametrize("case", ["utc", "+02:00", "sub_us"])
def test_string_vs_datetime64_bit_equal(case):
    tz = "+00:00" if case == "utc" else "+02:00"
    clocks = ["15:00:00", "15:00:01.123456", "15:47:12.000001", "16:00:00", "16:30:59.999999"]
    strings = [f"{_DAY}{c}{tz}" for c in clocks]
    types = pd.Series(["KickOff", "Play", "Play", "KickOff", "Play"], dtype=object)
    sections = pd.Series(["firstHalf", None, None, "secondHalf", None], dtype=object)
    stamps = pd.to_datetime(pd.Series(strings), format="ISO8601")
    if case == "sub_us":
        stamps = stamps + pd.to_timedelta([0, 999, 1, 0, 500], unit="ns")  # nanoseconds the floor drops
        floored, _r = derive_dfl_event_periods(stamps.dt.floor("us"), types, sections)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no "Discarding nonzero nanoseconds" warning
            got, _r2 = derive_dfl_event_periods(stamps, types, sections)
        assert_bits_equal(got, floored)
        return
    from_strings, _r = derive_dfl_event_periods(pd.Series(strings, dtype=object), types, sections)
    from_stamps, _r2 = derive_dfl_event_periods(stamps, types, sections)
    assert_bits_equal(from_stamps, from_strings)
    assert from_strings["period"].tolist() == [1, 1, 1, 2, 2]


@pytest.mark.skipif(sys.version_info[:2] != (3, 10), reason="Python 3.10's fromisoformat rules")
def test_py310_rejects_Z_and_7_digits_as_unparseable():
    rows = [_ko("firstHalf", "15:00:00"), ("Play", None, f"{_DAY}15:01:00Z"), ("Play", None, _t("15:01:00.1234567"))]
    out, report, _w = _derive(rows)
    assert out["period"].isna().tolist() == [False, True, True] and report.n_unparseable_time == 2


def test_shootout_period_5():
    out, _report, _w = _derive([_ko("penaltyShootout", "18:00:00"), _ev("18:01:00")])
    assert out["period"].tolist() == [5, 5]


def test_empty_series():
    empty = pd.Series([], dtype=object)
    out, report = derive_dfl_event_periods(empty, empty, empty)
    assert list(out.columns) == ["period", "timestamp_seconds"] and len(out) == 0
    assert str(out["period"].dtype) == "Int64" and out["timestamp_seconds"].dtype == np.float64
    assert report == DflPeriodReport({}, 0, 0, 0, (), ())


def test_no_kickoff_is_all_na():
    out, report, _w = _derive([_ev("15:00:00"), _ev("15:01:00")])
    assert out["period"].isna().all() and report.kickoff_times == {}


# ---- W8.2 / W8.3 the actor rule and the column contract -----------------------------------------------------------


@pytest.mark.parametrize(
    ("event_type", "attrs", "expected"),
    [
        ("TacklingGame", {"Winner": "W1", "Player": "P1"}, "W1"),
        ("TacklingGame", {"Player": "P1"}, "P1"),
        ("TacklingGame", {"Winner": "", "Player": "P1"}, "P1"),
        ("Play", {"Player": "P2"}, "P2"),
        ("Play", {}, ""),
        ("SomeNewDflTag", {"Player": "P3"}, "P3"),  # an unprefixed type: the default ["Player"]
    ],
)
def test_actor_rule(event_type, attrs, expected):
    assert _dfl_actor_rule(event_type, attrs) == expected


def test_column_contract():
    assert dfl_event_column("TacklingGame", "Winner") == "tackle_winner"
    assert dfl_event_column("KickOff", "Player", nested="Play") == "play_player"
    assert dfl_event_column("SomeNewDflTag", "Player") == "some_new_dfl_tag_player"
    assert dfl_actor_columns("TacklingGame") == ("tackle_winner", "tackle_player")
    assert dfl_actor_columns("KickOff") == ("kickoff_play_present", "kickoff_first_play_player", "kickoff_player")
    assert DFL_EVENT_TYPE_PREFIX["Play"] == "play" and DFL_NESTED_PREFIX["SavedShot"] == "shot_outcome"
    with pytest.raises(TypeError):
        DFL_EVENT_TYPE_PREFIX["Play"] = "x"  # type: ignore[index]


def _info() -> MatchInfo:
    return MatchInfo(
        home_team_id="DFL-CLU-H",
        away_team_id="DFL-CLU-A",
        player_team_map={"P1": "home", "P2": "away", "W1": "away"},
        gk_player_ids=frozenset(),
        competition_id="DFL-COM-1",
        season_id="DFL-SEA-1",
        pitch_x=105.0,
        pitch_y=68.0,
    )


def test_resolve_dfl_event_actors():
    events = pd.DataFrame(
        {
            "event_type": ["KickOff", "KickOff", "KickOff", "TacklingGame", "Play", "Play"],
            "kickoff_play_present": pd.array([True, False, True, pd.NA, pd.NA, pd.NA], dtype="boolean"),
            "kickoff_first_play_player": [
                "P1",
                None,
                "",
                None,
                None,
                None,
            ],  # the FIRST of two Plays; a Player-less Play
            "kickoff_player": [None, "P2", None, None, None, None],  # read only without a nested Play
            "tackle_winner": [None, None, None, "W1", None, None],
            "tackle_player": [None, None, None, "P1", None, None],
            "play_player": [None, None, None, None, "P2", "P99"],  # P99: not on the team sheet
        }
    )
    out = resolve_dfl_event_actors(events, _info())
    assert list(out.columns) == ["player_id", "team", "team_id_native"]
    assert out["player_id"].tolist() == ["P1", "P2", "", "W1", "P2", "P99"]
    assert out["team"].tolist() == ["home", "away", "unknown", "away", "away", "unknown"]
    assert out["team_id_native"].tolist() == ["DFL-CLU-H", "DFL-CLU-A", None, "DFL-CLU-A", "DFL-CLU-A", None]


def test_missing_kickoff_play_present_raises():
    events = pd.DataFrame({"event_type": ["KickOff"], "kickoff_player": ["P1"]})
    with pytest.raises(ValueError, match="kickoff_play_present"):
        resolve_dfl_event_actors(events, _info())


# ---- W8.6 the qualifier fill (R2) ---------------------------------------------------------------------------------


def test_qualifier_team_dtype_mismatch_detection():
    from silly_kicks.providers.sportec.parse import _resolve_idsse_team_from_qualifiers

    # an int CLU id in the qualifier column against str match-level ids: the raw `==` silently missed it
    df = pd.DataFrame(
        {
            "team": ["unknown", "unknown", "home"],
            "freekick_team": pd.Series([8, 9, None], dtype=object),
            "home_team_id_native": ["8", "8", "8"],
            "away_team_id_native": ["9", "9", "9"],
        }
    )
    _resolve_idsse_team_from_qualifiers(df)
    assert df["team"].tolist() == ["home", "away", "home"]
