"""SkillCorner converter data-quality fixes (2026-06-09).

BUG 1: time_seconds must be PERIOD-RELATIVE (ADR-017), not the continuous match clock.
       SkillCorner's `time_start` is the broadcast clock (2nd half shows 45:00+), so 2nd-half
       events parsed to ~2700-5800 s and could not link to the period-relative tracking frames.
BUG 2: goalkick result_id was hard-wired to `success`; it must use the same possession-based
       `same_team_next` logic as every other open-play/set-piece action.
"""

import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl.skillcorner import _to_period_relative, convert_to_actions

_META = {"pitch_length": 105.0, "pitch_width": 68.0, "home_team": {"id": "team_home"}}

# Defaults for every column convert_to_actions touches; per-row overrides via _events().
_COLS = dict(
    event_type="player_possession",
    start_type="pass_reception",
    end_type="pass",
    x_start=0.0,
    y_start=0.0,
    x_end=5.0,
    y_end=0.0,
    game_interruption_before="",
    game_interruption_after="",
    is_header=False,
    hand_pass=False,
    player_targeted_x_reception=float("nan"),
    player_targeted_y_reception=float("nan"),
    player_targeted_third_pass="",
    player_targeted_channel_pass="",
)


def _events(rows: list[dict]) -> pd.DataFrame:
    out = []
    for i, r in enumerate(rows):
        d = dict(_COLS)
        d.update(r)
        d.setdefault("event_id", f"e{i}")
        out.append(d)
    return pd.DataFrame(out)


class TestPeriodRelativeTime:
    def test_to_period_relative_rebases_each_period(self):
        # continuous broadcast clock -> period-relative via the nominal period-start offsets
        s = _to_period_relative(pd.Series([600.0, 2843.5, 5500.0, 6400.0]), pd.Series([1, 2, 3, 4]))
        assert list(s) == [600.0, 143.5, 100.0, 100.0]

    def test_convert_rebases_period2_time(self):
        events = _events(
            [
                dict(period=1, time_start="10:00.0", team_id="team_home", player_id="p1"),
                dict(period=2, time_start="50:00.0", team_id="team_home", player_id="p2"),
            ]
        )
        actions, _ = convert_to_actions(events, _META)
        p1 = actions[actions["period_id"] == 1]["time_seconds"]
        p2 = actions[actions["period_id"] == 2]["time_seconds"]
        assert p1.iloc[0] == pytest.approx(600.0)
        # 50:00 in period 2 is 5:00 of the 2nd half -> 300 s period-relative, NOT 3000 (the bug)
        assert p2.max() == pytest.approx(300.0)


class TestGoalkickResult:
    def test_goalkick_lost_to_opponent_is_fail(self):
        events = _events(
            [
                dict(
                    period=1,
                    time_start="10:00.0",
                    team_id="team_home",
                    player_id="gk1",
                    game_interruption_before="goal_kick_for",
                    x_start=-45.0,
                    x_end=-20.0,
                ),
                dict(period=1, time_start="10:05.0", team_id="team_away", player_id="p12"),
            ]
        )
        actions, _ = convert_to_actions(events, _META)
        gk = actions[actions["type_id"] == spadlconfig.actiontype_id["goalkick"]].iloc[0]
        assert gk["result_id"] == spadlconfig.result_id["fail"]  # opponent next -> lost

    def test_goalkick_retained_is_success(self):
        events = _events(
            [
                dict(
                    period=1,
                    time_start="10:00.0",
                    team_id="team_home",
                    player_id="gk1",
                    game_interruption_before="goal_kick_for",
                ),
                dict(period=1, time_start="10:05.0", team_id="team_home", player_id="p2"),
            ]
        )
        actions, _ = convert_to_actions(events, _META)
        gk = actions[actions["type_id"] == spadlconfig.actiontype_id["goalkick"]].iloc[0]
        assert gk["result_id"] == spadlconfig.result_id["success"]  # teammate next -> retained
