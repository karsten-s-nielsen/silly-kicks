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
    # Post-F2 (spec section 5): the goal kick is a DERIVED restart row (action_provenance "derived"),
    # keyed off the interruption marker on the possession before the stoppage -- not a relabel of the
    # receiving possession. result_id still follows the next-possession team (retained -> success,
    # lost -> fail), now on the derived row.
    def test_goalkick_lost_to_opponent_is_fail(self):
        events = _events(
            [
                dict(
                    period=1,
                    time_start="10:00.0",
                    team_id="team_home",
                    player_id="p0",
                    game_interruption_after="goal_kick_for",  # marker: team_home awarded a goal kick
                ),
                dict(
                    period=1,
                    time_start="10:05.0",
                    team_id="team_away",
                    player_id="p12",
                    game_interruption_before="goal_kick_against",  # team_away first-touches it (lost)
                ),
            ]
        )
        actions, _ = convert_to_actions(events, _META)
        gk = actions[actions["type_id"] == spadlconfig.actiontype_id["goalkick"]].iloc[0]
        assert gk["action_provenance"] == "derived"
        assert gk["team_id"] == "team_home"
        assert gk["result_id"] == spadlconfig.result_id["fail"]  # opponent next -> lost

    def test_goalkick_retained_is_success(self):
        events = _events(
            [
                dict(
                    period=1,
                    time_start="10:00.0",
                    team_id="team_home",
                    player_id="p0",
                    game_interruption_after="goal_kick_for",
                ),
                dict(
                    period=1,
                    time_start="10:05.0",
                    team_id="team_home",
                    player_id="p2",
                    game_interruption_before="goal_kick_for",  # team_home retains
                ),
            ]
        )
        actions, _ = convert_to_actions(events, _META)
        gk = actions[actions["type_id"] == spadlconfig.actiontype_id["goalkick"]].iloc[0]
        assert gk["action_provenance"] == "derived"
        assert gk["result_id"] == spadlconfig.result_id["success"]  # teammate next -> retained


class TestDerivedRestartPostProcessors:
    """Spec section 5.8: the post-processors must leave a derived restart row alone."""

    def _convert_corner(self):
        events = _events(
            [
                dict(
                    period=1,
                    time_start="10:00.0",
                    team_id="team_home",
                    player_id="p0",
                    game_interruption_after="corner_for",
                ),
                dict(
                    period=1,
                    time_start="10:05.0",
                    team_id="team_home",
                    player_id="p1",
                    game_interruption_before="corner_for",
                    x_start=40.0,
                    x_end=42.0,
                ),
            ]
        )
        return convert_to_actions(events, _META)[0]

    def test_derive_end_coordinates_leaves_derived_start_nan(self):
        actions = self._convert_corner()
        corner = actions[actions["type_id"] == spadlconfig.actiontype_id["corner_crossed"]].iloc[0]
        assert corner["action_provenance"] == "derived"
        # _derive_end_coordinates must not fill the NaN start of a derived restart (its end is set).
        assert pd.isna(corner["start_x"]) and pd.isna(corner["start_y"])
        assert pd.notna(corner["end_x"])

    def test_add_dribbles_does_not_bridge_derived_row(self):
        actions = self._convert_corner()
        dribbles = actions[actions["type_id"] == spadlconfig.actiontype_id["dribble"]]
        # A dribble bridging the NaN-start derived row would carry a NaN coordinate; none may.
        assert dribbles[["start_x", "start_y", "end_x", "end_y"]].notna().all().all()
