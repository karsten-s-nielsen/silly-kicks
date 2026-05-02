"""Unit tests for ``silly_kicks.spadl.orientation``.

Covers ``InputConvention``, ``to_spadl_ltr`` (per-convention dispatch + safety
properties), ``detect_input_convention`` (per-tier accuracy + ambiguous fallback),
and ``validate_input_convention`` (warn / raise / silent semantics).

End-to-end invariant tests against real provider fixtures live in
``tests/invariants/`` (PR-S22).
"""

from __future__ import annotations

import os
import warnings
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from silly_kicks.spadl.config import field_length, field_width
from silly_kicks.spadl.orientation import (
    ABSOLUTE_FRAME_HOME_RIGHT,
    PER_PERIOD_ABSOLUTE,
    POSSESSION_PERSPECTIVE,
    InputConvention,
    detect_input_convention,
    to_spadl_ltr,
    validate_input_convention,
)

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

HOME = 100
AWAY = 200


def _two_team_actions(
    *,
    home_x: float = 90.0,
    away_x: float = 90.0,
    home_y: float = 50.0,
    away_y: float = 50.0,
    home_team_id: int | str = HOME,
    away_team_id: int | str = AWAY,
    period_id: int = 1,
) -> pd.DataFrame:
    """Two rows: one home, one away. Useful for verifying mirror behaviour per row."""
    return pd.DataFrame(
        [
            {
                "team_id": home_team_id,
                "period_id": period_id,
                "start_x": home_x,
                "start_y": home_y,
                "end_x": home_x + 5,
                "end_y": home_y,
            },
            {
                "team_id": away_team_id,
                "period_id": period_id,
                "start_x": away_x,
                "start_y": away_y,
                "end_x": away_x + 5,
                "end_y": away_y,
            },
        ]
    )


def _four_period_actions(
    *,
    home_p1_x: float,
    away_p1_x: float,
    home_p2_x: float,
    away_p2_x: float,
) -> pd.DataFrame:
    return pd.concat(
        [
            _two_team_actions(home_x=home_p1_x, away_x=away_p1_x, period_id=1),
            _two_team_actions(home_x=home_p2_x, away_x=away_p2_x, period_id=2),
        ],
        ignore_index=True,
    )


def _synthetic_shot_events(
    *,
    convention: str,
    n_per_group: int = 30,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """Synthetic event set in one of three known conventions, x ∈ [0, 100]."""
    rng = np.random.default_rng(rng_seed)
    rows: list[dict[str, object]] = []
    if convention == "possession_perspective":
        for team in (HOME, AWAY):
            for period in (1, 2):
                for x in rng.uniform(60, 95, n_per_group):
                    rows.append({"game_id": 1, "team_id": team, "period_id": period, "start_x": x, "is_shot": True})
    elif convention == "absolute_no_switch":
        for period in (1, 2):
            for x in rng.uniform(60, 95, n_per_group):
                rows.append({"game_id": 1, "team_id": HOME, "period_id": period, "start_x": x, "is_shot": True})
            for x in rng.uniform(5, 40, n_per_group):
                rows.append({"game_id": 1, "team_id": AWAY, "period_id": period, "start_x": x, "is_shot": True})
    elif convention == "per_period_absolute":
        for x in rng.uniform(60, 95, n_per_group):
            rows.append({"game_id": 1, "team_id": HOME, "period_id": 1, "start_x": x, "is_shot": True})
        for x in rng.uniform(5, 40, n_per_group):
            rows.append({"game_id": 1, "team_id": AWAY, "period_id": 1, "start_x": x, "is_shot": True})
        for x in rng.uniform(5, 40, n_per_group):
            rows.append({"game_id": 1, "team_id": HOME, "period_id": 2, "start_x": x, "is_shot": True})
        for x in rng.uniform(60, 95, n_per_group):
            rows.append({"game_id": 1, "team_id": AWAY, "period_id": 2, "start_x": x, "is_shot": True})
    else:
        raise ValueError(convention)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# InputConvention enum
# ---------------------------------------------------------------------------


class TestInputConvention:
    def test_enum_has_three_values(self):
        assert {c.value for c in InputConvention} == {
            "possession_perspective",
            "absolute_frame_home_right",
            "per_period_absolute",
        }

    def test_module_level_aliases_match_enum(self):
        assert POSSESSION_PERSPECTIVE is InputConvention.POSSESSION_PERSPECTIVE
        assert ABSOLUTE_FRAME_HOME_RIGHT is InputConvention.ABSOLUTE_FRAME_HOME_RIGHT
        assert PER_PERIOD_ABSOLUTE is InputConvention.PER_PERIOD_ABSOLUTE

    def test_enum_is_str_subclass(self):
        # Allows "case == 'possession_perspective'" comparisons
        assert POSSESSION_PERSPECTIVE == "possession_perspective"


# ---------------------------------------------------------------------------
# to_spadl_ltr
# ---------------------------------------------------------------------------


class TestToSpadlLtrPossessionPerspective:
    def test_is_no_op_on_coords(self):
        actions = _two_team_actions(home_x=90, away_x=90, home_y=50, away_y=50)
        out = to_spadl_ltr(actions, input_convention=POSSESSION_PERSPECTIVE, home_team_id=HOME)
        # Both teams at high-x already, no change
        pd.testing.assert_frame_equal(out, actions)

    def test_returns_a_copy_not_the_same_object(self):
        actions = _two_team_actions()
        out = to_spadl_ltr(actions, input_convention=POSSESSION_PERSPECTIVE, home_team_id=HOME)
        assert out is not actions, "must return a copy to prevent caller mutation"


class TestToSpadlLtrAbsoluteFrame:
    def test_mirrors_away_team_x(self):
        # Absolute frame: home at x=90 (attacking right), away at x=10 (attacking left)
        actions = _two_team_actions(home_x=90, away_x=10)
        out = to_spadl_ltr(actions, input_convention=ABSOLUTE_FRAME_HOME_RIGHT, home_team_id=HOME)
        assert out.loc[out["team_id"] == HOME, "start_x"].iloc[0] == pytest.approx(90.0)
        assert out.loc[out["team_id"] == AWAY, "start_x"].iloc[0] == pytest.approx(field_length - 10)

    def test_mirrors_away_team_y(self):
        actions = _two_team_actions(home_x=90, away_x=10, home_y=50, away_y=50)
        out = to_spadl_ltr(actions, input_convention=ABSOLUTE_FRAME_HOME_RIGHT, home_team_id=HOME)
        assert out.loc[out["team_id"] == AWAY, "start_y"].iloc[0] == pytest.approx(field_width - 50)

    def test_mirrors_end_x_and_end_y(self):
        actions = _two_team_actions(home_x=90, away_x=10)
        out = to_spadl_ltr(actions, input_convention=ABSOLUTE_FRAME_HOME_RIGHT, home_team_id=HOME)
        # end_x was start_x + 5 = 15 for away; mirrored = 105 - 15 = 90
        assert out.loc[out["team_id"] == AWAY, "end_x"].iloc[0] == pytest.approx(field_length - 15)
        assert out.loc[out["team_id"] == AWAY, "end_y"].iloc[0] == pytest.approx(field_width - 50)

    def test_does_not_mutate_input(self):
        actions = _two_team_actions(home_x=90, away_x=10)
        actions_before = actions.copy()
        to_spadl_ltr(actions, input_convention=ABSOLUTE_FRAME_HOME_RIGHT, home_team_id=HOME)
        pd.testing.assert_frame_equal(actions, actions_before)

    def test_works_with_string_team_id_dfl_style(self):
        actions = _two_team_actions(
            home_x=90,
            away_x=10,
            home_team_id="DFL-CLU-HOME",
            away_team_id="DFL-CLU-AWAY",
        )
        out = to_spadl_ltr(
            actions,
            input_convention=ABSOLUTE_FRAME_HOME_RIGHT,
            home_team_id="DFL-CLU-HOME",
        )
        assert out.loc[out["team_id"] == "DFL-CLU-AWAY", "start_x"].iloc[0] == pytest.approx(field_length - 10)

    def test_no_away_rows_returns_unchanged_copy(self):
        # Edge case: only home team has actions
        actions = pd.DataFrame(
            [
                {
                    "team_id": HOME,
                    "period_id": 1,
                    "start_x": 90.0,
                    "start_y": 50.0,
                    "end_x": 95.0,
                    "end_y": 50.0,
                }
            ]
        )
        out = to_spadl_ltr(actions, input_convention=ABSOLUTE_FRAME_HOME_RIGHT, home_team_id=HOME)
        pd.testing.assert_frame_equal(out, actions)


class TestToSpadlLtrPerPeriod:
    def test_handles_halftime_switch(self):
        # Raw input (per-period absolute):
        #   P1: home at x=90 (attacking right), away at x=10 (attacking left)
        #   P2: home at x=10 (attacking left after switch), away at x=90 (attacking right)
        # After SPADL LTR (each team's actions oriented as if attacking right):
        #   P1: home stays at 90 (already LTR), away mirrors to 105-10=95
        #   P2: home mirrors to 105-10=95, away stays at 90
        # Net: ALL rows should land at high-x (> field_length/2 = 52.5)
        actions = _four_period_actions(home_p1_x=90, away_p1_x=10, home_p2_x=10, away_p2_x=90)
        flips = {1: True, 2: False}  # home attacks right in P1, left in P2
        out = to_spadl_ltr(
            actions,
            input_convention=PER_PERIOD_ABSOLUTE,
            home_team_id=HOME,
            home_attacks_right_per_period=flips,
        )
        # Invariant: after SPADL LTR, every team in every period attacks high-x.
        for _, row in out.iterrows():
            assert row["start_x"] > field_length / 2, (
                f"team={row['team_id']} period={row['period_id']} start_x={row['start_x']} should be > 52.5"
            )

    def test_per_period_handles_string_team_ids(self):
        actions = _four_period_actions(home_p1_x=90, away_p1_x=10, home_p2_x=10, away_p2_x=90)
        actions["team_id"] = actions["team_id"].map({HOME: "DFL-CLU-A", AWAY: "DFL-CLU-B"})
        flips = {1: True, 2: False}
        out = to_spadl_ltr(
            actions,
            input_convention=PER_PERIOD_ABSOLUTE,
            home_team_id="DFL-CLU-A",
            home_attacks_right_per_period=flips,
        )
        assert (out["start_x"] > field_length / 2).all()

    def test_handles_extra_time_periods(self):
        from silly_kicks.tracking._direction import home_attacks_right_per_period

        flips = home_attacks_right_per_period(
            home_team_start_left=True,
            home_team_start_left_extratime=False,
        )
        # P3 home attacks left (since home_team_start_left_extratime=False → False)
        rows = [
            {"team_id": HOME, "period_id": 3, "start_x": 10.0, "start_y": 50.0, "end_x": 15.0, "end_y": 50.0},
            {"team_id": AWAY, "period_id": 3, "start_x": 90.0, "start_y": 50.0, "end_x": 95.0, "end_y": 50.0},
        ]
        actions = pd.DataFrame(rows)
        out = to_spadl_ltr(
            actions,
            input_convention=PER_PERIOD_ABSOLUTE,
            home_team_id=HOME,
            home_attacks_right_per_period=flips,
        )
        # Both teams should now attack high-x in P3
        assert (out["start_x"] > 50).all(), out.to_dict("records")

    def test_requires_home_attacks_right_per_period(self):
        actions = _four_period_actions(home_p1_x=90, away_p1_x=10, home_p2_x=10, away_p2_x=90)
        with pytest.raises(ValueError, match="PER_PERIOD_ABSOLUTE requires"):
            to_spadl_ltr(actions, input_convention=PER_PERIOD_ABSOLUTE, home_team_id=HOME)

    def test_missing_period_raises(self):
        actions = _four_period_actions(home_p1_x=90, away_p1_x=10, home_p2_x=10, away_p2_x=90)
        with pytest.raises(ValueError, match="missing entries for periods"):
            to_spadl_ltr(
                actions,
                input_convention=PER_PERIOD_ABSOLUTE,
                home_team_id=HOME,
                home_attacks_right_per_period={1: True},  # missing P2
            )

    def test_does_not_mutate_input(self):
        actions = _four_period_actions(home_p1_x=90, away_p1_x=10, home_p2_x=10, away_p2_x=90)
        actions_before = actions.copy()
        to_spadl_ltr(
            actions,
            input_convention=PER_PERIOD_ABSOLUTE,
            home_team_id=HOME,
            home_attacks_right_per_period={1: True, 2: False},
        )
        pd.testing.assert_frame_equal(actions, actions_before)


# ---------------------------------------------------------------------------
# detect_input_convention
# ---------------------------------------------------------------------------


class TestDetectInputConvention:
    def test_detects_possession_perspective(self):
        events = _synthetic_shot_events(convention="possession_perspective", n_per_group=30)
        result = detect_input_convention(
            events,
            match_col="game_id",
            x_max=100,
            is_shot_col="is_shot",
        )
        assert result.convention is InputConvention.POSSESSION_PERSPECTIVE
        assert result.confidence == "high"

    def test_detects_absolute_no_switch(self):
        events = _synthetic_shot_events(convention="absolute_no_switch", n_per_group=30)
        result = detect_input_convention(
            events,
            match_col="game_id",
            x_max=100,
            is_shot_col="is_shot",
        )
        assert result.convention is InputConvention.ABSOLUTE_FRAME_HOME_RIGHT
        assert result.confidence == "high"

    def test_detects_per_period_absolute(self):
        events = _synthetic_shot_events(convention="per_period_absolute", n_per_group=30)
        result = detect_input_convention(
            events,
            match_col="game_id",
            x_max=100,
            is_shot_col="is_shot",
        )
        assert result.convention is InputConvention.PER_PERIOD_ABSOLUTE
        assert result.confidence == "high"

    def test_returns_ambiguous_on_too_few_shots(self):
        events = _synthetic_shot_events(convention="absolute_no_switch", n_per_group=2)
        result = detect_input_convention(
            events,
            match_col="game_id",
            x_max=100,
            is_shot_col="is_shot",
        )
        assert result.convention is None
        assert result.confidence in ("low", "ambiguous")

    def test_medium_confidence_on_5_to_9_shots(self):
        events = _synthetic_shot_events(convention="absolute_no_switch", n_per_group=7)
        result = detect_input_convention(
            events,
            match_col="game_id",
            x_max=100,
            is_shot_col="is_shot",
        )
        assert result.convention is InputConvention.ABSOLUTE_FRAME_HOME_RIGHT
        assert result.confidence == "medium"

    def test_high_confidence_requires_10_per_group(self):
        events_high = _synthetic_shot_events(convention="absolute_no_switch", n_per_group=10)
        result_high = detect_input_convention(
            events_high,
            match_col="game_id",
            x_max=100,
            is_shot_col="is_shot",
        )
        assert result_high.confidence == "high"

        events_just_under = _synthetic_shot_events(convention="absolute_no_switch", n_per_group=9)
        result_under = detect_input_convention(
            events_just_under,
            match_col="game_id",
            x_max=100,
            is_shot_col="is_shot",
        )
        assert result_under.confidence == "medium"

    def test_is_shot_col_filters(self):
        # Build events with shots clustered correctly but non-shots not
        rows = []
        rng = np.random.default_rng(0)
        # Shots: possession-perspective (both teams high)
        for team in (HOME, AWAY):
            for period in (1, 2):
                for x in rng.uniform(80, 95, 12):
                    rows.append({"game_id": 1, "team_id": team, "period_id": period, "start_x": x, "is_shot": True})
        # Non-shots: scattered noise
        for team in (HOME, AWAY):
            for period in (1, 2):
                for x in rng.uniform(0, 100, 50):
                    rows.append({"game_id": 1, "team_id": team, "period_id": period, "start_x": x, "is_shot": False})
        events = pd.DataFrame(rows)

        result_with_filter = detect_input_convention(
            events,
            match_col="game_id",
            x_max=100,
            is_shot_col="is_shot",
        )
        # Shots-only signal: clean possession_perspective
        assert result_with_filter.convention is InputConvention.POSSESSION_PERSPECTIVE

    def test_returns_low_when_fewer_than_2_reliable_groups(self):
        # Single (team, period) group with enough shots
        rows = [{"game_id": 1, "team_id": HOME, "period_id": 1, "start_x": 90.0, "is_shot": True}] * 10
        events = pd.DataFrame(rows)
        result = detect_input_convention(events, match_col="game_id", x_max=100, is_shot_col="is_shot")
        assert result.convention is None
        assert result.confidence == "low"

    def test_diagnostics_contain_groups(self):
        events = _synthetic_shot_events(convention="possession_perspective", n_per_group=30)
        result = detect_input_convention(
            events,
            match_col="game_id",
            x_max=100,
            is_shot_col="is_shot",
        )
        assert "groups" in result.diagnostics


# ---------------------------------------------------------------------------
# validate_input_convention
# ---------------------------------------------------------------------------


class TestValidateInputConvention:
    def test_warn_on_mismatch_default(self):
        events = _synthetic_shot_events(convention="per_period_absolute", n_per_group=30)
        with pytest.warns(UserWarning, match="declared=absolute_frame_home_right"):
            result = validate_input_convention(
                events,
                declared=ABSOLUTE_FRAME_HOME_RIGHT,
                match_col="game_id",
                x_max=100,
                is_shot_col="is_shot",
                on_mismatch="warn",
            )
        assert result.convention is InputConvention.PER_PERIOD_ABSOLUTE

    def test_raise_when_explicitly_requested(self):
        events = _synthetic_shot_events(convention="possession_perspective", n_per_group=30)
        with pytest.raises(ValueError, match="declared=absolute_frame_home_right"):
            validate_input_convention(
                events,
                declared=ABSOLUTE_FRAME_HOME_RIGHT,
                match_col="game_id",
                x_max=100,
                is_shot_col="is_shot",
                on_mismatch="raise",
            )

    def test_silent_when_match(self):
        events = _synthetic_shot_events(convention="absolute_no_switch", n_per_group=30)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = validate_input_convention(
                events,
                declared=ABSOLUTE_FRAME_HOME_RIGHT,
                match_col="game_id",
                x_max=100,
                is_shot_col="is_shot",
                on_mismatch="warn",
            )
        assert result.convention is InputConvention.ABSOLUTE_FRAME_HOME_RIGHT

    def test_silent_when_signal_too_weak(self):
        # 2 shots per group → ambiguous → defer silently
        events = _synthetic_shot_events(convention="possession_perspective", n_per_group=2)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = validate_input_convention(
                events,
                declared=ABSOLUTE_FRAME_HOME_RIGHT,
                match_col="game_id",
                x_max=100,
                is_shot_col="is_shot",
                on_mismatch="warn",
            )
        assert result.convention is None

    def test_silent_mode_suppresses_even_loud_mismatch(self):
        events = _synthetic_shot_events(convention="possession_perspective", n_per_group=30)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            validate_input_convention(
                events,
                declared=ABSOLUTE_FRAME_HOME_RIGHT,
                match_col="game_id",
                x_max=100,
                is_shot_col="is_shot",
                on_mismatch="silent",
            )

    def test_env_var_promotes_warn_to_raise(self):
        events = _synthetic_shot_events(convention="possession_perspective", n_per_group=30)
        with mock.patch.dict(os.environ, {"SILLY_KICKS_ASSERT_INVARIANTS": "1"}):
            with pytest.raises(ValueError, match="declared=absolute_frame_home_right"):
                # on_mismatch=None → resolves to "raise" under env var
                validate_input_convention(
                    events,
                    declared=ABSOLUTE_FRAME_HOME_RIGHT,
                    match_col="game_id",
                    x_max=100,
                    is_shot_col="is_shot",
                )

    def test_default_without_env_var_is_warn(self):
        events = _synthetic_shot_events(convention="possession_perspective", n_per_group=30)
        # Ensure the env var is not set
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SILLY_KICKS_ASSERT_INVARIANTS", None)
            with pytest.warns(UserWarning):
                validate_input_convention(
                    events,
                    declared=ABSOLUTE_FRAME_HOME_RIGHT,
                    match_col="game_id",
                    x_max=100,
                    is_shot_col="is_shot",
                )

    def test_requires_match_col_in_kwargs(self):
        events = _synthetic_shot_events(convention="possession_perspective", n_per_group=30)
        with pytest.raises(ValueError, match="'match_col' and 'x_max' must be"):
            validate_input_convention(
                events,
                declared=ABSOLUTE_FRAME_HOME_RIGHT,
                x_max=100,  # match_col missing
            )
