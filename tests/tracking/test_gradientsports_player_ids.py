"""TF-24 PR-A — GS tracking jersey->roster player-id resolution."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking.gradientsports import (
    GradientsportsRosterReport,
    add_gradientsports_player_ids,
)

HOME, AWAY = 366, 51


def _roster() -> pd.DataFrame:
    # team.id is string in raw GS; player.id is string; positionGroupType literal "GK".
    return pd.DataFrame(
        {
            "team_id": ["366", "366", "51", "51"],
            "shirt_number": ["8", "1", "10", "1"],
            "player_id": [8342, 8326, 940, 12],
            "position_group_type": ["AM", "GK", "FW", "GK"],
        }
    )


def _jersey_frames() -> pd.DataFrame:
    # 1 frame: home #8 (outfield), home #1 (GK), away #10 (outfield), away #1 (GK), ball.
    base = dict(
        game_id=1,
        period_id=1,
        frame_id=1,
        time_seconds=0.0,
        frame_rate=30.0,
        z=0.0,
        speed_native=1.0,
        ball_state="alive",
    )
    rows = [
        {**base, "team_side": "home", "jersey_number": "8", "is_ball": False, "x_centered": 0.0, "y_centered": 0.0},
        {**base, "team_side": "home", "jersey_number": "1", "is_ball": False, "x_centered": -40.0, "y_centered": 0.0},
        {**base, "team_side": "away", "jersey_number": "10", "is_ball": False, "x_centered": 5.0, "y_centered": 2.0},
        {**base, "team_side": "away", "jersey_number": "1", "is_ball": False, "x_centered": 40.0, "y_centered": 0.0},
        {**base, "team_side": None, "jersey_number": None, "is_ball": True, "x_centered": 1.0, "y_centered": 1.0},
    ]
    return pd.DataFrame(rows)


class TestHappyPath:
    def test_join_and_dtypes(self):
        frames = _jersey_frames()
        out, report = add_gradientsports_player_ids(frames, _roster(), home_team_id=HOME, away_team_id=AWAY)

        assert str(out["player_id"].dtype) == "Int64"
        assert str(out["team_id"].dtype) == "Int64"
        assert out["is_goalkeeper"].dtype == bool
        # home #8 -> 8342, home #1 -> 8326, away #10 -> 940, away #1 -> 12, ball -> NA
        assert out.loc[0, "player_id"] == 8342 and out.loc[0, "team_id"] == HOME
        assert out.loc[1, "player_id"] == 8326 and bool(out.loc[1, "is_goalkeeper"]) is True
        assert out.loc[2, "player_id"] == 940 and out.loc[2, "team_id"] == AWAY
        assert pd.isna(out.loc[4, "player_id"]) and pd.isna(out.loc[4, "team_id"])
        assert bool(out.loc[0, "is_goalkeeper"]) is False
        assert isinstance(report, GradientsportsRosterReport)
        assert report.n_player_rows == 4 and report.n_matched == 4 and report.n_unmatched == 0
        assert report.n_duplicate_roster_keys == 0

    def test_does_not_mutate_input(self):
        frames = _jersey_frames()
        before = frames.copy()
        add_gradientsports_player_ids(frames, _roster(), home_team_id=HOME, away_team_id=AWAY)
        pd.testing.assert_frame_equal(frames, before)

    def test_row_count_preserved(self):
        frames = _jersey_frames()
        out, _ = add_gradientsports_player_ids(frames, _roster(), home_team_id=HOME, away_team_id=AWAY)
        assert len(out) == len(frames)

    def test_missing_required_columns_raise(self):
        with pytest.raises(ValueError, match="jersey_frames"):
            add_gradientsports_player_ids(
                pd.DataFrame({"is_ball": [True]}), _roster(), home_team_id=HOME, away_team_id=AWAY
            )
        with pytest.raises(ValueError, match="roster"):
            add_gradientsports_player_ids(
                _jersey_frames(), pd.DataFrame({"team_id": [1]}), home_team_id=HOME, away_team_id=AWAY
            )


class TestRowAlignment:
    """C2 guard: resolution must be positionally exact on shuffled, multi-frame input
    (the order-safe .map contract — a reorder would misalign every player_id)."""

    def test_per_row_correct_on_shuffled_input(self):
        frames = pd.concat([_jersey_frames(), _jersey_frames().assign(frame_id=2)], ignore_index=True)
        frames = frames.sample(frac=1.0, random_state=7).reset_index(drop=True)  # shuffle rows
        out, _ = add_gradientsports_player_ids(frames, _roster(), home_team_id=HOME, away_team_id=AWAY)
        # expected per-row id from (team_side, jersey) on the SHUFFLED frame, computed independently
        expected = {("home", "8"): 8342, ("home", "1"): 8326, ("away", "10"): 940, ("away", "1"): 12}
        for i in range(len(out)):
            if bool(out.loc[i, "is_ball"]):
                assert pd.isna(out.loc[i, "player_id"])
                continue
            want = expected[(out.loc[i, "team_side"], out.loc[i, "jersey_number"])]  # type: ignore[index]
            assert out.loc[i, "player_id"] == want, f"row {i} misaligned"


class TestKeyNormalization:
    def test_format_drift_still_matches(self):
        # roster shirt as zero-padded string + int-typed + whitespace; normalization is
        # str().strip()-only (reconciles whitespace + int-vs-string, NOT zero-padding).
        frames = _jersey_frames()
        roster = _roster()
        roster["shirt_number"] = ["08", "1", 10, " 1 "]  # mixed str/int/padded/space
        frames.loc[0, "jersey_number"] = " 8 "
        out, _ = add_gradientsports_player_ids(frames, roster, home_team_id=HOME, away_team_id=AWAY)
        assert out.loc[2, "player_id"] == 940  # away "10" matches int-typed roster 10 -> "10"
        assert out.loc[3, "player_id"] == 12  # away "1" matches whitespace roster " 1 " -> "1"


class TestRosterUniqueness:
    def test_duplicate_key_no_explosion_and_warns(self):
        frames = _jersey_frames()
        roster = _roster()
        # inject a duplicate (team_id, shirt_number) for home #8
        dup = pd.DataFrame(
            {"team_id": ["366"], "shirt_number": ["8"], "player_id": [9999], "position_group_type": ["AM"]}
        )
        roster = pd.concat([roster, dup], ignore_index=True)
        with pytest.warns(UserWarning, match="duplicate"):
            out, report = add_gradientsports_player_ids(frames, roster, home_team_id=HOME, away_team_id=AWAY)
        assert len(out) == len(frames)  # no left-join explosion
        assert report.n_duplicate_roster_keys == 1
        assert out.loc[0, "player_id"] == 8342  # keep="first" -> original, not 9999


class TestGoalkeeper:
    def test_gk_vocab_drift_all_false_and_warns(self):
        frames = _jersey_frames()
        roster = _roster()
        roster["position_group_type"] = ["AM", "Goalkeeper", "FW", "Goalkeeper"]  # not literal "GK"
        with pytest.warns(UserWarning, match="no GK found"):
            out, _ = add_gradientsports_player_ids(frames, roster, home_team_id=HOME, away_team_id=AWAY)
        assert not out["is_goalkeeper"].any()  # pins the exact-"GK" literal

    def test_position_column_absent_all_false_and_warns(self):
        frames = _jersey_frames()
        roster = _roster().drop(columns=["position_group_type"])
        with pytest.warns(UserWarning, match="position_group_type"):
            out, _ = add_gradientsports_player_ids(frames, roster, home_team_id=HOME, away_team_id=AWAY)
        assert not out["is_goalkeeper"].any()


class TestUnmatchedAndDegenerate:
    def test_unmatched_jersey_is_na_not_zero(self):
        frames = _jersey_frames()
        frames.loc[0, "jersey_number"] = "99"  # no roster entry
        out, report = add_gradientsports_player_ids(frames, _roster(), home_team_id=HOME, away_team_id=AWAY)
        assert pd.isna(out.loc[0, "player_id"])  # NA, never 0
        assert (HOME, "99") in report.unmatched_jerseys
        assert report.n_unmatched == 1

    def test_degenerate_match_rate_warns(self):
        frames = _jersey_frames()
        # wrong team-id space: pass team ids that match nothing in the roster
        with pytest.warns(UserWarning, match="unmatched"):
            _out, report = add_gradientsports_player_ids(frames, _roster(), home_team_id=999, away_team_id=888)
        assert report.n_matched == 0

    def test_healthy_rate_no_degenerate_warning(self):
        frames = _jersey_frames()
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any UserWarning -> failure
            add_gradientsports_player_ids(frames, _roster(), home_team_id=HOME, away_team_id=AWAY)


def test_reexported_from_tracking():
    import silly_kicks.tracking as tk

    assert hasattr(tk, "add_gradientsports_player_ids")
    assert hasattr(tk, "GradientsportsRosterReport")
    assert "add_gradientsports_player_ids" in tk.__all__


class TestSyntheticEndToEndJoin:
    """In-CI proxy (H1): the resolved Int64 player_id must join BY VALUE to an
    INDEPENDENTLY-constructed int64 events SPADL id and yield nonzero carrier accuracy.
    NON-CIRCULAR (C1): action player_id/team_id come from the KNOWN fixture geometry +
    roster constants below, NOT from infer_ball_carrier's output. Limit (N4): both ids
    still trace to the same synthetic roster, so this proves join MECHANICS only; real
    id-derivation alignment is the env-gated real-data e2e."""

    HOME8, HOME1, AWAY10, AWAY1 = 8342, 8326, 940, 12

    def _fixture(self, n_frames=6):
        roster = _roster()
        rows, carrier_pid, carrier_team = [], {}, {}
        for fid in range(1, n_frames + 1):
            home_carries = fid % 2 == 0
            carrier_pid[fid] = self.HOME8 if home_carries else self.AWAY10
            carrier_team[fid] = HOME if home_carries else AWAY
            ball_x = -10.0 if home_carries else 10.0
            base = dict(
                game_id=1,
                period_id=1,
                frame_id=fid,
                time_seconds=fid * 0.1,
                frame_rate=30.0,
                z=0.0,
                speed_native=0.0,
                ball_state="alive",
            )
            rows += [
                {
                    **base,
                    "team_side": "home",
                    "jersey_number": "8",
                    "is_ball": False,
                    "x_centered": -10.0,
                    "y_centered": 0.0,
                },
                {
                    **base,
                    "team_side": "home",
                    "jersey_number": "1",
                    "is_ball": False,
                    "x_centered": -45.0,
                    "y_centered": 0.0,
                },
                {
                    **base,
                    "team_side": "away",
                    "jersey_number": "10",
                    "is_ball": False,
                    "x_centered": 10.0,
                    "y_centered": 0.0,
                },
                {
                    **base,
                    "team_side": "away",
                    "jersey_number": "1",
                    "is_ball": False,
                    "x_centered": 45.0,
                    "y_centered": 0.0,
                },
                {
                    **base,
                    "team_side": None,
                    "jersey_number": None,
                    "is_ball": True,
                    "x_centered": ball_x,
                    "y_centered": 0.0,
                },
            ]
        return pd.DataFrame(rows), roster, carrier_pid, carrier_team

    def test_independent_join_nonzero_accuracy_and_team_space(self):
        from silly_kicks.tracking import infer_ball_carrier, link_actions_to_frames
        from silly_kicks.tracking.gradientsports import convert_to_frames

        jersey_frames, roster, carrier_pid, carrier_team = self._fixture()
        resolved, _ = add_gradientsports_player_ids(jersey_frames, roster, home_team_id=HOME, away_team_id=AWAY)
        frames, _ = convert_to_frames(resolved, home_team_id=HOME, home_team_start_left=True, output_convention="ltr")
        carrier = infer_ball_carrier(frames)

        acts = []
        for fid, pid in carrier_pid.items():
            tid = carrier_team[fid]
            opp_pid, opp_tid = (self.AWAY1, AWAY) if tid == HOME else (self.HOME1, HOME)
            acts.append(
                dict(
                    game_id=1,
                    period_id=1,
                    time_seconds=fid * 0.1,
                    team_id=tid,
                    player_id=pid,
                    type_name="pass",
                    is_carrier=True,
                )
            )
            acts.append(
                dict(
                    game_id=1,
                    period_id=1,
                    time_seconds=fid * 0.1,
                    team_id=opp_tid,
                    player_id=opp_pid,
                    type_name="pass",
                    is_carrier=False,
                )
            )
        actions = pd.DataFrame(acts)
        actions["action_id"] = np.arange(len(actions))
        actions = actions.astype({"team_id": "int64", "player_id": "int64", "period_id": "int64"})

        pointers, _ = link_actions_to_frames(actions, frames)
        linked = (
            actions.merge(pointers[["action_id", "frame_id"]], on="action_id")
            .merge(
                carrier[["game_id", "period_id", "frame_id", "ball_carrier_player_id", "ball_carrier_team_id"]],
                on=["game_id", "period_id", "frame_id"],
                how="left",
            )
            .dropna(subset=["ball_carrier_player_id"])
        )

        # H1: carrier-actor actions match the inferred carrier by VALUE (independent int ids).
        carr_rows = linked[linked["is_carrier"]]
        acc = (carr_rows["player_id"] == carr_rows["ball_carrier_player_id"]).mean()
        assert acc > 0  # clean fixture -> expect 1.0; > 0 is the regression guard

        # H2b: team-id space correct + non-degenerate (subset, non-flaky) + mixed sameteam.
        carrier_teams = {int(t) for t in carr_rows["ball_carrier_team_id"].dropna().unique()}
        assert carrier_teams and carrier_teams <= {HOME, AWAY}
        sameteam = linked["team_id"] == linked["ball_carrier_team_id"]
        assert 0 < sameteam.mean() < 1


class TestOrientationDtypeInvariance:
    """ADR-019 regression guard (2026-06-09): convert_to_frames orientation must NOT depend on
    the dtype of home_team_id. A raw `team_id == home_team_id` silently matched ZERO players when
    the frame team_id was object-string and home_team_id was int -> team_attacking_direction
    mislabeled -> play_left_to_right double-flipped -> mis-oriented frames (the structural_sgm
    away-team blow-up root cause)."""

    def _resolved_object_string_team_id(self):
        from silly_kicks.tracking.gradientsports import convert_to_frames

        resolved, _ = add_gradientsports_player_ids(_jersey_frames(), _roster(), home_team_id=HOME, away_team_id=AWAY)
        resolved = resolved.copy()
        # Simulate the lakehouse frame: team_id is object-string ("366"/"51"), as real GS rosters are.
        resolved["team_id"] = resolved["team_id"].map(lambda v: str(int(v)) if pd.notna(v) else v).astype(object)
        return resolved, convert_to_frames

    def test_int_vs_str_home_team_id_identical(self):
        resolved, convert_to_frames = self._resolved_object_string_team_id()
        f_int, _ = convert_to_frames(resolved, home_team_id=366, home_team_start_left=True, output_convention="ltr")
        f_str, _ = convert_to_frames(resolved, home_team_id="366", home_team_start_left=True, output_convention="ltr")
        pd.testing.assert_frame_equal(f_int, f_str)

    def test_int_home_team_id_orients_home_to_attack_high_x(self):
        # period 1 + home_team_start_left=True -> home attacks +x, so the home GK (player 8326,
        # who defends x=0) sits at LOW x. Under the dtype bug the frame double-flips and the GK
        # lands at high x. Discriminating coord assertion on the int (buggy) path.
        resolved, convert_to_frames = self._resolved_object_string_team_id()
        f_int, _ = convert_to_frames(resolved, home_team_id=366, home_team_start_left=True, output_convention="ltr")
        gk = f_int[f_int["player_id"].astype(str) == "8326"]
        assert (gk["x"] < 52.5).all(), "home GK should be in its own (low-x) half when home attacks +x"
