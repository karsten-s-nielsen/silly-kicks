"""Tests for VAEP label windowing variants (TF-29)."""

import warnings

import pandas as pd
import pytest

import silly_kicks.spadl.config as spadl
import silly_kicks.vaep.labels as lab


def _make_actions(
    n: int = 10,
    *,
    goal_at: int | None = None,
    owngoal_at: int | None = None,
    possession_ids: list[int] | None = None,
    time_seconds: list[float] | None = None,
    period_ids: list[int] | None = None,
) -> pd.DataFrame:
    """Build a minimal SPADL-shaped actions DataFrame for windowing tests."""
    types = ["pass"] * n
    results = ["success"] * n
    if goal_at is not None:
        types[goal_at] = "shot"
        results[goal_at] = "success"
    if owngoal_at is not None:
        types[owngoal_at] = "shot"
        results[owngoal_at] = "owngoal"

    df = pd.DataFrame(
        {
            "game_id": [1] * n,
            "period_id": period_ids if period_ids else [1] * n,
            "action_id": list(range(n)),
            "type_id": [spadl.actiontype_id.get(t, 0) for t in types],
            "type_name": types,
            "result_id": [spadl.result_id.get(r, 0) for r in results],
            "result_name": results,
            "team_id": [1 if i % 2 == 0 else 2 for i in range(n)],
            "player_id": list(range(n)),
            "start_x": [50.0] * n,
            "start_y": [34.0] * n,
            "end_x": [60.0] * n,
            "end_y": [34.0] * n,
            "bodypart_id": [0] * n,
            "bodypart_name": ["foot"] * n,
            "time_seconds": time_seconds if time_seconds else [float(i * 3) for i in range(n)],
        }
    )
    if possession_ids is not None:
        df["possession_id"] = possession_ids
    return df


class TestWindowActionBackwardCompat:
    """window='action' must produce identical output to the original API."""

    def test_scores_action_default(self) -> None:
        actions = _make_actions(10, goal_at=5)
        old = lab.scores(actions, nr_actions=10)
        new = lab.scores(actions, nr_actions=10, window="action")
        pd.testing.assert_frame_equal(old, new)

    def test_concedes_action_default(self) -> None:
        actions = _make_actions(10, goal_at=5)
        old = lab.concedes(actions, nr_actions=10)
        new = lab.concedes(actions, nr_actions=10, window="action")
        pd.testing.assert_frame_equal(old, new)


class TestWindowPossession:
    def test_missing_possession_id_raises(self) -> None:
        actions = _make_actions(5)
        assert "possession_id" not in actions.columns
        with pytest.raises(ValueError, match="possession_id"):
            lab.scores(actions, window="possession")

    def test_scores_within_possession(self) -> None:
        # 3 possession chains: [0,1,2], [3,4,5], [6,7,8]
        # Goal at action 2 (chain 0), team_id=1
        # All even actions are team 1, odd are team 2
        actions = _make_actions(
            9,
            goal_at=2,
            possession_ids=[0, 0, 0, 1, 1, 1, 2, 2, 2],
        )
        scores = lab.scores(actions, window="possession")
        # Action 2 (team 1, goal) itself scores
        assert scores["scores"].iloc[2]
        # Action 0 (team 1) sees goal at action 2 (team 1, same team) -> scores
        assert scores["scores"].iloc[0]
        # Action 1 (team 2) sees goal at action 2 (team 1, different team) -> does NOT score
        assert not scores["scores"].iloc[1]
        # Actions in chains 1 and 2 should not score (no goal in their chain)
        assert not scores["scores"].iloc[3:].any()

    def test_concedes_within_possession(self) -> None:
        # Goal at action 2 (team 1); action 1 (team 2) should concede
        actions = _make_actions(
            6,
            goal_at=2,
            possession_ids=[0, 0, 0, 1, 1, 1],
        )
        concedes = lab.concedes(actions, window="possession")
        # Action 1 (team 2) sees goal at action 2 (team 1, different team) -> concedes
        assert concedes["concedes"].iloc[1]
        # Action 0 (team 1) sees goal at action 2 (team 1, same team) -> does NOT concede
        assert not concedes["concedes"].iloc[0]
        # Actions in chain 1: no goal -> no concedes
        assert not concedes["concedes"].iloc[3:].any()


class TestWindowTime:
    def test_missing_time_seconds_raises(self) -> None:
        actions = _make_actions(5)
        actions = actions.drop(columns=["time_seconds"])
        with pytest.raises(ValueError, match="time_seconds"):
            lab.scores(actions, window="time")

    def test_strict_boundary(self) -> None:
        """goal_time - action_time < window_seconds (strict inequality)."""
        # Goal at action 2, t=10.0. window_seconds=5.0
        # Action at t=5.0: 10-5=5.0, NOT < 5.0, should NOT score
        # Action at t=5.01: 10-5.01=4.99, < 5.0, should score
        actions = _make_actions(
            4,
            goal_at=2,
            time_seconds=[0.0, 5.0, 10.0, 15.0],
        )
        # All same team for simplicity
        actions["team_id"] = 1
        scores = lab.scores(actions, window="time", window_seconds=5.0)
        assert not scores["scores"].iloc[0]  # t=0, 10-0=10 >= 5
        assert not scores["scores"].iloc[1]  # t=5, 10-5=5.0, NOT < 5.0
        assert scores["scores"].iloc[2]  # t=10, goal action itself
        assert not scores["scores"].iloc[3]  # t=15, after goal

    def test_cross_period_no_bleed(self) -> None:
        """Goal in period 2 must not bleed into period 1."""
        actions = _make_actions(
            4,
            goal_at=3,
            time_seconds=[80.0, 89.0, 1.0, 5.0],
            period_ids=[1, 1, 2, 2],
        )
        actions["team_id"] = 1
        scores = lab.scores(actions, window="time", window_seconds=15.0)
        assert not scores["scores"].iloc[0]  # period 1, no goal in period 1
        assert not scores["scores"].iloc[1]  # period 1
        assert scores["scores"].iloc[2]  # period 2, within window of goal at t=5
        assert scores["scores"].iloc[3]  # the goal itself

    def test_unsorted_raises(self) -> None:
        """time_seconds must be non-decreasing within each period."""
        actions = _make_actions(
            3,
            time_seconds=[10.0, 5.0, 15.0],
        )
        with pytest.raises(ValueError, match="non-decreasing"):
            lab.scores(actions, window="time")


class TestXgWithNonActionWindows:
    """xg_column must work with possession and time modes, not just action."""

    def test_scores_possession_xg(self) -> None:
        actions = _make_actions(6, goal_at=2, possession_ids=[0, 0, 0, 1, 1, 1])
        actions["xg"] = [0.0, 0.0, 0.8, 0.0, 0.0, 0.0]
        result = lab.scores(actions, window="possession", xg_column="xg")
        assert result["scores"].iloc[0] == pytest.approx(0.8)  # same team as goal
        assert result["scores"].iloc[1] == pytest.approx(0.0)  # different team

    def test_concedes_possession_xg(self) -> None:
        actions = _make_actions(6, goal_at=2, possession_ids=[0, 0, 0, 1, 1, 1])
        actions["xg"] = [0.0, 0.0, 0.8, 0.0, 0.0, 0.0]
        result = lab.concedes(actions, window="possession", xg_column="xg")
        assert result["concedes"].iloc[1] == pytest.approx(0.8)  # different team
        assert result["concedes"].iloc[0] == pytest.approx(0.0)  # same team

    def test_scores_time_xg(self) -> None:
        actions = _make_actions(4, goal_at=2, time_seconds=[0.0, 5.0, 10.0, 15.0])
        actions["team_id"] = 1
        actions["xg"] = [0.0, 0.0, 0.7, 0.0]
        result = lab.scores(actions, window="time", window_seconds=15.0, xg_column="xg")
        assert result["scores"].iloc[1] == pytest.approx(0.7)  # within window
        assert result["scores"].iloc[2] == pytest.approx(0.7)  # goal itself


class TestNrActionsWarning:
    def test_warns_when_non_default_with_possession(self) -> None:
        actions = _make_actions(5, possession_ids=[0, 0, 0, 1, 1])
        with pytest.warns(UserWarning, match="nr_actions.*ignored"):
            lab.scores(actions, nr_actions=5, window="possession")

    def test_warns_when_non_default_with_time(self) -> None:
        actions = _make_actions(5)
        with pytest.warns(UserWarning, match="nr_actions.*ignored"):
            lab.scores(actions, nr_actions=5, window="time")

    def test_no_warning_when_default_nr_actions(self) -> None:
        actions = _make_actions(5, possession_ids=[0, 0, 0, 1, 1])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            lab.scores(actions, nr_actions=10, window="possession")


class TestGoalscoreFreeXfns:
    def test_xfns_default_no_goalscore_length(self) -> None:
        from silly_kicks.vaep.base import xfns_default, xfns_default_no_goalscore

        assert len(xfns_default_no_goalscore) == len(xfns_default) - 1

    def test_xfns_default_no_goalscore_excludes(self) -> None:
        from silly_kicks.vaep import features as fs
        from silly_kicks.vaep.base import xfns_default_no_goalscore

        assert fs.goalscore not in xfns_default_no_goalscore

    def test_xfns_default_no_goalscore_order(self) -> None:
        from silly_kicks.vaep import features as fs
        from silly_kicks.vaep.base import xfns_default, xfns_default_no_goalscore

        expected = [x for x in xfns_default if x is not fs.goalscore]
        assert xfns_default_no_goalscore == expected

    def test_hybrid_no_goalscore_length(self) -> None:
        from silly_kicks.vaep.hybrid import (
            hybrid_xfns_default,
            hybrid_xfns_default_no_goalscore,
        )

        assert len(hybrid_xfns_default_no_goalscore) == len(hybrid_xfns_default) - 1

    def test_hybrid_no_goalscore_excludes(self) -> None:
        from silly_kicks.vaep import features as fs
        from silly_kicks.vaep.hybrid import hybrid_xfns_default_no_goalscore

        assert fs.goalscore not in hybrid_xfns_default_no_goalscore

    def test_hybrid_no_goalscore_order(self) -> None:
        from silly_kicks.vaep import features as fs
        from silly_kicks.vaep.hybrid import (
            hybrid_xfns_default,
            hybrid_xfns_default_no_goalscore,
        )

        expected = [x for x in hybrid_xfns_default if x is not fs.goalscore]
        assert hybrid_xfns_default_no_goalscore == expected

    def test_feature_column_names_no_goalscore(self) -> None:
        from silly_kicks.vaep.base import xfns_default, xfns_default_no_goalscore
        from silly_kicks.vaep.features import feature_column_names

        cols_full = feature_column_names(list(xfns_default), 3)
        cols_no_gs = feature_column_names(list(xfns_default_no_goalscore), 3)
        assert len(cols_no_gs) < len(cols_full)
        assert not any("goalscore" in c for c in cols_no_gs)

    def test_reexport_from_vaep(self) -> None:
        from silly_kicks.vaep import (  # noqa: F401
            hybrid_xfns_default_no_goalscore,
            xfns_default_no_goalscore,
        )
