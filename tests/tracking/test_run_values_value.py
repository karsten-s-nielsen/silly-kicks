"""TF-35 run VALUATION (ADR-042).

The value oracle is closed-form rather than a reimplementation of the kernel. With a
CONSTANT threat surface ``xT == _XT_CONST`` the product ``pitch_control * threat``
collapses to ``_XT_CONST * pitch_control``, so a runner standing in space their team
fully controls must score exactly ``_XT_CONST`` -- a number written down by hand, not
recomputed from the surface under test.

Layout (period 1, home team 1 attacks "ltr", away team 2 attacks "rtl"): the away team
is parked in its own defensive third so home control saturates around the runners.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking import RunValueCoverageWarning
from silly_kicks.tracking._run_values import RunValuationParams, detect_off_ball_runs, value_off_ball_runs
from silly_kicks.xthreat import ExpectedThreat, values_at_points

_HOME, _AWAY = 1, 2
_RUNNER_TARGET, _RUNNER_DISRUPTIVE = 101, 102
_ACTOR = 103
_XT_CONST = 0.05
_OFFSETS = (-1.5, -1.0, -0.5, 0.0)
_T = 30.0


def _const_xt() -> ExpectedThreat:
    m = ExpectedThreat()
    m.xT[:] = _XT_CONST
    return m


def _frow(pid, team, gk, x, y, t, *, speed=0.0, is_ball=False):
    return {
        "game_id": 1,
        "period_id": 1,
        "frame_id": round(t * 10),
        "time_seconds": float(t),
        "frame_rate": 10.0,
        "player_id": pid,
        "team_id": team,
        "is_ball": is_ball,
        "is_goalkeeper": gk,
        "x": float(x),
        "y": float(y),
        "z": 0.0,
        "speed": speed,
        "vx": 0.0,
        "vy": 0.0,
        "speed_source": "native",
        "ball_state": "alive",
        "team_attacking_direction": None if is_ball else ("ltr" if team == _HOME else "rtl"),
        "confidence": None,
        "visibility": None,
        "source_provider": "synthetic",
        "is_goalkeeper_source": "native",
    }


def _frames(*, disruptive_absent_at_action: bool = False) -> pd.DataFrame:
    """Both runners advance 6 m at 7 m/s; every away player is parked past x=88.

    The ball (and the actor) sit deep at x=20 ON PURPOSE. Spearman zeroes a player's
    influence at any cell the BALL reaches before the player can (``_spearman.py``'s
    ball-travel-time filter), so a runner standing next to the ball has zero influence
    EVERYWHERE and the valuation oracle degenerates to 0. Keeping ~26 m between the ball
    and the runners gives the ball a ~2 s flight and lets the runners win their own space.
    """
    rows: list = []
    for off in _OFFSETS:
        t = _T + off
        frac = (off + 1.5) / 1.5
        rows.append(_frow(None, None, False, 20.0, 34.0, t, speed=8.0, is_ball=True))
        rows.append(_frow(_HOME, _HOME, True, 5.0, 34.0, t))
        rows.append(_frow(_ACTOR, _HOME, False, 20.0, 34.0, t))
        rows.append(_frow(_RUNNER_TARGET, _HOME, False, 40.0 + 6.0 * frac, 24.0, t, speed=7.0))
        if not (disruptive_absent_at_action and off == 0.0):
            rows.append(_frow(_RUNNER_DISRUPTIVE, _HOME, False, 40.0 + 6.0 * frac, 46.0, t, speed=7.0))
        rows.append(_frow(_AWAY, _AWAY, True, 100.0, 34.0, t))
        for pid, (x, y) in {201: (88.0, 24.0), 202: (90.0, 34.0), 203: (92.0, 46.0)}.items():
            rows.append(_frow(pid, _AWAY, False, x, y, t))
    return pd.DataFrame(rows)


def _actions(*, result_id: int = 1, receiver_is_opponent: bool = False) -> pd.DataFrame:
    """Action 10 is the pass; action 11 is the next touch, i.e. the receiver."""
    next_team = _AWAY if receiver_is_opponent else _HOME
    next_player = 201 if receiver_is_opponent else _RUNNER_TARGET
    return pd.DataFrame(
        {
            "game_id": [1, 1],
            "action_id": [10, 11],
            "period_id": [1, 1],
            "time_seconds": [_T, _T + 1.0],
            "team_id": pd.Series([_HOME, next_team], dtype="int64"),
            "player_id": pd.Series([_ACTOR, next_player], dtype="int64"),
            "start_x": [20.0, 60.0],
            "start_y": [34.0, 34.0],
            "end_x": [60.0, 90.0],
            "end_y": [34.0, 34.0],
            "type_id": [0, 0],
            "type_name": ["pass", "pass"],
            "result_id": pd.Series([result_id, 1], dtype="int64"),
            "result_name": ["success" if result_id == 1 else "fail", "success"],
            "bodypart_id": [0, 0],
            "bodypart_name": ["foot", "foot"],
        }
    )


def _valued(**frame_kwargs):
    actions, frames = (
        _actions(**{k: v for k, v in frame_kwargs.items() if k in ("result_id", "receiver_is_opponent")}),
        _frames(**{k: v for k, v in frame_kwargs.items() if k == "disruptive_absent_at_action"}),
    )
    runs = detect_off_ball_runs(actions, frames, home_team_id=_HOME)
    return runs, value_off_ball_runs(runs, actions, frames, _const_xt()), actions


def _by_player(valued: pd.DataFrame) -> dict:
    return {int(p): row for p, row in zip(valued["player_id"], valued.to_dict("records"), strict=True)}


class TestRoles:
    def test_both_runners_are_detected(self):
        runs, _valued_df, _a = _valued()
        assert set(runs["player_id"].astype("int64")) == {_RUNNER_TARGET, _RUNNER_DISRUPTIVE}, (
            "role assertions below would be vacuous without both runners"
        )

    def test_receiver_is_the_target_and_the_other_is_disruptive(self):
        _runs, valued, _a = _valued()
        by = _by_player(valued)
        assert by[_RUNNER_TARGET]["role"] == "target"
        assert bool(by[_RUNNER_TARGET]["is_receiver"]) is True
        assert by[_RUNNER_DISRUPTIVE]["role"] == "disruptive"
        assert bool(by[_RUNNER_DISRUPTIVE]["is_receiver"]) is False


class TestValueOracle:
    def test_run_value_is_the_constant_threat_where_control_saturates(self):
        """Closed form: max(control * const) == const once control reaches 1.0."""
        _runs, valued, _a = _valued()
        for pid in (_RUNNER_TARGET, _RUNNER_DISRUPTIVE):
            v = float(_by_player(valued)[pid]["run_value"])
            assert np.isfinite(v), f"runner {pid} was not valued"
            assert v <= _XT_CONST + 1e-12, "control cannot exceed 1.0, so value cannot exceed the constant"
            assert v == pytest.approx(_XT_CONST, rel=1e-3), (
                f"runner {pid} scored {v:.6g}; with saturated control the value must be the constant threat {_XT_CONST}"
            )

    def test_enabled_pass_credit_is_the_floored_xt_gain(self):
        _runs, valued, actions = _valued()
        xt = _const_xt()
        gain = values_at_points(xt, actions["end_x"], actions["end_y"]) - values_at_points(
            xt, actions["start_x"], actions["start_y"]
        )
        expected = max(0.0, float(gain[0]))
        credit = float(_by_player(valued)[_RUNNER_DISRUPTIVE]["enabled_pass_credit"])
        assert credit == pytest.approx(expected)

    def test_target_row_carries_no_pass_credit(self):
        """The credit is what the DISRUPTIVE runs enabled; the receiver is the pass itself."""
        _runs, valued, _a = _valued()
        assert pd.isna(_by_player(valued)[_RUNNER_TARGET]["enabled_pass_credit"])


class TestOffDomain:
    def test_failed_pass_is_off_domain(self):
        _runs, valued, _a = _valued(result_id=0)
        assert valued["run_value"].isna().all()
        assert valued["role"].isna().all()
        assert valued["is_receiver"].isna().all()

    def test_unresolved_receiver_is_off_domain(self):
        """Next touch belongs to the opponent -> no receiver -> nothing to value."""
        _runs, valued, _a = _valued(receiver_is_opponent=True)
        assert valued["run_value"].isna().all()
        assert valued["role"].isna().all()

    def test_on_domain_case_is_not_vacuous(self):
        """The off-domain assertions only mean something if the domain case DOES value."""
        _runs, valued, _a = _valued()
        assert valued["run_value"].notna().all()


class TestCoverageGap:
    def test_runner_absent_at_the_linked_frame_survives_with_nan(self):
        with pytest.warns(RunValueCoverageWarning, match="could not be valued"):
            _runs, valued, _a = _valued(disruptive_absent_at_action=True)
        by = _by_player(valued)
        assert _RUNNER_DISRUPTIVE in by, "the row must SURVIVE -- a visibility gap is not a deletion"
        assert pd.isna(by[_RUNNER_DISRUPTIVE]["run_value"]), "an absent runner must not be scored 0"
        assert by[_RUNNER_DISRUPTIVE]["role"] == "disruptive", "role is event-derived and stays assigned"
        assert np.isfinite(float(by[_RUNNER_TARGET]["run_value"])), "the visible runner is still valued"

    def test_full_coverage_does_not_warn(self):
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("error", RunValueCoverageWarning)
            _valued()


class TestGuards:
    def test_unfitted_xt_fails_loud(self):
        from sklearn.exceptions import NotFittedError

        actions, frames = _actions(), _frames()
        runs = detect_off_ball_runs(actions, frames, home_team_id=_HOME)
        with pytest.raises(NotFittedError):
            value_off_ball_runs(runs, actions, frames, ExpectedThreat())

    def test_empty_runs_returns_the_value_schema(self):
        actions, frames = _actions(), _frames()
        empty = detect_off_ball_runs(actions.iloc[:0], frames, home_team_id=_HOME)
        out = value_off_ball_runs(empty, actions, frames, _const_xt())
        for col in ("role", "is_receiver", "run_value", "enabled_pass_credit"):
            assert col in out.columns

    def test_input_runs_frame_is_not_mutated(self):
        actions, frames = _actions(), _frames()
        runs = detect_off_ball_runs(actions, frames, home_team_id=_HOME)
        before = runs.copy()
        out = value_off_ball_runs(runs, actions, frames, _const_xt())
        pd.testing.assert_frame_equal(runs, before)
        assert out is not runs

    def test_sprint_gate_keeps_both_runners(self):
        """Guard the fixture itself: 7 m/s must clear the 5.56 m/s default gate."""
        runs = detect_off_ball_runs(_actions(), _frames(), home_team_id=_HOME, params=RunValuationParams())
        assert len(runs) == 2


class TestAggregatorNaRoles:
    """Regression: the aggregator must survive runs whose action is OFF-domain.

    ``role`` is a nullable "string" column and is <NA> for every run belonging to an
    off-domain action, so a bare ``role == "target"`` yields a BooleanArray WITH NA and
    ``.to_numpy(dtype=bool)`` raises "cannot convert to 'bool'-dtype NumPy array with
    missing values". The standard fixtures never produced that shape (their only detected
    runs belonged to on-domain actions); the atomic mirror did, and it raised.
    """

    def test_off_domain_action_with_detected_runs_does_not_raise(self):
        import silly_kicks.tracking.features as tf

        actions, frames = _actions(result_id=0), _frames()  # failed pass -> off-domain
        runs = detect_off_ball_runs(actions, frames, home_team_id=_HOME)
        assert len(runs) > 0, "no detected runs -- the NA-role path would not be exercised"

        out = tf.add_off_ball_run_values(actions, frames, _const_xt(), home_team_id=_HOME)
        assert out["run_value_target"].isna().all()
        assert out["n_disruptive_runs"].isna().all()


class TestSafeIndexOfDtype:
    """ADR-019 regression: ``_safe_index_of`` resolves an id dtype-safely and degrades an
    absent/NA id to ``None``. Locks the dedup that routed it through ``ids_match`` -- a revert
    to a raw ``player_ids == player_id`` would miss across dtypes and fail the first case."""

    def test_str_query_resolves_int_ids(self):
        from silly_kicks.tracking._run_values import _safe_index_of

        assert _safe_index_of(np.array([101, 102, 103]), "102") == 1

    def test_int_query_resolves_object_string_ids(self):
        from silly_kicks.tracking._run_values import _safe_index_of

        assert _safe_index_of(np.array(["101", "102", "103"], dtype=object), 101) == 0

    def test_absent_na_and_none_degrade_to_none(self):
        from silly_kicks.tracking._run_values import _safe_index_of

        ids = np.array([101, 102, 103])
        assert _safe_index_of(ids, 999) is None  # absent
        assert _safe_index_of(ids, np.nan) is None  # NA id (ball rows) matches nothing
        assert _safe_index_of(None, 101) is None  # no decomposition

    def test_mixed_domain_frame_aggregates_only_the_on_domain_rows(self):
        """Non-vacuity: with one on-domain and one off-domain action, the aggregator
        must value the first and leave the second NA -- not fall over on either."""
        import silly_kicks.tracking.features as tf

        actions = pd.concat(
            [_actions(), _actions(result_id=0).assign(action_id=[20, 21], time_seconds=[_T, _T + 1.0])],
            ignore_index=True,
        )
        out = tf.add_off_ball_run_values(actions, _frames(), _const_xt(), home_team_id=_HOME)
        assert out.loc[out["action_id"] == 10, "run_value_target"].notna().all()
        assert out.loc[out["action_id"] == 20, "run_value_target"].isna().all()


class TestAtomicMirrorIsNotSilentlyDead:
    """The atomic mirror must actually VALUE actions, not return all-<NA>.

    ``_packing_atomic_adapter`` maps every non-domain atom to standard ``non_action``,
    which ``resolve_next_touch_receiver`` skips (a non-action is not a touch). That erases
    the ``receival`` atom -- the row carrying receiver identity -- so no receiver resolves
    and every TF-35 column comes back <NA> for every action. A mirror that returns all-<NA>
    still passes purity, id-dtype and nan-safety; only this test fails.
    """

    @staticmethod
    def _atomic_stream() -> pd.DataFrame:
        from silly_kicks.atomic.spadl import config as ac

        return pd.DataFrame(
            {
                "game_id": [1, 1],
                "period_id": [1, 1],
                "action_id": [10, 11],
                "time_seconds": [_T, _T + 1.0],
                "team_id": pd.Series([_HOME, _HOME], dtype="int64"),
                "player_id": pd.Series([_ACTOR, _RUNNER_TARGET], dtype="int64"),
                "x": [20.0, 60.0],
                "y": [34.0, 34.0],
                "dx": [40.0, 0.0],
                "dy": [0.0, 0.0],
                "type_id": [ac.actiontype_id["pass"], ac.actiontype_id["receival"]],
                "bodypart_id": [0, 0],
            }
        )

    def test_reception_atom_resolves_the_receiver(self):
        from silly_kicks.atomic.tracking.features import add_off_ball_run_values as atomic_add

        out = atomic_add(self._atomic_stream(), _frames(), _const_xt(), home_team_id=_HOME)
        assert out["run_value_target"].iloc[0] == pytest.approx(_XT_CONST, rel=1e-3), (
            "the pass atom was not valued -- its receiver did not resolve, so the whole "
            "atomic mirror is off-domain and emits <NA> everywhere"
        )
        assert int(out["n_disruptive_runs"].iloc[0]) == 1

    def test_the_reception_atom_itself_stays_off_domain(self):
        """Non-vacuity: re-typing receptions must not make THEM valued actions."""
        from silly_kicks.atomic.tracking.features import add_off_ball_run_values as atomic_add

        out = atomic_add(self._atomic_stream(), _frames(), _const_xt(), home_team_id=_HOME)
        assert pd.isna(out["run_value_target"].iloc[1])
        assert pd.isna(out["n_disruptive_runs"].iloc[1])

    def test_caller_columns_are_not_rewritten(self):
        """The adapter's synthesized std type_id must never leak into the output."""
        from silly_kicks.atomic.tracking.features import add_off_ball_run_values as atomic_add

        stream = self._atomic_stream()
        out = atomic_add(stream, _frames(), _const_xt(), home_team_id=_HOME)
        pd.testing.assert_series_equal(out["type_id"], stream["type_id"])


class TestMultiGameIsolation:
    """SPADL ``action_id`` restarts per game, so any action_id-only key leaks across games.

    Found by adversarial review (ADR-042 finding 2): ``value_off_ball_runs`` grouped runs by
    ``action_id`` alone and ``_run_values_at_actions`` collapsed positions the same way, so
    game 1's on-domain iteration wrote roles/values onto game 2's rows AND game 1's own
    counts absorbed game 2's runs.
    """

    @staticmethod
    def _two_games(second_game_result: int):
        a1, f1 = _actions(), _frames()
        a2, f2 = _actions(result_id=second_game_result), _frames()
        a2["game_id"] = 2
        f2["game_id"] = 2
        return (
            pd.concat([a1, a2], ignore_index=True),
            pd.concat([f1, f2], ignore_index=True),
        )

    def test_off_domain_game_is_not_valued_by_the_other_game(self):
        actions, frames = self._two_games(second_game_result=0)  # game 2's pass FAILED
        runs = detect_off_ball_runs(actions, frames, home_team_id=_HOME)
        assert (runs["game_id"] == 2).any(), "no game-2 runs detected -- the test is vacuous"

        valued = value_off_ball_runs(runs, actions, frames, _const_xt())
        g2 = valued[valued["game_id"] == 2]
        assert g2["run_value"].isna().all(), "game 2 is off-domain; its runs must not be valued"
        assert g2["role"].isna().all(), "game 2 must not inherit game 1's roles"

        g1 = valued[valued["game_id"] == 1]
        assert g1["run_value"].notna().all(), "game 1 must still be valued (non-vacuity)"

    def test_counts_are_not_inflated_by_the_other_game(self):
        import silly_kicks.tracking.features as tf

        actions, frames = self._two_games(second_game_result=1)  # both games on-domain
        out = tf.add_off_ball_run_values(actions, frames, _const_xt(), home_team_id=_HOME)
        assert len(out) == len(actions), "the provenance merge fanned out rows"

        for game in (1, 2):
            row = out[(out["game_id"] == game) & (out["action_id"] == 10)]
            assert len(row) == 1
            assert int(row["n_disruptive_runs"].iloc[0]) == 1, (
                f"game {game} action 10 has exactly ONE disruptive runner; a cross-game key reports 2"
            )


class TestAtomicXfnsIsNotSilentlyDead:
    """The atomic FACTORY needs the reception restoration too, not just the aggregator.

    No auto-enumerating gate covers this: the leakage guard only checks ``__name__`` and the
    shape check calls the ``frames=None`` branch, which returns NaN by design.
    """

    def test_atomic_xfn_produces_real_values_with_frames(self):
        from silly_kicks.atomic.tracking.features import off_ball_run_value_xfns as atomic_xfns

        stream = TestAtomicMirrorIsNotSilentlyDead._atomic_stream()
        xfn = atomic_xfns(_const_xt(), home_team_id=_HOME)[0]
        out = xfn([stream, stream, stream], _frames())
        assert out["run_value_target_a0"].notna().any(), (
            "the atomic xfn returned NaN everywhere -- the receival atom was hidden, so no "
            "receiver resolved and every action fell off-domain"
        )
        assert float(out["run_value_target_a0"].iloc[0]) == pytest.approx(_XT_CONST, rel=1e-3)

    def test_standard_and_atomic_xfns_agree_on_the_same_situation(self):
        """Cross-representation check: the same physical pass must value the same."""
        import silly_kicks.tracking.features as tf
        from silly_kicks.atomic.tracking.features import off_ball_run_value_xfns as atomic_xfns

        std_actions = _actions()
        std = tf.off_ball_run_value_xfns(_const_xt(), home_team_id=_HOME)[0](
            [std_actions, std_actions, std_actions], _frames()
        )
        atomic_stream = TestAtomicMirrorIsNotSilentlyDead._atomic_stream()
        atomic = atomic_xfns(_const_xt(), home_team_id=_HOME)[0](
            [atomic_stream, atomic_stream, atomic_stream], _frames()
        )
        assert float(atomic["run_value_target_a0"].iloc[0]) == pytest.approx(
            float(std["run_value_target_a0"].iloc[0]), rel=1e-6
        )


class TestGameIdDtypeMismatch:
    """A cross-dtype `game_id` must not silently void every run value (ADR-019).

    Found by final-review. The (game_id, period_id, frame_id) re-keying used a RAW tuple
    lookup, so on the documented lakehouse shape -- actions `game_id` int64, frames native
    string -- every `get_group` missed. Runs were DETECTED (detection uses `ids_equal`) but
    every `run_value` came back NaN, and the coverage warning misattributed the cause to a
    tracking-visibility gap. A wrong explanation is worse than a failure.
    """

    @staticmethod
    def _mismatched():
        actions, frames = _actions(), _frames()
        actions["game_id"] = actions["game_id"].astype("int64")
        frames["game_id"] = frames["game_id"].astype(str)
        return actions, frames

    def test_runs_are_valued_under_a_cross_dtype_game_id(self):
        actions, frames = self._mismatched()
        runs = detect_off_ball_runs(actions, frames, home_team_id=_HOME)
        assert len(runs) > 0, "no runs detected - the valuation assertion would be vacuous"

        valued = value_off_ball_runs(runs, actions, frames, _const_xt())
        assert valued["run_value"].notna().all(), (
            "every run_value is NaN under a cross-dtype game_id - the frame lookup is "
            "dtype-sensitive (ADR-019), and the coverage warning blames tracking visibility"
        )

    def test_matched_and_mismatched_dtypes_agree(self):
        """The values must be IDENTICAL, not merely non-NaN."""
        a_ok, f_ok = _actions(), _frames()
        ref = value_off_ball_runs(detect_off_ball_runs(a_ok, f_ok, home_team_id=_HOME), a_ok, f_ok, _const_xt())
        a_bad, f_bad = self._mismatched()
        got = value_off_ball_runs(detect_off_ball_runs(a_bad, f_bad, home_team_id=_HOME), a_bad, f_bad, _const_xt())
        np.testing.assert_allclose(
            got["run_value"].to_numpy(dtype="float64"),
            ref["run_value"].to_numpy(dtype="float64"),
            rtol=1e-12,
        )
