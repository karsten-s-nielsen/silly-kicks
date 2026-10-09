"""TF-58 Task 15: windows, stoppage evidence, phase assignment, and the fixture preconditions."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.coordination._columns import COORD_WINDOW_COLUMNS
from silly_kicks.coordination._report import CoordinationCoverageWarning
from silly_kicks.coordination._windows import (
    RESTART_TYPES,
    period_windows,
    phase_assignment,
    possession_windows_from_actions,
    possession_windows_from_frames,
    resolve_stoppages,
    validate_windows,
)
from tests.coordination._fixtures import make_coordination_actions, make_coordination_match


def _action(game, period, aid, t, team, name, result="success"):
    return {
        "game_id": game,
        "period_id": period,
        "action_id": aid,
        "time_seconds": float(t),
        "team_id": team,
        "player_id": team * 100 + 1,
        "type_id": spadlconfig.actiontypes.index(name),
        "result_id": spadlconfig.results.index(result),
        "start_x": 52.5,
        "start_y": 34.0,
        "end_x": 60.0,
        "end_y": 34.0,
        "bodypart_id": spadlconfig.bodyparts.index("foot"),
    }


# --------------------------------------------------------------------------- fixture preconditions (ADR-032)
def test_fixture_preconditions():
    frames = make_coordination_match(seconds=600.0, hz=10.0, oscillation_cpm=0.5)
    # planted oscillation: team A centroid_x spectral peak within one bin of oscillation_cpm.
    team_a = frames[(frames["team_id"] == 1) & (~frames["is_ball"])]
    cx = team_a.groupby("time_seconds")["x"].mean().to_numpy()
    cx = cx - cx.mean()
    freqs = np.fft.rfftfreq(cx.size, d=1.0 / 10.0)
    peak_hz = freqs[1 + int(np.argmax(np.abs(np.fft.rfft(cx))[1:]))]
    bin_cpm = 60.0 * (freqs[1] - freqs[0])
    assert abs(peak_hz * 60.0 - 0.5) <= bin_cpm
    # both directions present
    assert set(frames.loc[~frames["is_ball"], "team_attacking_direction"]) == {"ltr", "rtl"}
    # dead intervals surface in ball_state
    dead_frames = make_coordination_match(seconds=60.0, hz=10.0, dead_intervals=[(10.0, 40.0)])
    ball = dead_frames[dead_frames["is_ball"]]
    assert (ball["ball_state"].astype(object) == "dead").any()


# --------------------------------------------------------------------------- period / sliding
def test_period_windows_one_per_period_and_sliding_full_length_only():
    frames = make_coordination_match(seconds=30.0, hz=10.0)
    periods = period_windows(frames)
    assert len(periods) == 1
    assert periods["window_kind"].iloc[0] == "period"
    assert int(periods["window_id"].iloc[0]) == 0

    sliding = period_windows(frames, length_s=10.0, step_s=10.0)
    assert (sliding["window_kind"] == "sliding").all()
    # 30 s of frames -> [0,10),[10,20),[20,30); a partial [30,40) is dropped (C21)
    assert len(sliding) == 3
    assert list(sliding["window_id"]) == [0, 1, 2]
    assert (sliding["end_time_s"] - sliding["start_time_s"]).round(6).eq(10.0).all()


def test_period_windows_ignore_an_unused_categorical_id():
    # repo convention (groupby observed=True): a categorical game id carrying an unused category yields no phantom
    # window with NaN bounds -- one window per (game, period) that is actually there.
    frames = make_coordination_match(seconds=60.0, hz=10.0)
    cat = frames.assign(game_id=pd.Categorical(frames["game_id"], categories=[1, 2]))
    windows = period_windows(cat)
    assert len(windows) == 1
    assert windows["start_time_s"].notna().all() and windows["end_time_s"].notna().all()


def test_window_contract_columns_and_dtypes():
    frames = make_coordination_match(seconds=30.0, hz=10.0)
    actions = make_coordination_actions(frames)
    for windows in (
        period_windows(frames),
        possession_windows_from_actions(actions, frames),
        possession_windows_from_frames(frames),
    ):
        assert list(windows.columns) == list(COORD_WINDOW_COLUMNS)
        assert {c: str(d) for c, d in windows.dtypes.items()} == dict(COORD_WINDOW_COLUMNS)
        validate_windows(windows)


# --------------------------------------------------------------------------- possession from events
def test_possession_windows_end_at_terminal_event():
    frames = make_coordination_match(seconds=60.0, hz=10.0)
    actions = pd.DataFrame(
        [
            _action(1, 1, 0, 0.0, 1, "pass"),
            _action(1, 1, 1, 3.0, 1, "shot"),  # possession 0 (team 1) ends in a shot
            _action(1, 1, 2, 20.0, 1, "pass"),  # possession 1 (team 1)
            _action(1, 1, 3, 25.0, 2, "tackle"),  # possession 2 (team 2) -- ends possession 1
        ]
    )
    windows = possession_windows_from_actions(actions, frames).sort_values("start_time_s").reset_index(drop=True)
    shot_win = windows.iloc[0]
    assert shot_win["terminal_action"] == "shot"
    assert shot_win["attacking_team_id"] == 1
    assert shot_win["end_time_s"] == 3.0

    tackle_win = windows[windows["start_time_s"] == 20.0].iloc[0]
    assert tackle_win["terminal_action"] == "tackle"
    assert tackle_win["terminal_team_id"] == 2
    assert tackle_win["end_time_s"] == 25.0

    last_win = windows.iloc[-1]
    assert pd.isna(last_win["terminal_action"])
    assert pd.isna(last_win["terminal_team_id"])


def test_possession_windows_follow_time_not_a_non_chronological_action_id():
    # ADR-065 §3d: a persisted mart may carry a non-chronological action_id, so possession windows order the actions by
    # the robust (game, period, time_seconds, action_id) key -- relabelling the ids cannot move a window.
    frames = make_coordination_match(seconds=60.0, hz=10.0)
    chrono = pd.DataFrame(
        [
            _action(1, 1, 0, 0.0, 1, "pass"),
            _action(1, 1, 1, 3.0, 1, "shot"),
            _action(1, 1, 2, 20.0, 1, "pass"),
            _action(1, 1, 3, 25.0, 2, "tackle"),
        ]
    )
    scrambled = chrono.assign(action_id=[3, 1, 0, 2])  # the same actions; ids no longer in time order
    want = possession_windows_from_actions(chrono, frames).sort_values("start_time_s").reset_index(drop=True)
    got = possession_windows_from_actions(scrambled, frames).sort_values("start_time_s").reset_index(drop=True)
    pd.testing.assert_frame_equal(got, want)


def test_possession_windows_skip_a_period_without_tracking_and_say_so():
    # A per-match tracking hole (GS 10510/10511: events run through extra time, tracking stops after period 2; other
    # GS extra-time matches ship all four periods). A possession in a period the frames do not cover has no samples to
    # score, so it gets no window -- and the coverage warning names what was skipped (never a silent drop, never the
    # KeyError the period-end lookup used to raise).
    frames = make_coordination_match(seconds=60.0, hz=10.0)  # period 1 only
    tracked = [_action(1, 1, 0, 0.0, 1, "pass"), _action(1, 1, 1, 20.0, 2, "tackle")]
    extra_time = [_action(1, 3, 2, 5.0, 1, "pass"), _action(1, 3, 3, 9.0, 2, "tackle")]
    want = possession_windows_from_actions(pd.DataFrame(tracked), frames)
    with pytest.warns(CoordinationCoverageWarning, match=r"2 action\(s\).*game 1 period 3"):
        got = possession_windows_from_actions(pd.DataFrame(tracked + extra_time), frames)
    assert set(got["period_id"]) == {1}
    pd.testing.assert_frame_equal(got, want)


def test_possession_windows_keep_games_apart():
    # add_possessions restarts its counter per game, so possession k of game 1 and of game 2 share a possession_id: a
    # multi-game call must give each game exactly the windows a one-game call gives it (never a cross-game merge).
    one = make_coordination_match(seconds=60.0, hz=10.0)
    two = one.assign(game_id=2)
    frames = pd.concat([one, two], ignore_index=True)
    plays = [(0.0, 1, "pass"), (3.0, 1, "shot"), (20.0, 1, "pass"), (25.0, 2, "tackle")]
    game1 = pd.DataFrame([_action(1, 1, i, t, team, name) for i, (t, team, name) in enumerate(plays)])
    game2 = pd.DataFrame([_action(2, 1, 10 + i, t + 5.0, team, name) for i, (t, team, name) in enumerate(plays)])
    both = possession_windows_from_actions(pd.concat([game1, game2], ignore_index=True), frames)
    for game, actions, game_frames in ((1, game1, one), (2, game2, two)):
        alone = possession_windows_from_actions(actions, game_frames).reset_index(drop=True)
        got = both[both["game_id"] == game].reset_index(drop=True)
        assert len(alone) == 3  # non-vacuity: every game has several possessions to merge wrongly
        pd.testing.assert_frame_equal(got, alone)


def test_possession_windows_warn_nothing_when_every_period_is_tracked():
    frames = make_coordination_match(seconds=60.0, hz=10.0)
    actions = pd.DataFrame([_action(1, 1, 0, 0.0, 1, "pass"), _action(1, 1, 1, 20.0, 2, "tackle")])
    with warnings.catch_warnings():
        warnings.simplefilter("error", CoordinationCoverageWarning)
        windows = possession_windows_from_actions(actions, frames)
    assert len(windows) == 2


def test_links_override_action_times():
    frames = make_coordination_match(seconds=60.0, hz=10.0)
    actions = pd.DataFrame([_action(1, 1, 0, 0.0, 1, "pass"), _action(1, 1, 1, 30.0, 2, "pass")])
    # link action 0 to the frame at t=5.0 -> its windowing time becomes 5.0 (C23)
    fid = int(frames.loc[frames["time_seconds"] == 5.0, "frame_id"].iloc[0])
    links = pd.DataFrame([{"action_id": 0, "frame_id": fid}])
    plain = possession_windows_from_actions(actions, frames)
    linked = possession_windows_from_actions(actions, frames, links=links)
    assert plain["start_time_s"].min() == 0.0
    assert linked.sort_values("start_time_s")["start_time_s"].iloc[0] == 5.0


def test_links_override_action_times_across_id_dtypes():
    # ADR-019: a game id that is a string on the actions but an int on the frames, and link ids stored as strings,
    # are the same ids -- the frame time still overrides, never a silent fallback to the event time.
    frames = make_coordination_match(seconds=60.0, hz=10.0)
    actions = pd.DataFrame([_action(1, 1, 0, 0.0, 1, "pass"), _action(1, 1, 1, 30.0, 2, "pass")])
    actions["game_id"] = actions["game_id"].astype(str)
    fid = int(frames.loc[frames["time_seconds"] == 5.0, "frame_id"].iloc[0])
    links = pd.DataFrame({"action_id": ["0"], "frame_id": [str(fid)]})
    linked = possession_windows_from_actions(actions, frames, links=links)
    assert linked.sort_values("start_time_s")["start_time_s"].iloc[0] == 5.0


# --------------------------------------------------------------------------- possession from frames
def test_frames_possession_bridges_same_team_gaps_both_sides():
    from silly_kicks.coordination import _windows

    hz = 10.0
    t = np.arange(200) / hz
    # last team-1 frame at t=5.0, next at t=10.0 -> the same-team NA gap is exactly 5.0 s.
    team_bridge = np.where(t <= 5.0, 1.0, np.where(t < 10.0, np.nan, 1.0))
    assert len(_windows._spells(1, 1, t, team_bridge.astype(object), 5.0, hz, 3)) == 1  # 5.0 s bridges
    # last team-1 frame at t=4.9 -> the gap is 5.1 s and splits at gap = 5.0.
    team_split = np.where(t < 5.0, 1.0, np.where(t < 10.0, np.nan, 1.0))
    assert len(_windows._spells(1, 1, t, team_split.astype(object), 5.0, hz, 3)) == 2  # gap_s + 1/hz splits
    # a different-team gap always splits, however short.
    team_diff = np.where(t <= 5.0, 1.0, np.where(t < 5.5, np.nan, 2.0))
    assert len(_windows._spells(1, 1, t, team_diff.astype(object), 5.0, hz, 3)) == 2


# --------------------------------------------------------------------------- validation / C25
def test_mixed_possession_sources_refused():
    frames = make_coordination_match(seconds=20.0, hz=10.0)
    actions = make_coordination_actions(frames)
    ev = possession_windows_from_actions(actions, frames)
    tr = possession_windows_from_frames(frames)
    with pytest.raises(ValueError, match="never mixed"):
        validate_windows(pd.concat([ev, tr], ignore_index=True))


def test_caller_mixed_with_builder_refused():
    frames = make_coordination_match(seconds=20.0, hz=10.0)
    per = period_windows(frames)
    caller = per.copy()
    caller["window_source"] = "caller"
    with pytest.raises(ValueError, match="caller windows cannot mix"):
        validate_windows(pd.concat([per, caller], ignore_index=True))


def test_windows_index_must_be_unique():
    # A-50: the compute addresses each window by its index label; a duplicate label would return several rows. A
    # concatenation without reset_index is refused, a reset one accepted.
    frames = make_coordination_match(seconds=20.0, hz=10.0)
    dup = pd.concat([period_windows(frames), period_windows(frames, length_s=10.0, step_s=10.0)])  # no ignore_index
    assert not dup.index.is_unique  # fixture precondition (ADR-032)
    with pytest.raises(ValueError, match="index must be unique"):
        validate_windows(dup)
    validate_windows(dup.reset_index(drop=True))  # must not raise


def test_caller_windows_validated():
    frames = make_coordination_match(seconds=20.0, hz=10.0)
    per = period_windows(frames)
    bad_token = per.copy()
    bad_token["window_source"] = "bogus"
    with pytest.raises(ValueError, match="unknown window_source"):
        validate_windows(bad_token)
    with pytest.raises(ValueError, match="missing columns"):
        validate_windows(per.drop(columns=["start_time_s"]))


# --------------------------------------------------------------------------- n_phases (A-22, owner ruling 2026-10-04)
# A window's OWN n_phases decides its subdivision (spec 7.6, D3: "default 3 on possession windows"); NA means none.
# It must be NA or an integer >= 2: n = 1 would only duplicate the window's own row.
def _caller(frames, n_phases):
    w = period_windows(frames).copy()
    w["window_source"] = "caller"
    w["n_phases"] = n_phases
    return w


@pytest.mark.parametrize("bad", [0, -1, 1, 2.5, "3", True])
def test_window_n_phases_must_be_na_or_an_integer_of_at_least_two(bad):
    frames = make_coordination_match(seconds=20.0, hz=10.0)
    with pytest.raises(ValueError, match="n_phases"):
        validate_windows(_caller(frames, bad))


@pytest.mark.parametrize("good", [pd.NA, 2, 3, 7])
@pytest.mark.parametrize("dtype", ["Int64", "int64", "object"])
def test_window_n_phases_accepts_na_and_integers_of_at_least_two(good, dtype):
    if good is pd.NA and dtype == "int64":
        pytest.skip("a numpy int64 column cannot hold NA")
    frames = make_coordination_match(seconds=20.0, hz=10.0)
    w = _caller(frames, pd.NA)
    w["n_phases"] = pd.Series([good] * len(w), dtype=dtype, index=w.index)
    validate_windows(w)


@pytest.mark.parametrize("n_phases", [0, 1, -3])
def test_possession_builders_refuse_a_subdivision_below_two(n_phases):
    frames = make_coordination_match(seconds=60.0, hz=10.0)
    actions = make_coordination_actions(frames)
    with pytest.raises(ValueError, match="n_phases"):
        possession_windows_from_actions(actions, frames, n_phases=n_phases)
    with pytest.raises(ValueError, match="n_phases"):
        possession_windows_from_frames(frames, n_phases=n_phases)


def test_period_and_sliding_windows_carry_no_subdivision():
    frames = make_coordination_match(seconds=60.0, hz=10.0)
    for w in (period_windows(frames), period_windows(frames, length_s=20.0, step_s=10.0)):
        assert w["n_phases"].isna().all()
        assert str(w["n_phases"].dtype) == "Int64"


# --------------------------------------------------------------------------- stoppages
def test_stoppage_ball_state_intervals():
    frames = make_coordination_match(seconds=120.0, hz=10.0, provider="sportec", dead_intervals=[(20.0, 60.0)])
    ev = resolve_stoppages(frames, actions=None, provider="sportec", max_stoppage_s=25.0)
    assert ev.source == "ball_state"
    assert ev.n_splits == 1
    assert ev.dead_seconds == pytest.approx(40.0, abs=0.2)


def test_ball_state_dead_run_to_period_end_includes_the_last_frame():
    # A-49: a dead run reaching the last frame used to end at the last sample's time (one frame short). A 40 s
    # stoppage running to the period end (100..140 s on a 140 s match) must measure its full ~40 s, not 39.9 s.
    frames = make_coordination_match(seconds=140.0, hz=10.0, provider="sportec", dead_intervals=[(100.0, 140.0)])
    ev = resolve_stoppages(frames, actions=None, provider="sportec", max_stoppage_s=25.0)
    hi = float(ev.intervals[(1, 1)][0][1])
    assert hi > frames.loc[~frames["is_ball"].to_numpy(bool)].time_seconds.max() - 1e-9  # past the last frame time
    assert ev.dead_seconds == pytest.approx(40.0, abs=0.15)


def test_event_intervals_use_chronological_order_under_tied_times(monkeypatch):
    # A-49: _event_intervals sorts chronologically (ADR-065 time-then-action_id), not an unstable sort on time alone.
    sk = make_coordination_match(seconds=120.0, hz=10.0, provider="skillcorner")
    actions = make_coordination_actions(sk, restarts=[(60.0, "throw_in")])
    ev = resolve_stoppages(sk, actions=actions, provider="skillcorner", max_stoppage_s=1.0, mode="events")
    assert ev.n_splits >= 1  # the restart interval is detected under the robust order


def test_stoppage_precedence_each_source_reachable():
    sportec = make_coordination_match(seconds=120.0, hz=10.0, provider="sportec", dead_intervals=[(20.0, 60.0)])
    assert resolve_stoppages(sportec, actions=None, provider="sportec", max_stoppage_s=25.0).source == "ball_state"

    sk = make_coordination_match(seconds=120.0, hz=10.0, provider="skillcorner")
    actions = make_coordination_actions(sk, restarts=[(60.0, "throw_in")])
    assert resolve_stoppages(sk, actions=actions, provider="skillcorner", max_stoppage_s=1.0).source == "events"
    assert resolve_stoppages(sk, actions=None, provider="skillcorner", max_stoppage_s=1.0).source == "unavailable"
    # constant-alive skillcorner never reports ball_state (auto)
    assert resolve_stoppages(sk, actions=actions, provider="skillcorner", max_stoppage_s=1.0).source != "ball_state"


@pytest.mark.parametrize("restart", RESTART_TYPES)
def test_event_intervals_for_every_restart_type(restart):
    frames = make_coordination_match(seconds=60.0, hz=10.0, provider="idsse")
    actions = pd.DataFrame([_action(1, 1, 0, 0.0, 1, "pass"), _action(1, 1, 1, 30.0, 1, restart)])
    ev = resolve_stoppages(frames, actions=actions, provider="idsse", max_stoppage_s=25.0, mode="events")
    assert ev.n_splits == 1  # [0, 30] is a 30 s stoppage


def test_event_intervals_for_goals():
    frames = make_coordination_match(seconds=90.0, hz=10.0, provider="idsse")
    for result, name in (("success", "shot"), ("owngoal", "pass")):
        actions = pd.DataFrame([_action(1, 1, 0, 0.0, 1, name, result=result), _action(1, 1, 1, 40.0, 2, "pass")])
        ev = resolve_stoppages(frames, actions=actions, provider="idsse", max_stoppage_s=25.0, mode="events")
        assert ev.n_splits == 1  # goal -> next action is a 40 s stoppage


def test_only_longer_than_max_stoppage_splits():
    frames = make_coordination_match(seconds=60.0, hz=10.0, provider="idsse")
    exact = pd.DataFrame([_action(1, 1, 0, 0.0, 1, "pass"), _action(1, 1, 1, 25.0, 1, "throw_in")])
    assert resolve_stoppages(frames, actions=exact, provider="idsse", max_stoppage_s=25.0, mode="events").n_splits == 0
    over = pd.DataFrame([_action(1, 1, 0, 0.0, 1, "pass"), _action(1, 1, 1, 25.0 + 1e-3, 1, "throw_in")])
    assert resolve_stoppages(frames, actions=over, provider="idsse", max_stoppage_s=25.0, mode="events").n_splits == 1


def test_dead_ball_observed_unclassified_provider_raises():
    frames = make_coordination_match(seconds=20.0, hz=10.0, provider="idsse")
    with pytest.raises(ValueError, match="_DETECTION_AWARE_PROVIDERS"):
        resolve_stoppages(frames, actions=None, provider="wyscout", max_stoppage_s=25.0)


def test_explicit_mode_that_cannot_be_honoured_raises():
    sk = make_coordination_match(seconds=20.0, hz=10.0, provider="skillcorner")
    with pytest.raises(ValueError, match="cannot be honoured"):
        resolve_stoppages(sk, actions=None, provider="skillcorner", max_stoppage_s=25.0, mode="ball_state")


# --------------------------------------------------------------------------- phase assignment (C5)
def test_phase_assignment_rank_rule():
    assert list(phase_assignment(9, 3)) == [1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert list(phase_assignment(10, 3)) == [1, 1, 1, 2, 2, 2, 3, 3, 3, 3]
    assert list(phase_assignment(2, 3)) == [2, 3]
    assert phase_assignment(5, 3)[0] == 1  # the first sample is always assigned
