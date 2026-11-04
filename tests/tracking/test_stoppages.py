"""Event-derived stoppage intervals and the frame stoppage state (spec section 27.19; DL ruling R-C1, 2026-10-26)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.tracking._stoppages import (
    STOPPAGE_INTERVAL_COLUMNS,
    STOPPAGE_RULE_VERSION,
    STOPPAGE_SOURCES,
    derive_stoppage_state,
    event_stoppage_intervals,
    frame_stoppage_state,
    has_stoppage_state,
)


def _act(
    aid, t, name, *, result="success", provenance="native", t_kick=None, period=1, marker=None, team="A", frame_end=None
):
    """One action row; ``t_kick`` (the resolver's kick time) only when given, as in a resolver output. ``marker`` is the
    GI ``gi_interruption_after`` tag and ``frame_end`` its possession's GI frame_end (rule version 3, fix (A))."""
    row = {
        "game_id": 7,
        "period_id": period,
        "action_id": aid,
        "time_seconds": float(t),
        "team_id": team,
        "type_id": spadlconfig.actiontypes.index(name),
        "result_id": spadlconfig.results.index(result),
        "action_provenance": provenance,
        "gi_interruption_after": pd.NA if marker is None else marker,
        "gi_possession_frame_end": pd.NA if frame_end is None else int(frame_end),
    }
    if t_kick is not None:
        row["t_kick"] = float(t_kick)
    return row


def _ball(t, x, y, visibility: object = True, period=1, provider="skillcorner", rate=10.0):
    """Ball rows at the given times; ``x`` / ``y`` / ``visibility`` are scalars or per-row sequences."""
    t = np.asarray(t, dtype=float)
    n = len(t)
    return pd.DataFrame(
        {
            "game_id": 7,
            "period_id": period,
            "frame_id": np.arange(n),
            "time_seconds": t,
            "frame_rate": rate,
            "is_ball": True,
            "x": np.broadcast_to(np.asarray(x, dtype=float), (n,)).copy(),
            "y": np.broadcast_to(np.asarray(y, dtype=float), (n,)).copy(),
            "visibility": list(np.broadcast_to(np.asarray(visibility, dtype=object), (n,))),
            "ball_state": "alive",
            "source_provider": provider,
        }
    )


def _at(frames, state, t):
    return set(state[np.isclose(frames["time_seconds"].to_numpy(), t)])


def test_the_previous_action_instant_is_live():
    # The start is half-open: the pass at t = 10 (a shot or cross that goes out) is a live action. Mutant: a closed
    # start makes the frame at t = 10 dead.
    actions = pd.DataFrame([_act(1, 10.0, "pass"), _act(2, 20.0, "throw_in")])
    ivs = event_stoppage_intervals(actions, None, provider="gradientsports")
    assert list(ivs.columns) == list(STOPPAGE_INTERVAL_COLUMNS)
    row = ivs.iloc[0]
    assert (row["start"], row["end"], row["start_source"], row["end_source"]) == (
        10.0,
        20.0,
        "after_previous_action",
        "event",
    )
    frames = _ball(np.arange(9.5, 21.0, 0.5), 50.0, 30.0)
    state = frame_stoppage_state(frames, ivs)
    assert _at(frames, state, 10.0) == {"live"}
    assert _at(frames, state, 10.5) == {"dead"}
    assert _at(frames, state, 20.0) == {"live"}


def test_a_run_of_two_frames_beyond_a_line_by_the_ball_radius_sets_the_start():
    # One out frame (t = 11.0) is tracker noise; a frame 0.05 m beyond the line (t = 11.5) is still on the line
    # (Law 9: wholly crossed = beyond by the ball radius); the run from t = 12.0 opens the stoppage, closed at 12.0.
    t = np.round(np.arange(10.1, 20.0, 0.1), 3)
    y = np.full(len(t), 30.0)
    y[np.isclose(t, 11.0)] = -0.5
    y[np.isclose(t, 11.5) | np.isclose(t, 11.6)] = -0.05
    y[t >= 12.0 - 1e-9] = -0.2
    frames = _ball(t, 50.0, y)
    actions = pd.DataFrame([_act(1, 10.0, "pass"), _act(2, 20.0, "throw_in")])
    row = event_stoppage_intervals(actions, frames, provider="gradientsports").iloc[0]
    assert row["start_source"] == "ball_out"
    assert row["start"] == pytest.approx(12.0)
    state = frame_stoppage_state(frames, event_stoppage_intervals(actions, frames, provider="gradientsports"))
    assert _at(frames, state, 12.0) == {"dead"}
    assert _at(frames, state, 11.9) == {"live"}


def test_the_ball_radius_margin_is_in_real_metres():
    # 0.15 scaled units below the touchline: 0.15 m on a 68 m pitch (out), 0.075 m on a 34 m pitch (on the line).
    # Mutant: a margin in scaled units gives the same answer on both pitches.
    t = np.round(np.arange(10.1, 20.0, 0.1), 3)
    frames = _ball(t, 50.0, np.where(t >= 12.0 - 1e-9, -0.15, 30.0))
    actions = pd.DataFrame([_act(1, 10.0, "pass"), _act(2, 20.0, "throw_in")])
    full = event_stoppage_intervals(actions, frames, provider="gradientsports", pitch_width=68.0).iloc[0]
    half = event_stoppage_intervals(actions, frames, provider="gradientsports", pitch_width=34.0).iloc[0]
    assert full["start_source"] == "ball_out"
    assert half["start_source"] == "after_previous_action"


def test_a_goal_span_reads_the_goal_lines_only():
    # After the goal at t = 30 the ball crosses a touchline first (t = 31) and the goal line at t = 33: only the
    # goal line opens the goal span. Mutant: any line.
    t = np.round(np.arange(30.1, 60.0, 0.1), 3)
    x = np.full(len(t), 100.0)
    y = np.full(len(t), 34.0)
    y[(t >= 31.0 - 1e-9) & (t < 32.0)] = -1.0
    x[t >= 33.0 - 1e-9] = 105.5
    frames = _ball(t, x, y)
    actions = pd.DataFrame([_act(1, 30.0, "shot"), _act(2, 60.0, "pass")])
    ivs = event_stoppage_intervals(actions, frames, provider="gradientsports")
    row = ivs[ivs["kind"] == "goal"].iloc[0]
    assert row["start"] == pytest.approx(33.0)
    assert (row["start_source"], row["end"], row["end_source"]) == ("ball_out", 60.0, "event")


def test_a_skillcorner_restart_ends_at_the_resolved_kick_else_an_unknown_span():
    actions = pd.DataFrame(
        [
            _act(1, 10.0, "pass"),
            _act(2, 25.0, "throw_in", provenance="derived", t_kick=20.0),
            _act(3, 40.0, "pass"),
            _act(4, 50.0, "corner_crossed", provenance="derived"),
        ]
    )
    ivs = event_stoppage_intervals(actions, None, provider="skillcorner").set_index("anchor_action_id")
    assert (ivs.loc[2, "end"], ivs.loc[2, "end_source"]) == (20.0, "resolved_kick")
    assert (ivs.loc[4, "end"], ivs.loc[4, "end_source"]) == (50.0, "placeholder")
    assert (ivs["resolved_kick_reject_reason"] == "none").all()  # no kick at row 4: unresolved, not rejected
    frames = _ball(np.arange(9.0, 52.0, 0.5), 50.0, 30.0)
    state = frame_stoppage_state(frames, ivs.reset_index())
    assert _at(frames, state, 22.0) == {"live"}  # the delivery after the resolved kick is live
    assert _at(frames, state, 45.0) == {"unknown"}


_P, _R = 10.0, 25.0  # the previous action's instant and the derived throw-in's own time (its placeholder)


def _kick_case(t_kick, *, out_from=None, out_to: float = 20.0, visibility: object = True):
    """A derived throw-in at R with the resolved ``t_kick``; the ball is beyond the touchline from ``out_from`` to
    ``out_to`` (no ball-out when ``out_from`` is None). Returns the frames and the throw-in's span."""
    t = np.round(np.arange(10.1, 30.0, 0.1), 3)
    y = np.full(len(t), 30.0)
    if out_from is not None:
        y[(t >= out_from - 1e-9) & (t < out_to)] = -0.5
    frames = _ball(t, 50.0, y, visibility=visibility)
    actions = pd.DataFrame(
        [_act(1, _P, "pass"), _act(2, _R, "throw_in", provenance="derived", t_kick=t_kick), _act(3, 40.0, "pass")]
    )
    return frames, event_stoppage_intervals(actions, frames, provider="skillcorner").iloc[0]


@pytest.mark.parametrize(
    ("t_kick", "reason"),
    [
        (4.0, "before_previous_action"),
        (10.0, "before_previous_action"),  # K = P: the previous action's instant
        (10.05, "none"),  # just after P, no ball-out: the span (P, K), dead
        (25.0, "none"),  # K = R: the upper bound is closed
        (25.1, "after_restart_row"),  # after R: after the receiver's possession began
    ],
)
def test_rule_v2_the_kick_against_the_previous_action_and_the_restart_row(t_kick, reason):
    # DL ruling 2026-10-27 (Q1), both sides of each bound. A rejected kick keeps the start and ends at R (UNKNOWN).
    # Mutant: the v1 end (any finite kick) ends the span after R, so the restart row's own instant falls inside it.
    _frames, row = _kick_case(t_kick)
    assert row["resolved_kick_reject_reason"] == reason
    end = (t_kick, "resolved_kick") if reason == "none" else (_R, "placeholder")
    assert (row["start"], row["start_source"], row["end"], row["end_source"]) == (_P, "after_previous_action", *end)


@pytest.mark.parametrize(
    ("t_kick", "reason"), [(12.1, "none"), (12.0, "ball_out_after_kick"), (11.5, "ball_out_after_kick")]
)
def test_rule_v2_the_kick_against_the_first_ball_out(t_kick, reason):
    # B = 12.0, the first ball-out of start rule (b) in (P, R). B < K: the span [B, K), dead. B >= K: the only
    # ball-out follows the kick, so the kick is rejected and the span is (P, R), UNKNOWN, started after P. Mutant: the
    # built start (B, a span [B, R)) marks (P, B) live: the frame at 11.0 s.
    frames, row = _kick_case(t_kick, out_from=12.0)
    assert row["resolved_kick_reject_reason"] == reason
    state = frame_stoppage_state(frames, pd.DataFrame([row]))
    if reason == "none":
        assert (row["start_source"], row["start"], row["end"]) == ("ball_out", pytest.approx(12.0), t_kick)
        assert _at(frames, state, 11.0) == {"live"} and _at(frames, state, 12.0) == {"dead"}
    else:
        assert (row["start_source"], row["start"], row["end"], row["end_source"]) == (
            "after_previous_action",
            _P,
            _R,
            "placeholder",
        )
        assert _at(frames, state, 11.0) == {"unknown"} and _at(frames, state, 24.9) == {"unknown"}


@pytest.mark.parametrize(("out_from", "reason"), [(24.8, "ball_out_after_kick"), (24.9, "none"), (25.0, "none")])
def test_rule_v2_b_is_searched_in_p_to_r_open_at_r(out_from, reason):
    # DL answers 2026-10-27, item 2: B is searched in (P, R), open at R; a ball-out that starts at the restart row's
    # own time (a throw-in ball held beyond the touchline at the throw) is not the start of that stoppage. The ball
    # stays out from out_from to 30 s; K = 24.0 s. A run of the minimum length (2 frames) that starts at 24.8 s lies
    # inside (P, R): it is B, and B >= K rejects the kick. A run that starts at 24.9 s has its second frame AT R, and a
    # run that starts at R lies after it: neither is B, so the kick is accepted. Mutants: the window (P, R] fails the
    # 24.9 s case; a search that reads past R fails the 24.9 s and 25.0 s cases.
    _frames, row = _kick_case(24.0, out_from=out_from, out_to=30.0)
    assert row["resolved_kick_reject_reason"] == reason
    if reason == "none":
        assert (row["start"], row["start_source"], row["end"], row["end_source"]) == (
            _P,
            "after_previous_action",
            24.0,
            "resolved_kick",
        )


def test_rule_v2_records_the_detection_of_a_ball_out_after_the_kick():
    # The report splits "ball_out_after_kick" by a detected / extrapolated ball-out (DL ruling 2026-10-27, Q1).
    for visibility, detected in ((True, True), (False, False)):
        _frames, row = _kick_case(11.5, out_from=12.0, visibility=visibility)
        assert row["resolved_kick_reject_reason"] == "ball_out_after_kick"
        assert bool(row["rejected_ball_out_detected"]) is detected
        assert pd.isna(row["start_ball_detected"])  # the span starts after P, not at the ball-out
    _frames, kept = _kick_case(12.1, out_from=12.0)
    assert pd.isna(kept["rejected_ball_out_detected"])  # only a ball_out_after_kick reject carries the flag


def test_an_old_converter_skillcorner_restart_raises():
    actions = pd.DataFrame([_act(1, 10.0, "pass"), _act(2, 25.0, "throw_in", provenance="native")])
    with pytest.raises(ValueError, match="reconvert"):
        event_stoppage_intervals(actions, None, provider="skillcorner")


def test_a_native_skillcorner_penalty_ends_at_its_own_time():
    # The goals reconciliation re-types a native shot to shot_penalty and keeps it native (found in 3 staged
    # SkillCorner matches on 2026-10-27): the restart span ends at the shot's own time, "event"; t_kick is never read.
    for t_kick in (None, 69.5, 70.3):
        rows = [
            _act(1, 10.0, "foul"),
            _act(2, 70.0, "shot_penalty", provenance="native", t_kick=t_kick),
            _act(3, 80.0, "pass"),
        ]
        row = event_stoppage_intervals(pd.DataFrame(rows), None, provider="skillcorner").iloc[0]
        assert (row["kind"], row["end"], row["end_source"]) == ("shot_penalty", 70.0, "event"), t_kick


def test_a_derived_skillcorner_penalty_is_covered_by_the_goal_rule_only():
    # Rule version 3 (DL ruling 2026-10-27, pen_measure item 1; owner decision 2026-10-27): a derived shot_penalty is a
    # scored penalty anchored at the END of the possession that won it (the foul). It ends no restart span; the goal
    # rule makes (row, next action) dead, open at the row. Mutant: the restart-span rule kept for it gives a dead span
    # BEFORE the foul (the last live play, 69.9 s here).
    rows = [
        _act(1, 10.0, "foul"),
        _act(2, 70.0, "shot_penalty", provenance="derived", t_kick=70.3),
        _act(3, 80.0, "pass", team="B"),
    ]
    ivs = event_stoppage_intervals(pd.DataFrame(rows), None, provider="skillcorner")
    assert list(ivs["kind"]) == ["goal"]
    assert (ivs.iloc[0]["start"], ivs.iloc[0]["end"], ivs.iloc[0]["end_source"]) == (70.0, 80.0, "event")
    probe = pd.DataFrame({"game_id": 7, "period_id": 1, "time_seconds": [69.9, 70.0, 75.0]})
    assert list(frame_stoppage_state(probe, ivs)) == ["live", "live", "dead"]


def _marker_frames(fids):
    """Ball frames at 10 fps carrying the GI frame ids ``fids`` (time = frame / 10)."""
    fids = np.asarray(fids)
    f = _ball(fids / 10.0, 50.0, 30.0)
    f["frame_id"] = fids
    return f


def _marker_case(next_name, next_team, frames):
    """The possession that won a penalty (team A, starting at 50.0 s, GI frame_end 520 = 52.0 s), then the next
    action at 60.0 s. Returns the penalty-marker span."""
    rows = [
        _act(1, 40.0, "pass"),
        _act(2, 50.0, "dribble", marker="penalty_for", frame_end=520),
        _act(3, 60.0, next_name, team=next_team, result="fail"),
    ]
    ivs = event_stoppage_intervals(pd.DataFrame(rows), frames, provider="skillcorner")
    return ivs[ivs["kind"] == "penalty_marker"]


@pytest.mark.parametrize(
    "frames",
    [None, "dropped"],
    ids=["no frames", "frame id dropped"],
)
def test_the_penalty_marker_falls_back_to_the_row_time_open(frames):
    # DL ruling 2026-10-27 (the marker end time): when gi_possession_frame_end does not map (no frames, or the frame
    # was dropped -- here 519 and 521 exist, 520 does not), the span starts at the marker row's own time, open,
    # "after_previous_action". The taker's native shot next: DEAD, "event"; any other next action: UNKNOWN,
    # "next_action". Mutants: the fallback makes the shot branch UNKNOWN; the fallback gives no span; the mapping takes
    # the nearest frame (51.9 or 52.1) instead of an exact match.
    f = None if frames is None else _marker_frames([400, 519, 521, 600])
    for next_name, next_team, end_source in (("shot", "A", "event"), ("pass", "B", "next_action")):
        span = _marker_case(next_name, next_team, f)
        assert len(span) == 1, (frames, next_name)
        got = span.iloc[0]
        assert (got["start"], got["start_source"], got["end"], got["end_source"]) == (
            50.0,
            "after_previous_action",
            60.0,
            end_source,
        )
    probe = pd.DataFrame({"game_id": 7, "period_id": 1, "time_seconds": [50.0, 55.0]})
    shot = frame_stoppage_state(probe, _marker_case("shot", "A", f))
    other = frame_stoppage_state(probe, _marker_case("pass", "B", f))
    assert list(shot) == ["live", "dead"] and list(other) == ["live", "unknown"]


def test_the_penalty_marker_starts_at_the_mapped_possession_end_open():
    # The frame id maps exactly: the span starts at 52.0 s ("gi_possession_end"), open (the foul instant stays live).
    # Mutant: a closed start marks 52.0 s dead; a start at the row time marks 51.0 s not live.
    f = _marker_frames([400, 510, 520, 530, 600])
    span = _marker_case("shot", "A", f).iloc[0]
    assert (span["start"], span["start_source"], span["end_source"]) == (52.0, "gi_possession_end", "event")
    probe = pd.DataFrame({"game_id": 7, "period_id": 1, "time_seconds": [51.0, 52.0, 52.1]})
    assert list(frame_stoppage_state(probe, _marker_case("shot", "A", f))) == ["live", "live", "dead"]
    # the other team's action next (a kick-off after the penalty): UNKNOWN up to it
    other = _marker_case("pass", "B", f).iloc[0]
    assert (other["start"], other["end_source"]) == (52.0, "next_action")


def test_skillcorner_actions_without_the_marker_column_raise():
    # A converter older than the penalty marker (rule version 3): reconvert. Mutant: a silent default would leave the
    # foul -> kick stretch live.
    actions = pd.DataFrame([_act(1, 10.0, "pass"), _act(2, 25.0, "pass")]).drop(columns=["gi_interruption_after"])
    with pytest.raises(ValueError, match="reconvert"):
        event_stoppage_intervals(actions, None, provider="skillcorner")
    event_stoppage_intervals(actions, None, provider="gradientsports")  # other providers need no marker


def test_a_known_dead_span_wins_an_overlap_with_an_unknown_span():
    frames = _ball(np.arange(0.0, 30.0, 1.0), 50.0, 30.0)
    ivs = pd.DataFrame(
        {
            "game_id": 7,
            "period_id": 1,
            "start": [5.0, 10.0],
            "end": [20.0, 15.0],
            "start_source": "ball_out",
            "end_source": ["placeholder", "event"],
        }
    )
    state = frame_stoppage_state(frames, ivs)
    assert _at(frames, state, 7.0) == {"unknown"}
    assert _at(frames, state, 12.0) == {"dead"}
    assert _at(frames, state, 25.0) == {"live"}


def test_the_start_flag_marks_an_extrapolated_ball_frame():
    t = np.round(np.arange(10.1, 20.0, 0.1), 3)
    out = t >= 12.0 - 1e-9
    frames = _ball(t, 50.0, np.where(out, -0.5, 30.0), visibility=np.where(out, False, True))
    actions = pd.DataFrame([_act(1, 10.0, "pass"), _act(2, 20.0, "throw_in", provenance="derived")])
    row = event_stoppage_intervals(actions, frames, provider="skillcorner").iloc[0]
    assert row["start_source"] == "ball_out"
    assert row["start_ball_detected"] is False or bool(row["start_ball_detected"]) is False


def test_the_column_follows_the_observed_dead_ball_set():
    # IDSSE / Sportec: the column copies ball_state (alive -> live, dead -> dead) and needs no actions.
    frames = _ball(np.arange(0.0, 10.0, 0.04), 50.0, 30.0, provider="sportec", rate=25.0)
    frames.loc[frames["time_seconds"] >= 5.0, "ball_state"] = "dead"
    out = derive_stoppage_state(frames)
    assert "stoppage_state" not in frames.columns  # pure: the input is not mutated
    st = out["stoppage_state"].astype(object).to_numpy()
    t = out["time_seconds"].to_numpy()
    assert set(st[t < 5.0]) == {"live"} and set(st[t >= 5.0]) == {"dead"}
    assert set(out["stoppage_source"].astype(object)) == {"ball_state"}
    assert (out["ball_state"].astype(object) == frames["ball_state"].astype(object)).all()  # ball_state untouched
    # outside the set the constant "alive" cannot mask a dead ball: the actions are required
    with pytest.raises(ValueError, match="pass the actions"):
        derive_stoppage_state(_ball(np.arange(0.0, 10.0, 0.1), 50.0, 30.0))
    with pytest.raises(ValueError):
        derive_stoppage_state(_ball(np.arange(0.0, 10.0, 0.1), 50.0, 30.0, provider="statsbomb"))


def test_the_column_equals_the_core_spans():
    # The column is built from the single core (owner decision R-C2 item 8): the same input gives the same states.
    t = np.round(np.arange(9.0, 52.0, 0.1), 3)
    frames = _ball(t, 50.0, np.where((t >= 12.0 - 1e-9) & (t < 20.0), -0.5, 30.0))
    actions = pd.DataFrame(
        [
            _act(1, 10.0, "pass"),
            _act(2, 25.0, "throw_in", provenance="derived"),
            _act(3, 40.0, "pass"),
            _act(4, 50.0, "corner_crossed", provenance="derived"),
        ]
    )
    out = derive_stoppage_state(frames, actions=actions)
    core = frame_stoppage_state(frames, event_stoppage_intervals(actions, frames, provider="skillcorner"))
    assert list(out["stoppage_state"].astype(object)) == list(core)
    assert set(out["stoppage_source"].astype(object)) == {STOPPAGE_SOURCES[1]}
    report = out.attrs["stoppage_report"]
    assert report["rule_version"] == STOPPAGE_RULE_VERSION
    game = report["games"][next(iter(report["games"]))]
    assert game["end_source"] == {"placeholder": 2}
    assert len(report["spans"]) == 2


def test_the_report_is_json_serializable_with_a_populated_detection_flag():
    # The nullable `boolean` span column `start_ball_detected` is populated only when the ball-out is
    # OBSERVED (a detection-aware provider such as Gradient Sports in the field; here a ball that goes
    # undetected past the line). On `itertuples` a non-null value is `numpy.bool_`, whose type name is
    # `bool` under numpy 2.x and which `json.dumps` rejects. `stoppage_report` is a serialization mirror
    # (the parquet writer json-dumps `df.attrs` on every shard write), so a numpy scalar there breaks the
    # materialize. Red before the Python-native coercion in `derive_stoppage_state`.
    t = np.round(np.arange(10.1, 22.0, 0.1), 3)
    out_mask = t >= 12.0 - 1e-9
    frames = _ball(t, 50.0, np.where(out_mask, -0.5, 30.0), visibility=np.where(out_mask, False, True))
    actions = pd.DataFrame([_act(1, 10.0, "pass"), _act(2, 20.0, "throw_in", provenance="derived")])
    report = derive_stoppage_state(frames, actions=actions).attrs["stoppage_report"]
    flag_ix = list(report["span_columns"]).index("start_ball_detected")
    detected = [s[flag_ix] for s in report["spans"] if s[flag_ix] is not None]
    assert detected, "fixture must exercise a span with a populated start_ball_detected (else the test is vacuous)"
    assert all(type(v) is bool for v in detected), "start_ball_detected must coerce to a Python bool, not numpy.bool_"
    json.dumps(report)  # must not raise 'Object of type bool is not JSON serializable'


def test_the_column_refuses_actions_that_carry_t_kick():
    # Owner decision 2026-10-27: events_v<N> does not record the kicks, so a build with resolved kicks would carry the
    # label of a build without them; it is refused. An all-NaN t_kick is a resolver output too. The core keeps t_kick
    # (the R.2 probes). Mutant: the guard removed (the build labels resolved kicks events_v2).
    frames = _ball(np.arange(9.0, 52.0, 0.1), 50.0, 30.0)
    actions = pd.DataFrame([_act(1, 10.0, "pass"), _act(2, 25.0, "throw_in", provenance="derived", t_kick=20.0)])
    with pytest.raises(ValueError, match="carry t_kick"):
        derive_stoppage_state(frames, actions=actions)
    with pytest.raises(ValueError, match="carry t_kick"):
        derive_stoppage_state(frames, actions=actions.assign(t_kick=np.nan))
    assert event_stoppage_intervals(actions, frames, provider="skillcorner").iloc[0]["end_source"] == "resolved_kick"
    built = derive_stoppage_state(frames, actions=actions.drop(columns="t_kick"))
    assert set(built["stoppage_source"].astype(object)) == {STOPPAGE_SOURCES[1]}


def test_the_column_refuses_thinned_frames_and_mixed_providers():
    actions = pd.DataFrame([_act(1, 10.0, "pass"), _act(2, 20.0, "throw_in", provenance="derived")])
    thin = _ball(np.arange(0.0, 30.0, 0.5), 50.0, 30.0)  # every 5th frame of a 10 Hz feed
    with pytest.raises(ValueError, match="full-rate"):
        derive_stoppage_state(thin, actions=actions)
    mixed = _ball(np.arange(0.0, 30.0, 0.1), 50.0, 30.0)
    mixed.loc[mixed.index[:5], "source_provider"] = "metrica"
    with pytest.raises(ValueError, match="one provider per game"):
        derive_stoppage_state(mixed, actions=actions)


def test_the_reader_gate_fails_closed():
    sc = _ball(np.arange(0.0, 30.0, 0.1), 50.0, 30.0)
    with pytest.raises(ValueError, match="derive_stoppage_state"):
        has_stoppage_state(sc, where="reader")
    built = derive_stoppage_state(
        sc, actions=pd.DataFrame([_act(1, 10.0, "pass"), _act(2, 20.0, "throw_in", provenance="derived")])
    )
    assert has_stoppage_state(built, where="reader") is True
    old = built.copy()
    old["stoppage_source"] = (
        old["stoppage_source"].astype(object).replace(STOPPAGE_SOURCES[1], f"events_v{STOPPAGE_RULE_VERSION - 1}")
    )
    with pytest.raises(ValueError, match="not a current rule"):
        has_stoppage_state(old, where="reader")
    idsse = _ball(np.arange(0.0, 10.0, 0.04), 50.0, 30.0, provider="idsse", rate=25.0)
    assert has_stoppage_state(idsse, where="reader") is False  # the reader keeps its ball_state logic
    with pytest.raises(ValueError, match="source_provider"):
        has_stoppage_state(idsse.drop(columns="source_provider"), where="reader")


def test_freeze_frames_are_live_event_instants_and_still_need_the_column():
    # Owner decision 2026-10-26: an SB360 freeze frame sits at an action instant, live by the span rules. The column is
    # "live" / "event_instant" without actions (and even when ball_state says otherwise); the gate stays fail closed.
    # Mutant: the freeze frames take the observed-ball_state branch (dead stays dead) or the gate exempts them.
    snap = _ball([1.0, 7.0, 30.0], 50.0, 30.0, provider="snapshot")
    snap.loc[1, "ball_state"] = "dead"
    out = derive_stoppage_state(snap)
    assert set(out["stoppage_state"].astype(object)) == {"live"}
    assert set(out["stoppage_source"].astype(object)) == {"event_instant"}
    assert has_stoppage_state(out, where="reader") is True
    with pytest.raises(ValueError, match="freeze frames"):
        has_stoppage_state(snap, where="reader")


_TYPES = [
    "pass",
    "dribble",
    "cross",
    "shot",
    "throw_in",
    "corner_crossed",
    "freekick_short",
    "goalkick",
    "tackle",
    "shot_penalty",
]
_RESTARTS = {"throw_in", "corner_crossed", "freekick_short", "goalkick", "shot_penalty"}


@pytest.mark.parametrize("provider", ["gradientsports", "metrica", "skillcorner"])
def test_no_action_instant_is_ever_inside_a_span(provider):
    # The property behind the freeze-frame rule: over random action sequences (tied times, goals, own goals, restarts,
    # out-of-play ball runs), the core never marks an action instant as not live. SkillCorner restarts are derived,
    # with no kick (an unknown span) or a resolved kick ANYWHERE: before the previous action, at either action
    # instant, between them, after the placeholder, after the next action. No exclusion (rule version 2, DL ruling
    # 2026-10-27). Mutant: a closed start, a closed end, or the v1 end (any finite kick) fails it.
    rng = np.random.default_rng(46)
    n_accepted = n_rejected = 0
    for _ in range(200):
        n = int(rng.integers(3, 25))
        t = np.round(np.cumsum(rng.choice([0.0, 0.1, 0.5, 2.0, 7.0], size=n)) + 1.0, 1)
        names = list(rng.choice(_TYPES, size=n))
        rows = []
        for k in range(n):
            res = "success" if names[k] != "shot" else str(rng.choice(["success", "fail"]))
            res = "owngoal" if (names[k] == "pass" and rng.random() < 0.05) else res
            restart = names[k] in _RESTARTS
            tk = np.nan
            if provider == "skillcorner" and restart and k > 0 and rng.random() < 0.8:
                nxt = float(t[k + 1]) if k + 1 < n else float(t[k]) + 3.0
                tk = float(
                    rng.choice(
                        [
                            float(rng.uniform(float(t[0]) - 3.0, float(t[-1]) + 3.0)),
                            float(rng.uniform(t[k - 1], t[k])) if t[k] > t[k - 1] else float(t[k]),
                            float(rng.uniform(t[k], nxt)) if nxt > t[k] else float(t[k]) + 0.1,
                            float(t[k - 1]),
                            float(t[k]),
                            nxt,
                        ]
                    )
                )
            prov = "derived" if (provider == "skillcorner" and restart) else "native"
            # rule version 3: random penalty markers (the possession that won a penalty) with a random GI frame_end
            mark = "penalty_for" if (provider == "skillcorner" and not restart and rng.random() < 0.08) else None
            rows.append(
                _act(
                    k, t[k], names[k], result=res, provenance=prov, t_kick=tk, marker=mark,
                    team=str(rng.choice(["A", "B"])), frame_end=int(rng.integers(0, 200)) if mark else None,
                )
            )  # fmt: skip
        actions = pd.DataFrame(rows)
        grid = np.round(np.arange(0.5, float(t[-1]) + 1.0, 0.1), 1)
        y = np.where(rng.random(len(grid)) < 0.3, -1.0, 30.0)  # out-of-play ball runs open ball_out starts
        frames = _ball(grid, 50.0, y, provider=provider)
        ivs = event_stoppage_intervals(actions, frames, provider=provider)
        points = pd.DataFrame({"game_id": 7, "period_id": 1, "time_seconds": t})
        state = frame_stoppage_state(points, ivs)
        assert set(state) == {"live"}, (names, list(t), ivs.to_dict("records"))
        n_accepted += int((ivs["end_source"] == "resolved_kick").sum())
        n_rejected += int((ivs["resolved_kick_reject_reason"] != "none").sum())
    if provider == "skillcorner":  # non-vacuity: both branches of the kick rule ran
        assert n_accepted > 50 and n_rejected > 50, (n_accepted, n_rejected)
