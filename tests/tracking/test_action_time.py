"""The time step ``sync_action_times`` and the gap ``restart_gaps`` (action-time repair spec §4.3, §4.4, §4.17)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl.schema import restart_kind
from silly_kicks.tracking import _action_time
from silly_kicks.tracking._action_time import restart_gaps, sync_action_times

_T = spadlconfig.actiontype_id
_R = spadlconfig.result_id
FPS = 10.0


def _row(
    aid,
    t,
    type_name,
    team="H",
    *,
    src="tracking",
    prov="native",
    oid=None,
    player="p",
    start=(50.0, 34.0),
    end=(55.0, 34.0),
    result="success",
    period=1,
):
    return {
        "game_id": "g",
        "original_event_id": oid if oid is not None else f"e{aid}",
        "action_id": aid,
        "period_id": period,
        "time_seconds": float(t),
        "team_id": team,
        "player_id": player if prov == "native" else pd.NA,
        "start_x": start[0],
        "start_y": start[1],
        "end_x": end[0],
        "end_y": end[1],
        "type_id": _T[type_name],
        "result_id": _R[result],
        "bodypart_id": 0,
        "action_provenance": prov,
        "time_source": src,
    }


def _actions(*rows) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def _ball(points: list[tuple[float, float, float]], *, period=1, detected=True, provider="skillcorner") -> pd.DataFrame:
    """Ball rows: (t, x, y) per frame."""
    df = pd.DataFrame(points, columns=["time_seconds", "x", "y"])
    df["game_id"] = "g"
    df["period_id"] = period
    df["frame_id"] = (df["time_seconds"] * FPS).round().astype("int64")
    df["is_ball"] = True
    df["player_id"] = pd.array([pd.NA] * len(df), dtype="Int64")
    df["team_id"] = None
    df["visibility"] = detected
    df["source_provider"] = provider
    df["frame_rate"] = FPS
    return df


def _track(t0, t1, at=(60.0, 30.0)):
    """A ball in play: it moves 3 m/s along x (so it never forms a rest run)."""
    return [(round(t, 1), at[0] + 3.0 * ((t - t0) % 10.0), at[1]) for t in np.arange(t0, t1 + 1e-9, 0.1)]


def _rest_release(t_rest0, t_rest1, rest, away):
    """Ball at rest at `rest` from t_rest0 to t_rest1, then 3 m away 0.3 s later."""
    pts = [(round(t, 1), rest[0], rest[1]) for t in np.arange(t_rest0, t_rest1 + 1e-9, 0.1)]
    pts += [(round(t_rest1 + 0.3, 1), away[0], away[1]), (round(t_rest1 + 0.4, 1), away[0], away[1])]
    return pts


def _corner_scene(*, src="placeholder", rest=(104.8, 0.3), first_possession_team="H", end=(100.0, 4.0)):
    """a (A) ends at 20 s; the corner row for H; b (H) receives at 40 s and ends at 42 s."""
    acts = _actions(
        _row(0, 20.0, "pass", "A", oid="a"),
        _row(
            1,
            40.0 - (0.02 if src == "placeholder" else 0.03),
            "corner_crossed",
            "H",
            src=src,
            prov="derived",
            oid="b",
            start=(105.0, 68.0),
            end=end if src == "placeholder" else (105.0, 68.0),
        ),
        _row(2, 42.0, "pass", first_possession_team, oid="b"),
    )
    frames = _ball(_track(20.0, 29.9) + _rest_release(30.0, 31.0, rest, (101.0, 3.0)) + _track(31.5, 45.0))
    return acts, frames


def test_refuses_a_table_without_time_source():
    acts, frames = _corner_scene()
    with pytest.raises(ValueError, match="time_source"):
        sync_action_times(acts.drop(columns=["time_source"]), frames)


def test_corner_kick_found_in_the_gap():
    acts, frames = _corner_scene()
    out = sync_action_times(acts, frames)
    corner = out[out["type_id"].map(restart_kind) == "corner"].iloc[0]
    assert corner["time_source"] == "kick"
    assert corner["time_seconds"] == pytest.approx(31.0)
    # the rest point, in the restart team's frame (the attacking end)
    assert (corner["start_x"], corner["start_y"]) == pytest.approx((104.8, 0.3))
    # a corner with a first possession keeps its end (the reception) and is re-typed by the 15 m rule
    assert (corner["end_x"], corner["end_y"]) == (100.0, 4.0)
    assert corner["type_id"] == _T["corner_short"]


def test_kick_found_with_float_actions_game_id_and_int_frames_game_id():
    # RTR-IMPL A02 (ADR-019): a float-backed game id on the actions (7.0) and an int one on the frames (7) are the
    # same match; a str() join renders "7.0" against "7" and silently loses the kick
    acts, frames = _corner_scene()
    acts["game_id"] = 7.0
    frames["game_id"] = np.int64(7)
    corner = sync_action_times(acts, frames).pipe(lambda o: o[o["type_id"].map(restart_kind) == "corner"].iloc[0])
    assert corner["time_source"] == "kick"
    assert corner["time_seconds"] == pytest.approx(31.0)


def test_kick_found_with_int_actions_game_id_and_float_frames_game_id():
    # RTR-IMPL-05 (B r2) / A r2 R3: the reverse mix -- int on the actions (7), float on the frames (7.0). The round-1
    # str() ball filter rendered the frames as "7.0" against "7" and lost the kick; float actions with int frames
    # alone did not show it
    acts, frames = _corner_scene()
    acts["game_id"] = np.int64(7)
    frames["game_id"] = 7.0
    corner = sync_action_times(acts, frames).pipe(lambda o: o[o["type_id"].map(restart_kind) == "corner"].iloc[0])
    assert corner["time_source"] == "kick"


def test_elastic_runs_with_int_actions_game_id_and_float_frames_game_id(monkeypatch):
    acts = _actions(_row(0, 10.0, "pass", src="event"), _row(1, 20.0, "pass", src="event"))
    acts["game_id"] = np.int64(7)
    frames = _ball(_track(0.0, 30.0))
    frames["game_id"] = 7.0
    seen = []

    def fake_align(actions, frames, **kw):
        seen.append(len(frames))
        return pd.DataFrame({"action_id": [0], "elastic_frame_id": [123]})

    monkeypatch.setattr(_action_time, "align_events_to_frames", fake_align)
    sync_action_times(acts, frames)
    assert seen and seen[0] > 0


def test_elastic_runs_with_float_actions_game_id_and_int_frames_game_id(monkeypatch):
    acts = _actions(_row(0, 10.0, "pass", src="event"), _row(1, 20.0, "pass", src="event"))
    acts["game_id"] = 7.0
    frames = _ball(_track(0.0, 30.0))
    frames["game_id"] = np.int64(7)
    seen = []

    def fake_align(actions, frames, **kw):
        seen.append(len(frames))
        return pd.DataFrame({"action_id": [0], "elastic_frame_id": [123]})

    monkeypatch.setattr(_action_time, "align_events_to_frames", fake_align)
    out = sync_action_times(acts, frames).sort_values("original_event_id")
    assert seen and seen[0] > 0  # the match's frames reached ELASTIC
    assert out["time_source"].tolist() == ["elastic", "event"]


def test_undetected_ball_gives_no_kick():
    acts, frames = _corner_scene()
    frames["visibility"] = False
    out = sync_action_times(acts, frames)
    assert out.loc[out["type_id"].map(restart_kind) == "corner", "time_source"].tolist() == ["placeholder"]


def test_all_null_visibility_gives_no_kick():
    acts, frames = _corner_scene()
    frames["visibility"] = None
    out = sync_action_times(acts, frames)
    assert out.loc[out["type_id"].map(restart_kind) == "corner", "time_source"].tolist() == ["placeholder"]


def test_object_visibility_with_missing_values_raises_no_future_warning():
    # Supervisor note: `vis.fillna(False).astype(bool)` on an object column with missing values raises pandas'
    # "Downcasting object dtype arrays on .fillna" FutureWarning on every real SkillCorner match
    import warnings

    acts, frames = _corner_scene()
    frames["visibility"] = pd.Series([True if i % 7 else None for i in range(len(frames))], dtype=object)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        out = sync_action_times(acts, frames)
    assert len(out) == len(acts)


def test_flag_rest_inside_b_is_not_a_kick():
    acts, _ = _corner_scene()
    # the only rest at a flag lies after b's reception (40 s), inside b's possession
    frames = _ball(_track(20.0, 39.9) + _rest_release(40.5, 41.5, (104.8, 0.3), (101.0, 3.0)) + _track(42.0, 45.0))
    out = sync_action_times(acts, frames)
    assert out.loc[out["type_id"].map(restart_kind) == "corner", "time_source"].tolist() == ["placeholder"]


def test_order_only_corner_keeps_crossed_and_ends_at_the_rest_point():
    acts, frames = _corner_scene(src="order_only")
    # a goal follows: no follower row -> the follower cannot rest at a flag
    out = sync_action_times(acts, frames)
    corner = out[out["type_id"].map(restart_kind) == "corner"].iloc[0]
    assert corner["time_source"] == "kick"
    assert corner["type_id"] == _T["corner_crossed"]
    assert (corner["end_x"], corner["end_y"]) == pytest.approx((104.8, 0.3))


def test_order_only_corner_with_a_throw_in_follower_needs_a_later_run():
    acts, frames = _corner_scene(src="order_only")
    follower = _row(9, 40.0 - 0.02, "throw_in", "A", src="placeholder", prov="derived", oid="b", start=(np.nan, np.nan))
    acts = pd.concat([acts, pd.DataFrame([follower])], ignore_index=True)
    out = sync_action_times(acts, frames)  # the flag run is the last run of the gap -> not shown to be the corner's
    assert out.loc[out["type_id"] == _T["corner_crossed"], "time_source"].tolist() == ["order_only"]


def test_order_only_corner_goal_kick_follower_ignores_an_inferred_defensive_row_of_b():
    # spec 4.4: a goal-kick follower cannot rest at a flag, so the corner needs no later run. An inferred
    # defensive row of b carries b's event id too; it is not the follower.
    acts, frames = _corner_scene(src="order_only")
    follower = _row(9, 40.0 - 0.02, "goalkick", "A", src="placeholder", prov="derived", oid="b", start=(5.5, 34.0))
    inferred = _row(10, 40.0, "interception", "H", prov="derived", oid="b")
    acts = pd.concat([acts, pd.DataFrame([follower, inferred])], ignore_index=True)
    out = sync_action_times(acts, frames)
    assert out.loc[out["type_id"] == _T["corner_crossed"], "time_source"].tolist() == ["kick"]


def test_period_start_gap_starts_at_the_period_first_frame():
    acts = _actions(
        _row(0, 2000.0, "pass", "A", oid="a", period=1),
        _row(
            1,
            9.98,
            "corner_crossed",
            "H",
            src="placeholder",
            prov="derived",
            oid="b",
            period=2,
            start=(105.0, 68.0),
            end=(100.0, 4.0),
        ),
        _row(2, 12.0, "pass", "H", oid="b", period=2),
    )
    frames = pd.concat([_ball(_track(1990.0, 2000.0), period=1), _ball(_track(0.0, 15.0), period=2)])
    gaps = restart_gaps(acts, frames)
    assert gaps["gap_start"].tolist() == [pytest.approx(0.0)]
    assert gaps["gap_end"].tolist() == [pytest.approx(10.0)]


def test_penalty_kick_at_the_mark_last_run():
    acts = _actions(
        _row(0, 20.0, "pass", "H", oid="a"),
        _row(
            1,
            90.0 - 0.03,
            "shot_penalty",
            "H",
            src="order_only",
            prov="derived",
            oid="b",
            start=(94.0, 34.0),
            end=(94.0, 34.0),
        ),
        _row(2, 92.0, "pass", "A", oid="b"),
    )
    # a retake: two rests at the mark; the last one counts
    frames = _ball(
        _track(20.0, 29.9)
        + _rest_release(30.0, 31.0, (94.3, 34.2), (99.0, 34.0))
        + _track(31.5, 49.9, at=(80.0, 34.0))
        + _rest_release(50.0, 51.0, (94.2, 33.9), (99.0, 34.0))
        + _track(51.5, 95.0)
    )
    out = sync_action_times(acts, frames)
    pen = out[out["type_id"] == _T["shot_penalty"]].iloc[0]
    assert pen["time_source"] == "kick"
    assert pen["time_seconds"] == pytest.approx(51.0)
    assert (pen["start_x"], pen["start_y"]) == pytest.approx((94.2, 33.9))
    assert (pen["end_x"], pen["end_y"]) == pytest.approx((94.2, 33.9))


def test_period_end_penalty_gap_runs_to_the_period_last_frame():
    # RTR-IMPL A04(a): no non-derived row follows the penalty in its period -> gap_end is the period's last frame
    # and period_end is True (spec 4.4, 4.17)
    acts = _actions(
        _row(0, 20.0, "pass", "H", oid="a"),
        _row(1, 20.03, "shot_penalty", "H", src="order_only", prov="derived", oid="a", start=(94.0, 34.0),
             end=(94.0, 34.0)),
    )  # fmt: skip
    frames = _ball(_track(0.0, 60.0))
    gaps = restart_gaps(acts, frames)
    assert gaps["period_end"].tolist() == [True]
    assert gaps["gap_start"].tolist() == [pytest.approx(20.0)]
    assert gaps["gap_end"].tolist() == [pytest.approx(float(frames["time_seconds"].max()))]


def test_paired_penalty_rest_inside_b_is_not_a_kick():
    # RTR-IMPL A04(b): the bracket ends at b's reception (RTR-SPEC-12); a rest at the mark inside b is no kick
    acts = _actions(
        _row(0, 20.0, "pass", "H", oid="a"),
        _row(1, 40.0 - 0.02, "shot_penalty", "H", src="placeholder", prov="derived", oid="b", start=(94.0, 34.0),
             end=(94.0, 34.0)),
        _row(2, 42.0, "pass", "A", oid="b"),
    )  # fmt: skip
    frames = _ball(_track(20.0, 40.9) + _rest_release(41.0, 42.0, (94.2, 34.0), (99.0, 34.0)) + _track(42.5, 60.0))
    pen = sync_action_times(acts, frames).pipe(lambda o: o[o["type_id"] == _T["shot_penalty"]].iloc[0])
    assert pen["time_source"] == "placeholder"


def test_no_tracking_keeps_the_unmeasured_source():
    acts, _ = _corner_scene()
    empty = _ball([(0.0, 0.0, 0.0)]).iloc[0:0]
    out = sync_action_times(acts, empty)
    assert out.loc[out["type_id"].map(restart_kind) == "corner", "time_source"].tolist() == ["placeholder"]


def test_elastic_skipped_without_event_rows(monkeypatch):
    acts, frames = _corner_scene()
    called = []
    monkeypatch.setattr(_action_time, "align_events_to_frames", lambda *a, **k: called.append(1))
    sync_action_times(acts, frames)
    assert called == []


def test_elastic_sets_an_aligned_event_row(monkeypatch):
    acts = _actions(_row(0, 10.0, "pass", src="event"), _row(1, 20.0, "pass", src="event"))
    frames = _ball(_track(0.0, 30.0))

    def fake_align(actions, frames, **kw):
        return pd.DataFrame({"action_id": [0], "elastic_frame_id": [123]})  # row 0 aligned to 12.3 s; row 1 not

    monkeypatch.setattr(_action_time, "align_events_to_frames", fake_align)
    out = sync_action_times(acts, frames).sort_values("original_event_id")
    assert out["time_source"].tolist() == ["elastic", "event"]
    assert out["time_seconds"].tolist() == pytest.approx([12.3, 20.0])


def test_dribble_takes_the_midpoint_and_order_is_renumbered(monkeypatch):
    acts = _actions(
        _row(0, 10.0, "pass", src="event"),
        {**_row(1, 15.0, "dribble", src="derived"), "original_event_id": None},
        _row(2, 20.0, "pass", src="event"),
    )
    frames = _ball(_track(0.0, 30.0))
    monkeypatch.setattr(
        _action_time,
        "align_events_to_frames",
        lambda *a, **k: pd.DataFrame({"action_id": [2], "elastic_frame_id": [140]}),  # row 2 moves to 14.0 s
    )
    out = sync_action_times(acts, frames)
    assert out["action_id"].tolist() == [0, 1, 2]
    dribble = out[out["type_id"] == _T["dribble"]].iloc[0]
    assert dribble["time_seconds"] == pytest.approx((10.0 + 14.0) / 2)
    assert out["time_seconds"].is_monotonic_increasing


def test_step_keeps_the_key_of_every_row():
    acts, frames = _corner_scene()
    out = sync_action_times(acts, frames)

    def keys(df):
        return sorted(
            zip(df["original_event_id"], df["action_provenance"], df["type_id"].map(restart_kind), strict=True)
        )

    assert keys(out) == keys(acts)


def test_step_is_pure():
    acts, frames = _corner_scene()
    before = acts.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sync_action_times(acts, frames)
    pd.testing.assert_frame_equal(acts, before)
