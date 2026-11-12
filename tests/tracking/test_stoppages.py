"""The dead-ball state of tracking frames (silly_kicks.tracking._stoppages; ADR-130)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import silly_kicks.spadl.config as c
from silly_kicks.tracking._stoppages import (
    BALL_OUT_MARGIN_M,
    StoppageSpans,
    event_stoppage_spans,
    require_stoppage_state,
    skillcorner_stoppage_spans,
    training_frame_mask,
    with_stoppage_state,
)


def _frames(frame_ids, *, period=1, ball_xy=None, dt=0.1, game=7) -> pd.DataFrame:
    """One ball row and one player row per frame; the ball at the centre unless ``ball_xy`` gives it."""
    rows = []
    for k, f in enumerate(frame_ids):
        bx, by = (ball_xy or {}).get(f, (52.5, 34.0))
        base = {"game_id": game, "period_id": period, "frame_id": f, "time_seconds": k * dt, "ball_state": "alive"}
        rows.append({**base, "is_ball": True, "x": bx, "y": by, "player_id": pd.NA})
        rows.append({**base, "is_ball": False, "x": 40.0, "y": 30.0, "player_id": 9})
    return pd.DataFrame(rows)


def _state(frames: pd.DataFrame, spans) -> dict:
    out = with_stoppage_state(frames, spans=spans)
    one = out[out["is_ball"]]
    return dict(zip(one["frame_id"], one["stoppage_state"], strict=True))


def _pp(rows) -> pd.DataFrame:
    """SkillCorner possessions: (frame_start, frame_end, before marker, after marker)."""
    return pd.DataFrame(
        [
            {"event_type": "player_possession", "frame_start": s, "frame_end": e,
             "game_interruption_before": b, "game_interruption_after": a}
            for s, e, b, a in rows
        ]
    )  # fmt: skip


# ---- SkillCorner markers ------------------------------------------------------------------------------------


def test_a_marker_pair_gives_a_dead_gap_and_an_unknown_taker():
    sp = skillcorner_stoppage_spans(_pp([(10, 20, None, "throw_in_against"), (40, 50, "throw_in_for", None)]))
    assert sp.key == "frame_id"
    assert sp.spans.to_dict("records") == [
        {"start": 20, "end": 40, "state": "dead"},
        {"start": 40, "end": 51, "state": "unknown"},
    ]


@pytest.mark.parametrize("after,before", [("goal_for", None), (None, "corner_for")])
def test_one_marker_alone_gives_unknown_through_the_next_possession(after, before):
    sp = skillcorner_stoppage_spans(_pp([(10, 20, None, after), (40, 50, before, None)]))
    assert sp.spans.to_dict("records") == [{"start": 20, "end": 51, "state": "unknown"}]


def test_no_marker_gives_no_span_and_an_empty_table_gives_none():
    assert skillcorner_stoppage_spans(_pp([(10, 20, None, None), (40, 50, None, None)])).spans.empty
    assert skillcorner_stoppage_spans(_pp([])).spans.empty


def test_possessions_pair_by_frame_order_not_by_row_order():
    sp = skillcorner_stoppage_spans(_pp([(40, 50, "throw_in_for", None), (10, 20, None, "throw_in_against")]))
    assert sp.spans["state"].tolist() == ["dead", "unknown"]


# ---- Gradient Sports events --------------------------------------------------------------------------------


def _acts(rows) -> pd.DataFrame:
    return pd.DataFrame(
        [{"period_id": p, "time_seconds": t, "type_id": c.actiontype_id[ty], "result_id": c.result_id[r]}
         for p, t, ty, r in rows]
    )  # fmt: skip


@pytest.mark.parametrize(
    "restart",
    ["throw_in", "freekick_crossed", "freekick_short", "corner_crossed", "corner_short", "goalkick", "shot_penalty",
     "shot_freekick"],
)  # fmt: skip
def test_each_restart_type_closes_a_dead_span_from_the_previous_action(restart):
    sp = event_stoppage_spans(_acts([(1, 10.0, "pass", "fail"), (1, 20.0, restart, "success")]))
    assert sp.spans.to_dict("records") == [{"period_id": 1, "start": 10.0, "end": 20.0, "state": "dead"}]


@pytest.mark.parametrize("shot,result", [("shot", "success"), ("pass", "owngoal")])
def test_a_goal_opens_a_dead_span_to_the_next_action(shot, result):
    sp = event_stoppage_spans(_acts([(1, 10.0, shot, result), (1, 40.0, "pass", "success")]))
    assert sp.spans.to_dict("records") == [{"period_id": 1, "start": 10.0, "end": 40.0, "state": "dead"}]


def test_a_restart_or_goal_at_a_period_boundary_makes_no_cross_period_span():
    sp = event_stoppage_spans(_acts([(1, 10.0, "shot", "success"), (2, 0.0, "throw_in", "success")]))
    assert sp.spans.empty


# ---- with_stoppage_state -------------------------------------------------------------------------------------


def _sc_spans(rows):
    return skillcorner_stoppage_spans(_pp(rows))


def test_the_event_spans_mark_the_frames():
    fr = _frames(range(10, 61))
    st = _state(
        fr, _sc_spans([(10, 20, None, "throw_in_against"), (40, 50, "throw_in_for", None), (55, 60, None, None)])
    )
    assert st[15] == "live" and st[20] == "dead" and st[39] == "dead" and st[40] == "unknown" and st[50] == "unknown"
    assert st[51] == "live"


def test_frames_before_the_first_and_after_the_last_possession_are_dead():
    fr = _frames(range(0, 80))
    st = _state(fr, _sc_spans([(10, 20, None, None), (30, 60, None, None)]))
    assert {st[f] for f in range(0, 10)} == {"dead"} and st[10] == "live"
    assert st[60] == "live" and {st[f] for f in range(61, 80)} == {"dead"}


def test_period_edges_use_the_frames_period_not_an_event_label():
    # A second-half kick-off possession that the events would label period 1 still sits in the frames of period 2.
    fr = pd.concat([_frames(range(0, 50), period=1), _frames(range(100, 150), period=2)], ignore_index=True)
    st = _state(fr, _sc_spans([(5, 45, None, None), (110, 140, None, None)]))
    assert st[49] == "dead" and st[100] == "dead" and st[110] == "live" and st[145] == "dead"


def test_a_period_without_any_possession_is_unknown():
    fr = pd.concat([_frames(range(0, 50), period=1), _frames(range(100, 150), period=2)], ignore_index=True)
    st = _state(fr, _sc_spans([(5, 45, None, None)]))
    assert {st[f] for f in range(100, 150)} == {"unknown"}


@pytest.mark.parametrize(
    "xy",
    [(-0.12, 30.0), (105.12, 30.0), (50.0, -0.12), (50.0, 68.12)],
    ids=["left_goal_line", "right_goal_line", "near_touch_line", "far_touch_line"],
)
def test_a_ball_out_frame_is_dead_and_so_is_every_frame_to_the_next_possession(xy):
    fr = _frames(range(0, 100), ball_xy={30: xy, 31: xy})
    st = _state(fr, _sc_spans([(0, 35, None, None), (60, 70, None, None), (80, 99, None, None)]))
    assert st[29] == "live"
    assert {st[f] for f in range(30, 60)} == {"dead"}  # from the ball-out frame up to the next possession
    assert {st[f] for f in range(60, 71)} == {"unknown"}  # the next possession: the restart lies inside it
    assert st[71] == "live"


@pytest.mark.parametrize("xy", [(-0.10, 30.0), (105.10, 30.0), (50.0, -0.10), (50.0, 68.10)])
def test_a_ball_on_the_line_within_the_ball_radius_is_not_out(xy):
    fr = _frames(range(0, 100), ball_xy={30: xy})
    st = _state(fr, _sc_spans([(0, 35, None, None), (60, 70, None, None), (80, 99, None, None)]))
    assert st[30] == "live" and BALL_OUT_MARGIN_M == 0.11


def test_a_frame_without_a_ball_position_takes_no_state_from_the_ball_out_rule():
    fr = _frames(range(0, 100), ball_xy={30: (np.nan, np.nan)})
    st = _state(fr, _sc_spans([(0, 99, None, None)]))
    assert st[30] == "live"


def test_ball_out_on_gradient_sports_frames_runs_to_the_next_action_by_time():
    fr = _frames(range(0, 100), ball_xy={20: (106.0, 30.0)}, dt=0.1)  # frame k at k * 0.1 s
    sp = event_stoppage_spans(
        _acts(
            [
                (1, 0.0, "pass", "success"),
                (1, 5.0, "throw_in", "success"),
                (1, 7.0, "pass", "success"),
                (1, 9.9, "pass", "success"),
            ]
        )
    )
    st = _state(fr, sp)
    assert {st[f] for f in range(0, 50)} == {"dead"}  # the event span and the ball-out run to the restart
    assert {st[f] for f in range(50, 70)} == {"unknown"}  # the restart's own span, up to the next action
    assert st[70] == "live"


def test_unknown_wins_over_dead_and_dead_over_live():
    fr = _frames(range(0, 100))
    sp = StoppageSpans(
        "frame_id",
        pd.DataFrame({"start": [10, 15], "end": [30, 20], "state": ["dead", "unknown"]}),
        pd.DataFrame({"start": [0], "end": [100]}),
    )
    st = _state(fr, sp)
    assert st[12] == "dead" and st[16] == "unknown" and st[25] == "dead" and st[40] == "live"
    # The other order: a later, lower state never lowers an earlier, higher one.
    sp2 = StoppageSpans(
        "frame_id",
        pd.DataFrame({"start": [10, 15], "end": [20, 30], "state": ["unknown", "dead"]}),
        pd.DataFrame({"start": [0], "end": [100]}),
    )
    st2 = _state(fr, sp2)
    assert st2[16] == "unknown" and st2[25] == "dead"


def test_a_ball_out_run_inside_an_unknown_span_leaves_it_unknown():
    fr = _frames(range(0, 100), ball_xy={45: (110.0, 30.0)})
    st = _state(fr, _sc_spans([(0, 20, None, "throw_in_for"), (40, 50, None, None), (80, 99, None, None)]))
    assert st[45] == "unknown"  # the one-marker span [20, 51) is unknown; the ball-out rule must not lower it


def test_an_observed_provider_takes_ball_state():
    fr = _frames(range(0, 4))
    fr.loc[fr["frame_id"] == 2, "ball_state"] = "dead"
    out = with_stoppage_state(fr, spans=None)
    assert out.groupby("frame_id")["stoppage_state"].first().tolist() == ["live", "live", "dead", "live"]
    assert set(out["stoppage_source"]) == {"ball_state"}


def test_the_caller_index_is_kept():
    fr = _frames(range(0, 20)).set_axis(range(1000, 1040))
    out = with_stoppage_state(fr, spans=_sc_spans([(0, 19, None, None)]))
    assert out.index.equals(fr.index)


def test_the_input_frames_are_not_changed_and_ball_state_is_never_written():
    fr = _frames(range(0, 100), ball_xy={30: (110.0, 30.0)})
    before = fr.copy(deep=True)
    out = with_stoppage_state(fr, spans=_sc_spans([(0, 35, None, None), (60, 99, None, None)]))
    pd.testing.assert_frame_equal(fr, before)
    assert out is not fr and set(out["ball_state"]) == {"alive"} and len(out) == len(fr)
    assert set(out["stoppage_source"]) <= {"events", "ball_out", "period_edge", "no_coverage"}


# ---- training_frame_mask ---------------------------------------------------------------------------------------


def test_the_training_mask_drops_dead_unknown_and_frames_whose_horizon_reaches_unknown():
    fr = pd.DataFrame(
        {
            "game_id": 1,
            "period_id": 1,
            "time_seconds": [0.0, 1.0, 1.5, 2.0, 4.0, 5.5, 6.0],
            "stoppage_state": ["live", "live", "dead", "live", "live", "unknown", "live"],
        }
    )
    # 0.0: horizon [0, 1] holds a dead frame only -> kept. 4.0: [4, 5] clear -> kept. 5.5 unknown, 1.5 dead -> dropped.
    # 4.6 would reach 5.5; 6.0 is past it -> kept.
    assert training_frame_mask(fr, horizon_seconds=1.0).tolist() == [True, True, False, True, True, False, True]
    fr2 = fr.assign(time_seconds=[0.0, 1.0, 1.5, 2.0, 4.6, 5.5, 6.0])
    assert training_frame_mask(fr2, horizon_seconds=1.0).tolist()[4] is False


def test_the_training_mask_does_not_look_across_a_period():
    fr = pd.DataFrame(
        {"game_id": 1, "period_id": [1, 2], "time_seconds": [0.0, 0.5], "stoppage_state": ["live", "unknown"]}
    )
    assert training_frame_mask(fr, horizon_seconds=1.0).tolist() == [True, False]


# ---- implementation review A ------------------------------------------------------------------------------------


def test_a_ball_out_run_across_a_possession_runs_on_to_the_next_possession():
    # SGR-IMPL-01: possessions at 0-10, 30-40, 80-90, no markers; the ball out at 20-50, not tracked at 51-79.
    xy = {f: (110.0, 30.0) for f in range(20, 51)} | {f: (np.nan, np.nan) for f in range(51, 80)}
    st = _state(
        _frames(range(0, 101), ball_xy=xy), _sc_spans([(0, 10, None, None), (30, 40, None, None), (80, 90, None, None)])
    )
    assert st[19] == "live" and {st[f] for f in range(20, 30)} == {"dead"}
    assert {st[f] for f in range(30, 41)} == {"unknown"}  # the possession inside the run
    assert {st[f] for f in range(41, 80)} == {"dead"}  # the run binds again: dead up to the next possession
    assert {st[f] for f in range(80, 91)} == {"unknown"} and {st[f] for f in range(91, 101)} == {"dead"}


def test_a_marked_possession_without_frame_ids_gives_unknown_between_its_neighbours():
    # SGR-IMPL-02 (rule 3): the middle possession carries a marker and no frame ids.
    ev = _pp([(0, 10, None, None), (None, None, None, "throw_in_for"), (30, 40, None, None)])
    st = _state(_frames(range(0, 41)), skillcorner_stoppage_spans(ev))
    assert st[9] == "live" and {st[f] for f in range(10, 41)} == {"unknown"}


def test_spans_refuse_the_frames_of_two_matches():
    fr = pd.concat([_frames(range(0, 10), game=1), _frames(range(0, 10), game=2)], ignore_index=True)
    with pytest.raises(ValueError, match="ONE match"):
        with_stoppage_state(fr, spans=_sc_spans([(0, 9, None, None)]))


def test_a_staged_match_passes_without_a_provider_lookup():
    # A match that carries stoppage_state needs no provider classification (the toy probe provider "synthetic").
    require_stoppage_state(pd.DataFrame({"stoppage_state": ["live"]}), "synthetic")


def test_unstaged_frames_of_an_unclassified_provider_still_fail_closed():
    with pytest.raises(ValueError, match="unclassified provider 'synthetic'"):
        require_stoppage_state(pd.DataFrame({"ball_state": ["alive"]}), "synthetic")
