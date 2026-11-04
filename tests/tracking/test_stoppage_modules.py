"""GKDV, rest defence and off-ball runs drop the dead AND the unknown spans of the stoppage column.

TF-46 Part 2 spec section 27.19 (owner decision 2026-10-26, R-C2 fixtures item 7): a full-rate fixture on a provider
without an observed dead-ball state (SkillCorner), with the column built by ``derive_stoppage_state`` from actions that
give one dead span (a goal: the goal shot to the next action) and one unknown span (a restart without a resolved kick).
The dead span came from a resolved kick until 2026-10-27; the column now refuses actions that carry ``t_kick`` (owner
decision 2026-10-27), so a goal gives it, with the same bounds. Mutant for each module: it ignores the column (reads
the constant ``ball_state``), so neither span drops and the test fails.
"""

from __future__ import annotations

import pandas as pd

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.tracking import derive_stoppage_state, infer_ball_carrier, resolve_defended_goals
from tests.gkdv._fixtures import in_domain_frames

_DT = 0.04  # 25 Hz
_N = 80  # t = 1.00 .. 4.16 s


def _actions() -> pd.DataFrame:
    """goal shot 1.2 -> pass 1.6: dead (1.2, 1.6); pass 2.0 -> corner (derived, no kick): unknown (2.0, 2.6)."""
    rows = [
        (1, 1.2, "shot", "native"),
        (2, 1.6, "pass", "native"),
        (3, 2.0, "pass", "native"),
        (4, 2.6, "corner_crossed", "derived"),
    ]
    return pd.DataFrame(
        {
            "game_id": "100",
            "period_id": 1,
            "action_id": [r[0] for r in rows],
            "time_seconds": [r[1] for r in rows],
            "team_id": 2,
            "player_id": "a10",
            "type_id": [spadlconfig.actiontype_id[r[2]] for r in rows],
            "result_id": spadlconfig.result_id["success"],
            "bodypart_id": spadlconfig.bodypart_id["foot"],
            "start_x": 20.0,
            "start_y": 34.0,
            "end_x": 10.0,
            "end_y": 34.0,
            "action_provenance": [r[3] for r in rows],
            "gi_interruption_after": pd.NA,  # stoppage rule version 3 (fix (A)): no penalty marker
        }
    )


def _frames() -> pd.DataFrame:
    sc = "skillcorner"
    parts = [in_domain_frames(frame_id=i, timestamp=round(1.0 + i * _DT, 3), source_provider=sc) for i in range(_N)]
    frames = pd.concat(parts, ignore_index=True).drop(columns=["stoppage_state", "stoppage_source"], errors="ignore")
    frames["frame_rate"] = 1.0 / _DT
    # skillcorner is detection-aware (ADR-069/109): every row carries the detection flag. All detected, so the
    # TF-64 gkdv detection-gate is a no-op (gk_det == gk_ok) and this test isolates the stoppage-span drop.
    frames["visibility"] = True
    return derive_stoppage_state(frames, actions=_actions())


def _span_frames(frames: pd.DataFrame) -> tuple[set, set]:
    t = frames.drop_duplicates("frame_id").set_index("frame_id")["time_seconds"]
    return set(t[(t > 1.2) & (t < 1.6)].index), set(t[(t > 2.0) & (t < 2.6)].index)


def test_the_fixture_has_a_dead_and_an_unknown_span():
    # Precondition (ADR-056): both spans exist in the column and are non-empty.
    frames = _frames()
    dead, unknown = _span_frames(frames)
    st = frames.drop_duplicates("frame_id").set_index("frame_id")["stoppage_state"].astype(object)
    assert dead and unknown
    assert set(st[list(dead)]) == {"dead"} and set(st[list(unknown)]) == {"unknown"}
    assert (st.drop(list(dead | unknown)) == "live").all()


def _dropped_not_alive(dropped: pd.DataFrame) -> set:
    return set(dropped.loc[dropped["drop_reason"] == "ball_not_alive", "frame_id"])


def test_gkdv_drops_the_dead_and_the_unknown_span():
    from silly_kicks.gkdv import GkdvParams
    from silly_kicks.gkdv._engine import _apply_domain

    frames = _frames()
    _eligible, dropped = _apply_domain(
        frames, carrier=infer_ball_carrier(frames), goal_map=resolve_defended_goals(frames), params=GkdvParams()
    )
    dead, unknown = _span_frames(frames)
    assert _dropped_not_alive(dropped) == dead | unknown


def test_rest_defence_drops_the_dead_and_the_unknown_span():
    from silly_kicks.restdefense import RestDefenseParams
    from silly_kicks.restdefense._counterfactual import _apply_domain

    frames = _frames()
    carrier = infer_ball_carrier(frames)
    _eligible, dropped = _apply_domain(
        frames, carrier=carrier, goal_map=resolve_defended_goals(frames), params=RestDefenseParams(), which="keeper"
    )
    dead, unknown = _span_frames(frames)
    assert _dropped_not_alive(dropped) == dead | unknown


def test_off_ball_runs_leave_the_actions_in_the_dead_and_the_unknown_span_nan():
    from silly_kicks.tracking._off_ball_runs import _off_ball_runs_kernel

    frames = _frames()
    query = _actions().iloc[[2, 2, 2]].reset_index(drop=True)  # the pass row
    query["action_id"] = [11, 12, 13]
    query["time_seconds"] = [1.4, 2.3, 3.6]  # inside the dead span, inside the unknown span, live
    out = _off_ball_runs_kernel(query, frames)
    runners = out["n_off_ball_runners_pre_window"]
    assert runners.iloc[0] is pd.NA and runners.iloc[1] is pd.NA
    assert runners.iloc[2] is not pd.NA  # non-vacuity: the live action is scored
