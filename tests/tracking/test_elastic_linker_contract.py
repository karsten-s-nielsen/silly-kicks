"""Contract tests for link_actions_to_frames_elastic (the ELASTIC-NW alternative linker).

Pins the ADR-004 pointer schema + the deliberately-divergent semantics documented in the design
(spec section 6 / TF57-SPEC-01, TF57-SPEC-08): per-action ``n_candidate_frames`` (never a
per-episode constant), ``link_quality_score`` = NW confidence, ``tolerance_seconds`` = NaN, and the
confidence-gated low-coverage warning WITHOUT the time-base-mismatch hint.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from silly_kicks.tracking import link_actions_to_frames_elastic


def _alignable(extra_unlinkable: bool = False):
    """A controlled continuous-tracking scene the NW aligns: four team-1 players at fixed x, ball
    resting at each in turn and hopping between them. Actions = passes at the kick frames (one
    possession -> receptions between). ``extra_unlinkable`` appends a pass by a player absent from
    the frames (never links) to force low coverage."""
    xs = {"p1_0": 20.0, "p1_1": 45.0, "p1_2": 70.0, "p1_3": 90.0}

    def ball_x(f: int) -> float:
        if f <= 10:
            return 20.0
        if f <= 20:
            return 20.0 + 2.5 * (f - 10)
        if f <= 30:
            return 45.0
        if f <= 40:
            return 45.0 + 2.5 * (f - 30)
        if f <= 50:
            return 70.0
        if f <= 60:
            return 70.0 + 2.0 * (f - 50)
        return 90.0

    rows = []
    for f in range(71):
        t = f / 25.0
        rows.append(
            {
                "game_id": 1,
                "period_id": 1,
                "frame_id": f,
                "time_seconds": t,
                "player_id": None,
                "team_id": None,
                "x": ball_x(f),
                "y": 34.0,
                "z": float("nan"),
                "is_ball": True,
            }
        )
        for p, x in xs.items():
            rows.append(
                {
                    "game_id": 1,
                    "period_id": 1,
                    "frame_id": f,
                    "time_seconds": t,
                    "player_id": p,
                    "team_id": 1,
                    "x": x,
                    "y": 34.0,
                    "z": float("nan"),
                    "is_ball": False,
                }
            )
    frames = pd.DataFrame(rows)
    adf = {
        "action_id": [0, 1, 2],
        "game_id": [1, 1, 1],
        "period_id": [1, 1, 1],
        "time_seconds": [0.4, 1.2, 2.0],
        "player_id": ["p1_0", "p1_1", "p1_2"],
        "team_id": [1, 1, 1],
        "type_id": [0, 0, 0],
        "type_name": ["pass"] * 3,
        "result_name": ["success"] * 3,
    }
    if extra_unlinkable:
        adf["action_id"].append(3)
        adf["game_id"].append(1)
        adf["period_id"].append(1)
        adf["time_seconds"].append(2.4)
        adf["player_id"].append("p1_absent")  # never appears in frames -> cannot link
        adf["team_id"].append(1)
        adf["type_id"].append(0)
        adf["type_name"].append("pass")
        adf["result_name"].append("success")
    return frames, pd.DataFrame(adf)


def test_pointer_schema_and_dtypes():
    frames, actions = _alignable()
    ptr, report = link_actions_to_frames_elastic(actions, frames)
    assert list(ptr.columns) == [
        "action_id",
        "frame_id",
        "time_offset_seconds",
        "n_candidate_frames",
        "link_quality_score",
    ]
    assert str(ptr["action_id"].dtype) == "int64"
    assert str(ptr["frame_id"].dtype) == "Int64"
    assert str(ptr["time_offset_seconds"].dtype) == "float64"
    assert str(ptr["n_candidate_frames"].dtype) == "int64"
    assert str(ptr["link_quality_score"].dtype) == "float64"
    assert np.isnan(report.tolerance_seconds)
    assert len(ptr) == len(actions)


def test_frame_ids_are_real_frames():
    frames, actions = _alignable()
    ptr, _ = link_actions_to_frames_elastic(actions, frames)
    real = set(frames["frame_id"].tolist())
    assert set(ptr["frame_id"].dropna().astype(int)).issubset(real)


def test_n_candidate_frames_is_per_action_not_constant():
    # Both-sided guard for TF57-SPEC-01: a per-EPISODE-constant implementation (all actions in one
    # possession share a value) would make this uniform; the per-action membership count varies.
    frames, actions = _alignable()
    ptr, _ = link_actions_to_frames_elastic(actions, frames)
    assert ptr["n_candidate_frames"].nunique() > 1


def test_link_quality_score_is_confidence():
    frames, actions = _alignable()
    ptr, _ = link_actions_to_frames_elastic(actions, frames)
    linked = ptr[ptr["frame_id"].notna()]
    assert len(linked) > 0
    assert (linked["link_quality_score"] >= 0.0).all()
    assert (linked["link_quality_score"] <= 1.0).all()


def test_links_are_usable_as_add_star_pointers():
    # the whole point: the pointer frame is a drop-in for the links= kwarg (schema compatible)
    frames, actions = _alignable()
    ptr, report = link_actions_to_frames_elastic(actions, frames)
    assert report.link_rate > 0.0
    assert report.n_actions_in == len(actions)
    # the pointer frame is schema-compatible with the links= kwarg (drop-in for add_* aggregators)
    assert set(ptr.columns) == {
        "action_id",
        "frame_id",
        "time_offset_seconds",
        "n_candidate_frames",
        "link_quality_score",
    }


def test_low_coverage_fires_floor_without_time_base_hint():
    frames, actions = _alignable(extra_unlinkable=True)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        link_actions_to_frames_elastic(actions, frames, min_link_rate=0.99, on_low_coverage="warn")
    msgs = " ".join(str(x.message) for x in w)
    assert "below min_link_rate" in msgs
    assert "time-base mismatch" not in msgs


def test_empty_actions_returns_typed_empty():
    frames, _ = _alignable()
    empty = pd.DataFrame(
        columns=["action_id", "game_id", "period_id", "time_seconds", "player_id", "team_id", "type_id"]
    )
    ptr, report = link_actions_to_frames_elastic(empty, frames)
    assert len(ptr) == 0
    assert report.n_actions_in == 0
    assert np.isnan(report.tolerance_seconds)
