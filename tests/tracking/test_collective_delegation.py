"""Delegation parity + goal-lookup scaling for the vectorised collective kernel (TF-58 Task 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.tracking import add_team_shape, compute_defensive_line, resolve_defended_goals
from silly_kicks.tracking._gk_resolve import GoalMap
from tests._perf_structural import call_counter
from tests.tracking._legacy_collective_oracle import (
    legacy_compute_defensive_line,
    legacy_compute_team_shape,
)
from tests.tracking.test_action_ltr_mirror_invariance import _scenario


def test_add_team_shape_only_hull_area_moves(monkeypatch):
    actions, frames = _scenario()
    new = add_team_shape(actions, frames)
    # features.add_team_shape does a local `from ._team_shape import compute_team_shape`, so patch the source module.
    monkeypatch.setattr("silly_kicks.tracking._team_shape.compute_team_shape", legacy_compute_team_shape)
    old = add_team_shape(actions, frames)
    hull_cols = [c for c in new.columns if "convex_hull_area" in c]
    assert hull_cols, "no team_shape_convex_hull_area_* column found"
    other = [c for c in new.columns if c not in hull_cols]
    pd.testing.assert_frame_equal(new[other], old[other], check_exact=True, check_dtype=True)
    for c in hull_cols:
        np.testing.assert_allclose(
            new[c].to_numpy(dtype="float64"), old[c].to_numpy(dtype="float64"), rtol=1e-9, atol=0, equal_nan=True
        )


def test_restdefense_output_unchanged(monkeypatch):
    from silly_kicks.restdefense._compute import compute_rest_defense
    from tests.restdefense._fixtures import make_rest_defense_fixture

    actions, frames = make_rest_defense_fixture()
    new, new_report = compute_rest_defense(actions, frames)
    monkeypatch.setattr("silly_kicks.restdefense._compute.compute_team_shape", legacy_compute_team_shape)
    monkeypatch.setattr("silly_kicks.restdefense._compute.compute_defensive_line", legacy_compute_defensive_line)
    old, old_report = compute_rest_defense(actions, frames)
    pd.testing.assert_frame_equal(new, old, check_exact=True, check_dtype=True)
    assert new_report == old_report


def _line_frames(n_frames: int) -> pd.DataFrame:
    # 2 teams, 10 outfielders + 1 GK each, over n_frames frames. Team 1 defends x=0, team 2 defends x=105.
    rng = np.random.default_rng(0)
    rows = []
    for frame_id in range(n_frames):
        for team_id, (gk_x, base) in ((1, (4.0, 20.0)), (2, (101.0, 85.0))):
            rows.append((frame_id, team_id, True, gk_x, 34.0, "ltr" if team_id == 1 else "rtl"))
            for _ in range(10):
                rows.append(
                    (
                        frame_id,
                        team_id,
                        False,
                        base + rng.uniform(-5, 5),
                        rng.uniform(0, 68),
                        "ltr" if team_id == 1 else "rtl",
                    )
                )
    df = pd.DataFrame(rows, columns=["frame_id", "team_id", "is_goalkeeper", "x", "y", "team_attacking_direction"])
    df.insert(0, "period_id", 1)
    df.insert(0, "game_id", 1)
    df["is_ball"] = False
    df["player_id"] = np.arange(len(df))
    return df


def test_defensive_line_goal_lookups_scale_with_teams_not_frames(monkeypatch):
    counter = call_counter(monkeypatch, GoalMap, "get")
    seen = {}
    for n in (10, 100):
        frames = _line_frames(n)
        gm = resolve_defended_goals(frames)
        counter["n"] = 0  # reset after building the map (resolve does not call .get)
        compute_defensive_line(frames, goal_map=gm, n=4)
        seen[n] = counter["n"]
    # one lookup per (game, period, team) group with >= 3 players -- 2 teams -- regardless of frame count
    assert seen[10] == seen[100] == 2
