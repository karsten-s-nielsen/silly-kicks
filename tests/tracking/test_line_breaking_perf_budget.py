"""Structural performance guard for detect_line_breaking (TF-32).

Replaces a flaky wall-clock budget with a deterministic call-count invariant: the defensive
line is segmented with ONE ``scipy...linkage`` Ward clustering per frame, independent of
action/segment count. A regression to per-action / per-segment clustering is the real cost
blow-up. See tests/_perf_structural.py.
"""

import numpy as np
import pandas as pd
import pytest

from tests._perf_structural import call_counter


@pytest.fixture
def line_breaking_fixture():
    rows = []
    rows.append(
        dict(
            game_id=1,
            period_id=1,
            frame_id=1,
            time_seconds=1.0,
            frame_rate=25.0,
            player_id=0,
            team_id=np.nan,
            is_ball=True,
            is_goalkeeper=False,
            x=50.0,
            y=34.0,
            source_provider="synthetic",
            team_attacking_direction="ltr",
        )
    )
    rng = np.random.default_rng(42)
    for i in range(10):
        rows.append(
            dict(
                game_id=1,
                period_id=1,
                frame_id=1,
                time_seconds=1.0,
                frame_rate=25.0,
                player_id=50 + i,
                team_id=2,
                is_ball=False,
                is_goalkeeper=False,
                x=float(rng.uniform(40, 80)),
                y=float(rng.uniform(5, 63)),
                source_provider="synthetic",
                team_attacking_direction="ltr",
            )
        )
    rows.append(
        dict(
            game_id=1,
            period_id=1,
            frame_id=1,
            time_seconds=1.0,
            frame_rate=25.0,
            player_id=10,
            team_id=1,
            is_ball=False,
            is_goalkeeper=False,
            x=40.0,
            y=34.0,
            source_provider="synthetic",
            team_attacking_direction="ltr",
        )
    )
    frames = pd.DataFrame(rows)
    actions = pd.DataFrame(
        {
            "action_id": [0],
            "game_id": [1],
            "period_id": [1],
            "time_seconds": [1.0],
            "team_id": [1],
            "type_id": [0],
            "result_id": [1],
            "start_x": [40.0],
            "start_y": [34.0],
            "end_x": [70.0],
            "end_y": [34.0],
            "bodypart_id": [0],
            "player_id": [10],
        }
    )
    return actions, frames


def test_line_breaking_runs_one_ward_clustering(line_breaking_fixture, monkeypatch):
    from silly_kicks.tracking import _line_breaking

    actions, frames = line_breaking_fixture
    calls = call_counter(monkeypatch, _line_breaking, "linkage")

    result = _line_breaking.detect_line_breaking(actions, frames)

    assert result is not None
    assert calls["n"] == 1, (
        f"detect_line_breaking ran {calls['n']} Ward clusterings for one frame (expected 1). "
        "Per-action or per-segment clustering is the regression the wall-clock budget proxied."
    )
