"""Structural performance guard for compute_team_shape (TF-31).

Replaces a flaky wall-clock budget (the 5ms ceiling once measured 6.2ms on a shared Windows
runner and failed CI) with a deterministic call-count invariant: the per-frame Ward line
decomposition runs ONE ``scipy...linkage`` clustering, independent of player count. A
regression to per-player / repeated clustering is the real cost blow-up. See
tests/_perf_structural.py.
"""

import numpy as np
import pandas as pd
import pytest

from tests._perf_structural import call_counter


@pytest.fixture
def team_shape_frame():
    rows = []
    rng = np.random.default_rng(42)
    for i in range(10):
        rows.append(
            dict(
                game_id=1,
                period_id=1,
                frame_id=1,
                time_seconds=1.0,
                frame_rate=25.0,
                player_id=i + 10,
                team_id=1,
                is_ball=False,
                is_goalkeeper=False,
                x=float(rng.uniform(10, 90)),
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
    return pd.DataFrame(rows)


def test_team_shape_runs_one_ward_clustering(team_shape_frame, monkeypatch):
    from silly_kicks.tracking import _team_shape

    # _team_shape imports `linkage` at module scope (`from scipy... import linkage`), so patch
    # the _team_shape module attribute (the name the function resolves), not scipy.
    calls = call_counter(monkeypatch, _team_shape, "linkage")

    result = _team_shape.compute_team_shape(team_shape_frame, team_id=1)

    assert result is not None
    assert calls["n"] == 1, (
        f"compute_team_shape ran {calls['n']} Ward clusterings for one frame (expected 1). "
        "Per-player or repeated clustering is the O(n) regression the wall-clock budget proxied."
    )
