# tests/tracking/test_player_influence_perf_budget.py
"""Structural performance guard for compute_player_influence (TF-36 + TF-33).

Replaces a flaky wall-clock budget with a deterministic call-count invariant: the function
must build the per-frame pitch-control surface ONCE and reuse it across all 20 outfield
players (the ADR-008 cache contract its module docstring states). A regression to per-player
surface construction makes this O(players) — the real cost blow-up the old ms-budget only
caught indirectly. See tests/_perf_structural.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests._perf_structural import call_counter


def _make_22_player_frame():
    """Standard 22-player frame for benchmarking."""
    rng = np.random.default_rng(42)
    rows: list[dict] = []
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
            vx=5.0,
            vy=0.0,
            source_provider="synthetic",
            team_attacking_direction="ltr",
        )
    )
    for gk_pid, gk_tid, gk_x in [(1, 1, 3.0), (50, 2, 102.0)]:
        rows.append(
            dict(
                game_id=1,
                period_id=1,
                frame_id=1,
                time_seconds=1.0,
                frame_rate=25.0,
                player_id=gk_pid,
                team_id=gk_tid,
                is_ball=False,
                is_goalkeeper=True,
                x=gk_x,
                y=34.0,
                vx=0.0,
                vy=0.0,
                source_provider="synthetic",
                team_attacking_direction="ltr",
            )
        )
    for i in range(10):
        rows.append(
            dict(
                game_id=1,
                period_id=1,
                frame_id=1,
                time_seconds=1.0,
                frame_rate=25.0,
                player_id=10 + i,
                team_id=1,
                is_ball=False,
                is_goalkeeper=False,
                x=float(rng.uniform(10, 60)),
                y=float(rng.uniform(5, 63)),
                vx=float(rng.uniform(-3, 3)),
                vy=float(rng.uniform(-3, 3)),
                source_provider="synthetic",
                team_attacking_direction="ltr",
            )
        )
    for i in range(10):
        rows.append(
            dict(
                game_id=1,
                period_id=1,
                frame_id=1,
                time_seconds=1.0,
                frame_rate=25.0,
                player_id=60 + i,
                team_id=2,
                is_ball=False,
                is_goalkeeper=False,
                x=float(rng.uniform(45, 95)),
                y=float(rng.uniform(5, 63)),
                vx=float(rng.uniform(-3, 3)),
                vy=float(rng.uniform(-3, 3)),
                source_provider="synthetic",
                team_attacking_direction="ltr",
            )
        )
    return pd.DataFrame(rows)


@pytest.fixture
def fixture_22():
    from silly_kicks.xthreat import ExpectedThreat

    frame = _make_22_player_frame()
    xt = ExpectedThreat(l=16, w=12)
    xt.xT = np.tile(np.linspace(0.0, 1.0, 16), (12, 1))
    return frame, xt


def test_player_influence_builds_one_pitch_control_surface(fixture_22, monkeypatch):
    """compute_player_influence builds exactly ONE pitch-control surface for all 20 players."""
    from silly_kicks.tracking import _player_influence
    from silly_kicks.tracking.pitch_control import _cache

    frame, xt = fixture_22
    # Patch the cache's primitive (the symbol cache.surface() resolves), not the dispatch site.
    calls = call_counter(monkeypatch, _cache, "compute_pitch_control")

    result = _player_influence.compute_player_influence(frame, xt, attacking_team_id=1, home_team_id=1)

    assert len(result) == 20  # 20 outfield players
    assert calls["n"] == 1, (
        f"compute_player_influence built {calls['n']} pitch-control surfaces for 20 players "
        "(expected 1 — the per-frame cache invariant). A per-player regression makes this O(players)."
    )
