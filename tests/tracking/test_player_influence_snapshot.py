# tests/tracking/test_player_influence_snapshot.py
"""Snapshot hash test for compute_player_influence.

Multi-hash set pattern per feedback_multi_hash_snapshot_sets.md.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def _make_deterministic_frame():
    """Fixed-seed frame for snapshot reproducibility."""
    rng = np.random.default_rng(123)
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
    # Asymmetric 3v2 to exercise team-decomposition asymmetry
    for i in range(5):
        tid = 1 if i < 3 else 2
        rows.append(
            dict(
                game_id=1,
                period_id=1,
                frame_id=1,
                time_seconds=1.0,
                frame_rate=25.0,
                player_id=10 + i,
                team_id=tid,
                is_ball=False,
                is_goalkeeper=False,
                x=float(rng.uniform(10, 90)),
                y=float(rng.uniform(5, 63)),
                vx=float(rng.uniform(-2, 2)),
                vy=float(rng.uniform(-2, 2)),
                source_provider="synthetic",
                team_attacking_direction="ltr",
            )
        )
    return pd.DataFrame(rows)


def test_compute_player_influence_snapshot():
    from silly_kicks.tracking._player_influence import compute_player_influence
    from silly_kicks.xthreat import ExpectedThreat

    frame = _make_deterministic_frame()
    xt = ExpectedThreat(l=16, w=12)
    xt.xT = np.tile(np.linspace(0.0, 1.0, 16), (12, 1))

    result = compute_player_influence(
        frame,
        xt,
        attacking_team_id=1,
        attacks_rtl=False,
    )

    # Build a deterministic string representation
    parts = []
    for pid in sorted(result.keys(), key=str):
        pi = result[pid]
        parts.append(f"{pid}:{pi.off_ball_xt:.8f}:{pi.reachable_area_m2:.8f}")
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]

    # Multi-hash set for numpy runner drift
    # Update this set if numpy/scipy micro-versions cause ULP drift.
    valid_hashes: set[str] = {"dab140505e42a94a"}

    assert digest in valid_hashes, (
        f"Snapshot hash {digest!r} not in valid set {valid_hashes}. If numpy/scipy changed, add the new hash."
    )
