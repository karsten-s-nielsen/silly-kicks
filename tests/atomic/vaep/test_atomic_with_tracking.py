"""AtomicVAEP integration smoke test with tracking-aware features.

Loop 11 covers: AtomicVAEP(xfns=atomic_xfns_default + atomic_tracking_default_xfns)
.compute_features(...) reaches the tracking branch without errors. No AUC uplift
assertion (atomic synthetic noise floor is too high for a stable assertion).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.mark.slow
def test_atomic_vaep_with_tracking_compute_features():
    from silly_kicks.atomic.tracking.features import atomic_tracking_default_xfns
    from silly_kicks.atomic.vaep.base import AtomicVAEP, xfns_default

    rng = np.random.default_rng(0)
    n = 50
    actions = pd.DataFrame(
        {
            "game_id": [1] * n,
            "original_event_id": [None] * n,
            "action_id": list(range(1, n + 1)),
            "period_id": [1] * n,
            "time_seconds": [t * 0.5 for t in range(n)],
            "team_id": rng.choice([1, 2], size=n),
            "player_id": rng.choice([11, 12, 21, 22], size=n),
            "x": rng.uniform(0, 105, size=n),
            "y": rng.uniform(0, 68, size=n),
            "dx": rng.uniform(-10, 10, size=n),
            "dy": rng.uniform(-5, 5, size=n),
            "type_id": [0] * n,
            "bodypart_id": [0] * n,
        }
    )
    rows = []
    for _, a in actions.iterrows():
        fid = int(a["time_seconds"] * 10)
        rows.append(
            dict(
                game_id=1,
                period_id=1,
                frame_id=fid,
                time_seconds=a["time_seconds"],
                frame_rate=10.0,
                player_id=a["player_id"],
                team_id=a["team_id"],
                is_ball=False,
                is_goalkeeper=False,
                x=a["x"],
                y=a["y"],
                z=float("nan"),
                speed=2.0,
                speed_source="native",
                ball_state="alive",
                team_attacking_direction="ltr",
                confidence=None,
                visibility=None,
                source_provider="synth",
            )
        )
        rows.append(
            dict(
                game_id=1,
                period_id=1,
                frame_id=fid,
                time_seconds=a["time_seconds"],
                frame_rate=10.0,
                player_id=99,
                team_id=2 if a["team_id"] == 1 else 1,
                is_ball=False,
                is_goalkeeper=False,
                x=a["x"] + 5.0,
                y=a["y"],
                z=float("nan"),
                speed=1.5,
                speed_source="native",
                ball_state="alive",
                team_attacking_direction="ltr",
                confidence=None,
                visibility=None,
                source_provider="synth",
            )
        )
    frames = pd.DataFrame(rows)
    game = pd.Series({"game_id": 1, "home_team_id": 1, "away_team_id": 2})

    v = AtomicVAEP(xfns=list(xfns_default) + atomic_tracking_default_xfns)
    X = v.compute_features(game, actions, frames=frames)
    assert any("nearest_defender_distance_a0" in c for c in X.columns)
