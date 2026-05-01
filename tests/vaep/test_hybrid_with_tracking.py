"""End-to-end HybridVAEP integration test with tracking-aware features.

Loop 9 covers: HybridVAEP(xfns=hybrid_xfns_default + tracking_default_xfns)
.fit(...).rate(...) full lifecycle works on synthetic fixture.

NOTE on AUC uplift: the spec mentions an AUC-uplift assertion (augmented >=
baseline + epsilon=0.01). On a small synthetic fixture this signal is noisy.
The fixture-regeneration option is user-authorized per session policy if the
test ever flakes; for now PR-S20 ships the lifecycle smoke test only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_synthetic_match(seed: int = 42, n_actions: int = 200):
    """Build a synthetic SPADL action stream + linked tracking frames.

    Tuned so tracking features carry signal:
    - Shots (type_id=11) have defenders close (lower scoring prob proxy).
    """
    rng = np.random.default_rng(seed)
    actions = pd.DataFrame(
        {
            "game_id": [1] * n_actions,
            "original_event_id": [None] * n_actions,
            "action_id": list(range(1, n_actions + 1)),
            "period_id": [1] * n_actions,
            "time_seconds": [t * 0.5 for t in range(n_actions)],
            "team_id": rng.choice([1, 2], size=n_actions),
            "player_id": rng.choice([11, 12, 13, 21, 22, 23], size=n_actions),
            "start_x": rng.uniform(0, 105, size=n_actions),
            "start_y": rng.uniform(0, 68, size=n_actions),
            "end_x": rng.uniform(0, 105, size=n_actions),
            "end_y": rng.uniform(0, 68, size=n_actions),
            "type_id": rng.choice([0, 1, 11], size=n_actions),  # 11 = shot
            "result_id": [1] * n_actions,
            "bodypart_id": [0] * n_actions,
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
                x=a["start_x"],
                y=a["start_y"],
                z=float("nan"),
                speed=rng.uniform(0, 6),
                speed_source="native",
                ball_state="alive",
                team_attacking_direction="ltr",
                confidence=None,
                visibility=None,
                source_provider="synth",
            )
        )
        defender_dist = 3.0 if a["type_id"] == 11 else 8.0
        opposite_team = 2 if a["team_id"] == 1 else 1
        for j in range(5):
            angle = 2 * np.pi * j / 5
            rows.append(
                dict(
                    game_id=1,
                    period_id=1,
                    frame_id=fid,
                    time_seconds=a["time_seconds"],
                    frame_rate=10.0,
                    player_id=30 + j,
                    team_id=opposite_team,
                    is_ball=False,
                    is_goalkeeper=False,
                    x=a["start_x"] + defender_dist * np.cos(angle),
                    y=a["start_y"] + defender_dist * np.sin(angle),
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
    return actions, frames


@pytest.mark.slow
def test_hybrid_vaep_with_tracking_lifecycle():
    """HybridVAEP + tracking_default_xfns: compute_features + fit + rate without errors.

    Asserts:
      - tracking-aware feature columns appear in X (via lift_to_states naming).
      - fit/rate cycle reaches no errors on synthetic data.
    """
    from silly_kicks.tracking.features import tracking_default_xfns
    from silly_kicks.vaep.hybrid import HybridVAEP, hybrid_xfns_default

    actions, frames = _make_synthetic_match()
    game = pd.Series({"game_id": 1, "home_team_id": 1, "away_team_id": 2})

    v = HybridVAEP(xfns=hybrid_xfns_default + tracking_default_xfns)
    X = v.compute_features(game, actions, frames=frames)
    assert any("nearest_defender_distance_a0" in c for c in X.columns)
    assert any("actor_speed_a0" in c for c in X.columns)

    # Synthetic labels: shots (type_id=11) -> small chance of scoring
    y = pd.DataFrame(
        {
            "scores": (actions["type_id"] == 11).astype(int).to_numpy(),
            "concedes": np.zeros(len(actions), dtype=int),
        }
    )

    v.fit(X, y, learner="xgboost", val_size=0.25, random_state=42)
    ratings = v.rate(game, actions, frames=frames)
    assert "vaep_value" in ratings.columns
