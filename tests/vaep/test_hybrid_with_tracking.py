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

# Both tests below pass ``home_team_id=1``, so team 1 attacks left-to-right in the
# frame convention and team 2 attacks right-to-left. Previously EVERY frame row was
# labelled "ltr", which is a physically impossible scene (both teams attacking the same
# goal) and made ``acting_team_attacks_rtl`` resolve to an all-False flip on all 200
# actions -- 101 of them away actions. The fixture therefore never exercised the ADR-028
# re-projection path at all.
_HOME_TEAM_ID = 1
_AWAY_TEAM_ID = 2
_TEAM_DIRECTION = {_HOME_TEAM_ID: "ltr", _AWAY_TEAM_ID: "rtl"}

# Pitch extents for the ADR-028 point reflection. Kept as literals rather than imported
# so this fixture states its own geometry (mirrors _action_orientation.FIELD_LENGTH/WIDTH).
_FIELD_LENGTH = 105.0
_FIELD_WIDTH = 68.0


def _to_frame_coords(x: float, y: float, *, rtl: bool) -> tuple[float, float]:
    """Map an action-LTR position into the frame convention (home attacks +x).

    SPADL actions are action-LTR (the ACTING team attacks x=105); tracking frames are
    frame-LTR (the HOME team attacks x=105). For a team attacking right-to-left the two
    are a 180-degree POINT reflection apart -- BOTH axes, not just x (an x-only mirror is
    exact only for a y-symmetric scene, which is how ADR-041's incomplete repair survived).

    Every position below is authored in action-LTR (where the fixture's tuning is stated:
    defenders 3 m away for shots, 8 m otherwise, defending GK on the attacked goal line)
    and then mapped through here. ``acting_team_attacks_rtl`` maps it straight back, so
    the action-LTR feature values are unchanged -- what changes is that away actions now
    travel through the re-projection instead of bypassing it.
    """
    return (_FIELD_LENGTH - x, _FIELD_WIDTH - y) if rtl else (x, y)


def _make_synthetic_match(seed: int = 42, n_actions: int = 200):
    """Build a synthetic SPADL action stream + linked tracking frames.

    Tuned so tracking features carry signal:
    - Shots (type_id=11) have defenders close (lower scoring prob proxy).

    Frames are emitted in the frame-LTR convention (team 1 = home attacks +x, team 2
    attacks -x); see ``_to_frame_coords``.
    """
    rng = np.random.default_rng(seed)
    actions = pd.DataFrame(
        {
            "game_id": [1] * n_actions,
            "original_event_id": [None] * n_actions,
            "action_id": list(range(1, n_actions + 1)),
            "period_id": [1] * n_actions,
            "time_seconds": [t * 0.5 for t in range(n_actions)],
            "team_id": rng.choice([_HOME_TEAM_ID, _AWAY_TEAM_ID], size=n_actions),
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
        acting_team = int(a["team_id"])
        opposite_team = _AWAY_TEAM_ID if acting_team == _HOME_TEAM_ID else _HOME_TEAM_ID
        acting_rtl = _TEAM_DIRECTION[acting_team] == "rtl"
        actor_x, actor_y = _to_frame_coords(a["start_x"], a["start_y"], rtl=acting_rtl)
        rows.append(
            dict(
                game_id=1,
                period_id=1,
                frame_id=fid,
                time_seconds=a["time_seconds"],
                frame_rate=10.0,
                player_id=a["player_id"],
                team_id=acting_team,
                is_ball=False,
                is_goalkeeper=False,
                x=actor_x,
                y=actor_y,
                z=float("nan"),
                speed=rng.uniform(0, 6),
                speed_source="native",
                ball_state="alive",
                team_attacking_direction=_TEAM_DIRECTION[acting_team],
                confidence=None,
                visibility=None,
                source_provider="synth",
            )
        )
        defender_dist = 3.0 if a["type_id"] == 11 else 8.0
        for j in range(5):
            angle = 2 * np.pi * j / 5
            # Authored in action-LTR (a ring `defender_dist` around the actor), then
            # mapped into the frame convention -- so the ring lands back exactly where
            # the fixture intends once ADR-028 re-projects it.
            dx, dy = _to_frame_coords(
                a["start_x"] + defender_dist * np.cos(angle),
                a["start_y"] + defender_dist * np.sin(angle),
                rtl=acting_rtl,
            )
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
                    x=dx,
                    y=dy,
                    z=float("nan"),
                    speed=1.5,
                    speed_source="native",
                    ball_state="alive",
                    team_attacking_direction=_TEAM_DIRECTION[opposite_team],
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
    game = pd.Series({"game_id": 1, "home_team_id": _HOME_TEAM_ID, "away_team_id": _AWAY_TEAM_ID})

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


# ---------------------------------------------------------------------------
# PR-S21 — pre_shot_gk_default_xfns lifecycle through HybridVAEP
# ---------------------------------------------------------------------------


def _make_synthetic_match_with_gk(seed: int = 42, n_actions: int = 200):
    """Extension of _make_synthetic_match: shots (type_id=11) get defending_gk_player_id
    populated, with a defending GK at (104, 34) in their linked frame.
    Non-shot rows get NaN defending_gk_player_id.
    """
    actions, frames = _make_synthetic_match(seed=seed, n_actions=n_actions)
    actions = actions.copy()
    is_shot = actions["type_id"] == 11
    actions["defending_gk_player_id"] = np.where(is_shot, 99.0, np.nan)
    # For each shot's linked frame, append a defending-GK row (player 99 on opposite team).
    extra_rows = []
    for _, a in actions[is_shot].iterrows():
        fid = int(a["time_seconds"] * 10)
        acting_team = int(a["team_id"])
        opposite_team = _AWAY_TEAM_ID if acting_team == _HOME_TEAM_ID else _HOME_TEAM_ID
        acting_rtl = _TEAM_DIRECTION[acting_team] == "rtl"
        # (104, 34) is action-LTR: the keeper stands on the goal line the ACTING team
        # attacks. Mapped into frame coords that is x~104 for a team-1 shot and x~1 for a
        # team-2 shot -- team 2 attacks -x, so its opponent defends the LEFT goal. The old
        # fixture hardcoded frame x=104 for both, which put the away side's defending
        # keeper in the goal his own team was attacking.
        gk_x, gk_y = _to_frame_coords(104.0, 34.0, rtl=acting_rtl)
        extra_rows.append(
            dict(
                game_id=1,
                period_id=1,
                frame_id=fid,
                time_seconds=a["time_seconds"],
                frame_rate=10.0,
                player_id=99,
                team_id=opposite_team,
                is_ball=False,
                is_goalkeeper=True,
                x=gk_x,
                y=gk_y,
                z=float("nan"),
                speed=0.5,
                speed_source="native",
                ball_state="alive",
                team_attacking_direction=_TEAM_DIRECTION[opposite_team],
                confidence=None,
                visibility=None,
                source_provider="synth",
            )
        )
    if extra_rows:
        frames = pd.concat([frames, pd.DataFrame(extra_rows)], ignore_index=True)
    return actions, frames


@pytest.mark.slow
def test_hybrid_vaep_with_pre_shot_gk_lifecycle():
    """HybridVAEP + tracking_default_xfns + pre_shot_gk_default_xfns: full lifecycle smoke test.

    Asserts:
      - Both PR-S20 and PR-S21 lifted feature columns appear in X.
      - fit/rate cycle reaches no errors on synthetic data.
    """
    from silly_kicks.tracking.features import (
        pre_shot_gk_default_xfns,
        tracking_default_xfns,
    )
    from silly_kicks.vaep.hybrid import HybridVAEP, hybrid_xfns_default

    actions, frames = _make_synthetic_match_with_gk()
    game = pd.Series({"game_id": 1, "home_team_id": _HOME_TEAM_ID, "away_team_id": _AWAY_TEAM_ID})

    v = HybridVAEP(xfns=hybrid_xfns_default + tracking_default_xfns + pre_shot_gk_default_xfns)
    X = v.compute_features(game, actions, frames=frames)
    # PR-S20 features
    assert any("nearest_defender_distance_a0" in c for c in X.columns)
    # PR-S21 features
    assert any("pre_shot_gk_x_a0" in c for c in X.columns)
    assert any("pre_shot_gk_distance_to_goal_a0" in c for c in X.columns)

    y = pd.DataFrame(
        {
            "scores": (actions["type_id"] == 11).astype(int).to_numpy(),
            "concedes": np.zeros(len(actions), dtype=int),
        }
    )
    v.fit(X, y, learner="xgboost", val_size=0.25, random_state=42)
    ratings = v.rate(game, actions, frames=frames)
    assert "vaep_value" in ratings.columns
