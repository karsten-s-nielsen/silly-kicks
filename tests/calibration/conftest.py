"""Shared fixtures for the TF-24 calibration harness tests.

Built on the committed slim provider datasets (``tests/datasets/tracking/action_context_slim/``)
via the existing ``_provider_inputs`` helpers — real frame slices with the shot+keeper_save action
pattern, so the full ~15-aggregator enrichment chain exercises the same data the tracking-feature
tests already validate.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tests.tracking._provider_inputs import load_provider_frames, synthesize_actions

_HOME = "home_team"  # skillcorner slim team_id space is {"home_team", "away_team"}


def _single_match(provider: str = "skillcorner", n_actions: int = 10):
    """Return (actions, frames, home_team_id) for one slim match.

    Velocities are derived up front (vx/vy) so the bekkers_pi pressure flavor + velocity-aware
    carrier inference work — the calibration loaders must do the same (derive_velocities) for
    providers that don't emit native velocities.
    """
    from silly_kicks.tracking.preprocess import derive_velocities, smooth_frames

    frames = derive_velocities(smooth_frames(load_provider_frames(provider)))
    actions = synthesize_actions(frames, n_actions=n_actions)
    return actions, frames, _HOME


@pytest.fixture
def synth():
    """One slim match: (actions, frames, home_team_id)."""
    return _single_match()


@pytest.fixture
def frozen_xt(synth):
    """A FrozenXt fit on the synth actions (unit fixture only; production fits on a disjoint corpus)."""
    from silly_kicks.calibration._xt import fit_frozen_xt

    actions, _frames, _home = synth
    return fit_frozen_xt(actions, exclude_match_ids=set(), match_id_col="game_id", source="unit")


# --- Controlled carrier fixtures (Stage 1) ---------------------------------
# Tiny synthetic frames where we DICTATE the nearest-to-ball player, so carrier accuracy has a
# known ground truth (the slim data gives no controllable carrier).


def _carrier_match(game_id: str, *, correct: bool):
    """One match where the actor IS (correct) / is NOT (wrong) the player nearest the ball.

    Returns (actions, frames, home_team_id). Two outfield players + a GK per team. Player 10
    (home) sits ON the ball; the on-ball action is by player 10 (correct) or player 99 (wrong).
    """
    rows = []
    # frame 0 @ t=0: ball at (50, 34); player 10 ON ball; others far.
    layout = [
        ("ball", None, True, False, 50.0, 34.0),
        (10, 1, False, False, 50.0, 34.0),  # home outfield, on the ball
        (11, 1, False, True, 5.0, 34.0),  # home GK
        (20, 2, False, False, 80.0, 60.0),  # away outfield, far
        (21, 2, False, True, 100.0, 34.0),  # away GK
    ]
    for pid, team, is_ball, is_gk, x, y in layout:
        rows.append(
            {
                "game_id": game_id,
                "period_id": 1,
                "frame_id": 0,
                "time_seconds": 0.0,
                "frame_rate": 25.0,
                "player_id": pid,
                "team_id": team,
                "is_ball": is_ball,
                "is_goalkeeper": is_gk,
                "x": x,
                "y": y,
                "z": 0.0,
                "vx": 0.0,
                "vy": 0.0,
                "ball_state": "alive",
                "team_attacking_direction": "ltr",
                "source_provider": "synthetic",
            }
        )
    frames = pd.DataFrame(rows)
    actor = 10 if correct else 99  # 99 is not the nearest-to-ball player => wrong
    actions = pd.DataFrame(
        {
            "game_id": [game_id],
            "action_id": [1],
            "period_id": [1],
            "time_seconds": [0.0],
            "team_id": [1],
            "player_id": [actor],
            "type_name": ["pass"],  # carrier-actor type
            "type_id": [0],
            "result_id": [1],
            "bodypart_id": [0],
            "start_x": [50.0],
            "start_y": [34.0],
            "end_x": [60.0],
            "end_y": [34.0],
        }
    )
    return actions, frames, 1


@pytest.fixture
def synth_known_carrier():
    """One provider, one perfect-carrier match => accuracy 1.0."""
    return {"provA": [_carrier_match("provA_1", correct=True)]}


@pytest.fixture
def synth_two_providers_imbalanced():
    """provA: 100 perfect matches; provB: 1 all-wrong match => equal-weight mean = 0.5."""
    return {
        "provA": [_carrier_match(f"provA_{i}", correct=True) for i in range(100)],
        "provB": [_carrier_match("provB_1", correct=False)],
    }


@pytest.fixture
def stage1_fold():
    """A small two-match carrier fold for the CLI Stage-1 smoke."""
    return {"provA": [_carrier_match("provA_1", correct=True), _carrier_match("provA_2", correct=True)]}


# --- Stage-2 fixtures (subsampled slim frames + controlled goals) ----------
# A 24-action stream with a goal by each team (actions 8, 16) makes BOTH scores AND concedes
# non-degenerate within ONE match, so the LOMO train fold (1 match) can train XGBoost. Frames are
# subsampled to ~40 frames so the full enrichment chain runs in ~1-2 s per match (fast CI).

_SLIM_SUB = None  # module-level cache of the preprocessed slim frame subset


def _slim_subset(n_frames: int = 40):
    global _SLIM_SUB
    if _SLIM_SUB is None:
        from silly_kicks.tracking.preprocess import derive_velocities, smooth_frames

        f = derive_velocities(smooth_frames(load_provider_frames("skillcorner")))
        p1 = f[f["period_id"] == 1]
        keep_ids = sorted(p1["frame_id"].unique())[:n_frames]
        _SLIM_SUB = p1[p1["frame_id"].isin(keep_ids)].copy()
    return _SLIM_SUB


def _stage2_match(game_id: str, n_actions: int = 24):
    """Controlled action stream (goal per team) anchored on subsampled slim frames."""
    from silly_kicks.spadl import config as spadl_config

    frames = _slim_subset().copy()
    frames["game_id"] = game_id
    times = frames.drop_duplicates("frame_id").sort_values("frame_id")["time_seconds"].to_numpy()[:n_actions]
    n = len(times)
    teams = (["home_team", "away_team"] * (n // 2 + 1))[:n]
    pass_id, shot_id = spadl_config.actiontype_id["pass"], spadl_config.actiontype_id["shot"]
    succ, foot = spadl_config.result_id["success"], spadl_config.bodypart_id["foot"]
    type_ids = [pass_id] * n
    result_ids = [succ] * n
    # Goal by home @ action 8 and by away @ action 16 => both classes for scores AND concedes.
    for idx, team in ((8, "home_team"), (16, "away_team")):
        if idx < n:
            type_ids[idx], result_ids[idx], teams[idx] = shot_id, succ, team
    actions = pd.DataFrame(
        {
            "game_id": [game_id] * n,
            "action_id": list(range(1, n + 1)),
            "period_id": [1] * n,
            "time_seconds": times,
            "team_id": teams,
            "player_id": list(range(100, 100 + n)),
            "type_id": type_ids,
            "result_id": result_ids,
            "bodypart_id": [foot] * n,
            "start_x": [50.0] * n,
            "start_y": [34.0] * n,
            "end_x": [60.0] * n,
            "end_y": [34.0] * n,
        }
    )
    return actions, frames, _HOME


@pytest.fixture
def stage2_fold():
    """One provider, two non-degenerate matches => LOMO with 2 folds."""
    return {"skillcorner": [_stage2_match("m0"), _stage2_match("m1")]}
