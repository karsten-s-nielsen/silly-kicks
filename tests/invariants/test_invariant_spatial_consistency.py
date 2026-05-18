"""Physical invariant: play_left_to_right preserves ball-player distances.

After any coordinate normalization, Euclidean distances between entities in
the same frame must be preserved. This invariant catches bugs where some
entities (e.g., ball) are flipped differently from others (e.g., away-team
players), breaking spatial relationships.

Parametrized across multiple realistic scenarios: converter-style output
(home="ltr", away="rtl", ball=None), un-normalized frames, multi-period
frames, and Sportec-style string team IDs.
"""

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking.utils import play_left_to_right


def _build_tracking_frame(
    *,
    period_id: int,
    frame_id: int,
    home_team_id: int | str,
    away_team_id: int | str,
    home_dir: str,
    away_dir: str,
    n_home: int = 5,
    n_away: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a realistic tracking frame with N players per team + ball.

    Positions are sampled uniformly on the SPADL pitch (105x68). Ball is at
    a plausible position near one of the home players (the typical scenario
    for ball-carrier features).
    """
    rng = np.random.default_rng(seed + period_id * 1000 + frame_id)
    rows: list[dict] = []
    base = {
        "game_id": 1,
        "period_id": period_id,
        "frame_id": frame_id,
        "time_seconds": frame_id / 25.0,
        "frame_rate": 25.0,
        "is_goalkeeper": False,
        "z": float("nan"),
        "speed": 0.0,
        "speed_source": "native",
        "ball_state": "alive",
        "confidence": None,
        "visibility": None,
        "source_provider": "test",
    }
    # Home players
    for i in range(n_home):
        rows.append(
            {
                **base,
                "player_id": f"HOME-{i}",
                "team_id": home_team_id,
                "is_ball": False,
                "x": float(rng.uniform(5, 100)),
                "y": float(rng.uniform(5, 63)),
                "team_attacking_direction": home_dir,
            }
        )
    # Away players
    for i in range(n_away):
        rows.append(
            {
                **base,
                "player_id": f"AWAY-{i}",
                "team_id": away_team_id,
                "is_ball": False,
                "x": float(rng.uniform(5, 100)),
                "y": float(rng.uniform(5, 63)),
                "team_attacking_direction": away_dir,
            }
        )
    # Ball near a random home player
    home_idx = rng.integers(0, n_home)
    ball_x = rows[home_idx]["x"] + float(rng.uniform(-3, 3))
    ball_y = rows[home_idx]["y"] + float(rng.uniform(-3, 3))
    rows.append(
        {
            **base,
            "player_id": None,
            "team_id": None,
            "is_ball": True,
            "x": np.clip(ball_x, 0, 105),
            "y": np.clip(ball_y, 0, 68),
            "team_attacking_direction": None,  # converter-realistic
        }
    )
    return pd.DataFrame(rows)


def _all_pairwise_distances(frames: pd.DataFrame) -> pd.DataFrame:
    """Compute all pairwise distances between entities in each frame."""
    entities = frames[["period_id", "frame_id", "player_id", "is_ball", "x", "y"]].copy()
    entities["entity_id"] = entities.apply(lambda r: "BALL" if r["is_ball"] else str(r["player_id"]), axis=1)
    merged = entities.merge(entities, on=["period_id", "frame_id"], suffixes=("_a", "_b"))
    merged = merged[merged["entity_id_a"] < merged["entity_id_b"]]
    merged["dist"] = np.sqrt((merged["x_a"] - merged["x_b"]) ** 2 + (merged["y_a"] - merged["y_b"]) ** 2)
    return merged[["period_id", "frame_id", "entity_id_a", "entity_id_b", "dist"]]


# Scenarios parametrized to cover realistic converter output patterns
_SCENARIOS = [
    pytest.param(
        dict(home_dir="ltr", away_dir="rtl", home_team_id=100, away_team_id=200),
        id="converter-output-int-ids",
    ),
    pytest.param(
        dict(home_dir="ltr", away_dir="rtl", home_team_id="DFL-CLU-000008", away_team_id="DFL-CLU-000023"),
        id="converter-output-string-ids",
    ),
    pytest.param(
        dict(home_dir="rtl", away_dir="ltr", home_team_id=100, away_team_id=200),
        id="un-normalized-home-rtl",
    ),
]


@pytest.mark.parametrize("kwargs", _SCENARIOS)
def test_play_left_to_right_preserves_all_distances(kwargs):
    """All pairwise distances must be identical before and after play_left_to_right."""
    home_team_id = kwargs["home_team_id"]

    # Build multi-frame, multi-period data
    frames_list = []
    for pid in (1, 2):
        for fid in range(pid * 100, pid * 100 + 3):
            frames_list.append(
                _build_tracking_frame(
                    period_id=pid,
                    frame_id=fid,
                    **kwargs,
                )
            )
    frames = pd.concat(frames_list, ignore_index=True)

    raw_dists = _all_pairwise_distances(frames)
    ltr_frames = play_left_to_right(frames, home_team_id=home_team_id)
    ltr_dists = _all_pairwise_distances(ltr_frames)

    merged = raw_dists.merge(
        ltr_dists,
        on=["period_id", "frame_id", "entity_id_a", "entity_id_b"],
        suffixes=("_raw", "_ltr"),
    )
    delta = (merged["dist_raw"] - merged["dist_ltr"]).abs()
    max_delta = float(delta.max())
    assert max_delta < 0.001, f"play_left_to_right changed pairwise distances by up to {max_delta:.6f}m"


@pytest.mark.parametrize("kwargs", _SCENARIOS)
def test_home_direction_is_ltr_after_normalization(kwargs):
    """After play_left_to_right, home-team player rows must have direction='ltr'."""
    home_team_id = kwargs["home_team_id"]
    frames = _build_tracking_frame(period_id=1, frame_id=0, **kwargs)
    out = play_left_to_right(frames, home_team_id=home_team_id)
    home_dirs = out.loc[
        (~out["is_ball"]) & (out["team_id"] == home_team_id),
        "team_attacking_direction",
    ].unique()
    assert set(home_dirs) == {"ltr"}, f"Home directions after normalization: {home_dirs}"


@pytest.mark.parametrize("kwargs", _SCENARIOS)
def test_ball_direction_stays_none(kwargs):
    """Ball rows must retain team_attacking_direction=None after normalization."""
    home_team_id = kwargs["home_team_id"]
    frames = _build_tracking_frame(period_id=1, frame_id=0, **kwargs)
    out = play_left_to_right(frames, home_team_id=home_team_id)
    ball_dirs = out.loc[out["is_ball"], "team_attacking_direction"]
    assert ball_dirs.isna().all() or (ball_dirs == None).all()  # noqa: E711
