"""Shared helpers for synthetic tracking-fixture generators (PR-S19).

Reads ``tests/datasets/tracking/empirical_probe_baselines.json`` and emits
provider-shaped raw-input DataFrames for each adapter to consume. The probe
script ``scripts/probe_tracking_baselines.py`` produces the JSON; both files
are committed.

Synthetic motion is deterministic (uniform / piecewise-linear); only the
*structural parameters* (frame_rate, player counts, NaN signatures) are
calibrated against real data. Real datasets are not committed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
BASELINES_PATH = ROOT / "empirical_probe_baselines.json"
BASELINES = json.loads(BASELINES_PATH.read_text())


def get_provider_baseline(provider: str) -> dict[str, Any]:
    """Return per-provider stats dict from the committed JSON probe."""
    return BASELINES["providers"][provider]


def deterministic_uniform_motion(
    n_frames: int,
    frame_rate: float,
    n_players_per_team: int = 11,
    period_id: int = 1,
    t0: float = 0.0,
    seed: int = 42,
    *,
    inject_realistic_edge_cases: bool = False,
    edge_case_provider: str | None = None,
) -> pd.DataFrame:
    """Build a long-form provider-agnostic reference DataFrame.

    Each player moves at a deterministic uniform speed. Per-provider
    generators wrap this with provider-native column shaping (string vs.
    int identifiers, file-format dtypes).

    Returns a DataFrame with columns:
        period_id, frame_id, time_seconds, frame_rate,
        player_id, team_id, jersey, is_ball, is_goalkeeper,
        x_centered, y_centered, z, speed_native, ball_state.
    """
    rng = np.random.default_rng(seed)
    rows = []
    dt = 1.0 / frame_rate
    # Small bounded oscillation to give nontrivial per-frame dx/dy without
    # drifting off-pitch. Pitch (centered) is [-52.5, 52.5] x [-34, 34].
    n_total_frames_for_drift = max(n_frames - 1, 1)
    for i in range(n_frames):
        t = t0 + i * dt
        # Drift each player up to +/-3 m total over the sequence, varying by jersey.
        dy_drift_unit = (i / n_total_frames_for_drift) * 6.0 - 3.0
        for team_id, team_offset_x in ((100, -20.0), (200, 20.0)):
            for jersey in range(n_players_per_team):
                # Static positions: spread players across a 30 m x 24 m grid
                # within each half. Goalkeeper (jersey=0) stays near the goal line.
                col = jersey % 4
                rowj = jersey // 4
                base_x = team_offset_x + (col - 1.5) * 8.0  # +/-12 m around team center
                base_y = (rowj - 1.0) * 8.0  # 3 rows
                rows.append(
                    {
                        "period_id": period_id,
                        "frame_id": i,
                        "time_seconds": t,
                        "frame_rate": frame_rate,
                        "player_id": team_id * 100 + jersey,
                        "team_id": team_id,
                        "jersey": jersey,
                        "is_ball": False,
                        "is_goalkeeper": (jersey == 0),
                        "x_centered": base_x + 0.5 * np.sin(0.1 * i + jersey),
                        "y_centered": base_y + dy_drift_unit + 0.3 * np.cos(0.1 * i + jersey),
                        "z": float("nan"),
                        "speed_native": float(rng.uniform(2.0, 6.0)),
                        "ball_state": "alive",
                    }
                )
        # Ball traverses pitch length linearly, returns to start at end.
        ball_phase = (i / n_total_frames_for_drift) * 2.0 - 1.0  # in [-1, 1]
        rows.append(
            {
                "period_id": period_id,
                "frame_id": i,
                "time_seconds": t,
                "frame_rate": frame_rate,
                "player_id": None,
                "team_id": None,
                "jersey": None,
                "is_ball": True,
                "is_goalkeeper": False,
                "x_centered": ball_phase * 40.0,
                "y_centered": 5.0 * np.sin(0.05 * i),
                "z": 0.5,
                "speed_native": 10.0,
                "ball_state": "alive",
            }
        )
    df = pd.DataFrame(rows)
    if inject_realistic_edge_cases:
        df = _apply_realistic_edge_cases(df, edge_case_provider, rng)
    return df


def _apply_realistic_edge_cases(
    df: pd.DataFrame,
    provider: str | None,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Inject real-data-calibrated edge cases into a synthetic frame DataFrame.

    Edge cases applied (per empirical_probe_baselines.json):
      - Off-pitch tail on player y (~0.5% of player rows nudged out by 0.5-1 m).
      - NaN ball coordinates at the per-provider missing rate.
      - 5-second ball-out interval mid-sequence (ball_state="dead").
      - Ball-x off-pitch tail at the per-provider rate (throw-in artifact).

    The provider name selects the per-provider rates; a missing provider
    or missing entry falls back to a small default (~0.5%).
    """
    baseline: dict[str, Any] = BASELINES["providers"].get(provider, {}) if provider else {}
    n_rows = len(df)
    player_mask = (~df["is_ball"]).to_numpy(dtype=bool)
    ball_mask = df["is_ball"].to_numpy(dtype=bool)

    # Off-pitch player y tail: nudge a small fraction of player y_centered just
    # outside [-34, 34] to mimic touchline overruns / GK in goal.
    off_pitch_y_rate = float(baseline.get("off_pitch_y_rate") or 0.005)
    n_off_y = int(off_pitch_y_rate * player_mask.sum())
    if n_off_y > 0:
        idx_off_y = rng.choice(np.where(player_mask)[0], size=n_off_y, replace=False)
        df.loc[idx_off_y, "y_centered"] = df.loc[idx_off_y, "y_centered"].to_numpy() + rng.choice(
            [-1, 1], size=n_off_y
        ) * (34.5 + rng.random(n_off_y) * 0.5)

    # NaN ball coordinates per provider's nan_rate_ball_x.
    nan_ball_rate = float(baseline.get("nan_rate_ball_x") or 0.0)
    if nan_ball_rate > 0:
        n_nan = int(nan_ball_rate * ball_mask.sum())
        if n_nan > 0:
            idx_nan = rng.choice(np.where(ball_mask)[0], size=n_nan, replace=False)
            df.loc[idx_nan, "x_centered"] = float("nan")
            df.loc[idx_nan, "y_centered"] = float("nan")

    # Ball-x off-pitch tail (throw-in artifact).
    ball_off_x_rate = float(baseline.get("off_pitch_ball_x_rate") or 0.001)
    if ball_off_x_rate > 0:
        n_off_x = int(ball_off_x_rate * ball_mask.sum())
        if n_off_x > 0:
            idx_off_x = rng.choice(np.where(ball_mask)[0], size=n_off_x, replace=False)
            df.loc[idx_off_x, "x_centered"] = rng.choice([-1, 1], size=n_off_x) * (53.0 + rng.random(n_off_x) * 1.0)

    # Ball-out interval (5 s in the middle of the sequence).
    if "frame_rate" in df.columns and n_rows > 0:
        fr = float(df["frame_rate"].iloc[0])
        mid_t = float(df["time_seconds"].iloc[n_rows // 2])
        out_mask = df["time_seconds"].between(mid_t, mid_t + min(5.0, n_rows / fr / 4))
        df.loc[out_mask, "ball_state"] = "dead"

    return df
