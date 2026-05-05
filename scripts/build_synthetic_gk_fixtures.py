"""Generate synthetic GK identification fixtures deterministically.

Usage::

    uv run python scripts/build_synthetic_gk_fixtures.py

Produces 3 parquet files in tests/datasets/tracking/synthetic/:
- gk_substitution.parquet (~30 KB) - multi-GK substitution scenario
- sweeper_keeper.parquet (~15 KB) - sweeper-keeper fallback case
- brief_outfielder.parquet (~20 KB) - n_frames filter exclusion case
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "tests" / "datasets" / "tracking" / "synthetic"


def _build_gk_substitution() -> pd.DataFrame:
    """Multi-GK: 2 teams x 11 outfielders + 1 starter GK + 1 sub GK each.

    Starter plays period 1 (~750 frames); sub plays period 2 (~750 frames).
    Both GKs have realistic positional behavior (pa_dwell~0.7, dist~10m).
    """
    rows = []
    frame_rate = 25.0

    for team_idx, team_id in enumerate(["home", "away"]):
        gk_x = 5.0 if team_idx == 0 else 100.0  # own goal end

        for period_id in [1, 2]:
            gk_player = f"gk_starter_{team_id}" if period_id == 1 else f"gk_sub_{team_id}"

            for frame_id in range(750):
                time_seconds = (period_id - 1) * 45 * 60 + frame_id / frame_rate

                # GK row
                rows.append(
                    {
                        "game_id": "gk_sub_match",
                        "period_id": period_id,
                        "frame_id": (period_id - 1) * 750 + frame_id,
                        "time_seconds": time_seconds,
                        "frame_rate": frame_rate,
                        "player_id": gk_player,
                        "team_id": team_id,
                        "is_ball": False,
                        "is_goalkeeper": False,  # Algorithm must derive
                        "x": np.clip(gk_x + np.random.normal(0, 2), 0, 105),
                        "y": np.clip(34.0 + np.random.normal(0, 3), 0, 68),
                        "z": 0.0,
                        "speed": 0.5,
                        "speed_source": "native",
                        "ball_state": "alive",
                        "team_attacking_direction": "ltr" if team_idx == 0 else "rtl",
                        "confidence": None,
                        "visibility": None,
                        "source_provider": "synthetic",
                    }
                )

                # 10 outfielders
                for i in range(10):
                    outfielder_x = np.clip(30.0 + i * 5 + np.random.normal(0, 3), 0, 105)
                    rows.append(
                        {
                            "game_id": "gk_sub_match",
                            "period_id": period_id,
                            "frame_id": (period_id - 1) * 750 + frame_id,
                            "time_seconds": time_seconds,
                            "frame_rate": frame_rate,
                            "player_id": f"outfielder_{team_id}_{i}",
                            "team_id": team_id,
                            "is_ball": False,
                            "is_goalkeeper": False,
                            "x": outfielder_x,
                            "y": np.clip(10.0 + i * 5 + np.random.normal(0, 2), 0, 68),
                            "z": 0.0,
                            "speed": 3.0 + np.random.uniform(0, 2),
                            "speed_source": "native",
                            "ball_state": "alive",
                            "team_attacking_direction": "ltr" if team_idx == 0 else "rtl",
                            "confidence": None,
                            "visibility": None,
                            "source_provider": "synthetic",
                        }
                    )

        # Ball rows
        for period_id in [1, 2]:
            for frame_id in range(750):
                time_seconds = (period_id - 1) * 45 * 60 + frame_id / frame_rate
                rows.append(
                    {
                        "game_id": "gk_sub_match",
                        "period_id": period_id,
                        "frame_id": (period_id - 1) * 750 + frame_id,
                        "time_seconds": time_seconds,
                        "frame_rate": frame_rate,
                        "player_id": None,
                        "team_id": None,
                        "is_ball": True,
                        "is_goalkeeper": False,
                        "x": np.clip(52.5 + np.random.normal(0, 10), 0, 105),
                        "y": np.clip(34.0 + np.random.normal(0, 10), 0, 68),
                        "z": 0.0,
                        "speed": 5.0,
                        "speed_source": "native",
                        "ball_state": "alive",
                        "team_attacking_direction": None,
                        "confidence": None,
                        "visibility": None,
                        "source_provider": "synthetic",
                    }
                )

    return pd.DataFrame(rows)


def _build_sweeper_keeper() -> pd.DataFrame:
    """Sweeper-keeper GK (pa_dwell~0.25, dist~18m). Strict fails, fallback fires."""
    rows = []
    frame_rate = 25.0
    n_frames = 500

    for frame_id in range(n_frames):
        time_seconds = frame_id / frame_rate

        # Sweeper-keeper at x=18 (outside PA but closest to goal)
        rows.append(
            {
                "game_id": "sweeper_match",
                "period_id": 1,
                "frame_id": frame_id,
                "time_seconds": time_seconds,
                "frame_rate": frame_rate,
                "player_id": "sweeper_gk",
                "team_id": "home",
                "is_ball": False,
                "is_goalkeeper": False,
                "x": 18.0 + np.random.normal(0, 3),
                "y": 34.0 + np.random.normal(0, 5),
                "z": 0.0,
                "speed": 2.0,
                "speed_source": "native",
                "ball_state": "alive",
                "team_attacking_direction": "ltr",
                "confidence": None,
                "visibility": None,
                "source_provider": "synthetic",
            }
        )

        # 10 outfielders spread across midfield
        for i in range(10):
            rows.append(
                {
                    "game_id": "sweeper_match",
                    "period_id": 1,
                    "frame_id": frame_id,
                    "time_seconds": time_seconds,
                    "frame_rate": frame_rate,
                    "player_id": f"outfielder_{i}",
                    "team_id": "home",
                    "is_ball": False,
                    "is_goalkeeper": False,
                    "x": 40.0 + i * 5,
                    "y": 10.0 + i * 5,
                    "z": 0.0,
                    "speed": 4.0,
                    "speed_source": "native",
                    "ball_state": "alive",
                    "team_attacking_direction": "ltr",
                    "confidence": None,
                    "visibility": None,
                    "source_provider": "synthetic",
                }
            )

        # Ball row
        rows.append(
            {
                "game_id": "sweeper_match",
                "period_id": 1,
                "frame_id": frame_id,
                "time_seconds": time_seconds,
                "frame_rate": frame_rate,
                "player_id": None,
                "team_id": None,
                "is_ball": True,
                "is_goalkeeper": False,
                "x": np.clip(52.5 + np.random.normal(0, 10), 0, 105),
                "y": np.clip(34.0 + np.random.normal(0, 10), 0, 68),
                "z": 0.0,
                "speed": 5.0,
                "speed_source": "native",
                "ball_state": "alive",
                "team_attacking_direction": None,
                "confidence": None,
                "visibility": None,
                "source_provider": "synthetic",
            }
        )

    return pd.DataFrame(rows)


def _build_brief_outfielder() -> pd.DataFrame:
    """Standard GK + brief outfielder (<30% frames) near goal. Filter excludes brief sub."""
    rows = []
    frame_rate = 25.0
    n_frames = 500
    brief_start = 450  # Brief sub appears in last 50 frames (10%)

    for frame_id in range(n_frames):
        time_seconds = frame_id / frame_rate

        # Standard GK (full coverage)
        rows.append(
            {
                "game_id": "brief_match",
                "period_id": 1,
                "frame_id": frame_id,
                "time_seconds": time_seconds,
                "frame_rate": frame_rate,
                "player_id": "real_gk",
                "team_id": "home",
                "is_ball": False,
                "is_goalkeeper": False,
                "x": 5.0 + np.random.normal(0, 1),
                "y": 34.0 + np.random.normal(0, 2),
                "z": 0.0,
                "speed": 1.0,
                "speed_source": "native",
                "ball_state": "alive",
                "team_attacking_direction": "ltr",
                "confidence": None,
                "visibility": None,
                "source_provider": "synthetic",
            }
        )

        # Brief outfielder appears only in last 50 frames, positioned in PA
        if frame_id >= brief_start:
            rows.append(
                {
                    "game_id": "brief_match",
                    "period_id": 1,
                    "frame_id": frame_id,
                    "time_seconds": time_seconds,
                    "frame_rate": frame_rate,
                    "player_id": "brief_sub_near_goal",
                    "team_id": "home",
                    "is_ball": False,
                    "is_goalkeeper": False,
                    "x": 8.0,  # In PA
                    "y": 34.0,
                    "z": 0.0,
                    "speed": 2.0,
                    "speed_source": "native",
                    "ball_state": "alive",
                    "team_attacking_direction": "ltr",
                    "confidence": None,
                    "visibility": None,
                    "source_provider": "synthetic",
                }
            )

        # 9 outfielders (or 10 when brief sub not present)
        n_outfielders = 9 if frame_id >= brief_start else 10
        for i in range(n_outfielders):
            rows.append(
                {
                    "game_id": "brief_match",
                    "period_id": 1,
                    "frame_id": frame_id,
                    "time_seconds": time_seconds,
                    "frame_rate": frame_rate,
                    "player_id": f"outfielder_{i}",
                    "team_id": "home",
                    "is_ball": False,
                    "is_goalkeeper": False,
                    "x": 40.0 + i * 5,
                    "y": 15.0 + i * 5,
                    "z": 0.0,
                    "speed": 4.0,
                    "speed_source": "native",
                    "ball_state": "alive",
                    "team_attacking_direction": "ltr",
                    "confidence": None,
                    "visibility": None,
                    "source_provider": "synthetic",
                }
            )

        # Ball row
        rows.append(
            {
                "game_id": "brief_match",
                "period_id": 1,
                "frame_id": frame_id,
                "time_seconds": time_seconds,
                "frame_rate": frame_rate,
                "player_id": None,
                "team_id": None,
                "is_ball": True,
                "is_goalkeeper": False,
                "x": np.clip(52.5 + np.random.normal(0, 10), 0, 105),
                "y": np.clip(34.0 + np.random.normal(0, 10), 0, 68),
                "z": 0.0,
                "speed": 5.0,
                "speed_source": "native",
                "ball_state": "alive",
                "team_attacking_direction": None,
                "confidence": None,
                "visibility": None,
                "source_provider": "synthetic",
            }
        )

    return pd.DataFrame(rows)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Set seed for reproducibility
    np.random.seed(42)

    fixtures = {
        "gk_substitution.parquet": _build_gk_substitution,
        "sweeper_keeper.parquet": _build_sweeper_keeper,
        "brief_outfielder.parquet": _build_brief_outfielder,
    }

    for filename, builder in fixtures.items():
        df = builder()
        path = OUTPUT_DIR / filename
        df.to_parquet(path, index=False)
        print(f"Wrote {path.relative_to(REPO_ROOT)} ({len(df):,} rows, {path.stat().st_size / 1024:.1f} KB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
