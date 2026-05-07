"""Generate synthetic PFF-shaped raw tracking input.

PFF input shape (mirrors what callers parse from .jsonl.bz2 + flatten):
  - game_id (int), period_id (int), frame_id (int), time_seconds (float)
  - frame_rate (float, ~30 Hz)
  - player_id (Int64 nullable, NaN on ball rows), team_id (Int64 nullable)
  - is_ball, is_goalkeeper
  - x_centered, y_centered (float, PFF meters; 0 at pitch center)
  - z (float, populated for ball rows on most frames)
  - speed_native (NaN — PFF does not supply native speed)
  - ball_state (object, "alive" | "dead")
  - jersey (str, real shirt numbers like "8", "23")

Real PFF data properties (validated against 10502.jsonl.bz2):
  - frameNum is globally unique across periods (P1: 5366+, P2: 90535+)
  - 11 players per team per frame (not 22)
  - jerseyNum is str type with real shirt numbers
  - No speed field in raw tracking JSONL
  - confidence: LOW/MEDIUM/HIGH per player
  - visibility: VISIBLE/ESTIMATED per player
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_TESTS_DIR = Path(__file__).resolve().parents[3]
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from datasets.tracking._generator_common import (  # noqa: E402
    deterministic_uniform_motion,
    get_provider_baseline,
)

OUT_DIR = Path(__file__).resolve().parent
BASELINE = get_provider_baseline("pff")
FRAME_RATE = float(BASELINE.get("frame_rate_p50") or 30.0)

# Real PFF jersey numbers from WC2022 match 10502 (home team first 100 frames).
# Used to make synthetic data structurally match real data (str type, realistic values).
_HOME_JERSEYS = ["2", "4", "5", "8", "10", "14", "15", "17", "21", "22", "23"]
_AWAY_JERSEYS = ["1", "3", "6", "7", "9", "11", "13", "16", "18", "19", "25"]
_CONFIDENCE_VALUES = ["LOW", "MEDIUM", "HIGH"]
_VISIBILITY_VALUES = ["VISIBLE", "ESTIMATED"]


def _to_pff_shape(ref: pd.DataFrame, *, game_id: int = 10501) -> pd.DataFrame:
    """Shape generator output to match real PFF flattened schema."""
    out = ref.copy()
    out["game_id"] = game_id
    out["player_id"] = out["player_id"].astype("Int64")
    out["team_id"] = out["team_id"].astype("Int64")

    # Map sequential jersey indices to realistic string jersey numbers.
    rng = np.random.default_rng(42)
    jersey_map: dict[tuple[int | None, int | None], str] = {}
    for _, row in out[~out["is_ball"]].drop_duplicates(["team_id", "jersey"]).iterrows():
        tid = row["team_id"]
        j = row["jersey"]
        pool = _HOME_JERSEYS if tid == 100 else _AWAY_JERSEYS
        idx = int(j) if pd.notna(j) else 0
        jersey_map[(tid, int(j) if pd.notna(j) else None)] = pool[idx % len(pool)]

    def _map_jersey(row: pd.Series) -> str | None:
        if row["is_ball"]:
            return None
        return jersey_map.get((row["team_id"], int(row["jersey"]) if pd.notna(row["jersey"]) else None))

    out["jersey"] = out.apply(_map_jersey, axis=1)

    # Add confidence and visibility (present in real PFF data).
    player_mask = ~out["is_ball"]
    out["confidence"] = pd.NA
    out["visibility"] = pd.NA
    out.loc[player_mask, "confidence"] = rng.choice(_CONFIDENCE_VALUES, size=player_mask.sum())
    out.loc[player_mask, "visibility"] = rng.choice(_VISIBILITY_VALUES, size=player_mask.sum(), p=[0.85, 0.15])

    return out


def main() -> None:
    # --- tiny: 3 seconds, single period ---
    # Real PFF frameNum starts at ~5000, not 0.
    tiny_ref = deterministic_uniform_motion(
        n_frames=int(3 * FRAME_RATE),
        frame_rate=FRAME_RATE,
        period_id=1,
        frame_id_offset=5000,
        speed_native=False,
    )
    tiny = _to_pff_shape(tiny_ref)
    tiny.to_parquet(OUT_DIR / "tiny.parquet", index=False)

    # --- medium_halftime: 30s P1 + 30s P2, globally unique frame_ids ---
    # Real data: P1 starts ~5366, P2 starts ~90535 (large gap between periods).
    p1_n_frames = int(30 * FRAME_RATE)
    p1 = deterministic_uniform_motion(
        n_frames=p1_n_frames,
        frame_rate=FRAME_RATE,
        period_id=1,
        t0=0.0,
        seed=1,
        frame_id_offset=5000,
        speed_native=False,
    )
    p2 = deterministic_uniform_motion(
        n_frames=int(30 * FRAME_RATE),
        frame_rate=FRAME_RATE,
        period_id=2,
        t0=0.0,
        seed=2,
        frame_id_offset=5000 + p1_n_frames + 1000,  # gap mimics real halftime break
        speed_native=False,
    )
    medium = pd.concat([_to_pff_shape(p1), _to_pff_shape(p2)], ignore_index=True)
    dead_mask = (medium["period_id"] == 1) & (medium["time_seconds"].between(10, 15))
    medium.loc[dead_mask, "ball_state"] = "dead"
    medium.to_parquet(OUT_DIR / "medium_halftime.parquet", index=False)

    # --- realistic: 20s P1, edge cases ---
    realistic = deterministic_uniform_motion(
        n_frames=int(20 * FRAME_RATE),
        frame_rate=FRAME_RATE,
        period_id=1,
        t0=0.0,
        seed=11,
        frame_id_offset=5000,
        speed_native=False,
        inject_realistic_edge_cases=True,
        edge_case_provider="pff",
    )
    realistic = _to_pff_shape(realistic)
    realistic.to_parquet(OUT_DIR / "realistic.parquet", index=False)

    print(f"Wrote {OUT_DIR / 'tiny.parquet'} ({len(tiny)} rows)")
    print(f"Wrote {OUT_DIR / 'medium_halftime.parquet'} ({len(medium)} rows)")
    print(f"Wrote {OUT_DIR / 'realistic.parquet'} ({len(realistic)} rows)")


if __name__ == "__main__":
    main()
