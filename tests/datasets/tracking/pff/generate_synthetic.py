"""Generate synthetic PFF-shaped raw tracking input.

PFF input shape (mirrors what callers parse from .jsonl.bz2 + flatten):
  - game_id (int), period_id (int), frame_id (int), time_seconds (float)
  - frame_rate (float, ~30 Hz)
  - player_id (Int64 nullable, NaN on ball rows), team_id (Int64 nullable)
  - is_ball, is_goalkeeper
  - x_centered, y_centered (float, PFF meters; 0 at pitch center)
  - z (float, populated for ball rows on most frames)
  - speed_native (float, m/s, supplied by PFF)
  - ball_state (object, "alive" | "dead")
"""

from __future__ import annotations

import sys
from pathlib import Path

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


def _to_pff_shape(ref: pd.DataFrame, *, game_id: int = 10501) -> pd.DataFrame:
    out = ref.copy()
    out["game_id"] = game_id
    out["player_id"] = out["player_id"].astype("Int64")
    out["team_id"] = out["team_id"].astype("Int64")
    return out


def main() -> None:
    tiny_ref = deterministic_uniform_motion(
        n_frames=int(3 * FRAME_RATE),
        frame_rate=FRAME_RATE,
        period_id=1,
    )
    tiny = _to_pff_shape(tiny_ref)
    tiny.to_parquet(OUT_DIR / "tiny.parquet", index=False)

    p1 = deterministic_uniform_motion(
        n_frames=int(30 * FRAME_RATE),
        frame_rate=FRAME_RATE,
        period_id=1,
        t0=0.0,
        seed=1,
    )
    p2 = deterministic_uniform_motion(
        n_frames=int(30 * FRAME_RATE),
        frame_rate=FRAME_RATE,
        period_id=2,
        t0=0.0,
        seed=2,
    )
    medium = pd.concat([_to_pff_shape(p1), _to_pff_shape(p2)], ignore_index=True)
    dead_mask = (medium["period_id"] == 1) & (medium["time_seconds"].between(10, 15))
    medium.loc[dead_mask, "ball_state"] = "dead"
    medium.to_parquet(OUT_DIR / "medium_halftime.parquet", index=False)

    realistic = deterministic_uniform_motion(
        n_frames=int(20 * FRAME_RATE),
        frame_rate=FRAME_RATE,
        period_id=1,
        t0=0.0,
        seed=11,
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
