"""Generate synthetic Sportec-shaped raw input for tracking adapter tests.

Reads ``empirical_probe_baselines.json`` for sportec stats. Emits two
parquet files: ``tiny.parquet`` (~3 s) and ``medium_halftime.parquet``
(~60 s spanning HT).

Sportec input shape (mirrors what callers parse from DFL Position XML):
  - period_id, frame_id, time_seconds, frame_rate
  - player_id (DFL PersonId string), team_id (DFL TeamId string), game_id (DFL MatchId string)
  - is_ball, is_goalkeeper
  - x_centered, y_centered (DFL pitch-centered meters)
  - z, speed_native (DFL FrameSet/Frame.S, m/s)
  - ball_state ("alive" | "dead", from DFL BallStatus)

Run ``python tests/datasets/tracking/sportec/generate_synthetic.py``
during PR-S19 development to (re)build the parquets. Re-running the
script is deterministic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Allow running as a standalone script (the parent ``tests`` dir is
# added to sys.path automatically when imported via pytest).
_TESTS_DIR = Path(__file__).resolve().parents[3]
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from datasets.tracking._generator_common import (  # noqa: E402
    deterministic_uniform_motion,
    get_provider_baseline,
)

OUT_DIR = Path(__file__).resolve().parent
BASELINE = get_provider_baseline("sportec")
FRAME_RATE = float(BASELINE.get("frame_rate_p50") or 25.0)


def _to_sportec_shape(ref: pd.DataFrame, *, game_id: str = "DFL-MAT-0001") -> pd.DataFrame:
    out = ref.copy()
    out["game_id"] = game_id
    out["player_id"] = out["player_id"].apply(
        lambda v: f"DFL-OBJ-{int(v):05d}" if pd.notna(v) else None,
    )
    out["team_id"] = out["team_id"].apply(
        lambda v: f"DFL-CLU-{int(v):04d}" if pd.notna(v) else None,
    )
    return out


def main() -> None:
    tiny_ref = deterministic_uniform_motion(
        n_frames=int(3 * FRAME_RATE),
        frame_rate=FRAME_RATE,
        period_id=1,
    )
    tiny = _to_sportec_shape(tiny_ref)
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
    medium = pd.concat([_to_sportec_shape(p1), _to_sportec_shape(p2)], ignore_index=True)
    dead_mask = (medium["period_id"] == 1) & (medium["time_seconds"].between(10, 15))
    medium.loc[dead_mask, "ball_state"] = "dead"
    medium.to_parquet(OUT_DIR / "medium_halftime.parquet", index=False)

    # realistic.parquet: 20 s of period 1 with empirical-baseline-calibrated
    # edge cases (off-pitch player tail, ball-out interval, ball-x throw-in
    # tail). Covers the failure modes the cross-provider parity gate should
    # tolerate but the strict synthetic gate may not.
    realistic = deterministic_uniform_motion(
        n_frames=int(20 * FRAME_RATE),
        frame_rate=FRAME_RATE,
        period_id=1,
        t0=0.0,
        seed=11,
        inject_realistic_edge_cases=True,
        edge_case_provider="sportec",
    )
    realistic = _to_sportec_shape(realistic)
    realistic.to_parquet(OUT_DIR / "realistic.parquet", index=False)

    print(f"Wrote {OUT_DIR / 'tiny.parquet'} ({len(tiny)} rows)")
    print(f"Wrote {OUT_DIR / 'medium_halftime.parquet'} ({len(medium)} rows)")
    print(f"Wrote {OUT_DIR / 'realistic.parquet'} ({len(realistic)} rows)")


if __name__ == "__main__":
    main()
