"""Synthetic in-memory SkillCorner TrackingDataset builder for kloppy-gateway tests."""

from __future__ import annotations

import sys
from pathlib import Path

from kloppy.domain import Provider, TrackingDataset

_TESTS_DIR = Path(__file__).resolve().parents[3]
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from datasets.tracking._generator_common import get_provider_baseline  # noqa: E402
from datasets.tracking.metrica.generate_synthetic import (  # noqa: E402
    _build_synthetic_tracking_dataset,
)

BASELINE = get_provider_baseline("skillcorner")
FRAME_RATE = float(BASELINE.get("frame_rate_p50") or 10.0)


def build_skillcorner_tracking_dataset(
    n_frames: int = 30,
    *,
    inject_realistic_edge_cases: bool = False,
) -> TrackingDataset:
    """Build a synthetic SkillCorner TrackingDataset (default 30 frames at 10 Hz = 3 s).

    Set ``inject_realistic_edge_cases=True`` for empirical-baseline-
    calibrated NaN ball coords + off-pitch / ball-out / ball-x-throw-in
    edge cases.
    """
    return _build_synthetic_tracking_dataset(
        provider=Provider.SKILLCORNER,
        frame_rate=FRAME_RATE,
        n_frames=n_frames,
        inject_realistic_edge_cases=inject_realistic_edge_cases,
        edge_case_provider="skillcorner",
    )


if __name__ == "__main__":
    ds = build_skillcorner_tracking_dataset()
    print(f"Built TrackingDataset with {len(ds.records)} frames, {len(ds.metadata.teams[0].players)} home players")
    ds_realistic = build_skillcorner_tracking_dataset(n_frames=200, inject_realistic_edge_cases=True)
    print(f"Realistic variant: {len(ds_realistic.records)} frames")
