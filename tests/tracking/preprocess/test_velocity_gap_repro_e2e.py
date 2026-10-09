"""TF-65 §4.5(2): named real repros via pining (owner-tier broadcast; @e2e, skip-if-token-unset).

GS 10502 p1 is THE ball-gap repro (~930-frame non-detection at 57169-58109). WC2022 3851 is NOT a
ball gap -- it is the single-frame PLAYER case named in _velocity.py:71-73 (away #10, exactly 1 frame
in p2) -> velocity NaN. These run owner-side with PINING_FOR_THE_DATA_TOKEN; CI skips them.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.e2e

_SKIP = pytest.mark.skipif(not os.environ.get("PINING_FOR_THE_DATA_TOKEN"), reason="pining token unset (owner-tier GS)")


def _load_gs_frames_via_pining(match_id: str):
    # Raw tracking frames for one GS match via the pining loader (urllib two-step, owner token from
    # PINING_FOR_THE_DATA_TOKEN). No committed GS raw, no local path.
    import sys
    from pathlib import Path

    scripts_dir = Path(__file__).resolve().parents[3] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from _loader_pining import list_match_refs, load_match

    refs = list_match_refs(providers=["gradientsports"], match_ids={"gradientsports": [str(match_id)]})
    ref = next(r for r in refs if str(r.match_id) == str(match_id))
    frames = load_match(ref, events_only=False).frames
    assert frames is not None, "full GS load must build frames"  # narrows DataFrame | None
    return frames


@_SKIP
def test_gs_10502_ball_gap_no_spike():
    frames = _load_gs_frames_via_pining("10502")
    from silly_kicks.tracking.preprocess import PreprocessConfig, derive_velocities, smooth_frames

    cfg = PreprocessConfig.for_provider("gradientsports")
    v = derive_velocities(smooth_frames(frames, config=cfg), config=cfg)
    ball = v[(v["is_ball"].astype(bool)) & (v["period_id"] == 1)]
    assert (ball["speed"].dropna() <= cfg.max_plausible_speed).all()  # the 500-850 m/s fabrication is gone


@_SKIP
def test_gs_3851_single_frame_player_nan():
    frames = _load_gs_frames_via_pining("3851")
    from silly_kicks.tracking.preprocess import PreprocessConfig, derive_velocities, smooth_frames

    cfg = PreprocessConfig.for_provider("gradientsports")
    v = derive_velocities(smooth_frames(frames, config=cfg), config=cfg)
    outfield_p2 = v[(~v["is_ball"].astype(bool)) & (v["period_id"] == 2)]
    single = outfield_p2.groupby("player_id").filter(lambda g: len(g) == 1)
    assert single["speed"].isna().all()  # single-frame player -> NaN velocity (no fabrication)
