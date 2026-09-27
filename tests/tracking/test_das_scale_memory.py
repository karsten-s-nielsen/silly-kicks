"""DAS scale + memory guards (ADR-107, spec §13).

* ``pack_frames`` + ``compute_das`` scan work grows sub-quadratically in the frame count (a
  rescan-in-loop or per-frame ``frames[frames.frame_id == fid]`` would go quadratic).
* ``compute_das``'s WORKING memory (transient per-chunk buffers, i.e. peak minus the output
  arrays) is bounded by ``chunk_size``, NOT by the total frame count -- 10x the frames at the
  same chunk_size must not grow the working peak.
"""

from __future__ import annotations

import tracemalloc

import pandas as pd
import pytest

from silly_kicks.tracking._das_engine import compute_das
from silly_kicks.tracking._das_pack import pack_frames
from silly_kicks.tracking._das_params import DAS_PARAMS
from tests._perf_structural import assert_subquadratic_growth, rows_scanned_counter
from tests.tracking._das_helpers import single_frame


def _synth(n_frames: int, *, n_per_team: int = 4) -> pd.DataFrame:
    """``n_frames`` synthetic frames (each a ``single_frame`` with its own frame_id + ``dir`` col)."""
    return pd.concat(
        [single_frame(frame=fid, n1=n_per_team, n2=n_per_team, seed=fid) for fid in range(n_frames)],
        ignore_index=True,
    )


def test_pack_and_compute_das_scan_is_subquadratic():
    def measure_work(n: int) -> int:
        frames = _synth(n)
        with rows_scanned_counter() as c:
            packed = pack_frames(frames, attacking_direction_col="dir")
            compute_das(packed, DAS_PARAMS, engine="numpy")
        return c["n"]

    assert_subquadratic_growth(measure_work, sizes=(64, 256, 1024), label="pack_frames+compute_das")


def _working_peak_bytes(n_frames: int, *, chunk_size: int) -> int:
    frames = _synth(n_frames)
    packed = pack_frames(frames, attacking_direction_col="dir")
    tracemalloc.start()
    try:
        res = compute_das(packed, DAS_PARAMS, chunk_size=chunk_size, engine="numpy")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    output = int(res.team_as.nbytes + res.team_das.nbytes + res.player_as.nbytes + res.player_das.nbytes)
    return int(peak) - output


@pytest.mark.slow
def test_working_memory_is_bounded_by_chunk_not_total_frames():
    chunk = 256
    small = _working_peak_bytes(2_000, chunk_size=chunk)
    big = _working_peak_bytes(20_000, chunk_size=chunk)
    assert big <= 1.1 * small, (
        f"working peak grew with total frames at fixed chunk_size={chunk}: "
        f"2k={small} bytes, 20k={big} bytes (ratio {big / max(small, 1):.2f}) -- the per-chunk "
        "buffers must not scale with the total frame count."
    )
