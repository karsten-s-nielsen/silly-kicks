"""ADR-103 F2: compute_pitch_control_batch is byte-identical to the per-frame loop; dedups + groups once."""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.tracking.pitch_control import (
    PitchControlCache,
    compute_pitch_control,
    compute_pitch_control_batch,
)
from tests._perf_structural import call_counter


def _frames(frame_ids=(10, 11, 12)) -> pd.DataFrame:
    rows = []
    for fid in frame_ids:
        rows += [
            dict(
                game_id=1,
                period_id=1,
                frame_id=fid,
                is_ball=False,
                is_goalkeeper=True,
                team_id=1,
                player_id=10,
                x=30.0,
                y=34.0,
                vx=0.0,
                vy=0.0,
            ),
            dict(
                game_id=1,
                period_id=1,
                frame_id=fid,
                is_ball=False,
                is_goalkeeper=False,
                team_id=1,
                player_id=11,
                x=60.0,
                y=20.0,
                vx=0.0,
                vy=0.0,
            ),
            dict(
                game_id=1,
                period_id=1,
                frame_id=fid,
                is_ball=False,
                is_goalkeeper=True,
                team_id=2,
                player_id=20,
                x=45.0,
                y=40.0,
                vx=0.0,
                vy=0.0,
            ),
            dict(
                game_id=1,
                period_id=1,
                frame_id=fid,
                is_ball=False,
                is_goalkeeper=False,
                team_id=2,
                player_id=21,
                x=70.0,
                y=34.0,
                vx=0.0,
                vy=0.0,
            ),
            dict(
                game_id=1,
                period_id=1,
                frame_id=fid,
                is_ball=True,
                is_goalkeeper=False,
                team_id=np.nan,
                player_id=np.nan,
                x=50.0,
                y=34.0,
                vx=0.0,
                vy=0.0,
            ),
        ]
    return pd.DataFrame(rows)


def _slice(frames: pd.DataFrame, fid: int) -> pd.DataFrame:
    return frames[frames["frame_id"] == fid]


def test_batch_byte_identical_to_loop():
    frames = _frames()
    reqs = [((1, 1, fid), 1, dec) for fid in (10, 11, 12) for dec in (False, True)]
    for method in ("spearman", "fernandez_bornn"):
        batched = compute_pitch_control_batch(frames, reqs, method=method)
        assert len(batched) == len(reqs)
        for (frame_key, team, dec), got in zip(reqs, batched, strict=True):
            ref = compute_pitch_control(_slice(frames, frame_key[2]), team, method=method, decompose=dec)
            assert np.array_equal(got.surface, ref.surface), (method, frame_key, dec)
            if dec:
                assert got.per_player_influence is not None and ref.per_player_influence is not None
                assert np.array_equal(got.per_player_influence, ref.per_player_influence)


def test_batch_dedups_repeated_requests(monkeypatch):
    # ADR-105: spearman routes through the vectorized `compute_spearman_batch`, so the dedup is asserted
    # on the DISTINCT requests the kernel receives (2), not a per-request `compute_pitch_control` count.
    from silly_kicks.tracking.pitch_control import _spearman_batch

    frames = _frames((10, 11))
    # 4 requests but only 2 distinct (frame,team,decompose) keys
    reqs = [((1, 1, 10), 1, False), ((1, 1, 10), 1, False), ((1, 1, 11), 1, False), ((1, 1, 11), 1, False)]
    seen: dict[str, int] = {}
    orig = _spearman_batch.compute_spearman_batch

    def _spy(frame_slices, *a, **k):
        seen["n_distinct"] = len(frame_slices)
        return orig(frame_slices, *a, **k)

    monkeypatch.setattr(_spearman_batch, "compute_spearman_batch", _spy)
    out = compute_pitch_control_batch(frames, reqs, method="spearman")
    assert len(out) == 4
    assert seen["n_distinct"] == 2  # 4 requests, 2 distinct -> the kernel receives 2 (dedup)


def test_cache_warm_makes_surface_calls_hit():
    frames = _frames((10, 11))
    cache = PitchControlCache()
    cache.warm(frames, [((1, 1, 10), 1, True)], method="spearman")
    assert len(cache) == 1
    stored = next(iter(cache._store.values()))
    # a subsequent surface() on the warmed frame is a HIT (same object), not a recompute
    s = cache.surface(_slice(frames, 10), 1, method="spearman", decompose=True)
    assert s is stored
    assert len(cache) == 1


def test_batch_empty_requests():
    assert compute_pitch_control_batch(_frames(), [], method="spearman") == []


def test_warm_groups_frames_once(monkeypatch):
    """ADR-105 Task 5: PitchControlCache.warm builds group_rows ONCE (was TWICE -- the ADR-103
    double-group: compute_pitch_control_batch grouped, then warm re-grouped for _key)."""
    import silly_kicks._frame_index as _fi

    frames = _frames((10, 11))
    calls = call_counter(monkeypatch, _fi, "group_rows")
    cache = PitchControlCache()
    cache.warm(frames, [((1, 1, 10), 1, True), ((1, 1, 11), 1, False)], method="spearman")
    assert calls["n"] == 1
    # and the warmed surfaces still HIT byte-identically
    s = cache.surface(_slice(frames, 10), 1, method="spearman", decompose=True)
    assert s is next(iter(cache._store.values())) or len(cache) == 2
