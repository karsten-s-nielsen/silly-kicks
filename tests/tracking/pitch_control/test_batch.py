"""ADR-103 F2: compute_pitch_control_batch is byte-identical to the per-frame loop; dedups + groups once."""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.tracking import pitch_control as pc
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
    frames = _frames((10, 11))
    # 4 requests but only 2 distinct (frame,team,decompose) keys
    reqs = [((1, 1, 10), 1, False), ((1, 1, 10), 1, False), ((1, 1, 11), 1, False), ((1, 1, 11), 1, False)]
    calls = call_counter(monkeypatch, pc._dispatch, "compute_pitch_control")
    out = compute_pitch_control_batch(frames, reqs, method="spearman")
    assert len(out) == 4
    assert calls["n"] == 2  # each distinct surface computed ONCE (dedup)


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
