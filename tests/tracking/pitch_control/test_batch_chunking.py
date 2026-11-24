"""W3: memory-bounded chunks and a public frame index (spec 2026-11-08 tracking-batch §4.3, plan Task 8)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts._bitwise import assert_bits_equal
from silly_kicks.tracking.pitch_control import (
    PitchControlFrameIndex,
    build_pitch_control_index,
    compute_pitch_control,
    compute_pitch_control_batch,
    compute_pitch_control_batch_from_index,
)
from tests._perf_structural import call_counter
from tests._prod_fixtures import make_production_frames

_KEYS = ("game_id", "period_id", "frame_id")
_FIELDS = ("surface", "attacking_team_id", "per_player_influence", "player_ids", "player_team_ids")


def _frames() -> pd.DataFrame:
    frames = make_production_frames(n_frames=9, n_periods=2)
    rng = np.random.default_rng(20261111)
    n = len(frames)
    vx, vy = rng.normal(0.0, 2.5, n), rng.normal(0.0, 2.5, n)
    return frames.assign(vx=vx, vy=vy)


def _requests(frames: pd.DataFrame) -> list:
    keys = [tuple(k) for k in frames[list(_KEYS)].drop_duplicates().itertuples(index=False, name=None)]
    reqs = [(k, team, i % 2 == 0) for i, k in enumerate(keys) for team in (5, 6)]
    return reqs + reqs[:3]  # duplicates spread over the chunks


def _assert_same(new: list, old: list) -> None:
    assert len(new) == len(old) > 0
    for n, o in zip(new, old, strict=True):
        for field in _FIELDS:
            a, b = getattr(n, field), getattr(o, field)
            if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
                assert_bits_equal(a, b)
            else:
                assert a == b and type(a) is type(b), field


@pytest.mark.parametrize("method", ["spearman", "voronoi", "fernandez_bornn"])
@pytest.mark.parametrize("size", [1, 7, 2048, None])
def test_chunk_sizes_bit_identical(method, size):
    frames = _frames()
    requests = _requests(frames)
    one_chunk = compute_pitch_control_batch(frames, requests, method=method, max_frames_per_chunk=None)
    chunked = compute_pitch_control_batch(frames, requests, method=method, max_frames_per_chunk=size)
    _assert_same(chunked, one_chunk)


@pytest.mark.parametrize("method", ["spearman", "voronoi", "fernandez_bornn"])
def test_index_path_equals_frames_path(method):
    frames = _frames()
    requests = _requests(frames)
    index = build_pitch_control_index(frames)
    assert isinstance(index, PitchControlFrameIndex)
    first = compute_pitch_control_batch_from_index(index, requests[:5], method=method, max_frames_per_chunk=3)
    second = compute_pitch_control_batch_from_index(index, requests[5:], method=method, max_frames_per_chunk=3)
    _assert_same(first + second, compute_pitch_control_batch(frames, requests, method=method))


@pytest.mark.parametrize("decompose", [False, True])
def test_ball_as_first_row_of_its_frame(decompose):
    """Reviewer C TBC-TEST-11 (probe M2): the index finds a frame's ball row also when it is the frame's FIRST row.
    The batch call equals the per-frame public call, field by field, on frames whose ball row comes first."""
    frames = _frames()
    ball_last = ~frames["is_ball"].astype(bool)
    ball_first = (
        frames.assign(_ball_last=ball_last)
        .sort_values([*_KEYS, "_ball_last"], kind="stable")
        .drop(columns="_ball_last")
        .reset_index(drop=True)
    )
    grouped = ball_first.groupby(list(_KEYS), sort=False)
    assert grouped.head(1)["is_ball"].astype(bool).all()  # precondition: every frame starts with its ball row
    assert not frames.groupby(list(_KEYS), sort=False).head(1)["is_ball"].astype(bool).all()  # the order moved
    keys = [tuple(k) for k in ball_first[list(_KEYS)].drop_duplicates().itertuples(index=False, name=None)]
    requests = [(k, team, decompose) for k in keys for team in (5, 6)]
    batch = compute_pitch_control_batch(ball_first, requests, method="spearman")
    per_frame = [
        compute_pitch_control(grouped.get_group(k), team, method="spearman", decompose=d) for k, team, d in requests
    ]
    _assert_same(batch, per_frame)


def test_group_rows_called_once_per_call(monkeypatch):
    from silly_kicks.tracking.pitch_control import _dispatch

    frames = _frames()
    calls = call_counter(monkeypatch, _dispatch, "group_rows")
    compute_pitch_control_batch(frames, _requests(frames), method="spearman", max_frames_per_chunk=2)
    assert calls["n"] == 1  # however many chunks
    index_calls = call_counter(monkeypatch, _dispatch, "group_rows")
    index = build_pitch_control_index(frames)
    compute_pitch_control_batch_from_index(index, _requests(frames), max_frames_per_chunk=2)
    compute_pitch_control_batch_from_index(index, _requests(frames), max_frames_per_chunk=5)
    assert index_calls["n"] == 1  # the index groups once, for every wave


@pytest.mark.parametrize("bad", [0, -1, 2.5, True])
def test_invalid_chunk_size_raises(bad):
    frames = _frames()
    with pytest.raises(ValueError, match="max_frames_per_chunk"):
        compute_pitch_control_batch(frames, _requests(frames), max_frames_per_chunk=bad)
    with pytest.raises(ValueError, match="max_frames_per_chunk"):
        compute_pitch_control_batch_from_index(
            build_pitch_control_index(frames), _requests(frames), max_frames_per_chunk=bad
        )


def test_zero_requests_returns_empty_list():  # Review Focus 1
    frames = _frames()
    assert compute_pitch_control_batch(frames, []) == []
    assert compute_pitch_control_batch_from_index(build_pitch_control_index(frames), []) == []
    assert compute_pitch_control_batch(frames.iloc[:0], []) == []
