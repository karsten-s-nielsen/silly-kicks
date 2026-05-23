"""Parity tests: numba kernel produces identical output to Python fallback.

numba is a test dependency (pyproject.toml [test] extra) — these always run in CI.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest


def _build_dense_fixture(
    *,
    n_frames: int = 5,
    max_players: int = 3,
    has_velocity: bool = True,
    include_dead: bool = False,
    include_nan_velocity: bool = False,
    n_segments: int = 1,
) -> dict:
    """Build dense pre-indexed arrays for kernel tests.

    Returns a dict of all arrays needed by _carrier_loop_numpy / _carrier_loop_numba.
    """
    bx = np.array([50.0] * n_frames)
    by = np.array([34.0] * n_frames)
    ball_dead = np.zeros(n_frames, dtype=np.bool_)
    if include_dead and n_frames >= 3:
        ball_dead[1] = True  # frame 1 is dead

    px = np.full((n_frames, max_players), np.nan)
    py = np.full((n_frames, max_players), np.nan)
    pvx = np.full((n_frames, max_players), np.nan)
    pvy = np.full((n_frames, max_players), np.nan)
    player_slots = np.full((n_frames, max_players), -1, dtype=np.int64)
    n_valid = np.zeros(n_frames, dtype=np.int64)

    for f in range(n_frames):
        # Player 0: at (51, 34) — 1m from ball
        px[f, 0] = 51.0
        py[f, 0] = 34.0
        pvx[f, 0] = 0.0
        pvy[f, 0] = 0.0
        player_slots[f, 0] = 0
        # Player 1: at (52, 34) — 2m from ball
        px[f, 1] = 52.0
        py[f, 1] = 34.0
        pvx[f, 1] = 0.0
        pvy[f, 1] = 0.0
        player_slots[f, 1] = 1
        n_valid[f] = 2

    if include_nan_velocity and n_frames >= 4:
        pvx[3, 0] = np.nan
        pvy[3, 0] = np.nan

    seg_starts = np.array([0], dtype=np.int64)
    seg_ends = np.array([n_frames], dtype=np.int64)
    if n_segments == 2 and n_frames >= 4:
        mid = n_frames // 2
        seg_starts = np.array([0, mid], dtype=np.int64)
        seg_ends = np.array([mid, n_frames], dtype=np.int64)

    return dict(
        bx=bx,
        by=by,
        ball_dead=ball_dead,
        px=px,
        py=py,
        pvx=pvx,
        pvy=pvy,
        player_slots=player_slots,
        n_valid=n_valid,
        seg_starts=seg_starts,
        seg_ends=seg_ends,
        tolerance_m=3.0,
        beta=0.5,
        gamma=1.0,
        has_velocity=has_velocity,
    )


class TestPythonKernelBasic:
    def test_basic_winner_selection(self):
        """Closest player wins in distance-only mode."""
        from silly_kicks.tracking._ball_carrier import _carrier_loop_numpy

        arrays = _build_dense_fixture(n_frames=1, has_velocity=False)
        winner_slot, winner_dist = _carrier_loop_numpy(**arrays)
        assert winner_slot[0] == 0  # player 0 at 1m beats player 1 at 2m
        np.testing.assert_allclose(winner_dist[0], 1.0, atol=1e-9)

    def test_dead_ball_produces_minus_one(self):
        """Dead ball frames produce winner_slot=-1."""
        from silly_kicks.tracking._ball_carrier import _carrier_loop_numpy

        arrays = _build_dense_fixture(n_frames=3, include_dead=True, has_velocity=False)
        winner_slot, winner_dist = _carrier_loop_numpy(**arrays)
        assert winner_slot[1] == -1
        assert np.isnan(winner_dist[1])

    def test_hysteresis_retains_incumbent(self):
        """Incumbent keeps carrier even when slightly farther."""
        from silly_kicks.tracking._ball_carrier import _carrier_loop_numpy

        arrays = _build_dense_fixture(n_frames=2, has_velocity=False)
        # Frame 0: player 0 at 1m wins.
        # Frame 1: swap — player 0 at 2m, player 1 at 1.8m.
        # Difference (0.2m) < gamma (1.0m) → incumbent (0) retained.
        arrays["px"][1, 0] = 52.0  # player 0 now at 2m
        arrays["py"][1, 0] = 34.0
        arrays["px"][1, 1] = 51.8  # player 1 now at 1.8m
        arrays["py"][1, 1] = 34.0
        winner_slot, _ = _carrier_loop_numpy(**arrays)
        assert winner_slot[0] == 0
        assert winner_slot[1] == 0  # incumbent retained

    def test_segment_boundary_resets_incumbent(self):
        """New segment resets incumbent — no carry-over across periods."""
        from silly_kicks.tracking._ball_carrier import _carrier_loop_numpy

        arrays = _build_dense_fixture(n_frames=4, n_segments=2, has_velocity=False)
        # Segment 0 frames [0,1]: player 0 wins both (closer).
        # Segment 1 frames [2,3]: player 1 closer, should win (no incumbent carry-over).
        arrays["px"][2, 0] = 52.5
        arrays["px"][2, 1] = 50.5
        arrays["px"][3, 0] = 52.5
        arrays["px"][3, 1] = 50.5
        winner_slot, _ = _carrier_loop_numpy(**arrays)
        assert winner_slot[0] == 0  # seg 0
        assert winner_slot[2] == 1  # seg 1 — no incumbent

    def test_nan_velocity_treated_as_zero(self):
        """NaN velocity → 0.0 velocity-toward-ball, not NaN propagation."""
        from silly_kicks.tracking._ball_carrier import _carrier_loop_numpy

        arrays = _build_dense_fixture(n_frames=5, include_nan_velocity=True, has_velocity=True)
        winner_slot, winner_dist = _carrier_loop_numpy(**arrays)
        # Frame 3 has NaN velocity on player 0 — should still produce a valid winner
        assert winner_slot[3] >= 0
        assert not np.isnan(winner_dist[3])

    def test_tiebreak_lowest_slot(self):
        """Equal scores → lowest player_slots value wins."""
        from silly_kicks.tracking._ball_carrier import _carrier_loop_numpy

        arrays = _build_dense_fixture(n_frames=1, has_velocity=False)
        # Both players at same distance
        arrays["px"][0, 0] = 51.0
        arrays["px"][0, 1] = 51.0
        arrays["py"][0, 0] = 34.0
        arrays["py"][0, 1] = 34.0
        winner_slot, _ = _carrier_loop_numpy(**arrays)
        assert winner_slot[0] == 0  # slot 0 < slot 1


class TestPreIndexRoundTrip:
    def test_int_player_ids(self):
        """Integer pid → slot → pid round-trip is identity."""
        from silly_kicks.tracking._ball_carrier import _pre_index_frames
        from tests.tracking.test_ball_carrier import _make_carrier_frame

        frames = _make_carrier_frame(
            players=[
                dict(pid=10, tid=1, x=51.0, y=34.0, vx=0.0, vy=0.0),
                dict(pid=20, tid=2, x=52.0, y=34.0, vx=0.0, vy=0.0),
            ],
        )
        result = _pre_index_frames(frames)
        slot_to_pid = result["slot_to_pid"]
        pid_to_slot = result["pid_to_slot"]
        for pid, slot in pid_to_slot.items():
            assert slot_to_pid[slot] == pid

    def test_string_player_ids(self):
        """String pid (Sportec DFL-OBJ-*) → slot → pid round-trip is identity."""
        from silly_kicks.tracking._ball_carrier import _pre_index_frames
        from tests.tracking.test_ball_carrier import _make_carrier_frame

        frames = _make_carrier_frame(
            players=[
                dict(pid="DFL-OBJ-0001", tid="T1", x=51.0, y=34.0, vx=0.0, vy=0.0),
                dict(pid="DFL-OBJ-0002", tid="T2", x=52.0, y=34.0, vx=0.0, vy=0.0),
            ],
        )
        result = _pre_index_frames(frames)
        slot_to_pid = result["slot_to_pid"]
        pid_to_slot = result["pid_to_slot"]
        for pid, slot in pid_to_slot.items():
            assert slot_to_pid[slot] == pid


class TestNumbaParity:
    """Numba kernel must produce bit-identical output to Python fallback."""

    @pytest.mark.parametrize(
        "scenario",
        [
            dict(n_frames=5, has_velocity=False, include_dead=False, include_nan_velocity=False, n_segments=1),
            dict(n_frames=5, has_velocity=True, include_dead=False, include_nan_velocity=False, n_segments=1),
            dict(n_frames=5, has_velocity=False, include_dead=True, include_nan_velocity=False, n_segments=1),
            dict(n_frames=5, has_velocity=True, include_dead=False, include_nan_velocity=True, n_segments=1),
            dict(n_frames=6, has_velocity=True, include_dead=False, include_nan_velocity=False, n_segments=2),
        ],
    )
    def test_parity(self, scenario):
        from silly_kicks.tracking._ball_carrier import _carrier_loop_numpy
        from silly_kicks.tracking._ball_carrier_numba import _carrier_loop_numba

        arrays = _build_dense_fixture(**scenario)
        numpy_slot, numpy_dist = _carrier_loop_numpy(**arrays)
        numba_slot, numba_dist = _carrier_loop_numba(**arrays)
        np.testing.assert_array_equal(numpy_slot, numba_slot)
        # NaN == NaN comparison for dist
        for i in range(len(numpy_dist)):
            if np.isnan(numpy_dist[i]):
                assert np.isnan(numba_dist[i]), f"Frame {i}: numpy=NaN, numba={numba_dist[i]}"
            else:
                np.testing.assert_allclose(numpy_dist[i], numba_dist[i], rtol=1e-12)


class TestFallbackPath:
    """Verify Python fallback produces correct results when _HAS_NUMBA=False."""

    @patch("silly_kicks.tracking._ball_carrier._HAS_NUMBA", False)
    def test_basic_carrier_without_numba(self):
        from silly_kicks.tracking._ball_carrier import infer_ball_carrier
        from tests.tracking.test_ball_carrier import _make_carrier_frame

        frames = _make_carrier_frame(
            ball_x=50.0,
            ball_y=34.0,
            players=[
                dict(pid=10, tid=1, x=51.0, y=34.0, vx=0.0, vy=0.0),
                dict(pid=20, tid=2, x=53.0, y=34.0, vx=0.0, vy=0.0),
            ],
        )
        result = infer_ball_carrier(frames)
        assert result["ball_carrier_player_id"].iloc[0] == 10
        assert result["ball_carrier_team_id"].iloc[0] == 1

    @patch("silly_kicks.tracking._ball_carrier._HAS_NUMBA", False)
    def test_hysteresis_without_numba(self):
        from silly_kicks.tracking._ball_carrier import infer_ball_carrier
        from tests.tracking.test_ball_carrier import _concat_frames, _make_carrier_frame

        f1 = _make_carrier_frame(
            frame_id=1,
            ball_x=50.0,
            ball_y=34.0,
            players=[
                dict(pid=10, tid=1, x=51.0, y=34.0, vx=0.0, vy=0.0),
                dict(pid=20, tid=1, x=53.0, y=34.0, vx=0.0, vy=0.0),
            ],
        )
        f2 = _make_carrier_frame(
            frame_id=2,
            ball_x=50.0,
            ball_y=34.0,
            players=[
                dict(pid=10, tid=1, x=52.0, y=34.0, vx=0.0, vy=0.0),
                dict(pid=20, tid=1, x=51.8, y=34.0, vx=0.0, vy=0.0),
            ],
        )
        frames = _concat_frames(f1, f2)
        result = infer_ball_carrier(frames, gamma=1.0)
        carriers = result.sort_values("frame_id")["ball_carrier_player_id"].tolist()
        assert carriers == [10, 10]

    @patch("silly_kicks.tracking._ball_carrier._HAS_NUMBA", False)
    def test_dead_ball_without_numba(self):
        import pandas as pd

        from silly_kicks.tracking._ball_carrier import infer_ball_carrier
        from tests.tracking.test_ball_carrier import _make_carrier_frame

        frames = _make_carrier_frame(
            ball_state="dead",
            players=[dict(pid=10, tid=1, x=51.0, y=34.0, vx=0.0, vy=0.0)],
        )
        result = infer_ball_carrier(frames)
        assert pd.isna(result["ball_carrier_player_id"].iloc[0])


@pytest.mark.e2e
def test_bench_infer_ball_carrier_gs_match(benchmark):
    """Carrier inference for a full GS match should complete in <2s end-to-end.

    Uses pytest-benchmark for statistical rigor. Performance assertion on
    benchmark.stats.stats.mean, matching all existing perf budget tests.
    """
    from silly_kicks.tracking._ball_carrier import infer_ball_carrier
    from tests.tracking._provider_inputs import load_provider_frames

    frames = load_provider_frames("gradientsports")
    result = benchmark(infer_ball_carrier, frames)
    assert len(result) > 0
    assert list(result.columns) == [
        "game_id",
        "period_id",
        "frame_id",
        "ball_carrier_player_id",
        "ball_carrier_distance_m",
        "ball_carrier_team_id",
    ]
    # End-to-end target: <2s per spec. Using 3s budget for CI variance.
    assert benchmark.stats.stats.mean < 3.0


@pytest.mark.e2e
def test_numba_numpy_parity_real_gs_match():
    """Full GS match: numba and numpy fallback produce identical DataFrames.

    This is the production-scale parity gate — synthetic fixtures cannot catch
    data-shape-dependent divergence between the single-pass numba kernel and
    the two-pass numpy fallback.
    """
    import pandas as pd

    from silly_kicks.tracking._ball_carrier import infer_ball_carrier
    from tests.tracking._provider_inputs import load_provider_frames

    frames = load_provider_frames("gradientsports")

    # Run with numba (default when available)
    result_numba = infer_ball_carrier(frames)

    # Run with numpy fallback
    with patch("silly_kicks.tracking._ball_carrier._HAS_NUMBA", False):
        result_numpy = infer_ball_carrier(frames)

    pd.testing.assert_frame_equal(result_numba, result_numpy)
