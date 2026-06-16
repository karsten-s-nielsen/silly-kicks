"""Unit tests for PitchControlCache (TF-7 shared-surface perf).

The real compute_pitch_control is stubbed with a counting sentinel so these
tests exercise only the cache's keying/hit/miss/bypass logic — no simulation.
"""

from __future__ import annotations

import pandas as pd
import pytest

from silly_kicks.tracking.pitch_control import PitchControlCache, SpearmanParams, VoronoiParams


def _frame(frame_id: int = 100, game_id: int = 1, period_id: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [game_id, game_id, game_id],
            "period_id": [period_id, period_id, period_id],
            "frame_id": [frame_id, frame_id, frame_id],
            "player_id": [1, 2, 3],
            "team_id": [1, 1, 2],
            "x": [10.0, 20.0, 30.0],
            "y": [10.0, 20.0, 30.0],
            "vx": [0.0, 0.0, 0.0],
            "vy": [0.0, 0.0, 0.0],
            "is_ball": [False, False, False],
        }
    )


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list:
    """Replace compute_pitch_control with a counting sentinel-returning stub."""
    import silly_kicks.tracking.pitch_control._cache as cache_mod

    recorded: list = []

    def stub(frame, attacking_team_id, *, method="spearman", params=None, decompose=False, ball_position=None):
        recorded.append((attacking_team_id, method, params, decompose, ball_position))
        return object()  # unique per call -> identity reveals cache hits

    monkeypatch.setattr(cache_mod, "compute_pitch_control", stub)
    return recorded


class TestPitchControlCache:
    def test_same_frame_team_is_cached(self, calls: list) -> None:
        cache = PitchControlCache()
        s1 = cache.surface(_frame(100), 1)
        s2 = cache.surface(_frame(100), 1)
        assert s1 is s2
        assert len(calls) == 1

    def test_different_team_not_shared(self, calls: list) -> None:
        cache = PitchControlCache()
        cache.surface(_frame(100), 1)
        cache.surface(_frame(100), 2)
        assert len(calls) == 2

    def test_different_frame_id_not_shared(self, calls: list) -> None:
        cache = PitchControlCache()
        cache.surface(_frame(100), 1)
        cache.surface(_frame(101), 1)
        assert len(calls) == 2

    def test_none_and_default_params_share(self, calls: list) -> None:
        """params=None and an explicit default must collide (same surface)."""
        cache = PitchControlCache()
        s1 = cache.surface(_frame(100), 1, params=None)
        s2 = cache.surface(_frame(100), 1, params=SpearmanParams())
        assert s1 is s2
        assert len(calls) == 1

    def test_distinct_params_not_shared(self, calls: list) -> None:
        cache = PitchControlCache()
        cache.surface(_frame(100), 1, params=SpearmanParams(sigma=0.45))
        cache.surface(_frame(100), 1, params=SpearmanParams(sigma=0.9))
        assert len(calls) == 2

    def test_decompose_in_key(self, calls: list) -> None:
        cache = PitchControlCache()
        cache.surface(_frame(100), 1, decompose=False)
        cache.surface(_frame(100), 1, decompose=True)
        assert len(calls) == 2

    def test_method_in_key(self, calls: list) -> None:
        cache = PitchControlCache()
        cache.surface(_frame(100), 1, method="spearman")
        cache.surface(_frame(100), 1, method="voronoi", params=VoronoiParams())
        assert len(calls) == 2

    def test_ball_position_in_key(self, calls: list) -> None:
        cache = PitchControlCache()
        cache.surface(_frame(100), 1, ball_position=None)
        cache.surface(_frame(100), 1, ball_position=(52.5, 34.0))
        assert len(calls) == 2

    def test_unidentifiable_frame_bypasses_cache(self, calls: list) -> None:
        """A frame spanning multiple frame_ids cannot be keyed -> never cached."""
        multi = pd.concat([_frame(100), _frame(101)], ignore_index=True)
        cache = PitchControlCache()
        s1 = cache.surface(multi, 1)
        s2 = cache.surface(multi, 1)
        assert s1 is not s2
        assert len(calls) == 2

    def test_missing_identity_columns_bypasses_cache(self, calls: list) -> None:
        frame = _frame(100).drop(columns=["frame_id"])
        cache = PitchControlCache()
        cache.surface(frame, 1)
        cache.surface(frame, 1)
        assert len(calls) == 2

    def test_unknown_method_bypasses_cache(self, calls: list) -> None:
        """Unknown method bypasses keying (real compute would raise its own error)."""
        cache = PitchControlCache()
        cache.surface(_frame(100), 1, method="bogus")  # type: ignore[arg-type]
        cache.surface(_frame(100), 1, method="bogus")  # type: ignore[arg-type]
        assert len(calls) == 2


def _pc_frames() -> pd.DataFrame:
    """One identifiable frame (3v3 + ball) for real (voronoi) pitch control."""
    rows = []
    for pid in range(1, 4):
        rows.append(
            {
                "game_id": 1,
                "period_id": 1,
                "frame_id": 500,
                "player_id": pid,
                "team_id": 1,
                "x": 30.0 + pid * 5,
                "y": 30.0 + pid * 3,
                "is_ball": False,
                "is_goalkeeper": False,
            }
        )
    for pid in range(4, 7):
        rows.append(
            {
                "game_id": 1,
                "period_id": 1,
                "frame_id": 500,
                "player_id": pid,
                "team_id": 2,
                "x": 70.0 + pid,
                "y": 30.0 + pid,
                "is_ball": False,
                "is_goalkeeper": False,
            }
        )
    rows.append(
        {
            "game_id": 1,
            "period_id": 1,
            "frame_id": 500,
            "player_id": 99,
            "team_id": None,
            "x": 50.0,
            "y": 34.0,
            "is_ball": True,
            "is_goalkeeper": False,
        }
    )
    return pd.DataFrame(rows)


def _pc_actions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "action_id": [0, 1],
            "game_id": [1, 1],
            "period_id": [1, 1],
            "team_id": [1, 1],
            "start_x": [50.0, 55.0],
            "start_y": [34.0, 30.0],
            # pitch_control_at_action samples the action DESTINATION (ADR-032); end_* is a standard
            # SPADL column (every action carries it).
            "end_x": [60.0, 70.0],
            "end_y": [30.0, 40.0],
            "type_id": [0, 0],
        }
    )


def _pc_links() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "action_id": [0, 1],
            "frame_id": [500, 500],
            "time_offset_seconds": [0.0, 0.0],
            "n_candidate_frames": [1, 1],
            "link_quality_score": [1.0, 1.0],
        }
    )


class TestCrossFamilySharing:
    """A shared cache reuses canonical surfaces across different aggregators (end-to-end)."""

    def test_shared_cache_reused_across_two_aggregators(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import silly_kicks.tracking.pitch_control._cache as cache_mod

        real = cache_mod.compute_pitch_control
        n = {"calls": 0}

        def counting(*args, **kwargs):
            n["calls"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(cache_mod, "compute_pitch_control", counting)

        from silly_kicks.tracking.features import add_pitch_control, pitch_control_at_action

        frames, actions, links = _pc_frames(), _pc_actions(), _pc_links()
        cache = PitchControlCache()

        # First consumer: 2 actions on the same (frame 500, team 1) -> 1 compute.
        add_pitch_control(actions, frames, method="voronoi", links=links, pitch_control_cache=cache)
        after_first = n["calls"]
        assert after_first == 1

        # Second consumer, SAME cache + same frame/team/method -> all cache hits.
        pitch_control_at_action(actions, frames, method="voronoi", links=links, pitch_control_cache=cache)
        assert n["calls"] == after_first, "shared cache should be reused across families"

    def test_cache_does_not_change_output(self) -> None:
        from silly_kicks.tracking.features import add_pitch_control

        frames, actions, links = _pc_frames(), _pc_actions(), _pc_links()
        r_plain = add_pitch_control(actions, frames, method="voronoi", links=links)
        r_cached = add_pitch_control(
            actions, frames, method="voronoi", links=links, pitch_control_cache=PitchControlCache()
        )
        pd.testing.assert_frame_equal(r_plain, r_cached)
