"""TF-57 rev-3 — in-play (alive-segment) episode grouping.

The paper's alignment episodes are *consecutive in-play frames* (``ball_state == "alive"``
segments), NOT ``spadl.add_possessions`` possessions (the as-built proxy). Spec §3.1/§7: the
"control (reception)" virtual event fires for consecutive actions *in the same episode* by
different players, so the episode key must be the alive-segment id — feeding both the episode
partition and the reception rule.

This module gates the three new helpers + the wiring:
  * ``_inplay_segments``      — per (game, period) maximal alive-run time intervals.
  * ``_assign_segments``      — nearest-segment index per action time (dead-span actions → nearest).
  * ``_inplay_episode_ids``   — per-action episode id (the alive-segment assignment) used by the
                                assembler as the episode/possession partition key.
  * ``align_events_to_frames`` — now partitions by alive-segments (reads ``ball_state``).

The headline in-play win (pooled W2 0.845 → 0.856 across the 3-match CC-BY corpus) is validated
on the DGX; these gates pin the mechanism on the committed 1-match oracle + synthetic fixtures.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from silly_kicks.id_compat import canonical_id
from silly_kicks.spadl.utils import add_possessions
from silly_kicks.tracking._elastic_sync import (
    ElasticSyncParams,
    _assign_segments,
    _inplay_episode_ids,
    _inplay_segments,
    align_events_to_frames,
)

_ORACLE = pathlib.Path(__file__).resolve().parents[1] / "datasets" / "elastic_sync" / "j03wmx_slice"


def _frames_with_ball_state(period_patterns: dict[int, list[str]], *, game_id=1, rate=25.0):
    """Minimal long-form frames: one ball + one player row per frame, sharing ball_state/time.

    ``period_patterns`` maps period_id -> list of ball_state tokens for frames 0, 1, 2, ... in that
    period. ``time_seconds = frame_id / rate``. Frame ids are offset by 100*period to keep periods
    distinct (the native-frame-number case).
    """
    rows = []
    for pid, pattern in period_patterns.items():
        for i, state in enumerate(pattern):
            fid = 100 * pid + i
            t = fid / rate
            for is_ball, pdid in ((True, pd.NA), (False, 7)):
                rows.append(
                    {
                        "game_id": game_id,
                        "period_id": pid,
                        "frame_id": fid,
                        "time_seconds": t,
                        "ball_state": state,
                        "is_ball": is_ball,
                        "player_id": pdid,
                    }
                )
    return pd.DataFrame(rows)


def _make_alignable_bs(dead_frames=frozenset()):
    """Deterministic 3-pass scenario (ball rests at p1_0@20, departs f10, arrives p1_1@45 f20,
    departs f30, arrives p1_2@70 f40, departs f50) WITH a ``ball_state`` column: frames in
    ``dead_frames`` are 'dead', the rest 'alive'. Positions -- hence candidate detection -- are
    IDENTICAL regardless of ball_state, so any alignment difference is attributable to the
    alive-segment episode partition alone."""
    xs = {"p1_0": 20.0, "p1_1": 45.0, "p1_2": 70.0, "p1_3": 90.0}

    def ball_x(f: int) -> float:
        if f <= 10:
            return 20.0
        if f <= 20:
            return 20.0 + 25.0 * (f - 10) / 10.0
        if f <= 30:
            return 45.0
        if f <= 40:
            return 45.0 + 25.0 * (f - 30) / 10.0
        if f <= 50:
            return 70.0
        if f <= 60:
            return 70.0 + 20.0 * (f - 50) / 10.0
        return 90.0

    rows = []
    for f in range(71):
        t = f / 25.0
        state = "dead" if f in dead_frames else "alive"
        rows.append(
            {
                "game_id": 1,
                "period_id": 1,
                "frame_id": f,
                "time_seconds": t,
                "player_id": pd.NA,
                "team_id": pd.NA,
                "x": ball_x(f),
                "y": 34.0,
                "z": float("nan"),
                "is_ball": True,
                "ball_state": state,
            }
        )
        for p, x in xs.items():
            rows.append(
                {
                    "game_id": 1,
                    "period_id": 1,
                    "frame_id": f,
                    "time_seconds": t,
                    "player_id": p,
                    "team_id": 1,
                    "x": x,
                    "y": 34.0,
                    "z": float("nan"),
                    "is_ball": False,
                    "ball_state": state,
                }
            )
        rows.append(
            {
                "game_id": 1,
                "period_id": 1,
                "frame_id": f,
                "time_seconds": t,
                "player_id": "p2_0",
                "team_id": 2,
                "x": 5.0,
                "y": 60.0,
                "z": float("nan"),
                "is_ball": False,
                "ball_state": state,
            }
        )
    frames = pd.DataFrame(rows)
    actions = pd.DataFrame(
        {
            "action_id": [0, 1, 2],
            "game_id": [1, 1, 1],
            "period_id": [1, 1, 1],
            "time_seconds": [10 / 25, 30 / 25, 50 / 25],
            "player_id": ["p1_0", "p1_1", "p1_2"],
            "team_id": [1, 1, 1],
            "type_id": [0, 0, 0],
            "type_name": ["pass", "pass", "pass"],
            "result_name": ["success"] * 3,
        }
    )
    return frames, actions


class TestInplaySegments:
    def test_finds_maximal_alive_runs(self):
        # frames 100..109: alive,alive,dead,dead,alive,alive,alive,dead,alive,alive
        pattern = ["alive", "alive", "dead", "dead", "alive", "alive", "alive", "dead", "alive", "alive"]
        frames = _frames_with_ball_state({1: pattern})
        segs = _inplay_segments(frames)
        key = (canonical_id(1), 1)
        assert key in segs
        arr = segs[key]
        # three runs: frames[0,1], frames[4,5,6], frames[8,9] -> time intervals (rate=25 => 0.04 s/frame)
        expected = np.array([[100 / 25, 101 / 25], [104 / 25, 106 / 25], [108 / 25, 109 / 25]])
        assert arr.shape == (3, 2)
        np.testing.assert_allclose(arr, expected)

    def test_keyed_per_game_period_runs_do_not_cross_periods(self):
        frames = _frames_with_ball_state({1: ["alive", "alive", "dead"], 2: ["alive", "alive", "alive", "alive"]})
        segs = _inplay_segments(frames)
        assert (canonical_id(1), 1) in segs and (canonical_id(1), 2) in segs
        assert segs[(canonical_id(1), 1)].shape == (1, 2)  # one run in P1
        assert segs[(canonical_id(1), 2)].shape == (1, 2)  # one run in P2 (separate period)

    def test_all_dead_period_omitted(self):
        frames = _frames_with_ball_state({1: ["dead", "dead", "dead"]})
        assert (canonical_id(1), 1) not in _inplay_segments(frames)

    def test_missing_ball_state_or_empty_returns_empty(self):
        assert _inplay_segments(pd.DataFrame()) == {}
        frames = _frames_with_ball_state({1: ["alive", "alive"]}).drop(columns=["ball_state"])
        assert _inplay_segments(frames) == {}


class TestAssignSegments:
    def test_inside_between_outside(self):
        segs = np.array([[0.0, 0.04], [0.16, 0.24], [0.32, 0.36]])
        times = np.array([0.02, 0.08, 0.14, 0.20, 0.50, -0.10])
        # 0.02 in seg0; 0.08 nearer seg0; 0.14 nearer seg1; 0.20 in seg1; 0.50 nearer seg2; -0.10 nearer seg0
        np.testing.assert_array_equal(_assign_segments(times, segs), np.array([0, 0, 1, 1, 2, 0]))

    def test_empty_segments_returns_zero(self):
        np.testing.assert_array_equal(_assign_segments(np.array([0.1, 0.2]), np.empty((0, 2))), np.array([0, 0]))


class TestInplayEpisodeIdsOnOracle:
    def test_segment_episodes_are_coarser_than_possessions(self):
        """Non-vacuous: the alive-segment episode partition MERGES open-play turnovers that
        ``add_possessions`` splits, so it has strictly fewer episode boundaries — the paper's
        in-play grouping. Also non-degenerate (>= 2 distinct episodes)."""
        actions = pd.read_parquet(_ORACLE / "actions.parquet")
        frames = pd.read_parquet(_ORACLE / "frames.parquet")
        actions = actions.sort_values(["time_seconds", "action_id"]).reset_index(drop=True)

        ep = _inplay_episode_ids(actions, frames, params=ElasticSyncParams())
        assert len(ep) == len(actions)
        assert ep.nunique() >= 2, "episode partition is degenerate on the oracle"

        poss = add_possessions(actions.copy())["possession_id"]
        ep_codes = pd.factorize(ep.to_numpy())[0]
        poss_codes = pd.factorize(poss.to_numpy())[0]
        ep_bounds = int((np.diff(ep_codes) != 0).sum())
        poss_bounds = int((np.diff(poss_codes) != 0).sum())
        assert ep_bounds < poss_bounds, (
            f"alive-segment episodes ({ep_bounds} boundaries) should be coarser than "
            f"possessions ({poss_bounds}) — in-play grouping merges open-play turnovers"
        )


class TestAlignPartitionsByAliveSegment:
    def test_alignment_depends_on_ball_state(self):
        """Wiring gate (both-sided, NON-VACUOUS): the assembler partitions episodes by alive
        segments, so a dead gap that SPLITS two passes into different episodes MUST change the
        alignment. Here a dead gap between pass 0 (f10) and pass 1 (f30) puts pass 0 alone in its
        episode, removing its same-episode reception (f20 -> NA), while candidate detection
        (position-only) is unchanged. Under the old possession partition ball_state was ignored and
        the two outputs were identical (RED). A self-contained synthetic (not the oracle slice), so
        the dependency is guaranteed by construction rather than incidental to a real span."""
        frames_alive, actions = _make_alignable_bs()
        frames_split, _ = _make_alignable_bs(dead_frames=frozenset(range(14, 27)))
        r_alive = align_events_to_frames(actions, frames_alive).set_index("action_id")
        r_split = align_events_to_frames(actions, frames_split).set_index("action_id")
        assert not r_alive.equals(r_split), (
            "alignment did not change when a dead gap split the episode — the episode partition is "
            "not reading alive segments (in-play grouping not wired)"
        )
        # Non-vacuity: the specific change is pass 0 LOSING its same-episode reception (f20 -> NA).
        assert pd.notna(r_alive.at[0, "elastic_receive_frame_id"]), "pass 0 reception should fire when alive"
        assert pd.isna(r_split.at[0, "elastic_receive_frame_id"]), "pass 0 reception should vanish when split off"
