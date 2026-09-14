"""Tests for silly_kicks.tracking._elastic_sync — ELASTIC sync (Kim et al. 2025)."""

from __future__ import annotations

from collections import namedtuple

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._elastic_sync import (
    Candidate,
    ElasticSyncParams,
    Event,
    _clip_linear,
    _detect_candidate_frames,
    _enrich_events,
    _FrameLookups,
    _map_category,
    _needleman_wunsch,
    _score,
    align_events_to_frames,
    extract_ball_features,
)

_StubEvent = namedtuple("_StubEvent", "player_id category")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tracking_frames(
    n_frames: int = 50,
    n_players_per_team: int = 3,
    frame_rate: int = 25,
) -> pd.DataFrame:
    """Build minimal tracking frames with ball + players."""
    rows = []
    rng = np.random.RandomState(42)
    for fid in range(n_frames):
        t = fid / frame_rate
        # Ball with some movement
        bx = 50.0 + 2.0 * np.sin(fid * 0.3) + rng.normal(0, 0.2)
        by = 34.0 + 1.0 * np.cos(fid * 0.2) + rng.normal(0, 0.1)
        rows.append(
            {
                "game_id": 1,
                "period_id": 1,
                "frame_id": fid,
                "time_seconds": t,
                "player_id": None,
                "team_id": None,
                "x": bx,
                "y": by,
                "is_ball": True,
            }
        )
        # Players
        for tid in [1, 2]:
            for pid_idx in range(n_players_per_team):
                px = 30.0 + tid * 25 + rng.normal(0, 3)
                py = 15.0 + pid_idx * 15 + rng.normal(0, 2)
                rows.append(
                    {
                        "game_id": 1,
                        "period_id": 1,
                        "frame_id": fid,
                        "time_seconds": t,
                        "player_id": f"p{tid}_{pid_idx}",
                        "team_id": tid,
                        "x": px,
                        "y": py,
                        "is_ball": False,
                    }
                )
    return pd.DataFrame(rows)


def _make_actions(n: int = 5) -> pd.DataFrame:
    """Build minimal SPADL-like actions for alignment (team_id required by add_possessions)."""
    return pd.DataFrame(
        {
            "action_id": range(n),
            "game_id": [1] * n,
            "period_id": [1] * n,
            "time_seconds": np.linspace(0.2, 1.6, n),
            "player_id": [f"p1_{i % 3}" for i in range(n)],
            "team_id": [1] * n,
            "type_id": [0] * n,  # 0 == "pass"
        }
    )


def _make_idsse_like_frames(
    n_frames: int = 100,
    n_players_per_team: int = 3,
    frame_rate: int = 25,
    frame_offset: int = 10000,
) -> pd.DataFrame:
    """Frames with a native (non-zero) frame_id origin but 0-based period time.

    Mirrors IDSSE/Sportec: period-1 ``frame_id`` numbered from 10000 while
    ``time_seconds`` is period-elapsed (0-based). Used to regress the
    frame-id-origin assumption in ``align_events_to_frames`` — the frame
    window AND the frame->time conversion must derive from the frames' own
    ``(frame_id, time_seconds)`` relationship, not from ``time * frame_rate``.
    """
    frames = _make_tracking_frames(
        n_frames=n_frames,
        n_players_per_team=n_players_per_team,
        frame_rate=frame_rate,
    )
    frames["frame_id"] = frames["frame_id"] + frame_offset
    return frames


def _make_idsse_two_period_frames(n_frames: int = 80) -> pd.DataFrame:
    """IDSSE-like two-period frames: period 1 from 10000, period 2 from 100000.

    Mirrors real Sportec/IDSSE numbering (period 2 frames numbered from 100000)
    while ``time_seconds`` is period-elapsed (0-based) in BOTH periods — so each
    period needs its own (frame_id, time) fit.
    """
    p1 = _make_idsse_like_frames(n_frames=n_frames, frame_offset=10000)
    p2 = _make_idsse_like_frames(n_frames=n_frames, frame_offset=100000)
    p2["period_id"] = 2
    return pd.concat([p1, p2], ignore_index=True)


def _make_two_period_actions() -> pd.DataFrame:
    """Actions spread across both periods (period-elapsed times)."""
    return pd.DataFrame(
        {
            "action_id": range(6),
            "game_id": [1] * 6,
            "period_id": [1, 1, 1, 2, 2, 2],
            "time_seconds": [0.2, 0.8, 1.4, 0.2, 0.8, 1.4],
            "player_id": [f"p1_{i % 3}" for i in range(6)],
            "team_id": [1] * 6,
            "type_id": [0] * 6,
        }
    )


def _make_touch_fixture(
    spike_frame: int = 10,
    toucher: str = "p1_0",
    dist_m: float = 1.0,
    ball_z: float = float("nan"),
    n_frames: int = 30,
    frame_rate: int = 25,
) -> pd.DataFrame:
    """Single-period frames: a stationary ball, `toucher` closest to it AT `spike_frame`
    (a V-shaped distance -> local minimum), a far constant other player, configurable ball z.

    Guarantees a candidate frame at `spike_frame` via the min-player-distance signal, so the
    feasibility gate (<=3 m, ball height <=4 m) is what the tests exercise.
    """
    rows = []
    for fid in range(n_frames):
        t = fid / frame_rate
        bx, by = 50.0, 34.0  # stationary ball -> no accel/boundary candidate; only min-dist
        rows.append(
            {
                "game_id": "1",
                "period_id": 1,
                "frame_id": fid,
                "time_seconds": t,
                "player_id": None,
                "team_id": None,
                "x": bx,
                "y": by,
                "z": ball_z,
                "is_ball": True,
            }
        )
        d = dist_m + 2.0 * abs(fid - spike_frame)  # min == dist_m at spike_frame
        rows.append(
            {
                "game_id": "1",
                "period_id": 1,
                "frame_id": fid,
                "time_seconds": t,
                "player_id": toucher,
                "team_id": 1,
                "x": bx + d,
                "y": by,
                "z": float("nan"),
                "is_ball": False,
            }
        )
        rows.append(
            {
                "game_id": "1",
                "period_id": 1,
                "frame_id": fid,
                "time_seconds": t,
                "player_id": "p2_0",
                "team_id": 2,
                "x": 5.0,
                "y": 5.0,
                "z": float("nan"),
                "is_ball": False,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests — ElasticSyncParams
# ---------------------------------------------------------------------------


class TestElasticSyncParams:
    def test_frozen(self):
        params = ElasticSyncParams()
        with pytest.raises(AttributeError):
            params.min_confidence = 0.9  # type: ignore[misc]

    def test_dataclass_defaults_are_paper_intent_set_constants(self):
        """The ``ElasticSyncParams`` DATACLASS fields are paper-derived intent-set constants
        (rates / distances / thresholds; ``for_provider`` empty per ADR-009). This does NOT cover the
        scorer's DISCRIMINATIVE weights -- ``w_ba/w_pbd/w_kd/w_dyn`` + the directional slope term are
        OpenEvolve-TUNED literals inside ``_score`` (ADR-093, owner-approved 2026-09-14), NOT paper
        constants and deliberately NOT in this dataclass (see the ``_score`` docstring + NOTICE). So
        the name is scoped to the dataclass, not the whole scorer."""
        p = ElasticSyncParams()
        assert p.frame_rate == 25
        assert p.touch_distance_m == pytest.approx(3.0)
        assert p.ball_height_max_m == pytest.approx(4.0)
        assert p.accel_clip_max == pytest.approx(30.0)
        assert p.slope_window_seconds == pytest.approx(0.2)
        assert p.slope_clip_mps == pytest.approx(7.0)
        assert p.repeat_penalty == pytest.approx(-0.1)
        assert p.event_gap_penalty == pytest.approx(0.0)
        assert p.candidate_gap_penalty == pytest.approx(0.0)
        assert p.min_confidence == pytest.approx(0.5)

    def test_greedy_fields_removed(self):
        p = ElasticSyncParams()
        assert not hasattr(p, "accel_weight")
        assert not hasattr(p, "proximity_weight")
        assert not hasattr(p, "window_seconds")


# ---------------------------------------------------------------------------
# Tests — extract_ball_features
# ---------------------------------------------------------------------------


class TestExtractBallFeatures:
    def test_output_columns(self):
        frames = _make_tracking_frames()
        bf = extract_ball_features(frames)
        assert set(bf.columns) == {
            "game_id",
            "period_id",
            "frame_id",
            "ball_x",
            "ball_y",
            "ball_speed",
            "ball_accel",
        }

    def test_one_row_per_frame(self):
        frames = _make_tracking_frames(n_frames=20)
        bf = extract_ball_features(frames)
        assert len(bf) == 20

    def test_speed_nonnegative(self):
        frames = _make_tracking_frames()
        bf = extract_ball_features(frames)
        assert (bf["ball_speed"] >= 0).all()

    def test_accel_nonnegative(self):
        frames = _make_tracking_frames()
        bf = extract_ball_features(frames)
        assert (bf["ball_accel"] >= 0).all()

    def test_first_frame_zero_speed(self):
        """First frame in a period has zero speed/accel (no prior)."""
        frames = _make_tracking_frames()
        bf = extract_ball_features(frames)
        assert bf.iloc[0]["ball_speed"] == pytest.approx(0.0)
        assert bf.iloc[0]["ball_accel"] == pytest.approx(0.0)

    def test_empty_frames(self):
        frames = pd.DataFrame(
            columns=[
                "game_id",
                "period_id",
                "frame_id",
                "time_seconds",
                "x",
                "y",
                "is_ball",
            ]
        )
        bf = extract_ball_features(frames)
        assert len(bf) == 0

    def test_custom_frame_rate(self):
        frames = _make_tracking_frames(frame_rate=10)
        params = ElasticSyncParams(frame_rate=10)
        bf = extract_ball_features(frames, params=params)
        assert len(bf) > 0

    def test_multi_period(self):
        f1 = _make_tracking_frames(n_frames=10)
        f2 = _make_tracking_frames(n_frames=10)
        f2["period_id"] = 2
        f2["frame_id"] = f2["frame_id"] + 100
        frames = pd.concat([f1, f2], ignore_index=True)
        bf = extract_ball_features(frames)
        assert len(bf) == 20
        # Speed at period boundary should be 0
        p2_first = bf[bf["period_id"] == 2].iloc[0]
        assert p2_first["ball_speed"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests — candidate ball-touch frame detection
# ---------------------------------------------------------------------------


class TestCandidateDetection:
    def test_accel_spike_is_a_candidate(self):
        frames = _make_touch_fixture(spike_frame=10, toucher="p1_0", dist_m=1.0)
        cands = _detect_candidate_frames(frames, params=ElasticSyncParams())
        by_frame = {c.frame_id: c for c in cands[("1", 1)]}
        assert 10 in by_frame
        assert "p1_0" in by_frame[10].players
        assert isinstance(by_frame[10], Candidate)

    def test_far_player_excluded_by_feasibility_gate(self):
        frames = _make_touch_fixture(spike_frame=10, toucher="p1_0", dist_m=9.0)
        cands = _detect_candidate_frames(frames, params=ElasticSyncParams())
        with_toucher = {c.frame_id for c in cands.get(("1", 1), []) if "p1_0" in c.players}
        assert 10 not in with_toucher  # nearest player 9 m > 3 m -> no feasible pair

    def test_high_ball_excluded_when_z_present(self):
        frames = _make_touch_fixture(spike_frame=10, toucher="p1_0", dist_m=1.0, ball_z=6.0)
        cands = _detect_candidate_frames(frames, params=ElasticSyncParams())
        assert 10 not in {c.frame_id for c in cands.get(("1", 1), [])}

    def test_z_absent_does_not_gate(self):
        frames = _make_touch_fixture(spike_frame=10, toucher="p1_0", dist_m=1.0, ball_z=float("nan"))
        cands = _detect_candidate_frames(frames, params=ElasticSyncParams())
        assert 10 in {c.frame_id for c in cands.get(("1", 1), [])}

    def test_empty_frames_returns_empty(self):
        empty = pd.DataFrame(columns=["game_id", "period_id", "frame_id", "player_id", "x", "y", "z", "is_ball"])
        assert _detect_candidate_frames(empty, params=ElasticSyncParams()) == {}


# ---------------------------------------------------------------------------
# Tests — feature scoring
# ---------------------------------------------------------------------------


def _touch_lookups() -> _FrameLookups:
    """Actor p1_0 within 1 m of the ball at frame 10 (farther at neighbours); accel high at 10."""
    dist = {(10, "p1_0"): 1.0, (9, "p1_0"): 2.5, (11, "p1_0"): 2.5}
    return _FrameLookups(accel={10: 25.0}, dist=dist, frame_players={10: ["p1_0"]}, player_team={"p1_0": "A"})


def _departing_ball_lookups() -> _FrameLookups:
    """Actor at the ball at frame 10, ball departs afterwards (post-touch distance grows)."""
    dist = {
        (8, "p1_0"): 0.5,
        (9, "p1_0"): 0.5,
        (10, "p1_0"): 0.5,
        (11, "p1_0"): 1.5,
        (12, "p1_0"): 2.5,
        (13, "p1_0"): 3.0,
    }
    return _FrameLookups(accel={10: 20.0}, dist=dist, frame_players={10: ["p1_0"]}, player_team={"p1_0": "A"})


class TestScoring:
    def test_clip_linear_bounds(self):
        assert _clip_linear(-1, 0, 30) == 0.0
        assert _clip_linear(15, 0, 30) == pytest.approx(0.5)
        assert _clip_linear(40, 0, 30) == 1.0

    def test_actor_not_in_candidate_scores_zero(self):
        cand = Candidate("1", 1, 10, players=("p1_0",))
        ev = _StubEvent(player_id="p2_9", category="outgoing")
        assert _score(ev, cand, _touch_lookups(), params=ElasticSyncParams()) == 0.0

    def test_outgoing_high_when_ball_departs_after_touch(self):
        cand = Candidate("1", 1, 10, players=("p1_0",))
        ev = _StubEvent(player_id="p1_0", category="outgoing")
        s = _score(ev, cand, _departing_ball_lookups(), params=ElasticSyncParams())
        assert 0.0 < s <= 1.0

    def test_score_in_unit_interval(self):
        cand = Candidate("1", 1, 10, players=("p1_0",))
        for cat in ("outgoing", "incoming", "minor"):
            ev = _StubEvent(player_id="p1_0", category=cat)
            s = _score(ev, cand, _touch_lookups(), params=ElasticSyncParams())
            assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# Tests — event enrichment (virtual termination events)
# ---------------------------------------------------------------------------


def _events_df(specs) -> pd.DataFrame:
    """Build a minimal possession-tagged actions frame.

    Each spec is ``(type_name, player_id, possession_id)`` or
    ``(type_name, player_id, possession_id, result_name)``.
    """
    rows = []
    for i, spec in enumerate(specs):
        typ, player, poss = spec[0], spec[1], spec[2]
        result = spec[3] if len(spec) > 3 else "success"
        rows.append(
            {
                "action_id": i,
                "game_id": "1",
                "period_id": 1,
                "time_seconds": 0.5 * i,
                "player_id": player,
                "team_id": 1 if str(player).startswith("p1") else 2,
                "type_name": typ,
                "possession_id": poss,
                "result_name": result,
            }
        )
    return pd.DataFrame(rows)


class TestEventEnrichment:
    def test_category_mapping(self):
        assert _map_category("dribble") is None
        assert _map_category("non_action") is None
        assert _map_category("pass") == "outgoing"
        assert _map_category("throw_in") == "outgoing"
        assert _map_category("interception") == "incoming"
        assert _map_category("tackle") == "minor"

    def test_reception_inserted_for_same_possession_diff_player(self):
        evs = _enrich_events(_events_df([("pass", "p1_0", 1), ("pass", "p1_1", 1)]), params=ElasticSyncParams())
        assert [e.kind for e in evs] == ["real", "reception", "real"]
        assert evs[1].player_id == "p1_1"  # reception owned by the next actor
        assert evs[1].category == "incoming"

    def test_out_inserted_before_restart(self):
        evs = _enrich_events(_events_df([("pass", "p1_0", 1), ("throw_in", "p2_0", 2)]), params=ElasticSyncParams())
        assert [e.kind for e in evs] == ["real", "out", "real"]

    def test_goal_inserted_after_successful_shot_then_kickoff(self):
        evs = _enrich_events(
            _events_df([("shot", "p1_0", 1, "success"), ("pass", "p2_0", 2)]), params=ElasticSyncParams()
        )
        assert [e.kind for e in evs] == ["real", "goal", "real"]

    def test_no_insertion_same_player_continuation(self):
        evs = _enrich_events(_events_df([("pass", "p1_0", 1), ("take_on", "p1_0", 1)]), params=ElasticSyncParams())
        assert [e.kind for e in evs] == ["real", "real"]

    def test_excluded_types_dropped(self):
        evs = _enrich_events(
            _events_df([("pass", "p1_0", 1), ("dribble", "p1_0", 1), ("pass", "p1_1", 1)]),
            params=ElasticSyncParams(),
        )
        # dribble dropped -> two real events (pass, pass) with a reception between
        assert [e.kind for e in evs] == ["real", "reception", "real"]
        assert all(isinstance(e, Event) for e in evs)


# ---------------------------------------------------------------------------
# Tests — extended Needleman-Wunsch DP
# ---------------------------------------------------------------------------


class TestNeedlemanWunsch:
    def test_monotone_assignment(self):
        S = np.array([[0.9, 0.1, 0.0], [0.0, 0.9, 0.1], [0.0, 0.1, 0.9]])
        assert _needleman_wunsch(S, params=ElasticSyncParams()) == [0, 1, 2]

    def test_down_match_one_touch(self):
        # events 0 and 1 both best-match candidate 0 -> down-match assigns c0 to both
        S = np.array([[0.9, 0.0], [0.8, 0.05]])
        assert _needleman_wunsch(S, params=ElasticSyncParams()) == [0, 0]

    def test_order_preserving(self):
        S = np.array([[0.2, 0.9], [0.9, 0.2]])
        out = _needleman_wunsch(S, params=ElasticSyncParams())
        assigned = [c for c in out if c is not None]
        assert assigned == sorted(assigned)  # never assigns a later event to an earlier candidate

    def test_all_low_scores_event_gaps(self):
        S = np.zeros((2, 3))
        assert _needleman_wunsch(S, params=ElasticSyncParams()) == [None, None]

    def test_empty(self):
        assert _needleman_wunsch(np.zeros((0, 0)), params=ElasticSyncParams()) == []


# ---------------------------------------------------------------------------
# Tests — align_events_to_frames
# ---------------------------------------------------------------------------


_ELASTIC_COLS = {
    "action_id",
    "elastic_frame_id",
    "elastic_confidence",
    "elastic_error_seconds",
    "elastic_receive_frame_id",
    "elastic_receive_confidence",
    "elastic_receive_error_seconds",
}


def _make_alignable(frame_offset: int = 0, period_id: int = 1, frame_rate: int = 25):
    """A controlled continuous-tracking scene the NW can actually align.

    Four team-1 players at fixed x; the ball rests at each in turn and hops between them (accel
    peaks + the acting player within 3 m at each touch). Actions = three passes at the kick frames
    (10/30/50), same team -> one possession -> receptions between them. ``frame_offset`` mimics the
    native DFL numbering (period-relative ``time_seconds`` stays 0-based); ``player_id`` matches the
    frames so the actor-membership gate is live.
    """
    xs = {"p1_0": 20.0, "p1_1": 45.0, "p1_2": 70.0, "p1_3": 90.0}

    def ball_x(f: int) -> float:
        if f <= 10:
            return 20.0
        if f <= 20:
            return 20.0 + (45.0 - 20.0) * (f - 10) / 10.0
        if f <= 30:
            return 45.0
        if f <= 40:
            return 45.0 + (70.0 - 45.0) * (f - 30) / 10.0
        if f <= 50:
            return 70.0
        if f <= 60:
            return 70.0 + (90.0 - 70.0) * (f - 50) / 10.0
        return 90.0

    rows = []
    for f in range(71):
        t = f / frame_rate
        fid = f + frame_offset
        rows.append(
            {
                "game_id": 1,
                "period_id": period_id,
                "frame_id": fid,
                "time_seconds": t,
                "player_id": None,
                "team_id": None,
                "x": ball_x(f),
                "y": 34.0,
                "z": float("nan"),
                "is_ball": True,
            }
        )
        for p, x in xs.items():
            rows.append(
                {
                    "game_id": 1,
                    "period_id": period_id,
                    "frame_id": fid,
                    "time_seconds": t,
                    "player_id": p,
                    "team_id": 1,
                    "x": x,
                    "y": 34.0,
                    "z": float("nan"),
                    "is_ball": False,
                }
            )
        rows.append(
            {
                "game_id": 1,
                "period_id": period_id,
                "frame_id": fid,
                "time_seconds": t,
                "player_id": "p2_0",
                "team_id": 2,
                "x": 5.0,
                "y": 60.0,
                "z": float("nan"),
                "is_ball": False,
            }
        )
    frames = pd.DataFrame(rows)
    actions = pd.DataFrame(
        {
            "action_id": [0, 1, 2],
            "game_id": [1, 1, 1],
            "period_id": [period_id] * 3,
            "time_seconds": [10 / frame_rate, 30 / frame_rate, 50 / frame_rate],
            "player_id": ["p1_0", "p1_1", "p1_2"],
            "team_id": [1, 1, 1],
            "type_id": [0, 0, 0],
            "type_name": ["pass", "pass", "pass"],
            "result_name": ["success"] * 3,
        }
    )
    return frames, actions


class TestAlignEventsToFrames:
    def test_output_columns(self):
        frames, actions = _make_alignable()
        result = align_events_to_frames(actions, frames)
        assert set(result.columns) == _ELASTIC_COLS

    def test_confidence_in_unit_interval_or_nan(self):
        frames, actions = _make_alignable()
        result = align_events_to_frames(actions, frames)
        v = result["elastic_confidence"].dropna()
        assert (v >= 0.0).all() and (v <= 1.0).all()

    def test_error_nonnegative(self):
        frames, actions = _make_alignable()
        result = align_events_to_frames(actions, frames)
        v = result["elastic_error_seconds"].dropna()
        assert (v >= 0.0).all()

    def test_matched_frames_are_real_frames(self):
        frames, actions = _make_alignable()
        result = align_events_to_frames(actions, frames)
        assert len(result) > 0
        real = set(frames["frame_id"].tolist())
        matched = result["elastic_frame_id"].dropna().astype(int)
        assert set(matched).issubset(real)

    def test_reception_populated_for_pass_to_teammate(self):
        frames, actions = _make_alignable()
        result = align_events_to_frames(actions, frames)
        # the first pass (p1_0 -> p1_1) should get a reception frame
        row0 = result[result["action_id"] == 0]
        assert len(row0) == 1
        assert pd.notna(row0.iloc[0]["elastic_receive_frame_id"])

    def test_empty_actions(self):
        frames, _ = _make_alignable()
        actions = pd.DataFrame(
            columns=["action_id", "game_id", "period_id", "time_seconds", "player_id", "team_id", "type_id"]
        )
        result = align_events_to_frames(actions, frames)
        assert len(result) == 0
        assert set(result.columns) == _ELASTIC_COLS

    def test_empty_frames(self):
        _, actions = _make_alignable()
        frames = pd.DataFrame(
            columns=[
                "game_id",
                "period_id",
                "frame_id",
                "time_seconds",
                "x",
                "y",
                "z",
                "is_ball",
                "player_id",
                "team_id",
            ]
        )
        result = align_events_to_frames(actions, frames)
        assert len(result) == 0

    def test_min_confidence_filter(self):
        frames, actions = _make_alignable()
        loose = align_events_to_frames(actions, frames, params=ElasticSyncParams(min_confidence=0.01))
        strict = align_events_to_frames(actions, frames, params=ElasticSyncParams(min_confidence=0.999))
        assert len(strict) <= len(loose)

    def test_nw_golden_on_synthetic_frame(self):
        """Align-level EXACT-OUTPUT golden (spec §9 / plan Task 17 Step 1): the deleted greedy golden's
        replacement. On the deterministic ``_make_alignable`` scene (ball rests at p1_0@x20, departs
        f10, arrives p1_1@x45 f20, departs f30, arrives p1_2@x70 f40, departs f50), the NW DP must
        produce EXACTLY these start + reception frames -- a whole-alignment shift by a frame (the bug
        class this feature exists to prevent) fails HERE, where the property tests (columns / unit
        interval / reception-populated) all still pass. Values captured 2026-09-14 from the shipped
        engine; deterministic (synthetic scene has zero jitter -> unit confidence, zero error)."""
        frames, actions = _make_alignable()
        result = align_events_to_frames(actions, frames).sort_values("action_id").reset_index(drop=True)
        assert result["action_id"].tolist() == [0, 1, 2]
        # START: each pass aligns to its exact departure frame.
        assert [int(v) for v in result["elastic_frame_id"]] == [10, 30, 50]
        # RECEPTION: p1_0->p1_1 arrives f20; p1_1->p1_2 arrives f40; the last pass has no in-window
        # same-episode arrival -> NA (an inherent trailing-event edge, pinned so a regression is loud).
        recv = result["elastic_receive_frame_id"]
        assert int(recv.iloc[0]) == 20
        assert int(recv.iloc[1]) == 40
        assert pd.isna(recv.iloc[2])
        # exact matches on a zero-jitter synthetic scene -> unit confidence, zero error.
        assert (result["elastic_confidence"].dropna() == 1.0).all()
        assert (result["elastic_error_seconds"].dropna() == 0.0).all()
        assert (result["elastic_receive_confidence"].dropna() == 1.0).all()


class TestAlignEventsNonZeroFrameOrigin:
    """Regression: native-frame-numbered providers (IDSSE/Sportec).

    ``frame_id`` has a non-zero origin (10000+) while ``time_seconds`` is period-elapsed (0-based).
    The frame window AND the frame->time conversion must derive from the frames' own
    ``(frame_id, time_seconds)`` relationship, not from ``time * frame_rate``.
    """

    def test_aligned_frame_in_native_range(self):
        frames, actions = _make_alignable(frame_offset=10000)
        result = align_events_to_frames(actions, frames)
        assert len(result) > 0
        matched = result["elastic_frame_id"].dropna().astype(int)
        assert (matched >= 10000).all()
        assert (matched < 10100).all()

    def test_error_seconds_sane(self):
        """error_seconds uses the frames' frame->time relationship (not a ~400 s misread)."""
        frames, actions = _make_alignable(frame_offset=10000)
        result = align_events_to_frames(actions, frames)
        assert len(result) > 0
        assert (result["elastic_error_seconds"].dropna() <= 2.0).all()

    def test_multi_period_distinct_origins(self):
        """Each period's fit is independent: P1 -> 10000-range, P2 -> 100000-range."""
        f1, a1 = _make_alignable(frame_offset=10000, period_id=1)
        f2, a2 = _make_alignable(frame_offset=100000, period_id=2)
        a2 = a2.assign(action_id=a2["action_id"] + 100)
        frames = pd.concat([f1, f2], ignore_index=True)
        actions = pd.concat([a1, a2], ignore_index=True)
        result = align_events_to_frames(actions, frames)
        merged = result.merge(actions[["action_id", "period_id"]], on="action_id")
        p1 = merged[merged["period_id"] == 1]["elastic_frame_id"].dropna().astype(int)
        p2 = merged[merged["period_id"] == 2]["elastic_frame_id"].dropna().astype(int)
        assert len(p1) > 0 and len(p2) > 0
        assert (p1 >= 10000).all() and (p1 < 100000).all()
        assert (p2 >= 100000).all()

    def test_falls_back_when_time_seconds_absent(self):
        """No time_seconds column -> fall back to time*frame_rate (0-based providers)."""
        frames, actions = _make_alignable(frame_offset=0)
        frames = frames.drop(columns=["time_seconds"])
        result = align_events_to_frames(actions, frames)
        assert len(result) > 0
        assert (result["elastic_frame_id"].dropna().astype(int) < 100).all()


# ---------------------------------------------------------------------------
# Aggregator + VAEP factory tests
# ---------------------------------------------------------------------------


def _make_spadl_actions() -> pd.DataFrame:
    """Minimal SPADL-like actions for aggregator tests."""
    return pd.DataFrame(
        {
            "action_id": [0, 1, 2],
            "game_id": [1, 1, 1],
            "period_id": [1, 1, 1],
            "time_seconds": [0.2, 0.8, 1.4],
            "team_id": [1, 1, 2],
            "player_id": ["p1_0", "p1_1", "p2_0"],
            "start_x": [30.0, 40.0, 70.0],
            "start_y": [34.0, 34.0, 34.0],
            "end_x": [40.0, 50.0, 60.0],
            "end_y": [34.0, 34.0, 34.0],
            "type_id": [0, 0, 0],
            "type_name": ["pass", "pass", "pass"],
            "result_id": [1, 1, 1],
            "result_name": ["success", "success", "success"],
            "bodypart_id": [0, 0, 0],
            "bodypart_name": ["foot", "foot", "foot"],
        }
    )


class TestAddElasticSync:
    def test_enrichment_columns(self):
        """add_elastic_sync adds the 6 ELASTIC columns (start + reception)."""
        from silly_kicks.tracking.features import add_elastic_sync

        actions = _make_spadl_actions()
        frames = _make_tracking_frames()
        result = add_elastic_sync(actions, frames)
        expected_cols = {
            "elastic_frame_id",
            "elastic_confidence",
            "elastic_error_seconds",
            "elastic_receive_frame_id",
            "elastic_receive_confidence",
            "elastic_receive_error_seconds",
        }
        added = set(result.columns) - set(actions.columns)
        assert expected_cols.issubset(added)

    def test_row_count_preserved(self):
        """Row count unchanged after enrichment."""
        from silly_kicks.tracking.features import add_elastic_sync

        actions = _make_spadl_actions()
        frames = _make_tracking_frames()
        result = add_elastic_sync(actions, frames)
        assert len(result) == len(actions)

    def test_confidence_bounded(self):
        """elastic_confidence is in [0, 1] or NaN."""
        from silly_kicks.tracking.features import add_elastic_sync

        actions = _make_spadl_actions()
        frames = _make_tracking_frames()
        result = add_elastic_sync(actions, frames)
        vals = result["elastic_confidence"].dropna()
        if len(vals) > 0:
            assert vals.min() >= 0.0
            assert vals.max() <= 1.0


class TestElasticSyncXfns:
    def test_column_count(self):
        """elastic_sync_xfns produces 2 lifted xfns (6 VAEP columns)."""
        from silly_kicks.tracking.features import elastic_sync_xfns

        xfns = elastic_sync_xfns()
        assert len(xfns) == 2

    def test_introspection_nan(self):
        """xfns produce NaN in introspection mode (frames=None)."""
        from silly_kicks.tracking.features import elastic_sync_xfns

        xfns = elastic_sync_xfns()
        actions = pd.DataFrame(
            {
                "game_id": [1],
                "action_id": [0],
                "period_id": [1],
                "time_seconds": [10.0],
                "team_id": [1],
                "player_id": ["p1_0"],
                "start_x": [30.0],
                "start_y": [34.0],
                "end_x": [60.0],
                "end_y": [30.0],
                "type_id": [0],
                "type_name": ["pass"],
                "result_id": [1],
                "result_name": ["success"],
                "bodypart_id": [0],
                "bodypart_name": ["foot"],
            }
        )
        gamestates = [actions, actions, actions]
        for xfn in xfns:
            result = xfn(gamestates, None)
            assert result.isna().all().all() or result.isna().all()
