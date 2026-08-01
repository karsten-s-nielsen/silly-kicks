"""Tests for silly_kicks.tracking._ball_carrier.infer_ball_carrier."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest


def _make_carrier_frame(
    *,
    game_id=1,
    period_id=1,
    frame_id=1,
    ball_x=50.0,
    ball_y=34.0,
    ball_state: str | None = "alive",
    players: list[dict],
) -> pd.DataFrame:
    """Build a single-frame fixture for ball carrier tests.

    Each player dict: {pid, tid, x, y} + optional {vx, vy, is_goalkeeper}.
    Ball row omits vx/vy keys — pandas fills them with NaN via dict-union,
    so ``has_velocity`` (column-existence check) is True when any player
    dict includes vx/vy. Ball-row NaN velocity is intentional and doesn't
    affect carrier inference (ball rows are filtered out before scoring).
    """
    rows = []
    # Ball row
    rows.append(
        dict(
            game_id=game_id,
            period_id=period_id,
            frame_id=frame_id,
            time_seconds=frame_id * 0.04,
            frame_rate=25.0,
            player_id=np.nan,
            team_id=np.nan,
            is_ball=True,
            is_goalkeeper=False,
            x=ball_x,
            y=ball_y,
            ball_state=ball_state,
            source_provider="sportec",
            team_attacking_direction="ltr",
        )
    )
    for p in players:
        row = dict(
            game_id=game_id,
            period_id=period_id,
            frame_id=frame_id,
            time_seconds=frame_id * 0.04,
            frame_rate=25.0,
            player_id=p["pid"],
            team_id=p["tid"],
            is_ball=False,
            is_goalkeeper=p.get("is_goalkeeper", False),
            x=p["x"],
            y=p["y"],
            ball_state=ball_state,
            source_provider="sportec",
            team_attacking_direction="ltr",
        )
        if "vx" in p:
            row["vx"] = p["vx"]
            row["vy"] = p["vy"]
        rows.append(row)
    return pd.DataFrame(rows)


def _concat_frames(*frame_dfs: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(frame_dfs, ignore_index=True)


_RESULT_COLS = [
    "game_id",
    "period_id",
    "frame_id",
    "ball_carrier_player_id",
    "ball_carrier_distance_m",
    "ball_carrier_team_id",
]


class TestVelocityAwareScoring:
    def test_velocity_breaks_distance_tie(self):
        """Player farther away but moving toward ball wins over closer stationary player."""
        from silly_kicks.tracking._ball_carrier import infer_ball_carrier

        frames = _make_carrier_frame(
            ball_x=50.0,
            ball_y=34.0,
            players=[
                # Player 10: closer (1.5m) but stationary
                dict(pid=10, tid=1, x=51.5, y=34.0, vx=0.0, vy=0.0),
                # Player 20: farther (2.5m) but moving toward ball at 4 m/s
                # velocity toward ball = 4.0 m/s; score = 2.5 - 0.5*4 = 0.5
                # vs player 10: score = 1.5 - 0.5*0 = 1.5
                dict(pid=20, tid=1, x=52.5, y=34.0, vx=-4.0, vy=0.0),
            ],
        )
        # Logic test: beta is set explicitly so the velocity-weighting assertion is
        # independent of the calibrated default (beta=0.0 = pure distance, TF-24).
        # This behavior is only observable with beta>0.
        result = infer_ball_carrier(frames, beta=0.5)
        assert result["ball_carrier_player_id"].iloc[0] == 20


class TestHysteresis:
    def test_incumbent_retained_within_gamma(self):
        """Incumbent carrier kept when new candidate is only slightly better."""
        from silly_kicks.tracking._ball_carrier import infer_ball_carrier

        # Frame 1: player 10 is closest (1.0m), becomes carrier
        f1 = _make_carrier_frame(
            frame_id=1,
            ball_x=50.0,
            ball_y=34.0,
            players=[
                dict(pid=10, tid=1, x=51.0, y=34.0, vx=0.0, vy=0.0),
                dict(pid=20, tid=1, x=53.0, y=34.0, vx=0.0, vy=0.0),
            ],
        )
        # Frame 2: player 20 is now closer (1.8m vs 2.0m for player 10)
        # But difference (0.2m) < gamma (1.0m), so incumbent (10) stays
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

    def test_incumbent_overridden_when_exceeds_gamma(self):
        """New carrier wins when score difference exceeds gamma."""
        from silly_kicks.tracking._ball_carrier import infer_ball_carrier

        # Frame 1: player 10 at 2.0m
        f1 = _make_carrier_frame(
            frame_id=1,
            ball_x=50.0,
            ball_y=34.0,
            players=[
                dict(pid=10, tid=1, x=52.0, y=34.0, vx=0.0, vy=0.0),
                dict(pid=20, tid=1, x=55.0, y=34.0, vx=0.0, vy=0.0),
            ],
        )
        # Frame 2: player 20 now at 0.5m, player 10 at 2.5m
        # Difference = 2.0m > gamma(1.0m), so 20 takes over
        f2 = _make_carrier_frame(
            frame_id=2,
            ball_x=50.0,
            ball_y=34.0,
            players=[
                dict(pid=10, tid=1, x=52.5, y=34.0, vx=0.0, vy=0.0),
                dict(pid=20, tid=1, x=50.5, y=34.0, vx=0.0, vy=0.0),
            ],
        )
        frames = _concat_frames(f1, f2)
        result = infer_ball_carrier(frames, gamma=1.0)
        carriers = result.sort_values("frame_id")["ball_carrier_player_id"].tolist()
        assert carriers == [10, 20]

    def test_hysteresis_resets_on_dead_ball(self):
        """Dead-ball gap clears incumbent; next alive frame uses pure scoring."""
        from silly_kicks.tracking._ball_carrier import infer_ball_carrier

        f1 = _make_carrier_frame(
            frame_id=1,
            ball_x=50.0,
            ball_y=34.0,
            players=[
                dict(pid=10, tid=1, x=51.0, y=34.0, vx=0.0, vy=0.0),
                dict(pid=20, tid=1, x=53.0, y=34.0, vx=0.0, vy=0.0),
            ],
        )
        # Frame 2: dead ball
        f2 = _make_carrier_frame(
            frame_id=2,
            ball_x=50.0,
            ball_y=34.0,
            ball_state="dead",
            players=[
                dict(pid=10, tid=1, x=51.0, y=34.0, vx=0.0, vy=0.0),
                dict(pid=20, tid=1, x=53.0, y=34.0, vx=0.0, vy=0.0),
            ],
        )
        # Frame 3: alive again, player 20 is closer (0.8m vs 1.0m for 10)
        # No incumbent -> 20 wins on pure scoring (difference < gamma but no incumbent)
        f3 = _make_carrier_frame(
            frame_id=3,
            ball_x=50.0,
            ball_y=34.0,
            players=[
                dict(pid=10, tid=1, x=51.0, y=34.0, vx=0.0, vy=0.0),
                dict(pid=20, tid=1, x=50.8, y=34.0, vx=0.0, vy=0.0),
            ],
        )
        frames = _concat_frames(f1, f2, f3)
        result = infer_ball_carrier(frames, gamma=1.0)
        r = result.sort_values("frame_id")
        assert r.iloc[0]["ball_carrier_player_id"] == 10  # frame 1
        assert pd.isna(r.iloc[1]["ball_carrier_player_id"])  # frame 2 dead
        assert r.iloc[2]["ball_carrier_player_id"] == 20  # frame 3 no incumbent

    def test_hysteresis_resets_on_nan_carrier(self):
        """No-candidate frame clears incumbent."""
        from silly_kicks.tracking._ball_carrier import infer_ball_carrier

        f1 = _make_carrier_frame(
            frame_id=1,
            ball_x=50.0,
            ball_y=34.0,
            players=[
                dict(pid=10, tid=1, x=51.0, y=34.0, vx=0.0, vy=0.0),
            ],
        )
        # Frame 2: player too far (> tolerance 3m)
        f2 = _make_carrier_frame(
            frame_id=2,
            ball_x=50.0,
            ball_y=34.0,
            players=[
                dict(pid=10, tid=1, x=60.0, y=34.0, vx=0.0, vy=0.0),
            ],
        )
        # Frame 3: player 20 close, should win without hysteresis (incumbent was reset)
        f3 = _make_carrier_frame(
            frame_id=3,
            ball_x=50.0,
            ball_y=34.0,
            players=[
                dict(pid=10, tid=1, x=51.5, y=34.0, vx=0.0, vy=0.0),
                dict(pid=20, tid=1, x=51.0, y=34.0, vx=0.0, vy=0.0),
            ],
        )
        frames = _concat_frames(f1, f2, f3)
        result = infer_ball_carrier(frames, tolerance_m=3.0, gamma=1.0)
        r = result.sort_values("frame_id")
        assert r.iloc[0]["ball_carrier_player_id"] == 10
        assert pd.isna(r.iloc[1]["ball_carrier_player_id"])
        assert r.iloc[2]["ball_carrier_player_id"] == 20

    def test_first_frame_no_hysteresis(self):
        """First frame of period: pure scoring, no gamma bonus."""
        from silly_kicks.tracking._ball_carrier import infer_ball_carrier

        frames = _make_carrier_frame(
            frame_id=1,
            ball_x=50.0,
            ball_y=34.0,
            players=[
                dict(pid=10, tid=1, x=51.5, y=34.0, vx=0.0, vy=0.0),
                dict(pid=20, tid=1, x=51.0, y=34.0, vx=0.0, vy=0.0),
            ],
        )
        result = infer_ball_carrier(frames, gamma=1.0)
        assert result["ball_carrier_player_id"].iloc[0] == 20  # closer wins


class TestDistanceOnlyFallback:
    def test_fallback_when_no_velocity_columns(self):
        """Correct carrier + UserWarning when vx/vy absent."""
        from silly_kicks.tracking._ball_carrier import infer_ball_carrier

        frames = _make_carrier_frame(
            ball_x=50.0,
            ball_y=34.0,
            players=[
                dict(pid=10, tid=1, x=51.0, y=34.0),  # no vx/vy
                dict(pid=20, tid=1, x=53.0, y=34.0),
            ],
        )
        # Columns vx/vy not present
        assert "vx" not in frames.columns
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = infer_ball_carrier(frames)
            assert any("vx/vy columns not found" in str(x.message) for x in w)
        assert result["ball_carrier_player_id"].iloc[0] == 10

    def test_distance_only_with_hysteresis(self):
        """Hysteresis applies even in distance-only mode."""
        from silly_kicks.tracking._ball_carrier import infer_ball_carrier

        f1 = _make_carrier_frame(
            frame_id=1,
            ball_x=50.0,
            ball_y=34.0,
            players=[
                dict(pid=10, tid=1, x=51.0, y=34.0),
                dict(pid=20, tid=1, x=53.0, y=34.0),
            ],
        )
        f2 = _make_carrier_frame(
            frame_id=2,
            ball_x=50.0,
            ball_y=34.0,
            players=[
                dict(pid=10, tid=1, x=52.0, y=34.0),  # 2.0m
                dict(pid=20, tid=1, x=51.8, y=34.0),  # 1.8m, but delta < gamma
            ],
        )
        frames = _concat_frames(f1, f2)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = infer_ball_carrier(frames, gamma=1.0)
        carriers = result.sort_values("frame_id")["ball_carrier_player_id"].tolist()
        assert carriers == [10, 10]  # incumbent retained


class TestEdgeCases:
    def test_dead_ball_nan(self):
        from silly_kicks.tracking._ball_carrier import infer_ball_carrier

        frames = _make_carrier_frame(
            ball_state="dead",
            players=[dict(pid=10, tid=1, x=51.0, y=34.0, vx=0.0, vy=0.0)],
        )
        result = infer_ball_carrier(frames)
        assert pd.isna(result["ball_carrier_player_id"].iloc[0])

    def test_ball_state_nan_treated_as_alive(self):
        from silly_kicks.tracking._ball_carrier import infer_ball_carrier

        frames = _make_carrier_frame(
            ball_state=None,
            players=[dict(pid=10, tid=1, x=51.0, y=34.0, vx=0.0, vy=0.0)],
        )
        result = infer_ball_carrier(frames)
        assert result["ball_carrier_player_id"].iloc[0] == 10

    def test_no_ball_row_nan(self):
        from silly_kicks.tracking._ball_carrier import infer_ball_carrier

        frames = _make_carrier_frame(
            players=[dict(pid=10, tid=1, x=51.0, y=34.0, vx=0.0, vy=0.0)],
        )
        # Remove ball row — player rows remain, so unique_frames still has
        # this (game_id, period_id, frame_id), but ball_pos merge yields NaN.
        frames = frames[~frames["is_ball"]].reset_index(drop=True)
        result = infer_ball_carrier(frames)
        assert len(result) == 1
        assert pd.isna(result["ball_carrier_player_id"].iloc[0])

    def test_ball_coords_nan(self):
        from silly_kicks.tracking._ball_carrier import infer_ball_carrier

        frames = _make_carrier_frame(
            ball_x=np.nan,
            ball_y=np.nan,
            players=[dict(pid=10, tid=1, x=51.0, y=34.0, vx=0.0, vy=0.0)],
        )
        result = infer_ball_carrier(frames)
        assert pd.isna(result["ball_carrier_player_id"].iloc[0])

    def test_no_candidates_within_tolerance(self):
        from silly_kicks.tracking._ball_carrier import infer_ball_carrier

        frames = _make_carrier_frame(
            ball_x=50.0,
            ball_y=34.0,
            players=[dict(pid=10, tid=1, x=60.0, y=34.0, vx=0.0, vy=0.0)],
        )
        result = infer_ball_carrier(frames, tolerance_m=3.0)
        assert pd.isna(result["ball_carrier_player_id"].iloc[0])

    def test_gk_as_carrier(self):
        from silly_kicks.tracking._ball_carrier import infer_ball_carrier

        frames = _make_carrier_frame(
            ball_x=5.0,
            ball_y=34.0,
            players=[
                dict(pid=1, tid=1, x=5.5, y=34.0, vx=0.0, vy=0.0, is_goalkeeper=True),
                dict(pid=10, tid=1, x=15.0, y=34.0, vx=0.0, vy=0.0),
            ],
        )
        result = infer_ball_carrier(frames)
        assert result["ball_carrier_player_id"].iloc[0] == 1

    def test_tiebreak_determinism(self):
        from silly_kicks.tracking._ball_carrier import infer_ball_carrier

        frames = _make_carrier_frame(
            ball_x=50.0,
            ball_y=34.0,
            players=[
                dict(pid=20, tid=1, x=51.0, y=34.0, vx=0.0, vy=0.0),
                dict(pid=10, tid=1, x=51.0, y=34.0, vx=0.0, vy=0.0),
            ],
        )
        result = infer_ball_carrier(frames)
        assert result["ball_carrier_player_id"].iloc[0] == 10  # lowest pid

    def test_empty_frames(self):
        from silly_kicks.tracking._ball_carrier import infer_ball_carrier

        frames = pd.DataFrame(
            columns=[
                "game_id",
                "period_id",
                "frame_id",
                "time_seconds",
                "frame_rate",
                "player_id",
                "team_id",
                "is_ball",
                "is_goalkeeper",
                "x",
                "y",
                "ball_state",
                "source_provider",
                "team_attacking_direction",
            ]
        )
        result = infer_ball_carrier(frames)
        assert list(result.columns) == _RESULT_COLS
        assert len(result) == 0

    def test_set_piece_transition(self):
        from silly_kicks.tracking._ball_carrier import infer_ball_carrier

        # Frame 1: dead ball
        f1 = _make_carrier_frame(
            frame_id=1,
            ball_x=50.0,
            ball_y=34.0,
            ball_state="dead",
            players=[
                dict(pid=10, tid=1, x=50.5, y=34.0, vx=0.0, vy=0.0),
                dict(pid=20, tid=1, x=50.3, y=34.0, vx=0.0, vy=0.0),
            ],
        )
        # Frame 2: alive — kicker (20) closer
        f2 = _make_carrier_frame(
            frame_id=2,
            ball_x=50.0,
            ball_y=34.0,
            ball_state="alive",
            players=[
                dict(pid=10, tid=1, x=50.5, y=34.0, vx=0.0, vy=0.0),
                dict(pid=20, tid=1, x=50.2, y=34.0, vx=1.0, vy=0.0),
            ],
        )
        frames = _concat_frames(f1, f2)
        result = infer_ball_carrier(frames)
        r = result.sort_values("frame_id")
        assert pd.isna(r.iloc[0]["ball_carrier_player_id"])
        assert r.iloc[1]["ball_carrier_player_id"] == 20

    def test_multiple_ball_rows_mean(self):
        from silly_kicks.tracking._ball_carrier import infer_ball_carrier

        frames = _make_carrier_frame(
            ball_x=50.0,
            ball_y=34.0,
            players=[dict(pid=10, tid=1, x=52.0, y=35.0, vx=0.0, vy=0.0)],
        )
        # Add a second ball row with different position
        extra_ball = pd.DataFrame(
            [
                dict(
                    game_id=1,
                    period_id=1,
                    frame_id=1,
                    time_seconds=0.04,
                    frame_rate=25.0,
                    player_id=np.nan,
                    team_id=np.nan,
                    is_ball=True,
                    is_goalkeeper=False,
                    x=54.0,
                    y=36.0,
                    ball_state="alive",
                    source_provider="sportec",
                    team_attacking_direction="ltr",
                    vx=np.nan,
                    vy=np.nan,
                )
            ]
        )
        frames = pd.concat([frames, extra_ball], ignore_index=True)
        result = infer_ball_carrier(frames)
        # Mean ball pos = (52, 35); player at (52, 35) -> distance ~0
        # Just verify it produces a result without error
        assert result["ball_carrier_player_id"].iloc[0] == 10

    def test_multi_game_batch(self):
        from silly_kicks.tracking._ball_carrier import infer_ball_carrier

        f1 = _make_carrier_frame(
            game_id=1,
            frame_id=1,
            ball_x=50.0,
            ball_y=34.0,
            players=[dict(pid=10, tid=1, x=51.0, y=34.0, vx=0.0, vy=0.0)],
        )
        f2 = _make_carrier_frame(
            game_id=2,
            frame_id=1,
            ball_x=50.0,
            ball_y=34.0,
            players=[dict(pid=20, tid=2, x=51.5, y=34.0, vx=0.0, vy=0.0)],
        )
        frames = _concat_frames(f1, f2)
        result = infer_ball_carrier(frames)
        assert len(result) == 2
        g1 = result[result["game_id"] == 1].iloc[0]
        g2 = result[result["game_id"] == 2].iloc[0]
        assert g1["ball_carrier_player_id"] == 10
        assert g2["ball_carrier_player_id"] == 20


class TestReturnSchema:
    def test_output_columns(self):
        from silly_kicks.tracking._ball_carrier import infer_ball_carrier

        frames = _make_carrier_frame(
            players=[dict(pid=10, tid=1, x=51.0, y=34.0, vx=0.0, vy=0.0)],
        )
        result = infer_ball_carrier(frames)
        assert list(result.columns) == _RESULT_COLS

    def test_distance_bounded_by_tolerance(self):
        from silly_kicks.tracking._ball_carrier import infer_ball_carrier

        frames = _make_carrier_frame(
            players=[dict(pid=10, tid=1, x=51.0, y=34.0, vx=0.0, vy=0.0)],
        )
        result = infer_ball_carrier(frames, tolerance_m=3.0)
        valid = result["ball_carrier_distance_m"].dropna()
        assert (valid <= 3.0).all()

    def test_team_id_matches_carrier(self):
        from silly_kicks.tracking._ball_carrier import infer_ball_carrier

        frames = _make_carrier_frame(
            players=[
                dict(pid=10, tid=1, x=51.0, y=34.0, vx=0.0, vy=0.0),
                dict(pid=20, tid=2, x=53.0, y=34.0, vx=0.0, vy=0.0),
            ],
        )
        result = infer_ball_carrier(frames)
        assert result["ball_carrier_team_id"].iloc[0] == 1

    def test_fresh_range_index(self):
        from silly_kicks.tracking._ball_carrier import infer_ball_carrier

        frames = _make_carrier_frame(
            players=[dict(pid=10, tid=1, x=51.0, y=34.0, vx=0.0, vy=0.0)],
        )
        result = infer_ball_carrier(frames)
        assert list(result.index) == list(range(len(result)))


class TestCalibratedDefaults:
    """The carrier scoring defaults are TF-24 Optuna-calibrated (apply-PR PR-S79).

    ``beta``/``gamma`` were calibrated at the held ``tolerance_m=3.0`` against a 3-provider fold
    (Balanced + Gold-max agreed: beta≈0, gamma≈0.24). ``tolerance_m`` is held at 3.0 because the
    carrier-actor-action objective is under-determined on the radius. This is the single intentional
    guard on the calibrated values — if they change, the calibration record must change too. Logic
    tests elsewhere pass scoring params explicitly so they stay independent of these constants.
    """

    def test_infer_ball_carrier_calibrated_defaults(self):
        import inspect

        from silly_kicks.tracking._ball_carrier import infer_ball_carrier

        params = inspect.signature(infer_ball_carrier).parameters
        assert params["beta"].default == 0.0
        assert params["gamma"].default == 0.25
        assert params["tolerance_m"].default == 3.0  # held — objective-under-determined

    def test_ball_carrier_at_action_calibrated_defaults(self):
        import inspect

        from silly_kicks.tracking.features import ball_carrier_at_action

        params = inspect.signature(ball_carrier_at_action).parameters
        assert params["beta"].default == 0.0
        assert params["gamma"].default == 0.25
        assert params["tolerance_m"].default == 3.0


# Param sets spanning the current defaults, the recall-aware optimum region, and a tight
# radius — used by the cached-pre/links bit-identity tests.
_CACHE_EQUIV_PARAMS = [
    dict(tolerance_m=3.0, beta=0.5, gamma=1.0),
    dict(tolerance_m=7.0, beta=0.0, gamma=0.1),
    dict(tolerance_m=1.0, beta=1.0, gamma=2.0),
]


class TestCachedPreLinks:
    """A precomputed ``pre`` (and cached ``links``) is bit-identical to recomputing.

    The pre-index (long-form → dense numpy) is a pure function of ``frames`` and the linking
    depends only on the fixed link tolerance — both independent of the carrier-scoring params.
    Callers re-resolving carriers on the same frames with different params (the TF-24 sweep)
    cache these once; the result must match recomputing from scratch exactly.
    """

    def _multi_frame(self):
        f1 = _make_carrier_frame(
            frame_id=1,
            players=[
                dict(pid=10, tid=1, x=51.0, y=34.0, vx=0.5, vy=0.0),
                dict(pid=20, tid=1, x=53.0, y=34.0, vx=-1.0, vy=0.0),
                dict(pid=30, tid=2, x=57.0, y=34.0, vx=0.0, vy=0.0),
            ],
        )
        f2 = _make_carrier_frame(
            frame_id=2,
            players=[
                dict(pid=10, tid=1, x=52.0, y=34.0, vx=0.0, vy=0.0),
                dict(pid=20, tid=1, x=51.7, y=34.0, vx=-2.0, vy=0.0),
                dict(pid=30, tid=2, x=56.0, y=34.0, vx=0.0, vy=0.0),
            ],
        )
        f3 = _make_carrier_frame(
            frame_id=3,
            players=[
                dict(pid=10, tid=1, x=55.0, y=34.0, vx=0.0, vy=0.0),
                dict(pid=20, tid=1, x=50.6, y=34.0, vx=0.0, vy=0.0),
                dict(pid=30, tid=2, x=58.0, y=34.0, vx=0.0, vy=0.0),
            ],
        )
        return _concat_frames(f1, f2, f3)

    def test_infer_ball_carrier_cached_pre_identical(self):
        from silly_kicks.tracking._ball_carrier import _pre_index_frames, infer_ball_carrier

        frames = self._multi_frame()
        pre = _pre_index_frames(frames)
        for params in _CACHE_EQUIV_PARAMS:
            recomputed = infer_ball_carrier(frames, **params)  # type: ignore[arg-type]
            cached = infer_ball_carrier(frames, pre=pre, **params)
            pd.testing.assert_frame_equal(recomputed, cached)

    def test_ball_carrier_at_action_cached_pre_links_identical(self):
        from silly_kicks.tracking._ball_carrier import _pre_index_frames
        from silly_kicks.tracking.features import ball_carrier_at_action
        from silly_kicks.tracking.utils import link_actions_to_frames

        frames = self._multi_frame()
        actions = pd.DataFrame(
            {
                "game_id": [1, 1, 1],
                "action_id": [0, 1, 2],
                "period_id": [1, 1, 1],
                "time_seconds": [0.04, 0.08, 0.12],  # match frame_id 1/2/3 (= frame_id * 0.04)
                "team_id": [1, 1, 1],
                "player_id": [10, 20, 20],
            }
        )
        pre = _pre_index_frames(frames)
        links, _ = link_actions_to_frames(actions, frames, tolerance_seconds=0.2)
        for params in _CACHE_EQUIV_PARAMS:
            recomputed = ball_carrier_at_action(actions, frames, **params)  # type: ignore[arg-type]
            cached = ball_carrier_at_action(actions, frames, pre=pre, links=links, **params)
            pd.testing.assert_series_equal(recomputed, cached)


class TestActionCoupledWrapper:
    def _make_actions(self, n=2):
        return pd.DataFrame(
            {
                "game_id": [1] * n,
                "action_id": list(range(n)),
                "period_id": [1] * n,
                "time_seconds": [0.04 * (i + 1) for i in range(n)],
                "team_id": [1] * n,
                "player_id": [10] * n,
            }
        )

    def test_linked_carrier_matches(self):
        from silly_kicks.tracking.features import ball_carrier_at_action

        frames = _make_carrier_frame(
            frame_id=1,
            ball_x=50.0,
            ball_y=34.0,
            players=[
                dict(pid=10, tid=1, x=51.0, y=34.0, vx=0.0, vy=0.0),
                dict(pid=20, tid=2, x=53.0, y=34.0, vx=0.0, vy=0.0),
            ],
        )
        actions = self._make_actions(n=1)
        actions["time_seconds"] = [0.04]  # matches frame_id=1
        result = ball_carrier_at_action(actions, frames)
        assert len(result) == 1
        assert result.iloc[0] == 10

    def test_unlinked_action_nan(self):
        from silly_kicks.tracking.features import ball_carrier_at_action

        frames = _make_carrier_frame(
            frame_id=1,
            ball_x=50.0,
            ball_y=34.0,
            players=[
                dict(pid=10, tid=1, x=51.0, y=34.0, vx=0.0, vy=0.0),
            ],
        )
        actions = self._make_actions(n=1)
        actions["time_seconds"] = [999.0]  # no matching frame
        result = ball_carrier_at_action(actions, frames)
        assert pd.isna(result.iloc[0])


# ---------------------------------------------------------------------------
# Phase 0c.1 — derive_team_in_possession preserves ball_carrier_player_id
# ---------------------------------------------------------------------------


def _tiny_poss_frames() -> pd.DataFrame:
    rows = []
    for fid in range(6):
        rows.append(
            {
                "game_id": 1,
                "period_id": 1,
                "frame_id": fid,
                "time_seconds": fid / 25,
                "player_id": None,
                "team_id": None,
                "is_ball": True,
                "is_goalkeeper": False,
                "x": 50.0 + fid,
                "y": 34.0,
                "ball_state": "alive",
            }
        )
        for pid, tid in [(10, 1), (20, 2)]:
            rows.append(
                {
                    "game_id": 1,
                    "period_id": 1,
                    "frame_id": fid,
                    "time_seconds": fid / 25,
                    "player_id": pid,
                    "team_id": tid,
                    "is_ball": False,
                    "is_goalkeeper": False,
                    "x": 50.0 + fid + tid,
                    "y": 34.0,
                    "ball_state": "alive",
                }
            )
    return pd.DataFrame(rows)


def test_derive_team_in_possession_preserves_carrier_player_id():
    from silly_kicks.tracking import derive_team_in_possession, infer_ball_carrier

    frames = _tiny_poss_frames()
    carrier = infer_ball_carrier(frames)
    out = derive_team_in_possession(frames, carrier)
    assert "team_in_possession" in out.columns
    assert "ball_carrier_player_id" in out.columns  # NEW: carrier id preserved
    merged = out.merge(
        carrier[["game_id", "period_id", "frame_id", "ball_carrier_player_id"]],
        on=["game_id", "period_id", "frame_id"],
        suffixes=("", "_ref"),
    )
    assert (merged["ball_carrier_player_id"].fillna(-1) == merged["ball_carrier_player_id_ref"].fillna(-1)).all()


# ---------------------------------------------------------------------------
# ADR-043 -- derive_team_in_possession is IDEMPOTENT
#
# A bare merge suffixed a pre-existing team_in_possession to _x/_y, and every
# consumer that re-derives possession (_xshot_occurrence, _xcross_attempt) then
# died on KeyError: 'team_in_possession'. Live for any pipeline that enriches one
# frames object for several families -- exactly what `links` / `pitch_control_cache`
# exist to encourage.
# ---------------------------------------------------------------------------


def test_derive_team_in_possession_is_idempotent():
    """Second call is a no-op on the COLUMN SET -- no _x/_y suffixes, ever."""
    from silly_kicks.tracking import derive_team_in_possession, infer_ball_carrier

    frames = _tiny_poss_frames()
    carrier = infer_ball_carrier(frames)

    once = derive_team_in_possession(frames, carrier)
    twice = derive_team_in_possession(once, carrier)

    assert list(twice.columns) == list(once.columns)
    assert not any(c.endswith(("_x", "_y")) for c in twice.columns)
    assert "team_in_possession" in twice.columns
    # Non-vacuity: the column must actually carry possession, not be an all-NaN husk.
    assert once["team_in_possession"].notna().any()
    pd.testing.assert_frame_equal(twice, once)


def test_derive_team_in_possession_replaces_stale_column():
    """A pre-existing column is REPLACED by THIS carrier's answer, never preserved.

    Deliberate (see docstring): the output must always agree with the `carrier`
    argument just passed; a retained column from an earlier/different carrier is the
    train-serve-skew shape.
    """
    from silly_kicks.tracking import derive_team_in_possession, infer_ball_carrier

    frames = _tiny_poss_frames()
    carrier = infer_ball_carrier(frames)

    stale = frames.copy()
    stale["team_in_possession"] = 999  # a value THIS carrier can never produce
    stale["ball_carrier_player_id"] = 999

    out = derive_team_in_possession(stale, carrier)
    assert (out["team_in_possession"] != 999).all()
    assert (out["ball_carrier_player_id"].fillna(-1) != 999).all()
    pd.testing.assert_frame_equal(out, derive_team_in_possession(frames, carrier))


def test_double_enriched_frames_survive_xshot_consumer():
    """Downstream proof: a twice-enriched frames object must not KeyError.

    `prepare_xshot_training_data` re-derives possession internally and then reads
    `grp["team_in_possession"]` -- precisely where the suffixed column detonated.
    """
    from silly_kicks.tracking import derive_team_in_possession, infer_ball_carrier
    from silly_kicks.tracking._xshot_occurrence import prepare_xshot_training_data

    frames = _tiny_poss_frames()
    carrier = infer_ball_carrier(frames)
    double = derive_team_in_possession(derive_team_in_possession(frames, carrier), carrier)
    assert "team_in_possession" in double.columns  # non-vacuity: the enrichment IS present

    shots = pd.DataFrame(
        {
            "game_id": [1],
            "period_id": [1],
            "team_id": [1],
            "time_seconds": [0.1],
            "type_name": ["shot"],
        }
    )
    # MUST NOT raise KeyError: 'team_in_possession'
    X, y, groups = prepare_xshot_training_data(double, shots, home_team_id=1, attacking_third_only=False)
    assert len(X) == len(y) == len(groups)


# ---------------------------------------------------------------------------
# Carrier inference is ORIENTATION-INVARIANT (ADR-051 / TF-24 disposition)
# ---------------------------------------------------------------------------


def _point_reflect(frames: pd.DataFrame) -> pd.DataFrame:
    """ADR-028's 180 degree point reflection, applied per ADR-045's kind rules.

    Positions point-reflect (``x -> 105 - x``, ``y -> 68 - y``); velocities NEGATE; the direction
    LABEL swaps. Reflecting positions while leaving velocities alone would model every player running
    backwards -- the exact D1 defect ADR-045 records -- and would make this test measure that bug
    instead of the property it claims to check.
    """
    out = frames.copy()
    out["x"] = 105.0 - out["x"]
    out["y"] = 68.0 - out["y"]
    for v in ("vx", "vy"):
        if v in out.columns:
            out[v] = -out[v]
    if "team_attacking_direction" in out.columns:
        out["team_attacking_direction"] = out["team_attacking_direction"].map({"ltr": "rtl", "rtl": "ltr"})
    return out


def _contest(i: int, rng) -> list[dict]:
    """Two contenders arranged so the carrier VARIES at both betas and velocity DECIDES at beta>0.

    Sensitivity is the whole point, and an earlier draft had none. With one clearly-nearest player the
    distance term dominates and the velocity term never flips the winner -- MEASURED: a
    `_point_reflect` that forgets to negate velocities (the ADR-045 D1 defect) produced an identical
    carrier series, so the test passed against a planted bug. A second draft placed the two exactly
    equidistant, which made the carrier CONSTANT at the shipped `beta=0.0` (deterministic tie-break)
    and tripped this test's own non-vacuity guard.

    The arrangement below, per frame, with `lead = i % 2`:

    * `lead` sits NEARER (1.0 m) and is stationary  -> wins on distance alone.
    * the other sits FARTHER (1.6 m) but closes at 4 m/s -> wins once `beta * v_toward` counts
      (`1.6 - 1.5*4 = -4.4` beats `1.0`).

    So at `beta=0.0` the nearer contender wins and at `beta=1.5` the closing one does -- the winner
    alternates with frame parity in BOTH cases, and at `beta=1.5` it is the velocity term that picks
    it. Under a correct point reflection the velocities negate too, `v_toward` is preserved, and the
    winner is unchanged; under a reflection that moves positions only, `v_toward` flips sign and the
    winner changes -- which is exactly the defect this test must catch.
    """
    bx, by = 20.0 + 1.5 * i, 34.0 + 8.0 * np.sin(i / 3.0)
    lead = i % 2
    players = []
    for j in (0, 1):
        near = j == lead
        # Place both on the -x side of the ball so "closing" is +x for both; distance differs.
        dist = 1.0 if near else 1.6
        players.append(
            {
                "pid": 100 + j,
                "tid": 1,
                "x": float(np.clip(bx - dist, 0.5, 104.5)),
                "y": float(np.clip(by, 0.5, 67.5)),
                "vx": 0.0 if near else 4.0,  # the farther one closes hard
                "vy": 0.0,
            }
        )
    for j in range(2, 6):
        players.append(
            {
                "pid": 100 + j,
                "tid": 1 if j < 3 else 2,
                "x": float(np.clip(bx + rng.normal(0, 9), 0.5, 104.5)),
                "y": float(np.clip(by + rng.normal(0, 15), 0.5, 67.5)),
                "vx": float(rng.normal(0, 3)),
                "vy": float(rng.normal(0, 3)),
            }
        )
    return players


@pytest.mark.parametrize("beta", [0.0, 1.5], ids=["shipped-beta-0", "velocity-live"])
def test_carrier_inference_is_orientation_invariant(beta):
    """The measured basis for ADR-051's TF-24 "no re-sweep trigger" verdict.

    That disposition rests on `infer_ball_carrier`'s shipped params surviving the ADR-028 corrections
    even though they were TF-24-calibrated on a fold that included the unoriented SkillCorner frames.
    The argument is that carrier inference reads no orientation at all, so a mis-oriented fold gave
    the same answer. It was stated as a measurement in four documents with **no artifact, script or
    test anywhere in the repo** -- so it is asserted here, where CI can keep it true.

    BOTH betas are exercised, and that is not padding. The shipped default is `beta=0.0`
    (`DEFAULT_CARRIER_PARAMS`), which reduces `scores[ci] = cand_dists[ci] - beta * v_toward` to pure
    distance -- so at the shipped setting the velocity columns are INERT and the `_point_reflect`
    helper's velocity negation is never exercised. Measured: negating velocities changes nothing at
    `beta=0.0` and changes the carrier at `beta=1.5`. The first version of this test ran only the
    default, so it verified the positional half of the reflection and silently skipped the vector
    half -- exactly the ADR-045 rule it was written to respect. `beta=0.0` pins the production
    configuration; `beta=1.5` pins the invariance where velocity actually participates.
    """
    from silly_kicks.tracking import infer_ball_carrier

    rng = np.random.default_rng(0)
    frames = _concat_frames(
        *[
            _make_carrier_frame(
                frame_id=i,
                ball_x=float(20.0 + 1.5 * i),
                ball_y=float(34.0 + 8.0 * np.sin(i / 3.0)),
                # The player nearest the ball ROTATES with the frame index, so the carrier varies
                # instead of being one constant id. A constant answer would make the equality below
                # compare two constant Series and pass for almost any implementation.
                players=_contest(i, rng),
            )
            for i in range(40)
        ]
    )

    # NOTHING is asserted about warnings here, and that is deliberate after two failed attempts.
    # A `filterwarnings("ignore", ".*vx/vy.*")` sat here first and was INERT -- `_contest` sets vx/vy
    # on every player, so it could never match. It was then replaced by an
    # `OrientationUnresolvedWarning` assertion, which is EQUALLY inert: `_ball_carrier.py` contains
    # zero orientation references (that is the very property this test exists to demonstrate), so
    # the warning cannot be emitted and the assertion cannot fail. Swapping one unfalsifiable guard
    # for another is not a fix.
    #
    # The real guards are below and they can all fail: >=30 carriers resolved, >=2 distinct ids, the
    # series equality, and the <1e-9 distance bound -- plus the planted-defect check recorded in
    # `_contest`'s docstring.
    original = infer_ball_carrier(frames, beta=beta)
    reflected = infer_ball_carrier(_point_reflect(frames), beta=beta)

    assert len(original) == len(reflected) == 40, "fixture did not yield one row per frame"

    # NON-VACUITY, two ways. (a) carriers must actually resolve; (b) the answer must VARY across
    # frames, or the comparison is between two constant Series and proves nothing.
    ids = original["ball_carrier_player_id"]
    assert ids.notna().sum() >= 30, f"only {ids.notna().sum()}/40 frames resolved a carrier"
    # >= 2 is exactly "not constant", which is the property that keeps the equality below meaningful.
    # The contested fixture alternates two contenders by frame parity, so 2 is the designed answer;
    # demanding 3 would be an arbitrary bar the design cannot meet.
    assert ids.dropna().nunique() >= 2, (
        f"carrier is CONSTANT ({ids.dropna().nunique()} distinct id) -- the equality below is vacuous"
    )

    pd.testing.assert_series_equal(
        original["ball_carrier_player_id"].reset_index(drop=True),
        reflected["ball_carrier_player_id"].reset_index(drop=True),
        check_names=False,
    )
    # Unconditional: `ball_carrier_distance_m` is in this file's own _RESULT_COLS, so a membership
    # check would be dead today and would silently DELETE this assertion if the column were renamed.
    delta = float(
        np.nanmax(
            np.abs(original["ball_carrier_distance_m"].to_numpy() - reflected["ball_carrier_distance_m"].to_numpy())
        )
    )
    assert delta < 1e-9, f"carrier distance moved by {delta:g} under a pure reflection"
