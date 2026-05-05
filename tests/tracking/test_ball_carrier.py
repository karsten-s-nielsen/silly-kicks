"""Tests for silly_kicks.tracking._ball_carrier.infer_ball_carrier."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def _make_carrier_frame(
    *,
    game_id=1,
    period_id=1,
    frame_id=1,
    ball_x=50.0,
    ball_y=34.0,
    ball_state="alive",
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
        result = infer_ball_carrier(frames)
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
