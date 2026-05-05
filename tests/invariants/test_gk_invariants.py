"""Physical invariants for GK identification (PR-S26).

These tests verify that the GK identification algorithm maintains logical
consistency properties regardless of input data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.tracking._gk_identification import derive_goalkeepers


class TestGkInvariants:
    """Physical invariant tests for derive_goalkeepers."""

    def _make_team_frames(
        self,
        players: list[dict],
        n_frames: int = 100,
        game_id: str = "inv_match",
        team_id: str = "t1",
    ) -> pd.DataFrame:
        """Build synthetic frames for invariant testing."""
        rows = []
        for _frame_id in range(n_frames):
            for p in players:
                rows.append(
                    {
                        "game_id": game_id,
                        "team_id": team_id,
                        "player_id": p["player_id"],
                        "x": np.clip(p["x"], 0, 105),
                        "y": np.clip(p["y"], 0, 68),
                        "is_ball": False,
                        "is_goalkeeper": False,
                    }
                )
        return pd.DataFrame(rows)

    def test_at_least_one_gk_per_team(self):
        """Each team with players must have at least one GK identified."""
        # Multiple teams in same match
        players_t1 = [
            {"player_id": "gk_t1", "x": 5.0, "y": 34.0},
            {"player_id": "p1_t1", "x": 50.0, "y": 34.0},
        ]
        players_t2 = [
            {"player_id": "gk_t2", "x": 100.0, "y": 34.0},
            {"player_id": "p1_t2", "x": 50.0, "y": 34.0},
        ]
        df1 = self._make_team_frames(players_t1, team_id="team1")
        df2 = self._make_team_frames(players_t2, team_id="team2")
        frames = pd.concat([df1, df2], ignore_index=True)

        _, picks = derive_goalkeepers(frames)

        assert ("inv_match", "team1") in picks
        assert ("inv_match", "team2") in picks
        assert len(picks[("inv_match", "team1")]) >= 1
        assert len(picks[("inv_match", "team2")]) >= 1

    def test_gk_rows_flagged_correctly(self):
        """All rows for identified GK have is_goalkeeper=True."""
        players = [
            {"player_id": "gk1", "x": 5.0, "y": 34.0},
            {"player_id": "p2", "x": 50.0, "y": 34.0},
            {"player_id": "p3", "x": 60.0, "y": 34.0},
        ]
        frames = self._make_team_frames(players, n_frames=50)
        frames_out, picks = derive_goalkeepers(frames)

        gk_ids = picks[("inv_match", "t1")]
        for gk_id in gk_ids:
            gk_rows = frames_out[frames_out["player_id"] == gk_id]
            assert gk_rows["is_goalkeeper"].all(), f"GK {gk_id} has False rows"

    def test_non_gk_rows_not_flagged(self):
        """Rows for non-GK players have is_goalkeeper=False."""
        players = [
            {"player_id": "gk1", "x": 5.0, "y": 34.0},
            {"player_id": "p2", "x": 50.0, "y": 34.0},
        ]
        frames = self._make_team_frames(players, n_frames=50)
        frames_out, picks = derive_goalkeepers(frames)

        gk_ids = set(picks[("inv_match", "t1")])
        non_gk_rows = frames_out[~frames_out["player_id"].isin(gk_ids)]
        assert not non_gk_rows["is_goalkeeper"].any()

    def test_ball_rows_unchanged(self):
        """Ball rows (is_ball=True) remain unaffected by algorithm."""
        players = [{"player_id": "gk1", "x": 5.0, "y": 34.0}]
        frames = self._make_team_frames(players, n_frames=50)
        ball_rows = pd.DataFrame(
            {
                "game_id": ["inv_match"] * 50,
                "team_id": [None] * 50,
                "player_id": [None] * 50,
                "x": [52.5] * 50,
                "y": [34.0] * 50,
                "is_ball": [True] * 50,
                "is_goalkeeper": [False] * 50,
            }
        )
        frames = pd.concat([frames, ball_rows], ignore_index=True)

        frames_out, _ = derive_goalkeepers(frames)

        ball_out = frames_out[frames_out["is_ball"]]
        assert len(ball_out) == 50
        assert not ball_out["is_goalkeeper"].any()

    def test_picks_subset_of_players(self):
        """All identified GKs must be actual player_ids in the input."""
        players = [
            {"player_id": "alice", "x": 5.0, "y": 34.0},
            {"player_id": "bob", "x": 50.0, "y": 34.0},
            {"player_id": "charlie", "x": 60.0, "y": 34.0},
        ]
        frames = self._make_team_frames(players, n_frames=100)
        _, picks = derive_goalkeepers(frames)

        all_player_ids = set(frames["player_id"].dropna())
        for (_game_id, _team_id), gk_ids in picks.items():
            for gk_id in gk_ids:
                assert gk_id in all_player_ids, f"GK {gk_id} not in input"
