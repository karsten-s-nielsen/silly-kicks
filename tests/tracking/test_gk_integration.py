"""Integration tests for GK identification using synthetic fixtures (PR-S26)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from silly_kicks.tracking._gk_identification import derive_goalkeepers

SYNTHETIC_DIR = Path(__file__).resolve().parent.parent / "datasets" / "tracking" / "synthetic"


class TestGkBaselineRegression:
    """Regression tests against gk_baseline.json."""

    @pytest.fixture
    def baseline(self) -> dict:
        with open(SYNTHETIC_DIR / "gk_baseline.json") as f:
            return json.load(f)

    @pytest.mark.parametrize(
        "fixture_name",
        ["gk_substitution.parquet", "sweeper_keeper.parquet", "brief_outfielder.parquet"],
    )
    def test_picks_match_baseline(self, baseline: dict, fixture_name: str):
        """Verify algorithm picks match baseline JSON for all fixtures."""
        frames = pd.read_parquet(SYNTHETIC_DIR / fixture_name)
        _frames_out, picks = derive_goalkeepers(frames)

        expected = baseline["fixtures"][fixture_name]["expected_picks"]
        for key_str, expected_gks in expected.items():
            game_id, team_id = key_str.split("|")
            actual_gks = picks.get((game_id, team_id), [])
            assert set(actual_gks) == set(expected_gks), (
                f"{fixture_name} {key_str}: expected {expected_gks}, got {actual_gks}"
            )


class TestGkSubstitutionFixture:
    """Tests using gk_substitution.parquet fixture."""

    @pytest.fixture
    def frames(self) -> pd.DataFrame:
        return pd.read_parquet(SYNTHETIC_DIR / "gk_substitution.parquet")

    def test_multi_gk_detection_home_team(self, frames: pd.DataFrame):
        """Home team: both starter and sub detected as GK."""
        _frames_out, picks = derive_goalkeepers(frames)
        home_picks = picks.get(("gk_sub_match", "home"), [])
        assert set(home_picks) == {"gk_starter_home", "gk_sub_home"}

    def test_multi_gk_detection_away_team(self, frames: pd.DataFrame):
        """Away team: both starter and sub detected as GK."""
        _frames_out, picks = derive_goalkeepers(frames)
        away_picks = picks.get(("gk_sub_match", "away"), [])
        assert set(away_picks) == {"gk_starter_away", "gk_sub_away"}

    def test_is_goalkeeper_flag_set_correctly(self, frames: pd.DataFrame):
        """All detected GK rows have is_goalkeeper=True."""
        frames_out, picks = derive_goalkeepers(frames)
        for (game_id, team_id), gk_ids in picks.items():
            for gk_id in gk_ids:
                gk_rows = frames_out[
                    (frames_out["game_id"] == game_id)
                    & (frames_out["team_id"] == team_id)
                    & (frames_out["player_id"] == gk_id)
                ]
                assert gk_rows["is_goalkeeper"].all(), f"GK {gk_id} not flagged"

    def test_outfielders_not_flagged_as_gk(self, frames: pd.DataFrame):
        """Outfielders should not have is_goalkeeper=True."""
        frames_out, picks = derive_goalkeepers(frames)
        all_gks = set()
        for gk_ids in picks.values():
            all_gks.update(gk_ids)
        player_rows = frames_out[~frames_out["is_ball"]]
        outfielder_rows = player_rows[~player_rows["player_id"].isin(all_gks)]
        assert not outfielder_rows["is_goalkeeper"].any()


class TestSweeperKeeperFixture:
    """Tests using sweeper_keeper.parquet fixture."""

    @pytest.fixture
    def frames(self) -> pd.DataFrame:
        return pd.read_parquet(SYNTHETIC_DIR / "sweeper_keeper.parquet")

    def test_sweeper_gk_detected_via_fallback(self, frames: pd.DataFrame):
        """Sweeper-keeper (pa_dwell<0.4) detected via rank-sum fallback."""
        _frames_out, picks = derive_goalkeepers(frames)
        home_picks = picks.get(("sweeper_match", "home"), [])
        assert home_picks == ["sweeper_gk"]

    def test_sweeper_gk_flagged_correctly(self, frames: pd.DataFrame):
        """Sweeper-keeper rows have is_goalkeeper=True."""
        frames_out, _ = derive_goalkeepers(frames)
        sweeper_rows = frames_out[frames_out["player_id"] == "sweeper_gk"]
        assert sweeper_rows["is_goalkeeper"].all()


class TestBriefOutfielderFixture:
    """Tests using brief_outfielder.parquet fixture."""

    @pytest.fixture
    def frames(self) -> pd.DataFrame:
        return pd.read_parquet(SYNTHETIC_DIR / "brief_outfielder.parquet")

    def test_brief_sub_excluded_from_picks(self, frames: pd.DataFrame):
        """Brief substitute (<30% frames) excluded from candidate pool."""
        _frames_out, picks = derive_goalkeepers(frames)
        home_picks = picks.get(("brief_match", "home"), [])
        assert home_picks == ["real_gk"]
        assert "brief_sub_near_goal" not in home_picks

    def test_real_gk_flagged_correctly(self, frames: pd.DataFrame):
        """Real GK (full coverage) correctly flagged."""
        frames_out, _ = derive_goalkeepers(frames)
        real_gk_rows = frames_out[frames_out["player_id"] == "real_gk"]
        assert real_gk_rows["is_goalkeeper"].all()
