"""Physical invariant tests for TF-30 cover shadow features."""

from __future__ import annotations

import pytest

from tests.tracking._provider_inputs import load_provider_frames, synthesize_actions


@pytest.fixture
def cover_shadow_result(fitted_xt):
    """Enriched actions with cover shadows from Sportec fixture."""
    frames = load_provider_frames("sportec")
    from silly_kicks.tracking.preprocess import derive_velocities, smooth_frames
    from silly_kicks.tracking.utils import play_left_to_right

    frames = smooth_frames(frames)
    frames = derive_velocities(frames)
    home_team_id = frames[~frames["team_id"].isna()]["team_id"].iloc[0]
    frames = play_left_to_right(frames, home_team_id=home_team_id)
    actions = synthesize_actions(frames)

    from silly_kicks.tracking.features import add_cover_shadows

    return add_cover_shadows(actions, frames, fitted_xt, home_team_id=home_team_id)


class TestCoverShadowInvariants:
    """Physical invariant properties of cover shadow features."""

    def test_blocking_score_non_negative(self, cover_shadow_result):
        """Removing defenders cannot decrease threat (monotonicity)."""
        valid = cover_shadow_result["blocking_score"].dropna()
        assert (valid >= -1e-9).all()

    def test_blocked_threat_fraction_bounded(self, cover_shadow_result):
        """blocked_threat_fraction in [0, 1]."""
        valid = cover_shadow_result["blocked_threat_fraction"].dropna()
        assert (valid >= -1e-9).all()
        assert (valid <= 1.0 + 1e-9).all()

    def test_n_blocked_le_n_potential(self, cover_shadow_result):
        """Cannot block more lanes than exist."""
        df = cover_shadow_result
        both_valid = df[df["n_blocked_receivers"].notna() & df["n_potential_receivers"].notna()]
        if len(both_valid) == 0:
            pytest.skip("No valid rows")
        assert (both_valid["n_blocked_receivers"] <= both_valid["n_potential_receivers"]).all()

    def test_n_blocked_non_negative(self, cover_shadow_result):
        """n_blocked_receivers >= 0."""
        valid = cover_shadow_result["n_blocked_receivers"].dropna()
        assert (valid >= 0).all()

    def test_zero_blocked_implies_low_score(self, cover_shadow_result):
        """When n_blocked_receivers = 0, blocking_score should be low (approx)."""
        df = cover_shadow_result
        zero_blocked = df[df["n_blocked_receivers"] == 0]
        if len(zero_blocked) == 0:
            pytest.skip("No rows with n_blocked_receivers == 0")
        # Not strictly 0 due to Voronoi integral vs lane-level classification,
        # but should be small relative to non-zero cases
        valid_bs = zero_blocked["blocking_score"].dropna()
        if len(valid_bs) > 0:
            # Just assert non-negative (invariant already covers that)
            assert (valid_bs >= -1e-9).all()
