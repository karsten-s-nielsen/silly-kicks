"""Physical invariants for GK influence primitives (TF-15)."""

from __future__ import annotations

import numpy as np
import pytest

from tests.tracking._gk_test_helpers import _make_two_team_frame

# fitted_xt inherited from tests/conftest.py


@pytest.fixture(params=["spearman", "fernandez_bornn", "voronoi"])
def method(request):
    return request.param


class TestGkInfluenceInvariants:
    """Physical invariants that must hold across all configurations."""

    def test_share_in_unit_interval(self, method, fitted_xt):
        from silly_kicks.tracking._gk_influence import compute_gk_influence

        frame = _make_two_team_frame(
            home_positions=[(20, 20), (25, 30), (30, 40), (35, 50)],
            away_positions=[(80, 20), (85, 30), (80, 40), (85, 50)],
        )
        gi = compute_gk_influence(
            frame,
            attacking_team_id=2,
            gk_player_id=1,
            xt=fitted_xt,
            home_team_id=1,
            method=method,
        )
        if not np.isnan(gi.pitch_control_share_weighted):
            assert 0.0 <= gi.pitch_control_share_weighted <= 1.0

    def test_reachable_area_bounded(self, method, fitted_xt):
        from silly_kicks.tracking._gk_influence import compute_gk_influence

        frame = _make_two_team_frame(
            home_positions=[(20, 20), (25, 30), (30, 40), (35, 50)],
            away_positions=[(80, 20), (85, 30), (80, 40), (85, 50)],
        )
        gi = compute_gk_influence(
            frame,
            attacking_team_id=2,
            gk_player_id=1,
            xt=fitted_xt,
            home_team_id=1,
            method=method,
        )
        assert 0.0 <= gi.reachable_area_m2 <= 7140.0

    def test_min_leq_mean_closing_time(self, method, fitted_xt):
        from silly_kicks.tracking._gk_influence import compute_gk_influence

        frame = _make_two_team_frame(
            home_positions=[(20, 20), (25, 30), (30, 40), (35, 50)],
            away_positions=[(80, 20), (85, 30), (80, 40), (85, 50)],
        )
        gi = compute_gk_influence(
            frame,
            attacking_team_id=2,
            gk_player_id=1,
            xt=fitted_xt,
            home_team_id=1,
            method=method,
        )
        for zct in gi.closing_times.values():
            assert zct.min_s <= zct.mean_s

    def test_closing_time_non_negative(self, method, fitted_xt):
        from silly_kicks.tracking._gk_influence import compute_gk_influence

        frame = _make_two_team_frame(
            home_positions=[(20, 20), (25, 30), (30, 40), (35, 50)],
            away_positions=[(80, 20), (85, 30), (80, 40), (85, 50)],
        )
        gi = compute_gk_influence(
            frame,
            attacking_team_id=2,
            gk_player_id=1,
            xt=fitted_xt,
            home_team_id=1,
            method=method,
        )
        for zct in gi.closing_times.values():
            assert zct.min_s >= 0.0

    def test_closer_gk_lower_closing_time(self, fitted_xt):
        """GK closer to zone -> lower closing time (monotonicity)."""
        from silly_kicks.tracking._gk_influence import compute_gk_influence

        outfield = [(20, 20), (25, 30), (30, 40), (35, 50)]
        away_pos = [(80, 20), (85, 30), (80, 40), (85, 50)]

        frame_close = _make_two_team_frame(
            home_positions=outfield,  # type: ignore[arg-type]
            away_positions=away_pos,  # type: ignore[arg-type]
            home_gk_pos=(3.0, 34.0),
        )
        frame_far = _make_two_team_frame(
            home_positions=outfield,  # type: ignore[arg-type]
            away_positions=away_pos,  # type: ignore[arg-type]
            home_gk_pos=(30.0, 34.0),
        )
        gi_close = compute_gk_influence(
            frame_close,
            attacking_team_id=2,
            gk_player_id=1,
            xt=fitted_xt,
            home_team_id=1,
        )
        gi_far = compute_gk_influence(
            frame_far,
            attacking_team_id=2,
            gk_player_id=1,
            xt=fitted_xt,
            home_team_id=1,
        )
        assert gi_close.closing_times["six_yard_box"].min_s < gi_far.closing_times["six_yard_box"].min_s

    def test_string_typed_ids(self, fitted_xt):
        """M-2: String-typed team_id/player_id (DFL-OBJ-* style) must work."""
        from silly_kicks.tracking._gk_influence import compute_gk_influence

        frame = _make_two_team_frame(
            home_positions=[(20, 20), (25, 30), (30, 40), (35, 50)],
            away_positions=[(80, 20), (85, 30), (80, 40), (85, 50)],
        )
        # Convert IDs to strings (Sportec/kloppy DFL-OBJ-* style)
        # Use DFL-style prefixed IDs to avoid float→str ambiguity
        id_map = {}
        for pid in frame["player_id"].dropna().unique():
            id_map[pid] = f"DFL-OBJ-{int(pid):04d}"
        frame["player_id"] = frame["player_id"].map(id_map)

        tid_map = {}
        for tid in frame["team_id"].dropna().unique():
            tid_map[tid] = f"DFL-CLB-{int(tid):04d}"
        frame["team_id"] = frame["team_id"].map(tid_map)

        gi = compute_gk_influence(
            frame,
            attacking_team_id="DFL-CLB-0002",
            gk_player_id="DFL-OBJ-0001",
            xt=fitted_xt,
            home_team_id="DFL-CLB-0001",
        )
        assert 0.0 <= gi.pitch_control_share_weighted <= 1.0
        assert gi.reachable_area_m2 >= 0.0
