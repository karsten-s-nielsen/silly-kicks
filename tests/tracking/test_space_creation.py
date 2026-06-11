"""Tests for silly_kicks.tracking._space_creation — Space Creation (Fernandez 2018)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._space_creation import (
    SpaceCreationParams,
    compute_space_created,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_frame(
    n_per_team: int = 5,
    team_ids: tuple[int, int] = (1, 2),
) -> pd.DataFrame:
    """Build a minimal tracking frame with ball + outfield players."""
    rows = []
    rng = np.random.RandomState(42)
    for tid_idx, tid in enumerate(team_ids):
        for j in range(n_per_team):
            x = 20.0 + tid_idx * 60 + rng.uniform(-5, 5)
            y = 10.0 + j * 12 + rng.uniform(-2, 2)
            rows.append(
                {
                    "game_id": 1,
                    "period_id": 1,
                    "frame_id": 0,
                    "time_seconds": 0.0,
                    "player_id": tid * 100 + j,
                    "team_id": tid,
                    "x": x,
                    "y": y,
                    "vx": 0.5 * (1 - 2 * tid_idx),
                    "vy": 0.0,
                    "is_ball": False,
                    "is_goalkeeper": j == 0,
                }
            )
    # Ball
    rows.append(
        {
            "game_id": 1,
            "period_id": 1,
            "frame_id": 0,
            "time_seconds": 0.0,
            "player_id": None,
            "team_id": None,
            "x": 52.5,
            "y": 34.0,
            "vx": 0.0,
            "vy": 0.0,
            "is_ball": True,
            "is_goalkeeper": False,
        }
    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSpaceCreationParams:
    def test_frozen(self):
        params = SpaceCreationParams()
        with pytest.raises(AttributeError):
            params.pitch_length = 110.0  # type: ignore[misc]

    def test_defaults(self):
        params = SpaceCreationParams()
        assert params.pitch_length == 105.0
        assert params.pitch_width == 68.0


class TestComputeSpaceCreated:
    def test_output_schema(self):
        """Output has expected columns."""
        frame = _make_frame()
        result = compute_space_created(frame, attacking_team_id=1)
        assert set(result.columns) == {
            "player_id",
            "team_id",
            "space_created_m2",
            "space_destroyed_m2",
            "net_space_m2",
        }

    def test_one_row_per_attacking_player(self):
        """One row per attacking-team player (including GK)."""
        frame = _make_frame(n_per_team=5)
        result = compute_space_created(frame, attacking_team_id=1)
        # 5 players per team, all included in leave-one-out
        assert len(result) == 5

    def test_space_created_nonnegative(self):
        """space_created_m2 is always >= 0."""
        frame = _make_frame()
        result = compute_space_created(frame, attacking_team_id=1)
        assert (result["space_created_m2"] >= 0).all()

    def test_space_destroyed_nonnegative(self):
        """space_destroyed_m2 is always >= 0 (absolute value convention)."""
        frame = _make_frame()
        result = compute_space_created(frame, attacking_team_id=1)
        assert (result["space_destroyed_m2"] >= 0).all()

    def test_net_equals_created_minus_destroyed(self):
        """net_space_m2 = space_created_m2 - space_destroyed_m2."""
        frame = _make_frame()
        result = compute_space_created(frame, attacking_team_id=1)
        expected_net = result["space_created_m2"] - result["space_destroyed_m2"]
        np.testing.assert_array_almost_equal(result["net_space_m2"], expected_net)

    def test_custom_params(self):
        """Custom SpaceCreationParams are respected."""
        frame = _make_frame()
        params = SpaceCreationParams(pitch_length=120.0, pitch_width=80.0)
        result = compute_space_created(
            frame,
            attacking_team_id=1,
            params=params,
        )
        assert len(result) > 0

    def test_empty_attacking_team(self):
        """No outfield players -> empty DataFrame."""
        frame = _make_frame()
        # Use team 3 which doesn't exist
        result = compute_space_created(frame, attacking_team_id=3)
        assert len(result) == 0

    def test_explicit_ball_position(self):
        """Explicit ball_position overrides frame ball row."""
        frame = _make_frame()
        result = compute_space_created(
            frame,
            attacking_team_id=1,
            ball_position=(30.0, 20.0),
        )
        assert len(result) > 0


class TestLoopInvariantCorrectness:
    def test_hoisted_matches_naive(self):
        """Verify hoisted obso_multiplier gives same results as naive per-player OBSO."""
        from silly_kicks.tracking._obso import compute_obso_surface
        from silly_kicks.tracking.pitch_control import compute_pitch_control

        frame = _make_frame(n_per_team=3)
        atk_id = 1
        ball_pos = (52.5, 34.0)

        # Naive: compute full OBSO per removal
        baseline_pc = compute_pitch_control(frame, atk_id, ball_position=ball_pos)
        baseline_obso = compute_obso_surface(baseline_pc, ball_pos)

        atk_players = frame[
            (frame["team_id"] == atk_id) & (frame["is_ball"] != True)  # noqa: E712
        ]

        naive_deltas = []
        for _, player_row in atk_players.iterrows():
            pid = player_row["player_id"]
            removed = frame[
                ~((frame["player_id"] == pid) & (frame["is_ball"] != True))  # noqa: E712
            ]
            removed_pc = compute_pitch_control(removed, atk_id, ball_position=ball_pos)
            removed_obso = compute_obso_surface(removed_pc, ball_pos)
            naive_deltas.append(float(np.sum(baseline_obso.values - removed_obso.values)))

        # Hoisted: compute_space_created uses hoisted multiplier
        result = compute_space_created(frame, atk_id, ball_position=ball_pos)

        # The net_space should correlate with the naive delta direction
        # (signs should match for each player)
        for i, (_, row) in enumerate(result.iterrows()):
            hoisted_net = row["net_space_m2"]
            if naive_deltas[i] > 0:
                assert hoisted_net >= 0 or abs(hoisted_net) < 1e-6
            elif naive_deltas[i] < 0:
                assert hoisted_net <= 0 or abs(hoisted_net) < 1e-6

    @pytest.mark.parametrize("method", ["spearman", "fernandez_bornn"])
    def test_analytical_matches_naive(self, method):
        """Analytical PC delta matches naive N-recompute within tolerance."""
        from silly_kicks.tracking._obso import _get_default_grids, _interpolate_grid
        from silly_kicks.tracking._space_creation import (
            _analytical_leave_one_out,
            _naive_leave_one_out,
        )
        from silly_kicks.tracking.pitch_control import compute_pitch_control

        frame = _make_frame(n_per_team=3)
        atk_id = 1
        ball_pos = (52.5, 34.0)

        # Baseline with decomposition
        baseline = compute_pitch_control(
            frame,
            atk_id,
            method=method,
            decompose=True,
            ball_position=ball_pos,
        )
        ny, nx = baseline.surface.shape

        # OBSO multiplier (same as compute_space_created internals)
        tg, eg = _get_default_grids(None, None)
        ti = _interpolate_grid(tg, (ny, nx))
        ei = _interpolate_grid(eg, (ny, nx))
        gx, gy = np.asarray(baseline.grid_x), np.asarray(baseline.grid_y)
        xx, yy = np.meshgrid(gx, gy)
        dw = np.exp(-((xx - ball_pos[0]) ** 2) / (2.0 * 26.25**2) - (yy - ball_pos[1]) ** 2 / (2.0 * 17.0**2))
        et = ti * dw
        mt = np.max(et)
        if mt > 1e-10:
            et = et / mt
        obso_mult = et * ei
        baseline_obso = np.clip(np.asarray(baseline.surface) * obso_mult, 0.0, 1.0)
        dx = float(gx[1] - gx[0])
        dy = float(gy[1] - gy[0])
        cell_area = dx * dy

        atk_mask = (frame["team_id"] == atk_id) & (frame["is_ball"] != True)  # noqa: E712
        atk_players = frame.loc[atk_mask]

        analytical = _analytical_leave_one_out(
            baseline,
            baseline_obso,
            obso_mult,
            atk_id,
            atk_players,
            cell_area,
            method,
        )
        naive = _naive_leave_one_out(
            frame,
            baseline_obso,
            obso_mult,
            atk_id,
            atk_players,
            cell_area,
            method,
            ball_pos,
        )

        assert len(analytical) == len(naive)
        for a, n in zip(analytical, naive, strict=True):
            assert a["player_id"] == n["player_id"]
            np.testing.assert_allclose(
                a["space_created_m2"],
                n["space_created_m2"],
                atol=1e-6,
                err_msg=f"space_created mismatch for player {a['player_id']}",
            )
            np.testing.assert_allclose(
                a["space_destroyed_m2"],
                n["space_destroyed_m2"],
                atol=1e-6,
                err_msg=f"space_destroyed mismatch for player {a['player_id']}",
            )
            np.testing.assert_allclose(
                a["net_space_m2"],
                n["net_space_m2"],
                atol=1e-6,
                err_msg=f"net_space mismatch for player {a['player_id']}",
            )


# ---------------------------------------------------------------------------
# Aggregator + VAEP factory tests
# ---------------------------------------------------------------------------


def _make_actions_and_frames():
    """Minimal actions + frames for aggregator tests."""
    rng = np.random.RandomState(99)
    actions = pd.DataFrame(
        {
            "action_id": [0, 1],
            "game_id": [1, 1],
            "period_id": [1, 1],
            "time_seconds": [0.2, 0.6],
            "team_id": [1, 2],
            "player_id": [101, 201],
            "start_x": [30.0, 80.0],
            "start_y": [34.0, 34.0],
            "end_x": [50.0, 60.0],
            "end_y": [34.0, 34.0],
            "type_id": [0, 0],
            "type_name": ["pass", "pass"],
            "result_id": [1, 1],
            "result_name": ["success", "success"],
            "bodypart_id": [0, 0],
            "bodypart_name": ["foot", "foot"],
        }
    )
    frame_rows = []
    for fid in range(50):
        t = fid / 25.0
        for tid in [1, 2]:
            for j in range(4):
                frame_rows.append(
                    {
                        "game_id": 1,
                        "period_id": 1,
                        "frame_id": fid,
                        "time_seconds": t,
                        "source_provider": "test",
                        "player_id": tid * 100 + j,
                        "team_id": tid,
                        "x": 20 + (tid - 1) * 60 + rng.uniform(-3, 3),
                        "y": 10 + j * 15 + rng.uniform(-2, 2),
                        "vx": 0.5 * (1 - 2 * (tid - 1)),
                        "vy": 0.0,
                        "is_ball": False,
                        "is_goalkeeper": j == 0,
                    }
                )
        frame_rows.append(
            {
                "game_id": 1,
                "period_id": 1,
                "frame_id": fid,
                "time_seconds": t,
                "source_provider": "test",
                "player_id": None,
                "team_id": None,
                "x": 52.5,
                "y": 34.0,
                "vx": 0.0,
                "vy": 0.0,
                "is_ball": True,
                "is_goalkeeper": False,
            }
        )
    frames = pd.DataFrame(frame_rows)
    return actions, frames


class TestAddSpaceCreation:
    def test_enrichment_columns(self):
        """add_space_creation adds exactly the 3 team-side columns, all live.

        The ``*_opponent`` triplet was removed from the contract (4.22.2): it
        had been hard-coded NaN on every code path since introduction — a
        schema-only dead column. This gate asserts the remaining columns
        actually populate, so a dead column can't re-enter the contract.
        """
        from silly_kicks.tracking.features import add_space_creation

        actions, frames = _make_actions_and_frames()
        result = add_space_creation(actions, frames, home_team_id=1)
        expected_cols = {
            "space_created_m2_team",
            "space_destroyed_m2_team",
            "net_space_m2_team",
        }
        added = set(result.columns) - set(actions.columns)
        assert expected_cols.issubset(added)
        assert not {c for c in added if c.endswith("_opponent")}
        for col in expected_cols:
            assert result[col].notna().any(), f"{col} is all-NaN — dead contract column"

    def test_row_count_preserved(self):
        """Row count unchanged after enrichment."""
        from silly_kicks.tracking.features import add_space_creation

        actions, frames = _make_actions_and_frames()
        result = add_space_creation(actions, frames, home_team_id=1)
        assert len(result) == len(actions)


class TestSpaceCreationXfns:
    def test_column_count(self):
        """space_creation_xfns produces 3 lifted xfns (9 VAEP columns)."""
        from silly_kicks.tracking.features import space_creation_xfns

        xfns = space_creation_xfns(home_team_id=1)
        assert len(xfns) == 3

    def test_introspection_nan(self):
        """xfns produce NaN in introspection mode (frames=None)."""
        from silly_kicks.tracking.features import space_creation_xfns

        xfns = space_creation_xfns(home_team_id=1)
        actions = pd.DataFrame(
            {
                "game_id": [1],
                "action_id": [0],
                "period_id": [1],
                "time_seconds": [10.0],
                "team_id": [1],
                "player_id": [101],
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
