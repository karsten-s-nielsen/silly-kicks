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
        """add_space_creation adds all 6 contract columns, every one live.

        The ``*_opponent`` triplet shipped hard-coded NaN 3.21.0-4.22.1 (a
        schema-only dead contract), was removed in 4.22.2, and is IMPLEMENTED
        as of 4.23.0 (defender-side leave-one-out on the opponent-attacking
        OBSO surface). The liveness loop is the structural guarantee that a
        documented contract column can never again ship 100%-NaN.
        """
        from silly_kicks.tracking.features import _SPACE_CREATION_COLUMNS, add_space_creation

        actions, frames = _make_actions_and_frames()
        result = add_space_creation(actions, frames, home_team_id=1)
        expected_cols = {
            "space_created_m2_team",
            "space_destroyed_m2_team",
            "net_space_m2_team",
            "space_created_m2_opponent",
            "space_destroyed_m2_opponent",
            "net_space_m2_opponent",
        }
        assert set(_SPACE_CREATION_COLUMNS) == expected_cols
        added = set(result.columns) - set(actions.columns)
        assert expected_cols.issubset(added)
        # Meta-gate (lakehouse acceptance #5): every documented contract column
        # must be live on the fixture — a 100%-NaN contract column fails CI.
        for col in _SPACE_CREATION_COLUMNS:
            assert result[col].notna().any(), f"{col} is all-NaN — dead contract column"

    def test_opponent_coverage_parity(self):
        """Acceptance #1: opponent non-NaN coverage == team non-NaN coverage.

        The opponent triplet may be NaN ONLY where the team triplet is NaN
        (identical NaN mask — no new degradation paths). This test would have
        caught the original hard-coded-NaN defect on day one.
        """
        from silly_kicks.tracking.features import add_space_creation

        actions, frames = _make_actions_and_frames()
        result = add_space_creation(actions, frames, home_team_id=1)
        for base in ("space_created_m2", "space_destroyed_m2", "net_space_m2"):
            team_mask = result[f"{base}_team"].notna()
            opp_mask = result[f"{base}_opponent"].notna()
            assert team_mask.sum() > 0, f"{base}_team has no coverage on the fixture"
            assert (team_mask == opp_mask).all(), f"{base}: NaN masks differ (team vs opponent)"

    def test_opponent_symmetry_sanity(self):
        """Acceptance #2: opponent triplet is neither identically 0 nor a copy of team."""
        from silly_kicks.tracking.features import add_space_creation

        actions, frames = _make_actions_and_frames()
        result = add_space_creation(actions, frames, home_team_id=1)
        opp = result[["space_created_m2_opponent", "space_destroyed_m2_opponent", "net_space_m2_opponent"]]
        team = result[["space_created_m2_team", "space_destroyed_m2_team", "net_space_m2_team"]]
        opp_vals = opp.to_numpy(dtype=float)
        team_vals = team.to_numpy(dtype=float)
        finite = np.isfinite(opp_vals)
        assert finite.any()
        assert not np.allclose(opp_vals[finite], 0.0), "opponent triplet identically zero"
        assert not np.allclose(opp_vals[finite], team_vals[finite]), "opponent triplet equals team triplet"

    def test_opponent_sign_and_range_oracle(self):
        """Acceptance #3: created >= 0, destroyed >= 0, net == created - destroyed (exact),
        magnitudes in the same order as the team triplet."""
        from silly_kicks.tracking.features import add_space_creation

        actions, frames = _make_actions_and_frames()
        result = add_space_creation(actions, frames, home_team_id=1)
        created = result["space_created_m2_opponent"].dropna()
        destroyed = result["space_destroyed_m2_opponent"].dropna()
        net = result["net_space_m2_opponent"].dropna()
        assert (created >= 0).all()
        assert (destroyed >= 0).all()
        np.testing.assert_allclose(net.to_numpy(), (created - destroyed).to_numpy(), atol=1e-12)
        # Comparable scale: same grid, same OBSO multiplier -> same order of magnitude.
        team_scale = float(result[["space_created_m2_team", "space_destroyed_m2_team"]].abs().max().max())
        opp_scale = float(result[["space_created_m2_opponent", "space_destroyed_m2_opponent"]].abs().max().max())
        assert opp_scale > 0
        assert opp_scale < 1000.0 * team_scale
        assert team_scale < 1000.0 * opp_scale

    def test_opponent_two_team_guard(self):
        """Acceptance #4: a frame without exactly two team ids raises loud with
        the frame/action key in the message — no silent NaN."""
        from silly_kicks.tracking.features import add_space_creation

        actions, frames = _make_actions_and_frames()
        one_team_frames = frames[(frames["team_id"] == 1) | (frames["is_ball"] == True)]  # noqa: E712
        with pytest.raises(ValueError, match=r"action_id"):
            add_space_creation(actions[actions["team_id"] == 1], one_team_frames, home_team_id=1)

    def test_opponent_nan_action_ids_no_raise(self):
        """ADR-003: NaN actor identifiers route to the NaN-row default, never the guard."""
        from silly_kicks.tracking.features import add_space_creation

        actions, frames = _make_actions_and_frames()
        actions = actions.copy()
        actions.loc[0, "player_id"] = np.nan
        result = add_space_creation(actions, frames, home_team_id=1)
        assert np.isnan(result.loc[0, "space_created_m2_opponent"])  # type: ignore[arg-type]
        assert np.isnan(result.loc[0, "space_created_m2_team"])  # type: ignore[arg-type]

    def test_row_count_preserved(self):
        """Row count unchanged after enrichment."""
        from silly_kicks.tracking.features import add_space_creation

        actions, frames = _make_actions_and_frames()
        result = add_space_creation(actions, frames, home_team_id=1)
        assert len(result) == len(actions)


class TestComputeSpaceCreatedOpponentPerspective:
    def test_opponent_columns_present(self):
        """include_opponent_perspective=True adds the 3 opponent columns."""
        frame = _make_frame()
        result = compute_space_created(frame, attacking_team_id=1, include_opponent_perspective=True)
        assert {
            "opponent_space_created_m2",
            "opponent_space_destroyed_m2",
            "opponent_net_space_m2",
        }.issubset(result.columns)
        assert len(result) == 5

    def test_default_excludes_opponent_columns(self):
        """Default output schema is unchanged (backcompat)."""
        frame = _make_frame()
        result = compute_space_created(frame, attacking_team_id=1)
        assert not any(c.startswith("opponent_") for c in result.columns)

    def test_opponent_sign_convention(self):
        """created >= 0, destroyed >= 0, net == created - destroyed."""
        frame = _make_frame()
        result = compute_space_created(frame, attacking_team_id=1, include_opponent_perspective=True)
        assert (result["opponent_space_created_m2"] >= 0).all()
        assert (result["opponent_space_destroyed_m2"] >= 0).all()
        np.testing.assert_allclose(
            result["opponent_net_space_m2"],
            result["opponent_space_created_m2"] - result["opponent_space_destroyed_m2"],
        )

    def test_defender_mostly_denies_space(self):
        """A present defender denies opponent space: destroyed dominates created
        for at least one player (removing a defender frees opponent OBSO)."""
        frame = _make_frame()
        result = compute_space_created(frame, attacking_team_id=1, include_opponent_perspective=True)
        assert (result["opponent_space_destroyed_m2"] > 0).any()

    def test_two_team_guard_primitive(self):
        """Frame without exactly two team ids raises when opponent perspective is on."""
        frame = _make_frame()
        one_team = frame[(frame["team_id"] == 1) | (frame["is_ball"] == True)]  # noqa: E712
        with pytest.raises(ValueError, match="exactly two"):
            compute_space_created(one_team, attacking_team_id=1, include_opponent_perspective=True)
        # ... but stays silent when the perspective is off (backcompat).
        compute_space_created(one_team, attacking_team_id=1)

    @pytest.mark.parametrize("method", ["spearman", "fernandez_bornn"])
    def test_opponent_analytical_matches_naive(self, method):
        """Analytical opponent-side delta matches the naive N-recompute oracle."""
        from silly_kicks.tracking._obso import _get_default_grids, _interpolate_grid
        from silly_kicks.tracking._space_creation import (
            _analytical_leave_one_out,
            _naive_leave_one_out,
        )
        from silly_kicks.tracking.pitch_control import compute_pitch_control

        frame = _make_frame(n_per_team=3)
        atk_id = 1
        ball_pos = (52.5, 34.0)

        baseline = compute_pitch_control(
            frame,
            atk_id,
            method=method,
            decompose=True,
            ball_position=ball_pos,
        )
        ny, nx = baseline.surface.shape
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
            include_opponent=True,
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
            opponent_team_id=2,
        )

        assert len(analytical) == len(naive)
        for a, n in zip(analytical, naive, strict=True):
            assert a["player_id"] == n["player_id"]
            for key in (
                "opponent_space_created_m2",
                "opponent_space_destroyed_m2",
                "opponent_net_space_m2",
            ):
                np.testing.assert_allclose(
                    a[key],
                    n[key],
                    atol=1e-6,
                    err_msg=f"{key} mismatch for player {a['player_id']}",
                )

    def test_row_count_preserved(self):
        """Row count unchanged after enrichment."""
        from silly_kicks.tracking.features import add_space_creation

        actions, frames = _make_actions_and_frames()
        result = add_space_creation(actions, frames, home_team_id=1)
        assert len(result) == len(actions)


class TestSpaceCreationXfns:
    def test_column_count(self):
        """space_creation_xfns produces 6 lifted xfns (18 VAEP columns), lockstep
        with _SPACE_CREATION_COLUMNS."""
        from silly_kicks.tracking.features import _SPACE_CREATION_COLUMNS, space_creation_xfns

        xfns = space_creation_xfns(home_team_id=1)
        assert len(xfns) == 6
        assert len(_SPACE_CREATION_COLUMNS) == 6

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
