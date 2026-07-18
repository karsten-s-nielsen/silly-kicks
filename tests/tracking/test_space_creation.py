"""Tests for silly_kicks.tracking._space_creation — Space Creation (Fernandez 2018)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._space_creation import (
    SpaceCreationParams,
    compute_space_created,
)

# ADR-041 opt-out: deliberately exercises the synthetic placeholder EPV path (the default surface is this module's
# subject).
pytestmark = pytest.mark.filterwarnings("ignore::silly_kicks.tracking.SyntheticEPVWarning")

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
        """Lean contract (4.24.0): exactly the live measurement, no structural
        zeros, no redundant nets."""
        frame = _make_frame()
        result = compute_space_created(frame, attacking_team_id=1)
        assert set(result.columns) == {
            "player_id",
            "team_id",
            "space_created_m2",
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

    @pytest.mark.parametrize("method", ["spearman", "fernandez_bornn", "voronoi"])
    def test_retired_columns_never_emitted(self, method):
        """Lean-contract guard (4.24.0, lakehouse/owner decision): the retired
        always-zero/redundant columns (team destroyed, opponent created, both
        nets — structurally dead under the pointwise-monotone LOO) must never
        re-enter the output schema."""
        frame = _make_frame()
        result = compute_space_created(
            frame,
            attacking_team_id=1,
            include_opponent_perspective=True,
            pitch_control_method=method,  # type: ignore[arg-type]
        )
        retired = {
            "space_destroyed_m2",
            "net_space_m2",
            "opponent_space_created_m2",
            "opponent_space_destroyed_m2",
            "opponent_net_space_m2",
            "space_destroyed_m2_team",
            "space_created_m2_opponent",
        }
        assert not retired & set(result.columns)

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

        # Created mass should track the naive delta direction (monotone LOO:
        # the delta is one-signed, so created IS the whole measurement)
        for i, (_, row) in enumerate(result.iterrows()):
            hoisted_created = row["space_created_m2"]
            assert hoisted_created >= 0
            if naive_deltas[i] > 1e-6:
                assert hoisted_created > 0 or abs(naive_deltas[i]) < 1e-6

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
        """add_space_creation adds exactly the lean 2-column contract, both live.

        History: the ``*_opponent`` triplet shipped hard-coded NaN
        3.21.0-4.22.1, was removed in 4.22.2, implemented in 4.23.0 (but as an
        information-free mirror), and fixed + LEANED in 4.24.0 to the two live
        measurements (``space_created_m2``, ``space_denied_m2_opponent``).
        The liveness loop is the structural guarantee that a documented
        contract column can never again ship 100%-NaN.
        """
        from silly_kicks.tracking.features import _SPACE_CREATION_COLUMNS, add_space_creation

        actions, frames = _make_actions_and_frames()
        result = add_space_creation(actions, frames, home_team_id=1)
        expected_cols = {
            "space_created_m2",
            "space_denied_m2_opponent",
        }
        assert set(_SPACE_CREATION_COLUMNS) == expected_cols
        added = set(result.columns) - set(actions.columns)
        assert expected_cols.issubset(added)
        # No retired column may resurface at the aggregator level either.
        assert not {c for c in added if "destroyed" in c or "net_space" in c or c.endswith("_team")}
        # Meta-gate (lakehouse acceptance #5): every documented contract column
        # must be live on the fixture — a 100%-NaN contract column fails CI.
        for col in _SPACE_CREATION_COLUMNS:
            assert result[col].notna().any(), f"{col} is all-NaN — dead contract column"

    def test_opponent_coverage_parity(self):
        """Acceptance #1: the two live columns share an identical NaN mask
        (single-call design — no new degradation paths). This test would have
        caught the original hard-coded-NaN defect on day one.
        """
        from silly_kicks.tracking.features import add_space_creation

        actions, frames = _make_actions_and_frames()
        result = add_space_creation(actions, frames, home_team_id=1)
        created_mask = result["space_created_m2"].notna()
        denied_mask = result["space_denied_m2_opponent"].notna()
        assert created_mask.sum() > 0, "space_created_m2 has no coverage on the fixture"
        assert (created_mask == denied_mask).all(), "NaN masks differ (created vs denied)"

    def test_opponent_sign_and_range_oracle(self):
        """Acceptance #3: both live columns >= 0, magnitudes in the same order
        (shared grid/sigmas/method)."""
        from silly_kicks.tracking.features import add_space_creation

        actions, frames = _make_actions_and_frames()
        result = add_space_creation(actions, frames, home_team_id=1)
        created = result["space_created_m2"].dropna()
        denied = result["space_denied_m2_opponent"].dropna()
        assert (created >= 0).all()
        assert (denied >= 0).all()
        created_scale = float(created.abs().max())
        denied_scale = float(denied.abs().max())
        assert created_scale > 0
        assert denied_scale > 0
        assert denied_scale < 1000.0 * created_scale
        assert created_scale < 1000.0 * denied_scale

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
        assert np.isnan(result.loc[0, "space_denied_m2_opponent"])  # type: ignore[arg-type]
        assert np.isnan(result.loc[0, "space_created_m2"])  # type: ignore[arg-type]

    def test_nan_team_action_on_healthy_frame_returns_nan_row(self):
        """Acceptance #3 (GS sentinel fix): a NaN-team action on a healthy
        two-team frame routes to the NaN row — it must NOT reach the strict
        opponent-resolution guard. This is the downstream contract the GS
        converter's NaN-residue (formerly the sentinel '0') depends on: NaN is
        ``pd.isna``-routable; '0' was not, which is why the sentinel crashed."""
        from silly_kicks.tracking.features import _compute_space_creation_for_action

        actions, frames = _make_actions_and_frames()
        action_row = actions.iloc[0].copy()
        action_row["team_id"] = np.nan  # the GS null-actor residue
        frame = frames[frames["frame_id"] == frames["frame_id"].iloc[0]]
        # Sanity: the frame really is a healthy two-team frame (the guard WOULD
        # fire on a non-NaN unmatched id like the old sentinel '0').
        result = _compute_space_creation_for_action(action_row, frame, home_team_id=1)
        assert np.isnan(result["space_created_m2"])
        assert np.isnan(result["space_denied_m2_opponent"])

    def test_row_count_preserved(self):
        """Row count unchanged after enrichment."""
        from silly_kicks.tracking.features import add_space_creation

        actions, frames = _make_actions_and_frames()
        result = add_space_creation(actions, frames, home_team_id=1)
        assert len(result) == len(actions)


class TestOpponentMirrorDecoupling:
    """Lakehouse second-round acceptance (4.24.0): the opponent surface is weighted
    by the opponent's OWN attacking geometry (x-mirrored transition/EPV artifacts,
    ball-anchored distance weight unchanged), so the opponent LOO is a genuine
    independent measurement — NOT the algebraic negation of the team LOO that the
    4.23.0 shared-unmirrored multiplier degenerated to."""

    def test_opponent_not_a_mirror_of_team(self):
        """Acceptance #2 (anti-mirror): space_denied_m2_opponent must NOT equal
        space_created_m2 (the exact identity that held on 4.23.0)."""
        frame = _make_frame()
        r = compute_space_created(frame, attacking_team_id=1, include_opponent_perspective=True)
        assert not np.allclose(r["space_denied_m2_opponent"], r["space_created_m2"]), (
            "opponent denial is still the algebraic mirror of team creation"
        )

    def test_opponent_denied_live_under_spearman(self):
        """Acceptance #1 (live half): the rest-defense column is non-zero under
        the production method."""
        frame = _make_frame()
        r = compute_space_created(frame, attacking_team_id=1, include_opponent_perspective=True)
        assert (r["space_denied_m2_opponent"] > 0).any()

    def test_method_consistency_spearman_vs_voronoi(self):
        """Acceptance #4: spearman and voronoi opponent values on the same frame
        agree in sign and order of magnitude (one metric, two estimators — not two
        metrics under one name)."""
        frame = _make_frame()
        spearman = compute_space_created(frame, attacking_team_id=1, include_opponent_perspective=True)
        voronoi = compute_space_created(
            frame,
            attacking_team_id=1,
            include_opponent_perspective=True,
            pitch_control_method="voronoi",
        )
        s_total = float(spearman["space_denied_m2_opponent"].sum())
        v_total = float(voronoi["space_denied_m2_opponent"].sum())
        assert s_total > 0 and v_total > 0  # same sign (both denial-dominant)
        ratio = max(s_total, v_total) / min(s_total, v_total)
        assert ratio < 30.0, f"spearman vs voronoi opponent magnitudes diverge: {s_total} vs {v_total}"

    def test_mirrored_multiplier_geography(self):
        """The mirror must actually change the weighting geography: a defender near
        the OWN goal (where the opponent's mirrored EPV peaks) must register more
        opponent-space denial than the same defender placed at the opponent's goal
        end. Pins the orientation so a silent un-mirroring regression cannot pass."""
        frame = _make_frame()
        r = compute_space_created(frame, attacking_team_id=1, include_opponent_perspective=True)
        # Team-1 attacks LTR (toward x=105); the opponent attacks toward x=0, so the
        # mirrored EPV weights the LOW-x half. Team-1 players sit at x≈20-40 (own
        # half): the per-player ratio denied/created must NOT be a constant (a
        # constant ratio is exactly the unmirrored-multiplier mirror identity).
        ratios = r["space_denied_m2_opponent"] / r["space_created_m2"]
        finite = ratios[np.isfinite(ratios)]
        assert len(finite) >= 2
        assert float(finite.max() - finite.min()) > 1e-6, "constant ratio — multiplier not mirrored"


class TestComputeSpaceCreatedOpponentPerspective:
    def test_opponent_column_present(self):
        """include_opponent_perspective=True adds the rest-defense column."""
        frame = _make_frame()
        result = compute_space_created(frame, attacking_team_id=1, include_opponent_perspective=True)
        assert "space_denied_m2_opponent" in result.columns
        assert len(result) == 5

    def test_default_excludes_opponent_column(self):
        """Default output schema stays team-only."""
        frame = _make_frame()
        result = compute_space_created(frame, attacking_team_id=1)
        assert "space_denied_m2_opponent" not in result.columns

    def test_opponent_sign_convention(self):
        """denied >= 0 (denial is an absolute mass)."""
        frame = _make_frame()
        result = compute_space_created(frame, attacking_team_id=1, include_opponent_perspective=True)
        assert (result["space_denied_m2_opponent"] >= 0).all()

    def test_defender_denies_space(self):
        """A present defender denies opponent space (removing them frees opponent OBSO)."""
        frame = _make_frame()
        result = compute_space_created(frame, attacking_team_id=1, include_opponent_perspective=True)
        assert (result["space_denied_m2_opponent"] > 0).any()

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
        # Mirrored opponent multiplier (same construction as compute_space_created)
        et_opp = np.flip(ti, axis=1) * dw
        mt_opp = np.max(et_opp)
        if mt_opp > 1e-10:
            et_opp = et_opp / mt_opp
        obso_mult_opp = et_opp * np.flip(ei, axis=1)
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
            obso_multiplier_opponent=obso_mult_opp,
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
            obso_multiplier_opponent=obso_mult_opp,
        )

        assert len(analytical) == len(naive)
        for a, n in zip(analytical, naive, strict=True):
            assert a["player_id"] == n["player_id"]
            np.testing.assert_allclose(
                a["space_denied_m2_opponent"],
                n["space_denied_m2_opponent"],
                atol=1e-6,
                err_msg=f"space_denied_m2_opponent mismatch for player {a['player_id']}",
            )

    def test_row_count_preserved(self):
        """Row count unchanged after enrichment."""
        from silly_kicks.tracking.features import add_space_creation

        actions, frames = _make_actions_and_frames()
        result = add_space_creation(actions, frames, home_team_id=1)
        assert len(result) == len(actions)


class TestSpaceCreationXfns:
    def test_column_count(self):
        """space_creation_xfns produces 2 lifted xfns (6 VAEP columns), lockstep
        with _SPACE_CREATION_COLUMNS (lean contract, 4.24.0)."""
        from silly_kicks.tracking.features import _SPACE_CREATION_COLUMNS, space_creation_xfns

        xfns = space_creation_xfns(home_team_id=1)
        assert len(xfns) == 2
        assert len(_SPACE_CREATION_COLUMNS) == 2

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
