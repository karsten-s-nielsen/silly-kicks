"""Tests for shape graph formation detection (TF-39, Sotudeh 2026).

Tests cover:
  - Angular stability (thesis §3.3, equation 3.2)
  - Shape graph construction (Algorithm 1, thesis p.38)
  - Position inference (thesis Chapter 4, §4.1-4.3)
  - Aggregator + VAEP xfn wiring
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._shape_graph import (
    POSITION_LABEL_MATRIX,
    PositionLabel,
    ShapeGraph,
    _assign_levels_horizontal,
    _assign_levels_vertical,
    _compute_edge_stability,
    compute_shape_graph,
    infer_positions,
)

# ---------------------------------------------------------------------------
# Fixtures — reusable player arrangements
# ---------------------------------------------------------------------------


@pytest.fixture()
def positions_442() -> np.ndarray:
    """Canonical 4-4-2 formation.

    Defenders at x=20, midfielders at x=40, forwards at x=60.
    Each line spread across y = 10, 25, 43, 58 (or 25, 43 for forwards).
    """
    return np.array(
        [
            # Defenders (4)
            [20.0, 10.0],
            [20.0, 25.0],
            [20.0, 43.0],
            [20.0, 58.0],
            # Midfielders (4)
            [40.0, 10.0],
            [40.0, 25.0],
            [40.0, 43.0],
            [40.0, 58.0],
            # Forwards (2)
            [60.0, 25.0],
            [60.0, 43.0],
        ]
    )


@pytest.fixture()
def positions_352() -> np.ndarray:
    """3-5-2 formation for vertical distribution test."""
    return np.array(
        [
            # Defenders (3)
            [20.0, 15.0],
            [20.0, 34.0],
            [20.0, 53.0],
            # Midfielders (5)
            [40.0, 5.0],
            [40.0, 20.0],
            [40.0, 34.0],
            [40.0, 48.0],
            [40.0, 63.0],
            # Forwards (2)
            [60.0, 25.0],
            [60.0, 43.0],
        ]
    )


# ---------------------------------------------------------------------------
# Data type contracts
# ---------------------------------------------------------------------------


class TestDataTypes:
    """Verify frozen dataclass contracts."""

    def test_position_label_frozen(self) -> None:
        lbl = PositionLabel(vertical="B", horizontal="L", label="LB")
        with pytest.raises(AttributeError):
            lbl.vertical = "F"  # type: ignore[misc]

    def test_shape_graph_frozen(self) -> None:
        sg = ShapeGraph(
            edges=np.empty((0, 2), dtype=int),
            faces=[],
            stabilities=np.empty(0),
            points=np.empty((0, 2)),
        )
        with pytest.raises(AttributeError):
            sg.edges = np.empty((0, 2), dtype=int)  # type: ignore[misc]


class TestPositionLabelMatrix:
    """Verify the 5x5 position label matrix matches the thesis Figure 4.5b."""

    def test_all_25_positions_present(self) -> None:
        all_labels = set()
        for row in POSITION_LABEL_MATRIX.values():
            all_labels.update(row.values())
        assert len(all_labels) == 25

    def test_specific_labels(self) -> None:
        assert POSITION_LABEL_MATRIX["B"]["L"] == "LB"
        assert POSITION_LABEL_MATRIX["B"]["RC"] == "RCB"
        assert POSITION_LABEL_MATRIX["DM"]["L"] == "LWB"
        assert POSITION_LABEL_MATRIX["DM"]["C"] == "CDM"
        assert POSITION_LABEL_MATRIX["M"]["C"] == "CM"
        assert POSITION_LABEL_MATRIX["M"]["R"] == "RM"
        assert POSITION_LABEL_MATRIX["AM"]["L"] == "LWF"
        assert POSITION_LABEL_MATRIX["AM"]["C"] == "CAM"
        assert POSITION_LABEL_MATRIX["F"]["C"] == "CF"
        assert POSITION_LABEL_MATRIX["F"]["R"] == "RF"
        assert POSITION_LABEL_MATRIX["F"]["LC"] == "LCF"


# ---------------------------------------------------------------------------
# Angular Stability (thesis §3.3, equation 3.2)
# ---------------------------------------------------------------------------


class TestAngularStability:
    """Test angular stability per Sotudeh's equation 3.2."""

    def test_equilateral_diamond_high_stability(self) -> None:
        """Two equilateral triangles sharing an edge: alpha = 180 - 60 - 60 = 60."""
        points = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, np.sqrt(3.0)], [1.0, -np.sqrt(3.0)]])
        simplices = np.array([[0, 1, 2], [0, 1, 3]])
        stability = _compute_edge_stability(p_idx=0, q_idx=1, simplices=simplices, points=points)
        assert abs(stability - 60.0) < 1.0

    def test_near_degenerate_low_stability(self) -> None:
        """Near-flat triangle: large opposite angle yields low stability."""
        points = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 0.01], [5.0, -5.0]])
        simplices = np.array([[0, 1, 2], [0, 1, 3]])
        stability = _compute_edge_stability(p_idx=0, q_idx=1, simplices=simplices, points=points)
        assert stability < 45.0

    def test_boundary_edge_one_triangle(self) -> None:
        """Boundary edge: beta = 0, alpha = 180 - gamma = 120 for equilateral."""
        points = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, np.sqrt(3.0)]])
        simplices = np.array([[0, 1, 2]])
        stability = _compute_edge_stability(p_idx=0, q_idx=1, simplices=simplices, points=points)
        assert abs(stability - 120.0) < 1.0

    def test_cocircular_points_zero_stability(self) -> None:
        """Square on unit circle: opposite angles sum to 180, alpha = 0."""
        points = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        simplices = np.array([[0, 1, 2], [0, 2, 3]])
        stability = _compute_edge_stability(p_idx=0, q_idx=2, simplices=simplices, points=points)
        assert abs(stability) < 1.0


# ---------------------------------------------------------------------------
# Shape Graph Construction (Algorithm 1)
# ---------------------------------------------------------------------------


class TestComputeShapeGraph:
    """Test the full shape graph construction."""

    def test_classic_442_returns_stable_graph(self, positions_442: np.ndarray) -> None:
        sg = compute_shape_graph(positions_442)
        assert isinstance(sg, ShapeGraph)
        assert sg.edges.shape[1] == 2
        assert len(sg.edges) > 0
        assert all(s >= 45.0 for s in sg.stabilities)
        assert len(sg.faces) >= 1
        all_players: set[int] = set()
        for face in sg.faces:
            all_players.update(face)
        assert all_players == set(range(10))

    def test_three_players_triangle(self) -> None:
        positions = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 8.0]])
        sg = compute_shape_graph(positions)
        assert len(sg.edges) == 3

    def test_fewer_than_three_returns_empty(self) -> None:
        for n in (0, 1, 2):
            positions = np.array([[float(i), 0.0] for i in range(n)]) if n > 0 else np.empty((0, 2))
            sg = compute_shape_graph(positions)
            assert len(sg.edges) == 0
            assert len(sg.faces) == 0

    def test_collinear_players_empty_graph(self) -> None:
        positions = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0], [15.0, 0.0]])
        sg = compute_shape_graph(positions)
        assert isinstance(sg, ShapeGraph)
        assert len(sg.edges) == 0

    def test_no_edge_below_threshold(self, positions_442: np.ndarray) -> None:
        sg = compute_shape_graph(positions_442)
        for stability in sg.stabilities:
            assert stability >= 45.0

    def test_stabilities_match_edges(self) -> None:
        positions = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 8.0], [15.0, 5.0], [7.0, 12.0]])
        sg = compute_shape_graph(positions)
        assert len(sg.stabilities) == len(sg.edges)

    def test_points_preserved(self) -> None:
        positions = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 8.0]])
        sg = compute_shape_graph(positions)
        np.testing.assert_array_equal(sg.points, positions)

    def test_custom_threshold_strict_removes_more(self) -> None:
        positions = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 8.0], [15.0, 5.0], [7.0, 12.0]])
        sg_strict = compute_shape_graph(positions, stability_threshold=170.0)
        sg_lenient = compute_shape_graph(positions, stability_threshold=0.0)
        assert len(sg_lenient.edges) >= len(sg_strict.edges)

    def test_threshold_zero_keeps_all_delaunay_edges(self) -> None:
        positions = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 8.0], [15.0, 5.0], [7.0, 12.0]])
        sg = compute_shape_graph(positions, stability_threshold=0.0)
        from scipy.spatial import Delaunay

        tri = Delaunay(positions)
        delaunay_edges: set[tuple[int, int]] = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = int(simplex[i]), int(simplex[j])
                    delaunay_edges.add((min(a, b), max(a, b)))
        sg_edges = {(int(e[0]), int(e[1])) for e in sg.edges}
        assert sg_edges == delaunay_edges

    def test_352_produces_connected_graph(self, positions_352: np.ndarray) -> None:
        sg = compute_shape_graph(positions_352)
        assert len(sg.edges) > 0
        all_players = set()
        for face in sg.faces:
            all_players.update(face)
        assert all_players == set(range(10))

    def test_tie_breaking_does_not_overprune(self) -> None:
        angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        positions = np.column_stack([np.cos(angles), np.sin(angles)]) * 10.0
        sg = compute_shape_graph(positions)
        assert len(sg.edges) >= 4


# ---------------------------------------------------------------------------
# Position Inference (thesis Chapter 4)
# ---------------------------------------------------------------------------


class TestInferPositions:
    """Test position inference via recursive face-center decomposition."""

    def test_442_vertical_levels(self, positions_442: np.ndarray) -> None:
        sg = compute_shape_graph(positions_442)
        labels = infer_positions(sg, positions_442, attacking_direction=1.0)
        assert len(labels) == 10
        for i in range(4):
            assert labels[i].vertical == "B"
        for i in range(8, 10):
            assert labels[i].vertical == "F"

    def test_442_horizontal_levels(self, positions_442: np.ndarray) -> None:
        sg = compute_shape_graph(positions_442)
        labels = infer_positions(sg, positions_442, attacking_direction=1.0)
        def_horizontals = [labels[i].horizontal for i in range(4)]
        assert "L" in def_horizontals
        assert "R" in def_horizontals

    def test_reversed_attacking_direction(self, positions_442: np.ndarray) -> None:
        sg = compute_shape_graph(positions_442)
        labels_fwd = infer_positions(sg, positions_442, attacking_direction=1.0)
        labels_rev = infer_positions(sg, positions_442, attacking_direction=-1.0)
        for i in range(4):
            assert labels_fwd[i].vertical == "B"
            assert labels_rev[i].vertical == "F"

    def test_position_labels_follow_thesis_matrix(self, positions_442: np.ndarray) -> None:
        sg = compute_shape_graph(positions_442)
        labels = infer_positions(sg, positions_442, attacking_direction=1.0)
        for lbl in labels:
            assert lbl.vertical in {"B", "DM", "M", "AM", "F"}
            assert lbl.horizontal in {"L", "LC", "C", "RC", "R"}
            expected_label = POSITION_LABEL_MATRIX[lbl.vertical][lbl.horizontal]
            assert lbl.label == expected_label

    def test_empty_shape_graph_returns_empty(self) -> None:
        positions = np.array([[0.0, 0.0], [10.0, 0.0]])
        sg = compute_shape_graph(positions)
        labels = infer_positions(sg, positions, attacking_direction=1.0)
        assert labels == []

    def test_352_vertical_distribution(self, positions_352: np.ndarray) -> None:
        sg = compute_shape_graph(positions_352)
        labels = infer_positions(sg, positions_352, attacking_direction=1.0)
        assert len(labels) == 10
        for i in range(3):
            assert labels[i].vertical == "B"
        for i in range(8, 10):
            assert labels[i].vertical == "F"

    def test_all_labels_are_valid_matrix_entries(self, positions_442: np.ndarray) -> None:
        valid_labels = set()
        for row in POSITION_LABEL_MATRIX.values():
            valid_labels.update(row.values())
        sg = compute_shape_graph(positions_442)
        labels = infer_positions(sg, positions_442, attacking_direction=1.0)
        for lbl in labels:
            assert lbl.label in valid_labels


class TestVerticalLevelAssignment:
    """Test the vertical level assignment helper directly."""

    def test_three_distinct_x_groups(self) -> None:
        x_values = np.array([20.0, 20.0, 20.0, 20.0, 40.0, 40.0, 40.0, 40.0, 60.0, 60.0])
        face_centers_x = np.array([30.0, 50.0])
        levels = _assign_levels_vertical(x_values, face_centers_x)
        assert all(lv == "B" for lv in levels[:4])
        assert all(lv == "F" for lv in levels[8:10])

    def test_single_player_gets_level(self) -> None:
        x_values = np.array([50.0])
        face_centers_x = np.array([50.0])
        levels = _assign_levels_vertical(x_values, face_centers_x)
        assert len(levels) == 1
        assert levels[0] in {"B", "DM", "M", "AM", "F"}


class TestHorizontalLevelAssignment:
    """Test the horizontal level assignment helper directly."""

    def test_four_players_across(self) -> None:
        y_values = np.array([10.0, 25.0, 43.0, 58.0])
        face_centers_y = np.array([20.0, 40.0])
        levels = _assign_levels_horizontal(y_values, face_centers_y)
        assert levels[0] == "L"
        assert levels[3] == "R"


# ---------------------------------------------------------------------------
# Aggregator + VAEP xfn tests
# ---------------------------------------------------------------------------


def _make_frames_and_actions(
    positions_442: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build minimal frames + actions DataFrames for aggregator tests."""
    rows = []
    # Ball row
    rows.append(
        {
            "game_id": 1,
            "period_id": 1,
            "frame_id": 100,
            "time_seconds": 10.0,
            "player_id": -1,
            "team_id": pd.NA,
            "is_ball": True,
            "is_goalkeeper": False,
            "x": 40.0,
            "y": 34.0,
            "vx": 0.0,
            "vy": 0.0,
            "speed": 0.0,
            "ax": 0.0,
            "ay": 0.0,
            "is_visible": True,
            "jersey_number": pd.NA,
            "is_goalkeeper_source": "native",
            "source_provider": "test",
            "timestamp": pd.Timestamp("2026-01-01 00:00:10"),
        }
    )

    # Team 1 outfield (10 players from positions_442)
    for pid, (x, y) in enumerate(positions_442):
        rows.append(
            {
                "game_id": 1,
                "period_id": 1,
                "frame_id": 100,
                "time_seconds": 10.0,
                "player_id": 100 + pid,
                "team_id": 1,
                "is_ball": False,
                "is_goalkeeper": False,
                "x": x,
                "y": y,
                "vx": 0.0,
                "vy": 0.0,
                "speed": 0.0,
                "ax": 0.0,
                "ay": 0.0,
                "is_visible": True,
                "jersey_number": pid + 2,
                "is_goalkeeper_source": "native",
                "source_provider": "test",
                "timestamp": pd.Timestamp("2026-01-01 00:00:10"),
            }
        )

    # Team 2 outfield (10 players mirrored)
    for pid, (x, y) in enumerate(positions_442):
        rows.append(
            {
                "game_id": 1,
                "period_id": 1,
                "frame_id": 100,
                "time_seconds": 10.0,
                "player_id": 200 + pid,
                "team_id": 2,
                "is_ball": False,
                "is_goalkeeper": False,
                "x": 105.0 - x,
                "y": y,
                "vx": 0.0,
                "vy": 0.0,
                "speed": 0.0,
                "ax": 0.0,
                "ay": 0.0,
                "is_visible": True,
                "jersey_number": pid + 2,
                "is_goalkeeper_source": "native",
                "source_provider": "test",
                "timestamp": pd.Timestamp("2026-01-01 00:00:10"),
            }
        )

    frames = pd.DataFrame(rows)

    actions = pd.DataFrame(
        {
            "action_id": [0],
            "game_id": [1],
            "period_id": [1],
            "team_id": [1],
            "time_seconds": [10.0],
            "type_id": [0],
            "result_id": [1],
            "bodypart_id": [0],
            "start_x": [40.0],
            "start_y": [34.0],
            "end_x": [50.0],
            "end_y": [34.0],
            "player_id": [100],
        }
    )

    return frames, actions


def _multiframe_shape_graph(positions_442: np.ndarray, n_frames: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Multi-frame frames + a single action, for linked-frame restriction tests.

    Frame 100 keeps the base geometry; frames 101+ perturb player y so each
    frame's shape graph differs (proving a linked frame's metric is independent
    of the other frames present).
    """
    base, actions = _make_frames_and_actions(positions_442)
    parts = [base]
    for k in range(1, n_frames):
        f = base.copy()
        f["frame_id"] = 100 + k
        f["time_seconds"] = 10.0 + 0.04 * k
        outfield = ~f["is_ball"].astype(bool)
        f.loc[outfield, "y"] = f.loc[outfield, "y"] * (1.0 + 0.05 * k)
        parts.append(f)
    return pd.concat(parts, ignore_index=True), actions


def _links_to_frame(fid: int) -> pd.DataFrame:
    """Minimal link pointers mapping action 0 to a given frame_id."""
    return pd.DataFrame(
        {
            "action_id": [0],
            "frame_id": [fid],
            "time_offset_seconds": [0.0],
            "n_candidate_frames": [1],
            "link_quality_score": [1.0],
        }
    )


class TestShapeGraphLinkedFrameRestriction:
    """add_shape_graph must restrict the per-frame loop to linked frames when links given."""

    def test_restricts_compute_calls(self, positions_442: np.ndarray, monkeypatch: pytest.MonkeyPatch) -> None:
        import silly_kicks.tracking._shape_graph as sg_mod

        real = sg_mod.compute_shape_graph
        n = {"calls": 0}

        def spy(positions, *a, **k):
            n["calls"] += 1
            return real(positions, *a, **k)

        monkeypatch.setattr(sg_mod, "compute_shape_graph", spy)

        from silly_kicks.tracking.features import add_shape_graph

        frames, actions = _multiframe_shape_graph(positions_442, n_frames=5)
        add_shape_graph(actions, frames, links=_links_to_frame(100), home_team_id=1)
        # 1 linked frame x 2 teams, not 5 frames x 2 teams.
        assert n["calls"] == 2

    def test_no_links_computes_all_frames(self, positions_442: np.ndarray, monkeypatch: pytest.MonkeyPatch) -> None:
        import silly_kicks.tracking._shape_graph as sg_mod

        real = sg_mod.compute_shape_graph
        n = {"calls": 0}

        def spy(positions, *a, **k):
            n["calls"] += 1
            return real(positions, *a, **k)

        monkeypatch.setattr(sg_mod, "compute_shape_graph", spy)

        from silly_kicks.tracking.features import add_shape_graph

        frames, actions = _multiframe_shape_graph(positions_442, n_frames=5)
        add_shape_graph(actions, frames, home_team_id=1)  # no links -> full
        assert n["calls"] == 10

    def test_restricted_matches_single_frame(self, positions_442: np.ndarray) -> None:
        """Restricting among many frames == computing with only the linked frame present."""
        from silly_kicks.tracking.features import add_shape_graph

        frames, actions = _multiframe_shape_graph(positions_442, n_frames=5)
        links = _links_to_frame(100)
        single = frames[frames["frame_id"] == 100]

        r_multi = add_shape_graph(actions, frames, links=links, home_team_id=1)
        r_single = add_shape_graph(actions, single, links=links, home_team_id=1)

        cols = [c for c in r_multi.columns if c.startswith("shape_graph_")]
        pd.testing.assert_frame_equal(r_multi[cols], r_single[cols])


class TestAddShapeGraph:
    """Test add_shape_graph aggregator."""

    def test_produces_6_columns(self, positions_442: np.ndarray) -> None:
        from silly_kicks.tracking.features import add_shape_graph

        frames, actions = _make_frames_and_actions(positions_442)
        result = add_shape_graph(actions, frames, home_team_id=1)
        expected_cols = [
            "shape_graph_density_attacking",
            "shape_graph_n_edges_attacking",
            "shape_graph_mean_stability_attacking",
            "shape_graph_density_defending",
            "shape_graph_n_edges_defending",
            "shape_graph_mean_stability_defending",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_nan_safe(self, positions_442: np.ndarray) -> None:
        """Empty frames produce NaN output without crash."""
        from silly_kicks.tracking.features import add_shape_graph

        frames, actions = _make_frames_and_actions(positions_442)
        empty_frames = frames.iloc[:0]
        result = add_shape_graph(actions, empty_frames, home_team_id=1)
        assert len(result) == 1


class TestShapeGraphXfns:
    """Test shape_graph_xfns VAEP factory."""

    def test_column_count_18(self) -> None:
        from silly_kicks.tracking.features import shape_graph_xfns

        xfns = shape_graph_xfns("team_A")
        assert len(xfns) == 1

        fn = xfns[0]
        assert hasattr(fn, "_frame_aware")
        assert fn._frame_aware is True

    def test_introspection_nan(self) -> None:
        """xfn should emit NaN columns when frames=None (VAEP introspection)."""
        from silly_kicks.tracking.features import shape_graph_xfns

        xfns = shape_graph_xfns("team_A")
        fn = xfns[0]

        dummy_actions = pd.DataFrame(
            {
                "game_id": [1] * 3,
                "period_id": [1] * 3,
                "action_id": [0, 1, 2],
                "team_id": ["team_A"] * 3,
                "time_seconds": [0.0, 1.0, 2.0],
                "type_id": [0, 0, 0],
                "result_id": [1, 1, 1],
                "bodypart_id": [0, 0, 0],
                "start_x": [50.0] * 3,
                "start_y": [34.0] * 3,
                "end_x": [60.0] * 3,
                "end_y": [34.0] * 3,
                "player_id": [1, 2, 3],
            }
        )

        states = [dummy_actions] * 3
        result = fn(states, None)
        # 6 cols (3 metrics x 2 teams) x 3 states = 18
        assert result.shape[1] == 18
        assert result.isna().all().all()
