"""W5.6: team-relative position labels (spec 2026-11-08 tracking-batch section 4.5 W5.6, owner ruling R14; plan
Task 10A).

Every label is in the team's own frame, with SPADL handedness: for a team that attacks +x, "L" is high y; for a team
that attacks -x, "L" is low y. The detection tests were red at BASE_SHA. A frozen oracle pinned the base behaviour
next to them until the oracles ended (ruling C5, spec section 7); each test now pins the new labels only.

Two branches of the label path cannot fire on a graph that ``compute_shape_graph`` returns: the equal-frequency
fallback needs a graph with edges but no face that survives the Option B spread filter, and the pitch-thirds rule
needs a middle sub-graph with edges and no face of 3 or more vertices. ``compute_shape_graph`` never removes a face,
and every face of a team-sized frame spans more than 10 % of the lateral spread. So their fixtures are degenerate
graphs built here: a ``ShapeGraph`` with edges and no faces, which ``infer_positions`` accepts as public input.
"""

from __future__ import annotations

import contextlib
import itertools

import numpy as np
import pytest

from silly_kicks.tracking import _shape_graph as live
from silly_kicks.tracking._shape_graph import ShapeGraph, compute_shape_graph, infer_positions

_POSITIONS_442 = np.array(
    [
        [20.0, 10.0], [20.0, 25.0], [20.0, 43.0], [20.0, 58.0],  # defenders
        [40.0, 10.0], [40.0, 25.0], [40.0, 43.0], [40.0, 58.0],  # midfielders
        [60.0, 25.0], [60.0, 43.0],  # forwards
    ]
)  # fmt: skip


def _faceless(points: np.ndarray) -> ShapeGraph:
    """A degenerate graph: a path through ``points`` in x order, with edges and no faces."""
    order = np.argsort(points[:, 0], kind="stable")
    pairs = sorted((min(int(a), int(b)), max(int(a), int(b))) for a, b in itertools.pairwise(order))
    edges = np.array(pairs, dtype=int).reshape(-1, 2)
    return ShapeGraph(edges=edges, faces=[], stabilities=np.full(len(edges), 90.0), points=points)


def _horizontal(labels) -> list[str]:
    return [lbl.horizontal for lbl in labels]


# ---- D-lat: the lateral label of a +x team ---------------------------------------------------------------------


def test_dlat_plus_x_high_y_full_back_is_L():
    labels = infer_positions(compute_shape_graph(_POSITIONS_442), _POSITIONS_442, 1.0)
    defenders = range(4)
    high = max(defenders, key=lambda i: _POSITIONS_442[i, 1])
    low = min(defenders, key=lambda i: _POSITIONS_442[i, 1])
    assert labels[high].horizontal == "L"
    assert labels[low].horizontal == "R"


# ---- D-eqf: the equal-frequency fallback ------------------------------------------------------------------------

_EQF = np.array([[30.0, 8.0], [45.0, 21.0], [38.0, 33.0], [52.0, 47.0], [41.0, 60.0]])


def test_deqf_fixture_reaches_fallback():
    sg = _faceless(_EQF)
    assert len(sg.edges) > 0
    assert len(live._face_centroids(sg, axis=1)) == 0


@pytest.mark.parametrize("direction", [1.0, -1.0])
def test_deqf_fallback_team_relative(direction):
    # [-1.0] already passes at the base (pitch-absolute low y is "L"): a guard, not a detection test (A11-PLAN-92)
    labels = infer_positions(_faceless(_EQF), _EQF, direction)
    left = int(np.argmax(_EQF[:, 1])) if direction > 0 else int(np.argmin(_EQF[:, 1]))
    assert labels[left].horizontal == "L"


# ---- D-tree: the pitch-thirds rule of a -x team -----------------------------------------------------------------

# A team that attacks -x (its own goal is at x = 105) with a staggered middle three deep in its own third. The edges
# of the middle three are not steep on the vertical axis, so no bridging midpoint replaces the tree rule.
_DTREE = np.array(
    [
        [100.0, 8.0], [100.0, 24.0], [100.0, 44.0], [100.0, 60.0],  # back four
        [86.0, 30.0], [90.0, 34.0], [94.0, 38.0],  # the middle three
        [72.0, 12.0], [72.0, 56.0], [62.0, 34.0],  # forwards
    ]
)  # fmt: skip
_DTREE_MIDDLE = (4, 5, 6)


@contextlib.contextmanager
def _faceless_vertical_middle(module, record: list | None = None):
    """Inside ``module.infer_positions``: the vertical-axis Option C sub-graph is :func:`_faceless` (a tree). The
    lateral axis keeps the real sub-graph. With ``record``, it collects each vertical ``_is_tree`` result."""
    real_decompose, real_csg, real_tree = module._decompose_middle, module.compute_shape_graph, module._is_tree

    def tree(graph) -> bool:
        result = real_tree(graph)
        if record is not None:
            record.append(result)
        return result

    def decompose(*args, **kwargs):
        if kwargs.get("axis") != 0:
            return real_decompose(*args, **kwargs)
        module.compute_shape_graph, module._is_tree = _faceless, tree
        try:
            return real_decompose(*args, **kwargs)
        finally:
            module.compute_shape_graph, module._is_tree = real_csg, real_tree

    module._decompose_middle = decompose
    try:
        yield
    finally:
        module._decompose_middle = real_decompose


def test_dtree_fixture_reaches_tree_branch():
    record: list[bool] = []
    sg = compute_shape_graph(_DTREE)
    with _faceless_vertical_middle(live, record):
        infer_positions(sg, _DTREE, -1.0)
    assert True in record  # the vertical tree branch ran


def test_dtree_minus_x_deep_middle_is_DM():
    sg = compute_shape_graph(_DTREE)
    with _faceless_vertical_middle(live):
        labels = infer_positions(sg, _DTREE, -1.0)
    assert [labels[i].vertical for i in _DTREE_MIDDLE] == ["DM"] * 3  # 4.139 gave "AM": a second negation


# ---- the direction ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("direction", [float("nan"), 0.0, float("inf"), float("-inf")])
def test_direction_must_be_finite_nonzero(direction):
    with pytest.raises(ValueError, match="attacking_direction"):
        infer_positions(compute_shape_graph(_POSITIONS_442), _POSITIONS_442, direction)


# ---- mirror invariance and its premise --------------------------------------------------------------------------


def _team_frames_float32(seed: int, n_frames: int = 200) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_frames):
        n = int(rng.integers(3, 11))
        out.append(np.column_stack([rng.uniform(3.0, 102.0, n), rng.uniform(2.0, 66.0, n)]).astype(np.float32))
    return out


def _reflect(p: np.ndarray) -> np.ndarray:
    """The point reflection in float64: exact for float32 input."""
    p64 = p.astype(np.float64)
    return np.column_stack([105.0 - p64[:, 0], 68.0 - p64[:, 1]])


@pytest.mark.parametrize("seed", [20261111, 5])
def test_mirror_invariance(seed):
    n_labelled = n_lateral = 0
    for p in _team_frames_float32(seed):
        r = _reflect(p)
        minus = infer_positions(compute_shape_graph(p), p, -1.0)
        plus = infer_positions(compute_shape_graph(r), r, 1.0)
        assert minus == plus
        n_labelled += bool(minus)
        n_lateral += any(lbl.horizontal != "C" for lbl in minus)
    assert n_labelled >= 150 and n_lateral >= 150  # non-vacuity: labelled frames with non-central lateral labels


@pytest.mark.parametrize("seed", [20261111, 5])
def test_reflection_preserves_graph_faces(seed):
    for p in _team_frames_float32(seed):
        a, b = compute_shape_graph(p), compute_shape_graph(_reflect(p))
        assert {tuple(e) for e in a.edges.tolist()} == {tuple(e) for e in b.edges.tolist()}
        assert set(a.faces) == set(b.faces)


# ---- the -x lateral fallback (n = 9) and the -x two-player middle tie -------------------------------------------


def test_minus_x_lateral_fallback_n9():
    # spec W5.6: the fallback bins the ascending rank r of 68 - y as min(5r // 9, 4) onto R, RC, C, LC, L
    y = np.array([5.0, 12.0, 20.0, 27.0, 34.0, 41.0, 48.0, 56.0, 63.0])
    x = np.array([70.0, 55.0, 80.0, 62.0, 90.0, 48.0, 75.0, 66.0, 85.0])
    p = np.column_stack([x, y])
    labels = infer_positions(_faceless(p), p, -1.0)
    assert _horizontal(labels) == ["L", "LC", "LC", "C", "C", "RC", "RC", "R", "R"]  # 4.139: four players one level off


# A team that attacks -x whose lateral middle group is exactly two players with equal y (indices 2 and 3).
_TIE = np.array(
    [
        [90.0, 6.0], [90.0, 62.0],  # wide backs
        [80.0, 34.0], [60.0, 34.0],  # the two with exactly equal y
        [70.0, 14.0], [70.0, 54.0],  # wide midfielders
        [50.0, 10.0], [50.0, 58.0],  # wide forwards
    ]
)  # fmt: skip


def test_minus_x_two_player_middle_tie():
    sg = compute_shape_graph(_TIE)
    labels = infer_positions(sg, _TIE, -1.0)
    assert (labels[2].horizontal, labels[3].horizontal) == ("RC", "LC")  # the lower index gets "RC"; 4.139 swapped them
