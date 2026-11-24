"""Shape graph construction + tactical position inference (TF-39).

Implements the iterative edge-removal algorithm (Algorithm 1) and face-center
position decomposition (Chapter 4, §4.1-4.3) from Sotudeh (2026).

The shape graph is a sparse, stable subgraph of the Delaunay triangulation.
Unstable edges (low angular stability) are iteratively removed and their
incident faces merged, producing a clean proximity graph that filters the
"flicker" noise inherent in raw Delaunay triangulations of player positions.

Position inference decomposes player coordinates into a 5x5 grid of tactical
roles (5 vertical levels: B/DM/M/AM/F, 5 horizontal levels: L/LC/C/RC/R)
using recursive face-center decomposition along each axis.

See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, QhullError

from silly_kicks._frame_index import group_rows
from silly_kicks.id_compat import canonical_id
from silly_kicks.tracking._geometry import FIELD_LENGTH, to_goal_relative_x_array, to_goal_relative_y_array

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PositionLabel:
    """A tactical position label from the 5x5 decomposition.

    Attributes
    ----------
    vertical : str
        Vertical level — one of B, DM, M, AM, F.
    horizontal : str
        Horizontal level — one of L, LC, C, RC, R.
    label : str
        Combined label using thesis notation, e.g. "RCB" (not "B-RC").
    """

    vertical: str  # B | DM | M | AM | F
    horizontal: str  # L | LC | C | RC | R
    label: str  # e.g. "RCB"


@dataclass(frozen=True)
class ShapeGraph:
    """Result of the shape graph algorithm.

    Attributes
    ----------
    edges : np.ndarray
        (m, 2) array of player index pairs forming the stable subgraph.
    faces : list[frozenset[int]]
        List of frozensets, each containing player indices in one merged face.
    stabilities : np.ndarray
        (m,) array of angular stability values (degrees) per edge.
    points : np.ndarray
        (n, 2) original player positions.
    """

    edges: np.ndarray
    faces: list[frozenset[int]]
    stabilities: np.ndarray
    points: np.ndarray


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STABILITY_THRESHOLD: float = 45.0

# Vertical level labels (from back to front)
_VERTICAL_LEVELS: tuple[str, ...] = ("B", "DM", "M", "AM", "F")
# Horizontal level labels (from left to right)
# The lateral ladder in ascending y of the ATTACK frame (spec 2026-11-08 tracking-batch W5.6): there, high y is the
# left of a player who faces the attacked goal.
_HORIZONTAL_LEVELS: tuple[str, ...] = ("R", "RC", "C", "LC", "L")

# 5x5 position label matrix (thesis Figure 4.5b)
# Maps (vertical, horizontal) → compact position label.
POSITION_LABEL_MATRIX: Mapping[str, Mapping[str, str]] = {
    "B": {"L": "LB", "LC": "LCB", "C": "CB", "RC": "RCB", "R": "RB"},
    "DM": {"L": "LWB", "LC": "LDM", "C": "CDM", "RC": "RDM", "R": "RWB"},
    "M": {"L": "LM", "LC": "LCM", "C": "CM", "RC": "RCM", "R": "RM"},
    "AM": {"L": "LWF", "LC": "LAM", "C": "CAM", "RC": "RAM", "R": "RWF"},
    "F": {"L": "LF", "LC": "LCF", "C": "CF", "RC": "RCF", "R": "RF"},
}

# Steep bridging edge threshold (thesis §4.2): 90° ± 22.5°
_STEEP_ANGLE_LOW: float = 67.5  # degrees
_STEEP_ANGLE_HIGH: float = 112.5  # degrees

# Adaptive fallback threshold (Option D): if >60% of players in one level
_DEGENERATE_FRACTION: float = 0.6


# ---------------------------------------------------------------------------
# Angular stability (thesis §3.3, equation 3.2)
# ---------------------------------------------------------------------------


def _angle_at_vertex(
    vertex: np.ndarray,
    arm1: np.ndarray,
    arm2: np.ndarray,
) -> float:
    """Compute the angle at *vertex* formed by rays to *arm1* and *arm2*.

    Returns angle in degrees in [0, 180].
    """
    v1 = arm1 - vertex
    v2 = arm2 - vertex
    dot = float(np.dot(v1, v2))
    # np.linalg.norm of a 1-D real vector IS sqrt(x.dot(x)) in the vector's own dtype; spelling it out skips the
    # norm wrapper and keeps every bit (spec 2026-11-08 tracking-batch W5.1). The dot stays numpy's (a BLAS dot is
    # not guaranteed to round as `a*b + c*d` does).
    mag1 = float(np.sqrt(np.dot(v1, v1)))
    mag2 = float(np.sqrt(np.dot(v2, v2)))
    if mag1 < 1e-12 or mag2 < 1e-12:
        return 0.0
    ratio = dot / (mag1 * mag2)
    cos_val = -1.0 if ratio < -1.0 else (1.0 if ratio > 1.0 else ratio)  # np.clip of a float; NaN passes through
    return float(np.degrees(np.arccos(cos_val)))


def _compute_edge_stability(
    p_idx: int,
    q_idx: int,
    simplices: np.ndarray,
    points: np.ndarray,
) -> float:
    """Compute angular stability of edge (p, q) per Sotudeh equation 3.2.

    For an edge pq shared by two triangles with opposite vertices p' and q':
      gamma = angle at p' in triangle 1
      beta  = angle at q' in triangle 2
      alpha = 180 - (gamma + beta)

    Boundary edges (one incident triangle) have beta = 0 (the missing triangle
    contributes no opposite angle), so alpha = 180 - gamma.
    """
    edge_set = {p_idx, q_idx}

    # Find triangles incident to this edge and their opposite vertices
    opposite_angles: list[float] = []
    for simplex in simplices:
        simplex_verts = {int(simplex[0]), int(simplex[1]), int(simplex[2])}
        if edge_set.issubset(simplex_verts):
            # The opposite vertex is the one NOT in the edge
            opposite_idx = (simplex_verts - edge_set).pop()
            angle = _angle_at_vertex(points[opposite_idx], points[p_idx], points[q_idx])
            opposite_angles.append(angle)

    if not opposite_angles:
        return 180.0  # No incident triangles — treat as maximally stable

    # Sum of opposite angles (one or two triangles)
    angle_sum = sum(opposite_angles)
    return max(180.0 - angle_sum, 0.0)


# ---------------------------------------------------------------------------
# Edge extraction helpers
# ---------------------------------------------------------------------------


def _extract_edges(simplices: np.ndarray) -> set[tuple[int, int]]:
    """Extract unique edges from Delaunay simplices."""
    edges: set[tuple[int, int]] = set()
    for simplex in simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = int(simplex[i]), int(simplex[j])
                edges.add((min(a, b), max(a, b)))
    return edges


# ---------------------------------------------------------------------------
# Face state: merged faces, the vertex -> face index, and the opposite-angle cache
# ---------------------------------------------------------------------------


class _FaceState:
    """The faces of the shape graph during Algorithm 1, in creation order, with the indexes that make it fast.

    Spec 2026-11-08 tracking-batch W5.1: the speed-ups are by construction, so every value equals the historical
    list-scan code bit for bit.

    - The faces keep the historical list order: a merge drops the incident faces and appends the merged face, so a
      face's creation id orders the faces as the list did. A vertex -> face-id index finds the faces incident to an
      edge without a scan of all faces.
    - Opposite angles are cached per (edge, opposite vertex): an angle depends only on the three points, so the
      cached value is the one a recomputation gives. The cache is shared with :meth:`copy` (the tie-break
      simulation), which reads the same points.
    """

    def __init__(self, faces: list[frozenset[int]], points: np.ndarray) -> None:
        self._faces: dict[int, frozenset[int]] = dict(enumerate(faces))
        self._next_id = len(faces)
        self._by_vertex: dict[int, set[int]] = {}
        for fid, face in self._faces.items():
            for v in face:
                self._by_vertex.setdefault(v, set()).add(fid)
        self._points = points
        self._angles: dict[tuple[int, int, int], float] = {}

    def copy(self) -> _FaceState:
        """An independent face state over the same points; the angle cache is shared."""
        twin = object.__new__(_FaceState)
        twin._faces = dict(self._faces)
        twin._next_id = self._next_id
        twin._by_vertex = {v: set(ids) for v, ids in self._by_vertex.items()}
        twin._points = self._points
        twin._angles = self._angles
        return twin

    def faces(self) -> list[frozenset[int]]:
        """The faces in list order."""
        return list(self._faces.values())

    def _incident_ids(self, edge: tuple[int, int]) -> list[int]:
        return sorted(self._by_vertex.get(edge[0], set()) & self._by_vertex.get(edge[1], set()))

    def incident(self, edge: tuple[int, int]) -> list[frozenset[int]]:
        """The faces that contain both vertices of ``edge``, in list order."""
        return [self._faces[fid] for fid in self._incident_ids(edge)]

    def _opposite_angle(self, p: int, q: int, r: int) -> float:
        key = (p, q, r)
        angle = self._angles.get(key)
        if angle is None:
            pts = self._points
            angle = _angle_at_vertex(pts[r], pts[p], pts[q])
            self._angles[key] = angle
        return angle

    def stability(self, edge: tuple[int, int]) -> float:
        """Stability of ``edge`` from the merged faces (UpdateStabilities).

        After face merging, the opposite angle from the merged-face side becomes the **minimum** angle formed by
        (p, r, q) over all vertices r in that face (excluding p and q). This prevents over-pruning (thesis
        p.35-36, Figure 3.8).

        For each side of the edge, find the incident face and compute:
          - If the side is the external face: opposite angle = 0°
          - Else: opposite angle = min over r in face \\ {p,q} of angle(p, r, q)

        alpha = 180 - (angle_side1 + angle_side2)
        """
        p_idx, q_idx = edge
        edge_set = frozenset({p_idx, q_idx})
        side_angles: list[float] = []
        for face in self.incident(edge):
            other_verts = face - edge_set
            if not other_verts:
                # Degenerate face (just the edge) — contributes 0°
                side_angles.append(0.0)
                continue
            # Minimum angle at any opposite vertex in this face
            side_angles.append(min(self._opposite_angle(p_idx, q_idx, r) for r in other_verts))

        # Boundary: missing side contributes 0°
        while len(side_angles) < 2:
            side_angles.append(0.0)

        # Use the two incident face contributions
        return max(180.0 - side_angles[0] - side_angles[1], 0.0)

    def merge(self, edge: tuple[int, int]) -> None:
        """Merge the faces incident to ``edge`` into one face (appended last); a boundary edge merges nothing."""
        ids = self._incident_ids(edge)
        if len(ids) < 2:
            return  # Boundary edge — no merge needed
        merged: frozenset[int] = frozenset[int]().union(*[self._faces[fid] for fid in ids])
        for fid in ids:
            for v in self._faces.pop(fid):
                self._by_vertex[v].discard(fid)
        new_id = self._next_id
        self._next_id += 1
        self._faces[new_id] = merged
        for v in merged:
            self._by_vertex.setdefault(v, set()).add(new_id)

    def update_after_merge(
        self,
        removed_edge: tuple[int, int],
        active_edges: set[tuple[int, int]],
        edge_stability: dict[tuple[int, int], float],
    ) -> None:
        """Recompute stabilities for edges adjacent to a just-merged face.

        After removing *removed_edge* and merging its incident faces, find the merged face (the first face that
        contains both vertices of the removed edge) and recompute stability for all of its boundary edges that are
        still active. The edges are visited in ``active_edges`` order: an edge that tie-breaking took out of the
        queue re-enters ``edge_stability`` at its end, so the visiting order is part of the result.

        Mutates *edge_stability* in place.
        """
        incident = self.incident(removed_edge)
        if not incident:
            return
        merged_face = incident[0]
        for edge in list(active_edges):
            if edge[0] in merged_face and edge[1] in merged_face:
                edge_stability[edge] = self.stability(edge)


# ---------------------------------------------------------------------------
# Shape graph construction (Algorithm 1, thesis p.38)
# ---------------------------------------------------------------------------


def _empty_shape_graph(positions: np.ndarray) -> ShapeGraph:
    """Return an empty shape graph for degenerate inputs."""
    return ShapeGraph(
        edges=np.empty((0, 2), dtype=int),
        faces=[],
        stabilities=np.empty(0),
        points=positions,
    )


def compute_shape_graph(
    positions: np.ndarray,
    stability_threshold: float = _STABILITY_THRESHOLD,
) -> ShapeGraph:
    """Compute the shape graph of player positions (Sotudeh Algorithm 1).

    1. Compute Delaunay triangulation of outfield player (x, y) positions.
    2. Calculate angular stability for each edge.
    3. Find edges with minimal stability; if below threshold, remove them
       (with tie-breaking to prevent over-pruning).
    4. Recompute stabilities on affected edges via UpdateStabilities.
    5. Repeat until all remaining edges have stability >= threshold.

    Parameters
    ----------
    positions : np.ndarray
        (n, 2) array of outfield player (x, y) coordinates.
    stability_threshold : float
        Minimum angular stability in degrees (default 45.0).

    Returns
    -------
    ShapeGraph
        Stable edges, merged faces, and stability values.

    Examples
    --------
    Compute the stable subgraph for a set of player positions::

        import numpy as np
        from silly_kicks.tracking import compute_shape_graph

        positions = np.array([
            [20, 10], [20, 30], [20, 50],  # defenders
            [40, 20], [40, 40],             # midfielders
            [60, 25], [60, 35],             # forwards
        ], dtype=float)
        sg = compute_shape_graph(positions)
        print(sg.edges.shape, sg.stabilities.mean())

    See NOTICE for full bibliographic citations.
    """
    n = len(positions)
    if n < 3:
        return _empty_shape_graph(positions)

    try:
        tri = Delaunay(positions)
    except QhullError:
        warnings.warn(
            "Delaunay triangulation failed (likely collinear points)",
            stacklevel=2,
        )
        return _empty_shape_graph(positions)

    # Initialize edges and faces
    active_edges: set[tuple[int, int]] = _extract_edges(tri.simplices)

    # Initial faces: one per simplex
    faces = _FaceState([frozenset(int(v) for v in s) for s in tri.simplices], positions)

    # Compute initial stabilities using face-based method (UpdateStabilities).
    # Initially, faces are 1:1 with simplices, so this is equivalent to the
    # simplex-based method. Using faces from the start ensures consistency
    # after merges.
    edge_stability: dict[tuple[int, int], float] = {}
    for edge in active_edges:
        edge_stability[edge] = faces.stability(edge)

    # Iterative removal loop (Algorithm 1)
    # Note: edge_stability acts as the priority queue Q from the thesis.
    # Edges may be in active_edges but not in edge_stability (removed from Q
    # by tie-breaking but kept in the graph).
    while edge_stability:
        # Find minimum stability among edges still in the queue
        min_stability = min(edge_stability.values())
        if min_stability >= stability_threshold:
            break  # All queued edges are stable

        # Collect all edges with minimal stability (ties)
        min_edges = [e for e, s in edge_stability.items() if abs(s - min_stability) < 1e-10]

        if len(min_edges) > 1:
            # Tie-breaking (thesis Algorithm 1, lines 10-17)
            # Pick arbitrary e0, simulate its removal, check if any other
            # tied edge would become stable (>= threshold) after removal.
            e0 = min_edges[0]
            do_not_remove = False

            # Simulate removal of e0: merge faces and recompute stabilities
            sim_faces = faces.copy()
            sim_edges = set(active_edges)
            sim_edges.discard(e0)
            sim_faces.merge(e0)

            # Recompute stabilities on edges adjacent to the merged face
            sim_stability = dict(edge_stability)
            del sim_stability[e0]
            sim_faces.update_after_merge(e0, sim_edges, sim_stability)

            for e in min_edges:
                if e == e0:
                    continue
                if e in sim_stability and sim_stability[e] >= stability_threshold:
                    do_not_remove = True
                    break

            if do_not_remove:
                # Thesis footnote 2, Algorithm 1: if removing one tied edge would
                # stabilize another, keep ALL tied edges in the graph. Remove them
                # from the priority queue (so they're never reconsidered as minimum-
                # stability candidates) but leave them in active_edges (the graph).
                for e in min_edges:
                    edge_stability.pop(e, None)
                continue

        # Remove all tied edges and merge faces
        for e in min_edges:
            active_edges.discard(e)
            edge_stability.pop(e, None)
            faces.merge(e)
            # Update stabilities for edges adjacent to the newly merged face
            faces.update_after_merge(e, active_edges, edge_stability)

    # Build output — edges kept by tie-breaking may not be in edge_stability,
    # so recompute their stability from the final face state.
    remaining_edges = sorted(active_edges)
    if remaining_edges:
        stabilities = np.array(
            [edge_stability[e] if e in edge_stability else faces.stability(e) for e in remaining_edges]
        )
        edges_arr = np.array(remaining_edges, dtype=int).reshape(-1, 2)
    else:
        stabilities = np.empty(0)
        edges_arr = np.empty((0, 2), dtype=int)

    return ShapeGraph(
        edges=edges_arr,
        faces=faces.faces(),
        stabilities=stabilities,
        points=positions,
    )


# ---------------------------------------------------------------------------
# Face centroid and bridging edge helpers (Chapter 4)
# ---------------------------------------------------------------------------


def _face_centroids(
    shape_graph: ShapeGraph,
    axis: int | None = None,
    min_spread_frac: float = 0.1,
) -> np.ndarray:
    """Compute centroids of inter-line faces of the shape graph.

    Internal faces have >= 3 vertices (external/unbounded face excluded).

    When *axis* is specified (0 for x, 1 for y), faces whose vertex spread
    along that axis is less than *min_spread_frac* of the total player spread
    are excluded.  These "within-line" faces have centroids AT a positional
    line rather than BETWEEN lines, which skews the max/min face center
    boundaries used for level decomposition (Option B fix).

    A centroid sums its face's vertices in ascending vertex-index order, so it does not depend on the order in which
    the face's vertex set iterates (spec 2026-11-08 tracking-batch W5.6).
    """
    # Compute total player spread along the filter axis
    total_spread = 0.0
    if axis is not None:
        axis_values = shape_graph.points[:, axis]
        total_spread = float(np.ptp(axis_values))  # max - min

    centroids: list[np.ndarray] = []
    for face in shape_graph.faces:
        if len(face) < 3:
            continue
        pts = shape_graph.points[sorted(face)]  # ascending vertex index: independent of the face's set order

        # Option B: filter out within-line faces (low spread along decomposition axis)
        if axis is not None and total_spread > 1e-10:
            face_spread = float(np.ptp(pts[:, axis]))
            if face_spread < min_spread_frac * total_spread:
                continue

        centroids.append(pts.mean(axis=0))

    if not centroids:
        return np.empty((0, 2))
    return np.array(centroids)


def _bridging_edge_midpoints(
    middle_sg: ShapeGraph,
    axis: int,
) -> np.ndarray:
    """Get midpoints of steep bridging edges along *axis* (thesis §4.2).

    A bridging edge is one whose removal would disconnect the graph.  "Steep"
    means its slope is within 90° ± 22.5° relative to the decomposition axis
    (vertical axis=0 -> steep means roughly vertical edges; horizontal axis=1 ->
    roughly horizontal edges).

    These midpoints act as additional face centers when the middle shape graph
    lacks internal faces (tree-like structure).
    """
    if len(middle_sg.edges) == 0:
        return np.empty((0, 2))

    pts = middle_sg.points
    midpoints: list[np.ndarray] = []

    # Build adjacency for bridge detection
    adj: dict[int, set[int]] = {}
    for e in middle_sg.edges:
        a, b = int(e[0]), int(e[1])
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    edge_set = {(int(e[0]), int(e[1])) for e in middle_sg.edges}

    for e in middle_sg.edges:
        a, b = int(e[0]), int(e[1])

        # Bridge detection: remove edge, check if still connected via BFS
        remaining = edge_set - {(a, b), (b, a)}
        adj_minus: dict[int, set[int]] = {}
        for ea, eb in remaining:
            adj_minus.setdefault(ea, set()).add(eb)
            adj_minus.setdefault(eb, set()).add(ea)

        all_nodes = set(adj.keys())
        if not all_nodes:
            continue
        visited: set[int] = set()
        stack = [next(iter(all_nodes))]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            stack.extend(adj_minus.get(node, set()) - visited)

        is_bridge = len(visited) < len(all_nodes)
        if not is_bridge:
            continue

        # Check if edge is "steep" along the decomposition axis
        dx = float(pts[b][0] - pts[a][0])
        dy = float(pts[b][1] - pts[a][1])
        if abs(dx) < 1e-10 and abs(dy) < 1e-10:
            continue

        angle_deg = float(np.degrees(np.arctan2(abs(dy), abs(dx))))
        # For vertical decomposition (axis=0): steep = edge is roughly vertical (angle ~90°)
        # For horizontal decomposition (axis=1): steep = edge is roughly horizontal (angle ~0° or ~180°)
        if axis == 0:
            is_steep = _STEEP_ANGLE_LOW <= angle_deg <= _STEEP_ANGLE_HIGH
        else:
            is_steep = angle_deg <= (90.0 - _STEEP_ANGLE_LOW) or angle_deg >= (90.0 + (180.0 - _STEEP_ANGLE_HIGH))

        if is_steep:
            midpoints.append((pts[a] + pts[b]) / 2.0)

    if not midpoints:
        return np.empty((0, 2))
    return np.array(midpoints)


def _is_tree(sg: ShapeGraph) -> bool:
    """Check if the shape graph is a tree (no internal faces with >= 3 vertices)."""
    return all(len(face) < 3 for face in sg.faces)


# ---------------------------------------------------------------------------
# Level decomposition (thesis §4.1-4.3)
# ---------------------------------------------------------------------------


def _decompose_middle(
    middle_indices: list[int],
    values: np.ndarray,
    levels: list[str],
    positions: np.ndarray | None,
    axis: int,
    label_above: str,
    label_below: str,
    label_center: str,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> None:
    """Decompose middle players into 3 sub-levels using §4.1 + §4.2 special cases.

    *values* and *positions* are in the attack frame (:func:`_to_attack_frame`), so no flip is needed: ascending
    *values* run from ``label_below`` to ``label_above``.

    Modifies *levels* in place.
    """
    if len(middle_indices) < 2 or positions is None:
        return

    if len(middle_indices) == 2:
        # Two middle players — split by coordinate
        mid_vals = [(values[idx], idx) for idx in middle_indices]
        mid_vals.sort()
        levels[mid_vals[0][1]] = label_below
        levels[mid_vals[1][1]] = label_above
        return

    # Option C: build a NEW shape graph from middle players (thesis §4.1)
    middle_positions = positions[middle_indices]
    middle_sg = compute_shape_graph(middle_positions)

    if len(middle_sg.edges) > 0:
        mid_centroids = _face_centroids(middle_sg, axis=axis)

        # §4.2 — steep bridging edges: add their midpoints as face centers
        # when the middle shape graph is tree-like (few or no internal faces)
        bridge_mids = _bridging_edge_midpoints(middle_sg, axis=axis)
        if len(bridge_mids) > 0:
            if len(mid_centroids) > 0:
                mid_centroids = np.vstack([mid_centroids, bridge_mids])
            else:
                mid_centroids = bridge_mids

        if len(mid_centroids) > 0:
            mid_fc = mid_centroids[:, axis].copy()

            mid_fc_max = float(np.max(mid_fc))
            mid_fc_min = float(np.min(mid_fc))

            if mid_fc_max - mid_fc_min > 1e-10:
                # §4.2 — diamond: if all face centers and >= 1 player are in the
                # middle third between the highest and lowest middle players, use
                # middle thirds instead of face centers.
                mid_values = values[middle_indices]
                mid_player_min = float(np.min(mid_values))
                mid_player_max = float(np.max(mid_values))
                mid_range = mid_player_max - mid_player_min
                if mid_range > 1e-10:
                    third_low = mid_player_min + mid_range / 3.0
                    third_high = mid_player_min + 2.0 * mid_range / 3.0
                    all_fc_in_middle = all(third_low <= fc <= third_high for fc in mid_fc)
                    any_player_in_middle = any(third_low <= values[idx] <= third_high for idx in middle_indices)
                    if all_fc_in_middle and any_player_in_middle:
                        # Diamond case — use thirds instead of face centers
                        mid_fc_max = third_high
                        mid_fc_min = third_low

                # §4.2 — asymmetric middle groups: if one sub-group has >= 2 more
                # players than the other, the larger group stays center, the smaller
                # gets above/below label based on position relative to face centers.
                above = [idx for idx in middle_indices if values[idx] > mid_fc_max]
                below = [idx for idx in middle_indices if values[idx] < mid_fc_min]
                center = [idx for idx in middle_indices if mid_fc_min <= values[idx] <= mid_fc_max]

                if len(above) > 0 and len(below) > 0 and abs(len(above) - len(below)) >= 2:
                    # Asymmetric: larger group -> center label, smaller -> directional
                    if len(above) > len(below):
                        for idx in above:
                            levels[idx] = label_center
                        for idx in below:
                            levels[idx] = label_below
                    else:
                        for idx in below:
                            levels[idx] = label_center
                        for idx in above:
                            levels[idx] = label_above
                    for idx in center:
                        levels[idx] = label_center
                else:
                    # Standard split
                    for idx in above:
                        levels[idx] = label_above
                    for idx in below:
                        levels[idx] = label_below
                    for idx in center:
                        levels[idx] = label_center
                return

        # §4.2 — tree shapes: when middle shape graph is a tree (no internal faces),
        # use team centroid relative to pitch thirds.
        if _is_tree(middle_sg):
            _pitch_thirds_level(
                values,
                middle_indices,
                levels,
                pitch_dim=pitch_length if axis == 0 else pitch_width,
                label_above=label_above,
                label_below=label_below,
            )
            return

    # Fallback: use thirds of the middle player range
    mid_values = values[middle_indices]
    mid_min_val = float(np.min(mid_values))
    mid_max_val = float(np.max(mid_values))
    mid_range = mid_max_val - mid_min_val
    if mid_range > 1e-10:
        third_low = mid_min_val + mid_range / 3.0
        third_high = mid_min_val + 2.0 * mid_range / 3.0
        for idx in middle_indices:
            if values[idx] < third_low:
                levels[idx] = label_below
            elif values[idx] > third_high:
                levels[idx] = label_above


def _pitch_thirds_level(
    values: np.ndarray,
    middle_indices: list[int],
    levels: list[str],
    *,
    pitch_dim: float,
    label_above: str,
    label_below: str,
) -> None:
    """§4.2 tree shapes: label a middle group by its mean against the pitch thirds (attack-frame *values*).

    A mean in the lowest third (the team's own end on the vertical axis) gives ``label_below``, a mean in the highest
    third gives ``label_above``, and otherwise the group stays centre. Modifies *levels* in place.
    """
    centroid_val = float(np.mean(values[middle_indices]))
    third = pitch_dim / 3.0
    if centroid_val < third:
        for idx in middle_indices:
            levels[idx] = label_below
    elif centroid_val > 2.0 * third:
        for idx in middle_indices:
            levels[idx] = label_above
    # else: stays center


def _assign_levels_vertical(
    values: np.ndarray,
    face_centers: np.ndarray,
    shape_graph: ShapeGraph | None = None,
    positions: np.ndarray | None = None,
) -> list[str]:
    """Assign vertical levels (B/DM/M/AM/F) using face-center decomposition.

    Implements thesis §4.1 with Options B, C, D and all §4.2 special cases. *values*, *face_centers*, and
    *positions* are in the attack frame: ascending x runs from the team's own goal (``B``) to the attacked goal.
    """
    n = len(values)
    if n == 0:
        return []

    if len(face_centers) == 0:
        return _equal_frequency_assign(values, _VERTICAL_LEVELS)

    levels = ["M"] * n

    fc_max = float(np.max(face_centers))
    fc_min = float(np.min(face_centers))

    # Initial B/M/F split
    front_mask = values > fc_max
    back_mask = values < fc_min
    middle_mask = ~front_mask & ~back_mask

    for i in range(n):
        if front_mask[i]:
            levels[i] = "F"
        elif back_mask[i]:
            levels[i] = "B"

    # Option D: adaptive fallback — if >60% in M, tighten bounds using percentiles.
    m_count = sum(1 for lv in levels if lv == "M")
    if n >= 5 and m_count > _DEGENERATE_FRACTION * n and len(face_centers) >= 4:
        fc_p25 = float(np.percentile(face_centers, 25))
        fc_p75 = float(np.percentile(face_centers, 75))
        if fc_p75 - fc_p25 > 1e-10 and fc_p75 < fc_max and fc_p25 > fc_min:
            levels = ["M"] * n
            fc_max = fc_p75
            fc_min = fc_p25
            front_mask = values > fc_max
            back_mask = values < fc_min
            middle_mask = ~front_mask & ~back_mask
            for i in range(n):
                if front_mask[i]:
                    levels[i] = "F"
                elif back_mask[i]:
                    levels[i] = "B"

    # Middle decomposition: Option C + all §4.2 special cases
    middle_indices = [i for i in range(n) if middle_mask[i]]
    _decompose_middle(
        middle_indices,
        values,
        levels,
        positions,
        axis=0,
        label_above="AM",
        label_below="DM",
        label_center="M",
    )

    return levels


def _assign_levels_horizontal(
    values: np.ndarray,
    face_centers: np.ndarray,
    shape_graph: ShapeGraph | None = None,
    positions: np.ndarray | None = None,
) -> list[str]:
    """Assign horizontal levels (L/LC/C/RC/R) using face-center decomposition.

    Same algorithm as vertical (§4.1 + §4.2) but with horizontal labels. *values*, *face_centers*, and *positions*
    are in the attack frame, where high y is the left: the ladder in ascending y is ``R, RC, C, LC, L``.
    """
    n = len(values)
    if n == 0:
        return []

    if len(face_centers) == 0:
        return _equal_frequency_assign(values, _HORIZONTAL_LEVELS)

    levels = ["C"] * n

    fc_max = float(np.max(face_centers))
    fc_min = float(np.min(face_centers))

    left_mask = values > fc_max
    right_mask = values < fc_min
    middle_mask = ~left_mask & ~right_mask

    for i in range(n):
        if left_mask[i]:
            levels[i] = "L"
        elif right_mask[i]:
            levels[i] = "R"

    # Option D: adaptive fallback for horizontal
    c_count = sum(1 for lv in levels if lv == "C")
    if n >= 5 and c_count > _DEGENERATE_FRACTION * n and len(face_centers) >= 4:
        fc_p25 = float(np.percentile(face_centers, 25))
        fc_p75 = float(np.percentile(face_centers, 75))
        if fc_p75 - fc_p25 > 1e-10 and fc_p75 < fc_max and fc_p25 > fc_min:
            levels = ["C"] * n
            fc_max = fc_p75
            fc_min = fc_p25
            left_mask = values > fc_max
            right_mask = values < fc_min
            middle_mask = ~left_mask & ~right_mask
            for i in range(n):
                if left_mask[i]:
                    levels[i] = "L"
                elif right_mask[i]:
                    levels[i] = "R"

    # Middle decomposition: Option C + all §4.2 special cases
    middle_indices = [i for i in range(n) if middle_mask[i]]
    _decompose_middle(
        middle_indices,
        values,
        levels,
        positions,
        axis=1,
        label_above="LC",
        label_below="RC",
        label_center="C",
    )

    return levels


def _equal_frequency_assign(values: np.ndarray, levels: tuple[str, ...]) -> list[str]:
    """Fallback: assign level labels using equal-frequency binning.

    Used when no face centers are available (degenerate shape graph).
    """
    n = len(values)
    if n == 0:
        return []

    n_levels = len(levels)
    sorted_indices = np.argsort(values)
    assignments = [""] * n

    for rank, idx in enumerate(sorted_indices):
        bin_idx = min(rank * n_levels // n, n_levels - 1)
        assignments[int(idx)] = levels[bin_idx]

    return assignments


# ---------------------------------------------------------------------------
# Public API — position inference
# ---------------------------------------------------------------------------


def _to_attack_frame(positions: np.ndarray, attacking_direction: float) -> np.ndarray:
    """``positions`` in the frame of a team that attacks +x: the identity for ``attacking_direction > 0``, the ADR-051
    point reflection ``(105 - x, 68 - y)`` for ``< 0`` (float64). The only place that reads the direction."""
    goal_x = 0.0 if attacking_direction > 0 else FIELD_LENGTH
    return np.column_stack(
        [
            to_goal_relative_x_array(positions[:, 0], goal_x=goal_x),
            to_goal_relative_y_array(positions[:, 1], goal_x=goal_x),
        ]
    )


def infer_positions(
    shape_graph: ShapeGraph,
    positions: np.ndarray,
    attacking_direction: float,
) -> list[PositionLabel]:
    """Infer tactical positions from player positions via face-center decomposition.

    Vertical decomposition uses the attacking axis with face centroids as split boundaries. Horizontal decomposition
    uses the lateral axis similarly. This follows thesis Chapter 4, §4.1-4.3.

    Every label is in the team's OWN frame, with SPADL handedness (ADR-004; spec 2026-11-08 tracking-batch W5.6):

    - vertical: ``B`` is the team's own goal end, ``F`` the attacked goal end;
    - lateral: ``L`` is the left of a player who faces the attacked goal. For a team that attacks +x that is high y;
      for a team that attacks -x it is low y.

    The positions are first mapped to the attack frame (the ADR-051 point reflection ``(105 - x, 68 - y)`` for a
    team that attacks -x), and every label input comes from those positions: the player values, the face centroids
    (``shape_graph`` supplies only its face vertex sets and its edge count), the middle sub-graph, and the pitch
    thirds. So a frame and its point reflection, with the opposite direction, get the same labels.

    Parameters
    ----------
    shape_graph : ShapeGraph
        Computed shape graph (its faces provide decomposition boundaries).
    positions : np.ndarray
        (n, 2) array of player positions.
    attacking_direction : float
        ``> 0`` if the team attacks toward higher x, ``< 0`` if toward lower x.

    Returns
    -------
    list[PositionLabel]
        One per player. Empty if shape graph has no edges.

    Raises
    ------
    ValueError
        If ``attacking_direction`` is not finite or is zero (the direction is never guessed).

    Examples
    --------
    Label both teams of one frame; the away team attacks -x::

        import numpy as np
        from silly_kicks.tracking import compute_shape_graph, infer_positions

        home = np.array([
            [20, 10], [20, 30], [20, 50], [20, 58],  # defenders
            [40, 15], [40, 25], [40, 43], [40, 53],  # midfielders
            [60, 25], [60, 43],                        # forwards
        ], dtype=float)
        away = np.column_stack([105.0 - home[:, 0], 68.0 - home[:, 1]])  # the same shape, mirrored
        home_labels = infer_positions(compute_shape_graph(home), home, attacking_direction=1.0)
        away_labels = infer_positions(compute_shape_graph(away), away, attacking_direction=-1.0)
        # the same roles: each team's left back is "LB" in its own frame
        assert [lbl.label for lbl in home_labels] == [lbl.label for lbl in away_labels]

    See NOTICE for full bibliographic citations.
    """
    if not math.isfinite(attacking_direction) or attacking_direction == 0.0:
        raise ValueError(
            f"infer_positions: attacking_direction must be finite and non-zero, got {attacking_direction!r}"
        )
    if len(shape_graph.edges) == 0:
        return []

    n = len(positions)
    attack = _to_attack_frame(positions, attacking_direction)
    x = attack[:, 0].copy()
    y = attack[:, 1].copy()

    # Compute face centroids for decomposition boundaries, from the attack-frame positions.
    # Option B: filter out within-line faces per axis to get inter-line centroids.
    attack_graph = ShapeGraph(
        edges=shape_graph.edges, faces=shape_graph.faces, stabilities=shape_graph.stabilities, points=attack
    )
    centroids_v = _face_centroids(attack_graph, axis=0)  # filter on x-spread for vertical
    centroids_h = _face_centroids(attack_graph, axis=1)  # filter on y-spread for horizontal

    face_centers_x = centroids_v[:, 0].copy() if len(centroids_v) > 0 else np.empty(0)
    face_centers_y = centroids_h[:, 1].copy() if len(centroids_h) > 0 else np.empty(0)

    # Assign vertical and horizontal levels
    vertical = _assign_levels_vertical(x, face_centers_x, attack_graph, attack)
    horizontal = _assign_levels_horizontal(y, face_centers_y, attack_graph, attack)

    return [
        PositionLabel(
            vertical=vertical[i],
            horizontal=horizontal[i],
            label=POSITION_LABEL_MATRIX[vertical[i]][horizontal[i]],
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Public API — many team-frames in one call (spec 2026-11-08 tracking-batch W5.2-W5.3)
# ---------------------------------------------------------------------------

#: The ``shape_graph_source`` domain of :func:`compute_shape_graphs` (the ``DAS_SOURCE_VALUES`` precedent).
#: ``"degenerate"`` covers a Delaunay ``QhullError`` and a graph with no edges.
SHAPE_GRAPH_SOURCE_VALUES: frozenset[str] = frozenset({"ok", "too_few_players", "degenerate", "direction_unresolved"})

_TEAM_FRAME_KEYS = ("game_id", "period_id", "frame_id", "team_id")
_SUMMARY_METRICS = ("n_players", "n_edges", "density", "mean_stability", "shape_graph_source")
_LABEL_COLUMNS = ("vertical", "horizontal", "position_label")
_DIRECTION_SIGN = {"ltr": 1.0, "rtl": -1.0}


@dataclass(frozen=True)
class _TeamFrame:
    """One team-frame of the selection: its rows (positions into the selection), positions, and graph metrics."""

    rows: np.ndarray
    positions: np.ndarray
    graph: ShapeGraph | None
    n_edges: int
    density: float
    mean_stability: float
    source: str  # "ok", "too_few_players", or "degenerate": the graph does not depend on the direction


def _outfield_selection(frames: pd.DataFrame) -> pd.DataFrame:
    """The rows a shape graph is built from: not the ball, not a goalkeeper, ``x`` and ``y`` not NaN, in frame-row
    order."""
    keep = (
        ~frames["is_ball"].astype(bool).to_numpy()
        & ~frames["is_goalkeeper"].astype(bool).to_numpy()
        & frames["x"].notna().to_numpy()
        & frames["y"].notna().to_numpy()
    )
    return frames[keep]


def _graph_metrics(rows: np.ndarray, positions: np.ndarray, stability_threshold: float) -> _TeamFrame:
    """The graph of one team-frame and its summary metrics (the ``add_shape_graph`` definitions)."""
    n = len(positions)
    if n < 3:
        return _TeamFrame(rows, positions, None, 0, np.nan, np.nan, "too_few_players")
    sg = compute_shape_graph(positions, stability_threshold)
    n_edges = len(sg.edges)
    max_edges = n * (n - 1) / 2
    density = float(n_edges) / max_edges if max_edges > 0 else 0.0
    mean_stability = float(np.mean(sg.stabilities)) if n_edges > 0 else np.nan
    return _TeamFrame(rows, positions, sg, n_edges, density, mean_stability, "ok" if n_edges > 0 else "degenerate")


def _team_frame_graphs(
    frames: pd.DataFrame, *, stability_threshold: float = _STABILITY_THRESHOLD
) -> tuple[pd.DataFrame, list[tuple[tuple, _TeamFrame]]]:
    """The single per-team-frame core (spec W5.3): the outfield selection and, per (game_id, period_id, frame_id,
    team_id) in first-appearance order, ``(canonical key, team-frame)``.

    One ``group_rows`` over the selection (ADR-068); the Delaunay stays per team-frame. ``x`` and ``y`` are read as
    float64 and never cast to the schema dtype. A row with a missing key value is not selected (``group_rows``).
    The graph never reads the direction.
    """
    sel = _outfield_selection(frames)
    xy = sel[["x", "y"]].to_numpy(dtype=np.float64)
    groups = group_rows(sel[list(_TEAM_FRAME_KEYS)], _TEAM_FRAME_KEYS)
    out: list[tuple[tuple, _TeamFrame]] = []
    for key in groups.keys():
        rows = groups.positions(*key)
        out.append((key, _graph_metrics(rows, xy[rows], stability_threshold)))
    return sel, out


def _team_frame_direction(directions: np.ndarray | None, rows: np.ndarray, key_values: tuple) -> float | None:
    """The ``infer_positions`` direction of one team-frame from its selected rows' ``team_attacking_direction``:
    ``"ltr"`` -> +1.0, ``"rtl"`` -> -1.0, anything else -> None (never a guess). A team-frame whose rows mix values
    (null included) is a data defect: ``ValueError`` naming it (spec W5.2, TB-SPEC-57)."""
    if directions is None:
        return None
    values = directions[rows]
    missing = pd.isna(values)
    if missing.all():
        return None
    distinct = set(values[~missing].tolist())
    if missing.any() or len(distinct) > 1:
        game_id, period_id, frame_id, team_id = key_values
        seen = sorted(repr(v) for v in distinct) + (["None"] if missing.any() else [])
        raise ValueError(
            "compute_shape_graphs: team_attacking_direction mixes values within one team-frame "
            f"(game_id={game_id}, period_id={period_id}, frame_id={frame_id}, team_id={team_id}): {seen}"
        )
    return _DIRECTION_SIGN.get(next(iter(distinct)))


@dataclass(frozen=True)
class ShapeGraphBatch:
    """The result of :func:`compute_shape_graphs`: two DataFrames and a graph accessor.

    - ``frames_summary``: one row per (game_id, period_id, frame_id, team_id) with ``n_players``, ``n_edges``,
      ``density``, ``mean_stability``, and ``shape_graph_source`` (one of :data:`SHAPE_GRAPH_SOURCE_VALUES`).
    - ``players``: one row per selected outfield player of each team-frame, with ``player_id``,
      ``shape_graph_source`` (the team-frame's value), and the ``vertical`` / ``horizontal`` / ``position_label``
      labels (``<NA>`` unless the source is ``"ok"`` and positions were requested).

    Examples
    --------
    Look up the graph of one team-frame::

        from silly_kicks.tracking import compute_shape_graphs
        batch = compute_shape_graphs(frames)
        game_id, period_id, frame_id, team_id = batch.frames_summary.iloc[0][
            ["game_id", "period_id", "frame_id", "team_id"]
        ]
        sg = batch.graph(game_id, period_id, frame_id, team_id)
    """

    frames_summary: pd.DataFrame
    players: pd.DataFrame
    _positions: Mapping[tuple, np.ndarray] = field(repr=False, compare=False)
    _graphs: Mapping[tuple, ShapeGraph] | None = field(repr=False, compare=False)
    _stability_threshold: float = field(repr=False, compare=False)

    def graph(self, game_id, period_id, frame_id, team_id) -> ShapeGraph:
        """The shape graph of one team-frame (ids matched dtype-safely, ADR-019). Without ``retain_graphs`` it is
        recomputed from the retained positions, bit-identical to the batch's own graph. ``KeyError`` for a key
        that names no team-frame.

        Examples
        --------
        Recompute one team-frame's graph::

            sg = batch.graph(7, 1, 1200, 100)
            print(sg.edges.shape)
        """
        key = tuple(canonical_id(k) for k in (game_id, period_id, frame_id, team_id))
        if self._graphs is not None:
            return self._graphs[key]
        return compute_shape_graph(self._positions[key], self._stability_threshold)


def compute_shape_graphs(
    frames: pd.DataFrame,
    *,
    stability_threshold: float = _STABILITY_THRESHOLD,
    with_positions: bool = True,
    retain_graphs: bool = False,
) -> ShapeGraphBatch:
    """Shape graphs and position labels for every team-frame of ``frames`` in one call (TF-39; spec 2026-11-08
    tracking-batch W5.2).

    **Selection** is the one ``add_shape_graph`` uses: per (game_id, period_id, frame_id, team_id), the outfield
    rows (not the ball, not a goalkeeper, ``x`` / ``y`` not NaN) in frame-row order, with the teams from the frames'
    own ``team_id`` values. ``x`` and ``y`` are read as float64, never cast. A row with a missing key value is in no
    team-frame; the call counts such outfield rows in one ``UserWarning`` (spec AG4).

    **Direction** for the labels comes from ``team_attacking_direction``: ``"ltr"`` -> +1.0, ``"rtl"`` -> -1.0.
    Any other value (null included) gives ``<NA>`` labels and ``shape_graph_source="direction_unresolved"``; the
    graph metrics are still emitted. All selected rows of one team-frame must carry the same value (the goalkeeper
    and ball rows are not read); a mix raises ``ValueError`` naming the team-frame. The direction is read only when
    ``with_positions=True``. The labels are team-relative (``infer_positions``).

    **Sources**: ``"ok"``; ``"too_few_players"`` (fewer than 3 selected players); ``"degenerate"`` (a Delaunay
    ``QhullError`` or a graph with no edges); ``"direction_unresolved"``. A team-frame's labels are all set or all
    ``<NA>``.

    **Row order**: team-frames in the order of their first appearance in ``frames``, players in frame-row order. No
    id is sorted, so a mixed-dtype ``game_id`` cannot raise.

    **Memory**: with ``retain_graphs=False`` (the default) only the positions are kept, and
    :meth:`ShapeGraphBatch.graph` recomputes a graph on request. ``retain_graphs=True`` keeps every graph: about
    3.5 kB more per team-frame (``tracemalloc``, measured on the whole gradientsports match 10517), so the ~508k
    team-frames of a large match take about 1.8 GB more.

    **Chunked calls**: team-frames are independent, so for chunks that are contiguous row slices of ``frames`` split
    on whole (game_id, period_id, frame_id) frames, concatenating the per-chunk DataFrames equals one call on all
    the frames, row order included. That is the pattern for a caller with a deadline (the function has no stop
    callable).

    Parameters
    ----------
    frames : pd.DataFrame
        Long-form tracking frames (``TRACKING_FRAMES_COLUMNS``), any number of games.
    stability_threshold : float
        Minimum angular stability in degrees (default 45.0).
    with_positions : bool
        Compute the position labels (default True). With False the labels are ``<NA>`` and the direction is not read.
    retain_graphs : bool
        Keep every ``ShapeGraph`` (default False).

    Returns
    -------
    ShapeGraphBatch

    Examples
    --------
    Keep the labelled players only, with no id join (ADR-019)::

        from silly_kicks.tracking import compute_shape_graphs
        batch = compute_shape_graphs(frames)
        players = batch.players[batch.players["shape_graph_source"] == "ok"]

    Process a long match in chunks of whole frames, for a caller with a deadline::

        import pandas as pd
        frame_key = frames["period_id"].astype(str) + "/" + frames["frame_id"].astype(str)
        starts = frame_key.ne(frame_key.shift()).to_numpy().nonzero()[0]
        bounds = [*starts[::5000], len(frames)]
        parts = [compute_shape_graphs(frames.iloc[lo:hi]) for lo, hi in zip(bounds[:-1], bounds[1:])]
        players = pd.concat([p.players for p in parts], ignore_index=True)

    See NOTICE for full bibliographic citations.
    """
    sel, team_frames = _team_frame_graphs(frames, stability_threshold=stability_threshold)
    keys = sel[list(_TEAM_FRAME_KEYS)]
    n_missing_key = int(keys.isna().any(axis=1).sum())
    if n_missing_key:
        warnings.warn(
            f"compute_shape_graphs: {n_missing_key} outfield row(s) have a missing game_id, period_id, frame_id or "
            "team_id, so they are in no team-frame.",
            stacklevel=2,
        )
    directions = None
    if with_positions and "team_attacking_direction" in sel.columns:
        directions = sel["team_attacking_direction"].to_numpy(dtype=object)

    first_rows: list[int] = []
    player_rows: list[np.ndarray] = []
    metrics: dict[str, list] = {name: [] for name in _SUMMARY_METRICS}
    labels: dict[str, list] = {name: [] for name in _LABEL_COLUMNS}
    sources: list[str] = []
    positions: dict[tuple, np.ndarray] = {}
    graphs: dict[tuple, ShapeGraph] | None = {} if retain_graphs else None
    for key, tf in team_frames:
        n = len(tf.rows)
        first_rows.append(int(tf.rows[0]))
        player_rows.append(tf.rows)
        source = tf.source
        team_labels: list[PositionLabel] | None = None
        if with_positions:
            key_values = tuple(keys.iloc[int(tf.rows[0])])
            direction = _team_frame_direction(directions, tf.rows, key_values)
            if source == "ok" and direction is None:
                source = "direction_unresolved"
            if source == "ok" and tf.graph is not None:
                team_labels = infer_positions(tf.graph, tf.positions, direction)  # type: ignore[arg-type]
                if len(team_labels) != n:  # all-or-nothing (spec W5.2)
                    raise RuntimeError(f"compute_shape_graphs: {len(team_labels)} labels for {n} players at {key}")
        for name, value in zip(_SUMMARY_METRICS, (n, tf.n_edges, tf.density, tf.mean_stability, source), strict=True):
            metrics[name].append(value)
        sources.extend([source] * n)
        labels["vertical"].extend([lb.vertical for lb in team_labels] if team_labels else [pd.NA] * n)
        labels["horizontal"].extend([lb.horizontal for lb in team_labels] if team_labels else [pd.NA] * n)
        labels["position_label"].extend([lb.label for lb in team_labels] if team_labels else [pd.NA] * n)
        positions[key] = tf.positions
        if graphs is not None:
            graphs[key] = tf.graph if tf.graph is not None else compute_shape_graph(tf.positions, stability_threshold)

    summary = keys.take(np.asarray(first_rows, dtype=np.intp)).reset_index(drop=True)
    summary["n_players"] = np.asarray(metrics["n_players"], dtype=np.int64)
    summary["n_edges"] = np.asarray(metrics["n_edges"], dtype=np.int64)
    summary["density"] = np.asarray(metrics["density"], dtype=np.float64)
    summary["mean_stability"] = np.asarray(metrics["mean_stability"], dtype=np.float64)
    summary["shape_graph_source"] = pd.Series(metrics["shape_graph_source"], index=summary.index, dtype=object)

    all_rows = np.concatenate(player_rows) if player_rows else np.empty(0, dtype=np.intp)
    players = sel[[*_TEAM_FRAME_KEYS, "player_id"]].take(all_rows).reset_index(drop=True)
    players["shape_graph_source"] = pd.Series(sources, index=players.index, dtype=object)
    for name in _LABEL_COLUMNS:
        players[name] = pd.Series(labels[name], index=players.index, dtype=object)
    return ShapeGraphBatch(
        frames_summary=summary,
        players=players,
        _positions=positions,
        _graphs=graphs,
        _stability_threshold=stability_threshold,
    )
