"""Shape-and-structure MirrorEntry registrations (ADR-028 section 6).

Four aggregators: ``add_team_shape``, ``add_shape_graph``, ``add_structural_pass``, ``add_packing``.

Every class, tolerance and non-vacuity anchor below was MEASURED on ``canonical_scene()``, not
recalled -- the emitted column sets come from diffing ``out.columns`` against the input, and each
recorded tolerance basis quotes the observed base-vs-mirror delta.

Two of the four are D3 targets: ``_structural_pass.py:146`` and ``_packing.py:145`` both decide
which way the acting team attacks with ``same_id(attacking_team_id, home_team_id)`` -- identity, not
the frame label -- so Gate B moves them and is marked ``xfail`` until the re-key lands. The other two
accept ``home_team_id`` and never read it (``add_team_shape`` reprojects via ``_reproject_team_shape``,
which is label-keyed; ``add_shape_graph`` emits rigid-motion-invariant graph metrics), so their green
Gate B is the D3 evidence that the parameter is already dead there.

Discriminating power, MEASURED -- a green gate is worth only what a plant can move:

* ``add_team_shape`` -- real. The underlying ``compute_team_shape`` moves 20.2 m in ``centroid_x``
  and 44.0 m in ``defensive_line_height`` between the two legs; the emitted column is identical, so
  the reprojection is reconciling a genuine difference rather than restating one.
* ``add_structural_pass`` / ``add_packing`` -- real. Re-running the mirror leg WITHOUT the
  ``home_team_id`` swap (exactly what a mis-decided direction produces) moves ``structural_lbs`` by
  2, ``structural_sgm`` by 4.31, ``structural_sdi`` by 0.74, ``packing_made`` by 2 and
  ``packing_net`` by 1. This matters because ``packing_made``/``_net``/``_goal_threat`` are all 0 on
  the canonical scene: the base values are degenerate, the comparison is not.
* ``add_shape_graph`` -- NONE, and that is a property of the metric, not a gap in the fixture.
  ``compute_shape_graph`` on the point-reflected positions returns the identical edge count (16) and
  the identical mean stability to full precision, because Delaunay connectivity and unsigned angles
  are preserved by any isometry. No ADR-028 defect is expressible in these columns -- the entry is
  registered for completeness and to hold the classification if a future column does emit an
  absolute coordinate (``infer_positions``' pitch-absolute lateral label is not surfaced by any
  ``add_*`` today; see ``_shape_graph.py:877-880``).
"""

from __future__ import annotations

_PROVENANCE = ("frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score")
_PROVENANCE_REASON = "linkage provenance, not geometry"

_TEAM_SHAPE_METRICS = (
    "n_outfield_players",
    "centroid_x",
    "centroid_y",
    "convex_hull_area",
    "team_length",
    "team_width",
    "stretch_index",
    "defensive_line_height",
    "inter_line_gap_1",
    "inter_line_gap_2",
)


def register() -> None:
    from silly_kicks.tracking.features import (
        add_packing,
        add_shape_graph,
        add_structural_pass,
        add_team_shape,
    )
    from tests.tracking._mirror_registry import _entry

    # ------------------------------------------------------------------
    # add_team_shape -- 20 metric columns (10 metrics x attacking/defending) + provenance.
    # ------------------------------------------------------------------
    team_shape_columns = {
        f"team_shape_{metric}_{side}": "invariant"
        for metric in _TEAM_SHAPE_METRICS
        for side in ("attacking", "defending")
    }
    team_shape_columns.update({c: "exempt" for c in _PROVENANCE})
    _entry(
        "add_team_shape",
        lambda a, f, h: add_team_shape(a, f, home_team_id=h),
        team_shape_columns,
        tol=1e-9,
        basis=(
            "pure geometry (centroids, hull area, extents, Ward line heights) reprojected to "
            "action-LTR by the label-keyed _reproject_team_shape, so it is exact under a point "
            "reflection up to float accumulation; measured worst case 4.5e-13 on the canonical "
            "scene (convex_hull_area, shoelace summation), leaving ~3 orders of headroom at 1e-9"
        ),
        role="direction_only",
        non_vacuity=(
            "team_shape_centroid_x_attacking",
            "team_shape_centroid_y_attacking",
            "team_shape_convex_hull_area_attacking",
        ),
        exempt={c: _PROVENANCE_REASON for c in _PROVENANCE},
    )

    # ------------------------------------------------------------------
    # add_shape_graph -- 6 metric columns (3 metrics x attacking/defending) + provenance.
    # ------------------------------------------------------------------
    shape_graph_columns = {
        f"shape_graph_{metric}_{side}": "invariant"
        for metric in ("density", "n_edges", "mean_stability")
        for side in ("attacking", "defending")
    }
    shape_graph_columns.update({c: "exempt" for c in _PROVENANCE})
    _entry(
        "add_shape_graph",
        lambda a, f, h: add_shape_graph(a, f, home_team_id=h),
        shape_graph_columns,
        tol=1e-9,
        basis=(
            "Delaunay edge counts, their density ratio and the angular stability mean are all "
            "rigid-motion invariants, and a point reflection is a rigid motion composed with a "
            "reflection that preserves unsigned angles; measured EXACTLY 0.0 on every column of "
            "the canonical scene, so 1e-9 is float headroom rather than a fitted bound"
        ),
        role="direction_only",
        non_vacuity=(
            "shape_graph_density_attacking",
            "shape_graph_n_edges_attacking",
            "shape_graph_mean_stability_attacking",
        ),
        exempt={c: _PROVENANCE_REASON for c in _PROVENANCE},
    )

    # ------------------------------------------------------------------
    # add_structural_pass -- 3 metric columns (pass/cross rows only) + provenance.
    # ------------------------------------------------------------------
    _entry(
        "add_structural_pass",
        lambda a, f, h: add_structural_pass(a, f, home_team_id=h),
        {
            "structural_lbs": "invariant",
            "structural_sgm": "invariant",
            "structural_sdi": "invariant",
            **{c: "exempt" for c in _PROVENANCE},
        },
        tol=1e-9,
        basis=(
            "defender bypass counting, the inverse-Gaussian density difference and the "
            "centroid-distance difference are computed after mirroring defenders into the acting "
            "team's attack-positive frame, so all three are exact under a point reflection; "
            "measured EXACTLY 0.0 on the canonical scene's two in-domain pass rows"
        ),
        role="direction_only",
        non_vacuity=("structural_lbs", "structural_sgm", "structural_sdi"),
        exempt={c: _PROVENANCE_REASON for c in _PROVENANCE},
        defect_b="D3 re-key pending: identity-keyed direction (spec 4.3)",
    )

    # ------------------------------------------------------------------
    # add_packing -- 3 numeric + receiver id + secured flag + provenance.
    #
    # ``packing_secured`` is classed ``invariant`` because that is what it is -- a possession-
    # sequence boolean that cannot legitimately depend on which way the pitch is drawn -- but it is
    # all-<NA> on the canonical scene and Gate A therefore skips it. The cause is structural, not a
    # bug: ``features.py:1419`` masks ``secured`` to NA wherever ``packing_made < 1``, and the short
    # passes in this fixture bypass nobody. Recorded so the skip is a known hole rather than a
    # silently green column.
    # ------------------------------------------------------------------
    _entry(
        "add_packing",
        lambda a, f, h: add_packing(a, f, home_team_id=h),
        {
            "packing_made": "invariant",
            "packing_net": "invariant",
            "packing_goal_threat": "invariant",
            "packing_secured": "invariant",
            "packing_receiver_player_id": "exempt",
            **{c: "exempt" for c in _PROVENANCE},
        },
        tol=1e-9,
        basis=(
            "bypassed-defender counts and the back-line goal-threat count are integer geometry "
            "taken after the same attack-positive mirror as structural_pass, so they are exact "
            "under a point reflection; measured EXACTLY 0.0 on the canonical scene's two "
            "in-domain pass rows"
        ),
        role="direction_only",
        non_vacuity=("packing_made", "packing_net", "packing_goal_threat"),
        exempt={
            "packing_receiver_player_id": (
                "player identifier resolved from the next same-team touch -- an event-sequence "
                "label, not geometry, so it has no mirror image to compare against"
            ),
            **{c: _PROVENANCE_REASON for c in _PROVENANCE},
        },
        defect_b="D3 re-key pending: identity-keyed direction (spec 4.3)",
    )
