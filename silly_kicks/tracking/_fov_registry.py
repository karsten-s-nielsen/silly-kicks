"""Single-sourced FOV-observability: one convex region per metric, one engine (ADR-077).

Every region-based tracking metric answers "how much of the region I integrated over did the
provider actually observe?" against the action's ``visible_area`` polygon. Historically each
aggregator hand-rolled that question (ADR-062 did it inline in ``add_action_context``). This module
is the ONE engine: an :class:`ObservabilityEntry` binds a raw metric column to a convex-region
builder, :data:`OBSERVABILITY_REGISTRY` lists the entries per aggregator, and
:func:`append_observability_companions` emits the ``<column>_observed_fraction`` /
``_observed_source`` pair for each. The completeness gate (Task 8) reads every companioned column
from :func:`companioned_columns`, so the registry is the single source of truth.

Neutral by construction: this module imports ONLY ``_visibility``, ``_kernels``,
``silly_kicks.id_compat`` and the two neutral root primitives ``silly_kicks._polygon`` (the
convexity check the Andrienko oval needs) and ``silly_kicks.spadl.config`` (pitch dimensions for
the packing band) -- none of which reach ``pitch_control`` or ``_das`` (``_kernels``' own
top-level import closure -- ``_action_orientation`` / ``_gk_resolve`` / ``_gk_geometry`` /
``utils`` / ``feature_framework`` -- reaches neither either). It must NOT import ``pitch_control``
/ ``_das`` / ``features``, so those layers may depend on it without a cycle.

See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

import silly_kicks.spadl.config as spadlconfig
from silly_kicks._polygon import is_convex
from silly_kicks.id_compat import canonical_id

from . import _kernels
from ._visibility import (
    REGION_OBSERVATION_DEGENERATE_REGION,
    VISIBLE_AREA_OBSERVED,
    VISIBLE_AREA_UNLINKED,
    _polygons_by_action,
    classify_region_observation,
)

#: Identity sentinel: a builder returns this when NO measurable region exists for this action
#: (a NaN nearest-distance -> no radius; later, an unsupported/velocity pressure method -> no
#: convex ROI; an anchor_actor credit -> event-resolved; an unresolved defended end). The engine
#: maps it to ``degenerate_region``. Tested with ``is``, NEVER ``==`` -- ``np.zeros((16, 2)) ==
#: "degenerate"`` is an ARRAY and ``if <array>:`` RAISES (P2, executed).
_NO_REGION = object()


@dataclasses.dataclass
class RegionCtx:
    """Per-action arrays the region builders index with ``i``.

    ``sx``/``sy`` are the actor anchor (triangle / disk / oval builders); ``ex``/``ey`` the action's
    end (``receiver_disk`` and the packing band); ``nearest_dist`` the measured nearest-defender
    distance (``nearest_defender_disk``, populated by ``add_action_context`` only, else ``None``).
    The fixed pitch-zone builders consult NEITHER argument (the zone is keyed on the column's ROLE,
    not player geometry). Per-call params (radius, pressure method) arrive via ``extras`` so the
    registry entries stay STATIC.
    """

    sx: npt.NDArray[np.float64]
    sy: npt.NDArray[np.float64]
    ex: npt.NDArray[np.float64]
    ey: npt.NDArray[np.float64]
    nearest_dist: npt.NDArray[np.float64] | None
    extras: dict


@dataclasses.dataclass(frozen=True)
class ObservabilityEntry:
    """Binds a raw metric column to the convex region it integrates over.

    ``region(i, ctx)`` returns a convex ``(M, 2)`` polygon or :data:`_NO_REGION` (never a literal
    ``None``, P7). ``covers`` is the RAW metric columns this companion annotates; the default
    ``()`` means the companion covers exactly ``(column,)``. A non-default ``covers`` lets one
    companion stand for several raw columns (e.g. a team-shape x/y-per-role pair, or a rollup), so
    the completeness gate maps raw columns to companions correctly (R1).
    """

    column: str
    region: Callable  # (i, ctx) -> (M, 2) convex ndarray | _NO_REGION
    covers: tuple[str, ...] = ()


def triangle_to_goal(i, ctx: RegionCtx) -> np.ndarray:
    """The shot-to-goal-mouth triangle: actor start, both goalposts (goal fixed at x=105)."""
    return np.array(
        [
            [ctx.sx[i], ctx.sy[i]],
            [_kernels._GOAL_X, _kernels._GOAL_LEFT_POST_Y],
            [_kernels._GOAL_X, _kernels._GOAL_RIGHT_POST_Y],
        ]
    )


def receiver_disk(i, ctx: RegionCtx) -> np.ndarray:
    """Inscribed disk at the action's END of the receiver-zone radius (``ctx.extras``)."""
    return _kernels._inscribed_disk(float(ctx.ex[i]), float(ctx.ey[i]), float(ctx.extras["receiver_radius"]))


def nearest_defender_disk(i, ctx: RegionCtx):
    """Inscribed disk at the actor of radius = the measured nearest-defender distance.

    A NaN/non-finite distance (no opponent, or unlinked) has no radius -> :data:`_NO_REGION`
    (``degenerate_region``), never a fabricated fraction. Registered only for aggregators that
    populate ``nearest_dist`` (``add_action_context``); an absent ``nearest_dist`` array likewise
    degenerates rather than fabricating.
    """
    if ctx.nearest_dist is None:
        return _NO_REGION
    d = float(ctx.nearest_dist[i])
    if not np.isfinite(d):
        return _NO_REGION
    return _kernels._inscribed_disk(float(ctx.sx[i]), float(ctx.sy[i]), d)


#: Boundary samples for the Andrienko oval k-gon. 24 is dense enough that the default-parameter
#: oval (``d_back=3``, ``d_front=9``) is strictly convex at every vertex (verified numerically:
#: ``min|cross|`` stays well above 0), so :func:`_convex_hull` is a no-op on the common path.
_OVAL_K = 24


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Andrew's monotone-chain convex hull (counter-clockwise), pure-numpy.

    Used only as a fallback when a caller supplies extreme ``AndrienkoParams`` (a large
    ``d_front / d_back`` ratio) whose sampled oval is not perfectly convex -- Sutherland-Hodgman
    (via :func:`classify_region_observation`) clips against a CONVEX region only, so a non-convex
    ROI would RAISE. The hull is a conservative over-approximation of the dimpled tail; on the
    default (convex) oval it returns the same ring. Collinear vertices are dropped.
    """
    pts = np.asarray(points, dtype=float)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    keep = np.ones(len(pts), dtype=bool)
    keep[1:] = np.any(np.diff(pts, axis=0) != 0.0, axis=1)  # drop exact duplicates
    pts = pts[keep]
    if len(pts) < 3:
        return pts

    def _cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in pts[::-1]:
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def andrienko_oval_region(i, ctx: RegionCtx):
    """The Andrienko et al. 2017 directional pressure oval, as a convex k-gon (ADR-077, Trap 1).

    This is the FAITHFUL region the ``andrienko_oval`` pressure metric integrates over -- NOT an
    axis-aligned ellipse. It is centred on the actor ``(ctx.sx[i], ctx.sy[i])`` and oriented toward
    the goal (``ctx.extras['goal_x']/'goal_y']``). Reproducing ``_kernels._pressure_andrienko``'s
    in-zone boundary exactly, the polar radius at an angular offset ``phi`` from the actor->goal
    direction is::

        z    = (1 + cos(phi)) / 2
        L(phi) = d_back + (d_front - d_back) * (z**3 + 0.3 * z) / 1.3

    so the oval extends ``d_front`` toward the goal (``phi=0`` -> ``z=1`` -> ``L=d_front``) and
    ``d_back`` behind it (``phi=pi`` -> ``z=0`` -> ``L=d_back``). ``q`` (the in-zone WEIGHT
    exponent) does not enter the boundary and is deliberately not consulted. A non-finite actor
    position has no oval -> :data:`_NO_REGION`.
    """
    cx = float(ctx.sx[i])
    cy = float(ctx.sy[i])
    if not (np.isfinite(cx) and np.isfinite(cy)):
        return _NO_REGION
    ex = ctx.extras
    dx = float(ex["goal_x"]) - cx
    dy = float(ex["goal_y"]) - cy
    # Actor exactly at the goal: direction is ill-defined -> point toward +x (higher goal x),
    # matching the kernel's ``+1e-12`` epsilon behaviour. The oval is near-symmetric there anyway.
    phi_goal = 0.0 if (dx * dx + dy * dy) < 1e-18 else float(np.arctan2(dy, dx))
    d_front = float(ex["oval_d_front"])
    d_back = float(ex["oval_d_back"])
    off = np.linspace(0.0, 2.0 * np.pi, _OVAL_K, endpoint=False)
    z = (1.0 + np.cos(off)) / 2.0
    length = d_back + (d_front - d_back) * (z**3 + 0.3 * z) / 1.3
    ang = phi_goal + off
    poly = np.column_stack([cx + length * np.cos(ang), cy + length * np.sin(ang)])
    if not is_convex(poly):  # only reachable for extreme custom params; convex on the default oval
        poly = _convex_hull(poly)
    return poly


def link_zones_support_region(i, ctx: RegionCtx):
    """The Link et al. 2016 pressure metric's effective support, as a convex disk (ADR-077, N3).

    Link pressure is a piecewise-ANGULAR aggregation -- a head-on / lateral / hind zone, each with
    its OWN radius -- so it is NOT a single convex zone. Its effective support is nonetheless
    bounded: a defender beyond the LARGEST zone radius contributes 0 in every zone
    (``_kernels._pressure_link`` computes ``in_zone = d < r_zo`` with ``r_zo`` one of ``{r_hoz,
    r_lz, r_hz}``). So the conservative CONVEX outer bound is the disk of radius
    ``max(r_hoz, r_lz, r_hz)`` centred on the actor -- radially symmetric, hence orientation-free.
    ``ctx.extras['link_effective_radius']`` carries that radius (the caller reads it off
    ``LinkParams``). A non-finite actor position has no support region -> :data:`_NO_REGION`.
    """
    cx = float(ctx.sx[i])
    cy = float(ctx.sy[i])
    if not (np.isfinite(cx) and np.isfinite(cy)):
        return _NO_REGION
    return _kernels._inscribed_disk(cx, cy, float(ctx.extras["link_effective_radius"]))


def xt_gk_pressure_region(i, ctx: RegionCtx):
    """Method-dispatched pressure ROI for the xT-GK pressure-bearing columns (ADR-077, Task 5).

    ``compute_xt_gk`` derives both ``rho`` (``xt_gk_pressure``) and the PEV forward gain
    (``xt_gk_pev``) through ``pressure_on_actor(method=params.pressure_method)`` centred on the
    RESOLVED GK origin (``xt_gk_origin_x/_y``), NOT the raw ``start_x`` -- so the observability
    region is the SAME region that pressure method integrates over. The method is dispatched PER
    ACTION from ``ctx.extras['pressure_method']`` so the registry entry stays STATIC (P7):

    * ``andrienko_oval`` -> the directional oval (:func:`andrienko_oval_region`).
    * ``link_zones``     -> the effective-support disk (:func:`link_zones_support_region`).
    * ``bekkers_pi`` / anything unsupported -> :data:`_NO_REGION` (``degenerate_region``): Bekkers
      Pressing Intensity is a velocity-derived TTI model with no fixed spatial ROI, and on the
      velocity-less freeze frames this seam serves it is honest-NaN anyway -- there is no region
      to observe.
    """
    method = ctx.extras["pressure_method"]
    if method == "andrienko_oval":
        return andrienko_oval_region(i, ctx)
    if method == "link_zones":
        return link_zones_support_region(i, ctx)
    return _NO_REGION


#: Full pitch height for the packing band (``add_visible_area_coverage`` uses the same source).
_PITCH_WIDTH = float(spadlconfig.field_width)

#: The three ``add_packing`` columns that COUNT defenders inside a region (so they get an FOV
#: companion, P9). All three are x-interval tests over the full-height passer->receiver band:
#: ``packing_made`` counts ``(p_x, r_x]``, ``packing_net`` the direction-weighted ``(min, max]``
#: interval, ``packing_goal_threat`` the back-line count in ``(p_x, r_x]``. ``packing_receiver_
#: player_id`` (an id passthrough) and ``packing_secured`` (a boolean) are NOT counts; ``line_x``
#: is a POSITION (``bypassed.max()``) and is kernel-internal, not even emitted -- all three are
#: excluded. Derived by reading ``_packing.compute_packing_metrics`` and ``features.add_packing``.
#:
#: KNOWN LIMIT (manual discipline): there is no structural "is-region-count" signal on an
#: ``add_packing`` column, so a NEWLY-ADDED region-count column cannot be auto-detected -- a
#: contributor adding one MUST register it here AND in :data:`_AGGREGATE_FOV_SENSITIVE`. The drift
#: guard in ``tests/tracking/test_fov_companions.py`` only catches the OTHER direction (a rename that
#: leaves a name here stale) by pinning each name to a real ``add_packing`` output column.
_PACKING_REGION_COUNT_COLUMNS: tuple[str, ...] = ("packing_made", "packing_net", "packing_goal_threat")


def packing_zone_region(i, ctx: RegionCtx):
    """The packing passer->receiver x-band, as a convex rectangle (ADR-077, Trap 2).

    ``compute_packing_metrics`` is an x-interval test ONLY: all three region-counts live inside the
    full-height vertical band ``{ min(p_x, r_x) <= x <= max(p_x, r_x), 0 <= y <= field_width }`` in
    the same action-LTR frame the metric and the ``visible_area`` polygon share (goal at x=105; y
    is used ONLY for the action's direction angle, never as a band bound). ``p_x`` = ``ctx.sx[i]``
    (passer_x = start_x), ``r_x`` = ``ctx.ex[i]`` (receiver_x = end_x). A non-finite endpoint or a
    zero-width band (start_x == end_x) has no measurable region -> :data:`_NO_REGION`.
    """
    px = float(ctx.sx[i])
    rx = float(ctx.ex[i])
    if not (np.isfinite(px) and np.isfinite(rx)):
        return _NO_REGION
    lo, hi = (px, rx) if px <= rx else (rx, px)
    if hi <= lo:
        return _NO_REGION
    return np.array([[lo, 0.0], [hi, 0.0], [hi, _PITCH_WIDTH], [lo, _PITCH_WIDTH]])


# ---------------------------------------------------------------------------
# Fixed action-LTR pitch-zone regions for the aggregate-position metrics
# (defensive line, team-shape centroids, off-ball threat) -- ADR-077.
# ---------------------------------------------------------------------------
# These metrics are emitted in the SPADL action-LTR frame: the acting team attacks x=105 (ADR-028
# re-projects ``defensive_line_x`` / ``team_shape_centroid_*`` / ``off_ball_xt_team`` into it),
# which is the SAME frame the ``visible_area`` polygon lives in -- the only supplier, SB360, is
# action-LTR (``tests/sb360/_fixture.py``). So the DEFENDED end is FIXED: the acting team attacks
# the HIGH end and its opponent defends it, uniformly for every action. The observability zone is
# therefore a CONSTANT pitch band keyed only on the column's ROLE -- there is no per-action
# ``goal_map`` / team-id lookup. A ``goal_map``-keyed zone would return FRAME-coordinate ends and
# land on the OPPOSITE end from the action-LTR polygon for every away-possession action (the S1
# silent failure this cycle exists to prevent). The builders keep the ``(i, ctx)`` engine
# signature but consult NEITHER argument -- the zone is fixed geometry, never player-derived.

#: Defended third: ``x in [2/3 * field_length, field_length]``. The DEFENDING team defends x=105.
_DEFENDED_THIRD_LO = 2.0 * float(spadlconfig.field_length) / 3.0  # 70.0
#: Halves split at the pitch midline.
_MIDLINE = float(spadlconfig.field_length) / 2.0  # 52.5
#: Attacked-goal end (full pitch length).
_PITCH_LENGTH = float(spadlconfig.field_length)  # 105.0


def _pitch_band(lo: float, hi: float) -> np.ndarray:
    """A convex CCW rectangle spanning ``x in [lo, hi]``, full pitch height (``0..field_width``)."""
    return np.array([[lo, 0.0], [hi, 0.0], [hi, _PITCH_WIDTH], [lo, _PITCH_WIDTH]])


def defended_third_region(i, ctx: RegionCtx) -> np.ndarray:
    """The DEFENDING team's defended third ``x in [70, 105]``, action-LTR (ADR-077).

    Fixed zone (see the module note above): in action-LTR the acting team attacks x=105, so its
    opponent -- whose back line ``defensive_line_x`` measures -- defends the HIGH end and its
    third is ``[2/3 * field_length, field_length]``. Consults no ``goal_map`` / team-id: the
    ``visible_area`` polygon shares this action-LTR frame, so a ``goal_map`` end would mis-orient
    away-possession actions.
    """
    return _pitch_band(_DEFENDED_THIRD_LO, _PITCH_LENGTH)


def attacking_own_half_region(i, ctx: RegionCtx) -> np.ndarray:
    """The attacking/possession team's OWN half ``x in [0, 52.5]``, action-LTR (ADR-077).

    Fixed zone: the acting team attacks x=105, so its own half is the LOW end. Annotates the
    ``team_shape_centroid_*_attacking`` columns (the acting team's shape).
    """
    return _pitch_band(0.0, _MIDLINE)


def defending_own_half_region(i, ctx: RegionCtx) -> np.ndarray:
    """The defending team's OWN half ``x in [52.5, 105]``, action-LTR (ADR-077).

    Fixed zone: the defending team defends x=105, so its own half is the HIGH end. Annotates the
    ``team_shape_centroid_*_defending`` columns (the opponent's shape).
    """
    # Intentionally distinct from ``attacking_half_region`` despite the identical ``[52.5, 105]``
    # band: this is the DEFENDING team's OWN half (a different tactical role that coincides
    # geometrically on a standard pitch); kept separate so each registry entry reads its own role.
    return _pitch_band(_MIDLINE, _PITCH_LENGTH)


def attacking_half_region(i, ctx: RegionCtx) -> np.ndarray:
    """The attacking/possession team's attacking half ``x in [52.5, 105]``, action-LTR (ADR-077).

    Fixed zone: the acting team attacks x=105, so its attacking half is the HIGH end -- where its
    off-ball threat (``off_ball_xt_team``) concentrates.
    """
    # Intentionally distinct from ``defending_own_half_region`` despite the identical band: this is
    # the ACTING team's ATTACKING half (a different tactical role that coincides geometrically);
    # kept separate so each registry entry reads its own role.
    return _pitch_band(_MIDLINE, _PITCH_LENGTH)


# ---------------------------------------------------------------------------
# Task 6 (ADR-077): the CUSTOM defensive-credit rollup companion. Unlike every aggregator above --
# one convex region per emitted metric column, served by the generic engine -- ``add_defensive_
# credit`` emits ONE per-action rollup for the WHOLE credit family, because the region a credit
# integrates over is per-CREDIT (its resolution mode + anchor), not per-column. So the region is
# built BY MODE here and the per-action fraction is a credit-magnitude-weighted mean; the coverage
# is declared in ``_CUSTOM_COMPANION_COVERS`` (below), NOT in ``OBSERVABILITY_REGISTRY``.
# ---------------------------------------------------------------------------

#: Attacked goal centre in action-LTR (the shot->goal corridor + the credit resolution share this
#: frame; ``defensive_credit._resolution._GOAL_XY`` is the same point). Derived from ``spadlconfig``
#: like the pitch-band constants above, so they cannot drift from ``_resolution._GOAL_XY`` (105.0 /
#: field_width/2 == 34.0).
_DC_GOAL_X = float(spadlconfig.field_length)
_DC_GOAL_Y = float(spadlconfig.field_width) / 2.0

#: Resolution-mode string values (mirror ``defensive_credit._params.RESOLUTION_*``; a stable closed
#: vocabulary, kept as literals here to preserve this module's neutrality -- it imports nothing from
#: ``defensive_credit``. A drift guard in ``tests/tracking/test_fov_companions.py`` pins them to
#: ``RESOLUTION_VALUES``). ``anchor_actor`` is event-resolved (no region); ``lane`` is the corridor;
#: every OTHER token is a proximity DISK.
_DC_MODE_ANCHOR_ACTOR = "anchor_actor"
_DC_MODE_LANE = "lane"


def _defensive_credit_lane_region(origin_x, origin_y, cone_width_factor, max_t, min_half_width):
    """The shot->goal lane CORRIDOR (``_resolution._lane_blocker`` geometry) as a convex trapezoid.

    Reproduces the blocker's in-corridor admissible set: a defender qualifies when its fraction
    along the shot->goal lane ``t in [0, max_t]`` AND its perpendicular offset is
    ``<= max(cone_width_factor * shot_dist / 2 * t, min_half_width)``. That set is the trapezoid with
    corners at the origin ``+- min_half_width`` (the cone floor at ``t=0``) and, at ``t=max_t`` along
    the lane, ``+- the half-width there``. The far half-width ``>= min_half_width`` always, so the
    trapezoid never pinches -> strictly convex. Goal fixed at ``(105, 34)`` (action-LTR).
    """
    lane = np.array([_DC_GOAL_X - origin_x, _DC_GOAL_Y - origin_y], dtype=float)
    shot_dist = float(np.hypot(lane[0], lane[1]))
    if shot_dist < 1e-6:
        return _NO_REGION  # degenerate: shot from the goal line -> no lane direction
    u = lane / shot_dist
    perp = np.array([-u[1], u[0]], dtype=float)  # unit normal to the lane
    near_hw = float(min_half_width)  # the cone term is 0 at t=0, so the floor governs the near end
    far = np.array([origin_x, origin_y], dtype=float) + max_t * lane  # point at t=max_t along the lane
    far_hw = max(cone_width_factor * shot_dist / 2.0 * max_t, min_half_width)
    poly = np.array(
        [
            [origin_x + near_hw * perp[0], origin_y + near_hw * perp[1]],
            [origin_x - near_hw * perp[0], origin_y - near_hw * perp[1]],
            [far[0] - far_hw * perp[0], far[1] - far_hw * perp[1]],
            [far[0] + far_hw * perp[0], far[1] + far_hw * perp[1]],
        ]
    )
    if not is_convex(poly):  # unreachable on the floored trapezoid; refuse rather than mis-clip
        return _NO_REGION
    return poly


def defensive_credit_region_for_mode(
    mode: str,
    *,
    origin_x: float,
    origin_y: float,
    region_radius: float,
    lane_cone_width_factor: float,
    lane_max_t: float,
    lane_min_half_width_m: float,
):
    """The convex region a defensive-credit's resolution mode searched, in action-LTR (ADR-077).

    * ``anchor_actor`` -> :data:`_NO_REGION` (event-resolved passer/recoverer -- no spatial ROI, P2).
    * ``lane`` -> the shot->goal CORRIDOR trapezoid the lane blocker searched.
    * every other token (``nearest`` / ``all_within`` / ``all_within_beyond_nearest`` /
      ``nearest_fallback``) -> the proximity DISK of radius ``region_radius`` centred on the anchor.

    A non-finite origin, a degenerate lane, or (for the disk modes) a non-finite/non-positive radius
    has no measurable region -> :data:`_NO_REGION` (``degenerate_region``), never a fabricated
    fraction.
    """
    if mode == _DC_MODE_ANCHOR_ACTOR:
        return _NO_REGION
    ox, oy = float(origin_x), float(origin_y)
    if not (np.isfinite(ox) and np.isfinite(oy)):
        return _NO_REGION
    if mode == _DC_MODE_LANE:
        return _defensive_credit_lane_region(ox, oy, lane_cone_width_factor, lane_max_t, lane_min_half_width_m)
    r = float(region_radius)
    if not (np.isfinite(r) and r > 0.0):
        return _NO_REGION
    return _kernels._inscribed_disk(ox, oy, r)


def _rollup_credit_observed_fraction(observations):
    """Credit-magnitude-weighted mean of per-credit observed fractions for ONE action (ADR-077, P5).

    ``observations`` -- an iterable of ``(magnitude, observed_fraction, source, region_bearing)`` per
    DEFENDING credit. Returns ``(fraction, source)``:

    * ``region_bearing=False`` (an ``anchor_actor`` credit, or a degenerate region) is excluded from
      BOTH the numerator and the denominator -- it contributes no measurable region.
    * with >= 1 region-bearing credit actually OBSERVED, the fraction is
      ``sum(|magnitude| * observed_fraction) / sum(|magnitude|)`` over those credits, source
      ``observed``.
    * with region-bearing credits OBSERVED but every one of magnitude 0 (weight sum 0), there is no
      honest weighted fraction -> NaN + ``degenerate_region`` (NEVER ``observed`` with a NaN fraction,
      which would violate the observed => finite-fraction contract).
    * with region-bearing credits but NONE observed (all no_polygon / degenerate), NaN + the first
      such credit's token.
    * with NO region-bearing credit (an ``anchor_actor``-only action, or none at all), NaN +
      ``degenerate_region`` -- never a fabricated 1.0.
    """
    region_bearing = [o for o in observations if o[3]]
    if not region_bearing:
        return float("nan"), REGION_OBSERVATION_DEGENERATE_REGION
    observed = [
        (abs(float(m)), float(f)) for (m, f, s, _rb) in region_bearing if s == VISIBLE_AREA_OBSERVED and np.isfinite(f)
    ]
    total = sum(w for w, _ in observed)
    if observed and total > 0.0:
        return float(sum(w * f for w, f in observed) / total), VISIBLE_AREA_OBSERVED
    if observed:  # region-bearing credits WERE observed but all have magnitude 0 -> no honest fraction
        return float("nan"), REGION_OBSERVATION_DEGENERATE_REGION
    return float("nan"), region_bearing[0][2]


def append_observability_companions(out, actions, *, entries, visible_area, linked_ids, ctx: RegionCtx):
    """Emit ``<column>_observed_fraction`` / ``_observed_source`` for each ``entry``.

    One engine for every FOV companion. For each entry and each action: an action absent from
    ``linked_ids`` is ``unlinked`` (fraction stays NaN); a :data:`_NO_REGION` builder result is
    ``degenerate_region`` (NaN); otherwise :func:`classify_region_observation` scores the region
    against the action's polygon (a missing polygon classifies as ``no_polygon``).
    """
    polygons = _polygons_by_action(visible_area)
    n = len(actions)
    for e in entries:
        fracs = np.full(n, np.nan)
        sources: list[str] = []
        for i, aid in enumerate(actions["action_id"]):
            key = canonical_id(aid)
            if linked_ids is not None and key not in linked_ids:
                sources.append(VISIBLE_AREA_UNLINKED)
                continue
            region = e.region(i, ctx)
            if region is _NO_REGION:  # identity, NEVER == (P2)
                sources.append(REGION_OBSERVATION_DEGENERATE_REGION)
                continue
            frac, s = classify_region_observation(polygons.get(key), region)  # polygon None -> no_polygon
            fracs[i] = frac
            sources.append(s)
        out[f"{e.column}_observed_fraction"] = fracs
        out[f"{e.column}_observed_source"] = sources
    return out


#: Every metric's region is STATIC here; per-call params (radius, later a pressure method) arrive
#: via ``ctx.extras``, so the completeness gate (Task 8) reads every companioned column from ONE
#: source. The ``add_action_context`` entries are ordered ``(nearest, receiver, triangle)`` to
#: match the byte-identical emission order the ADR-062 helper established. Later tasks assign the
#: other aggregator keys.
OBSERVABILITY_REGISTRY: dict[str, tuple[ObservabilityEntry, ...]] = {
    "add_action_context": (
        ObservabilityEntry("nearest_defender_distance", nearest_defender_disk),
        ObservabilityEntry("receiver_zone_density", receiver_disk),  # reads ctx.extras["receiver_radius"]
        ObservabilityEntry("defenders_in_triangle_to_goal", triangle_to_goal),
    ),
    # Emitted ONLY when ``method == "andrienko_oval"`` -- the other pressure methods do not
    # produce this column, so there is nothing to companion (the aggregator gates the call).
    # ``ctx.extras`` carries ``oval_d_front`` / ``oval_d_back`` / ``goal_x`` / ``goal_y``.
    "add_pressure_on_actor": (ObservabilityEntry("pressure_on_actor__andrienko_oval", andrienko_oval_region),),
    # Voluntary companions: only the three region-COUNT columns (P9), one entry each, all sharing
    # the passer->receiver x-band region.
    "add_packing": tuple(ObservabilityEntry(col, packing_zone_region) for col in _PACKING_REGION_COUNT_COLUMNS),
    # Aggregate-position metrics on FIXED action-LTR pitch bands (ADR-077), keyed on the column's
    # ROLE, never a per-action goal_map (see the fixed-zone note above).
    # ``defensive_line_x`` = the DEFENDING team's back line -> its defended third (HIGH end).
    "add_defensive_line": (ObservabilityEntry("defensive_line_x", defended_third_region),),
    # ``add_team_shape`` emits FOUR centroid columns split by team ROLE on OPPOSITE ends -> TWO
    # companions, one per role. There is NO ``team_shape_centroid`` column, so each companion's
    # synthetic ``column`` name (``*_attacking`` / ``*_defending``) is mapped to its real x/y pair
    # via ``covers`` for the completeness gate (Task 8). ``attacking`` = the acting team's OWN half
    # (LOW end); ``defending`` = the opponent's OWN half (HIGH end).
    "add_team_shape": (
        ObservabilityEntry(
            "team_shape_centroid_attacking",
            attacking_own_half_region,
            covers=("team_shape_centroid_x_attacking", "team_shape_centroid_y_attacking"),
        ),
        ObservabilityEntry(
            "team_shape_centroid_defending",
            defending_own_half_region,
            covers=("team_shape_centroid_x_defending", "team_shape_centroid_y_defending"),
        ),
    ),
    # ``off_ball_xt_team`` = the ATTACKING/possession team's off-ball threat -> its attacking half
    # (HIGH end). The other player-influence columns are not region-based counts and are uncompanioned.
    "add_player_influence": (ObservabilityEntry("off_ball_xt_team", attacking_half_region),),
    # xT-GK pressure-bearing columns (Task 5): ``rho`` (xt_gk_pressure) AND the PEV forward gain
    # (xt_gk_pev) both flow through ``pressure_on_actor`` centred on the RESOLVED GK origin, so both
    # take the method-dispatched pressure ROI. The GK-geometry / completion columns
    # (xt_gk_base/_rav/_dzv) are NOT pressure-region-dependent -> uncompanioned; the composite
    # ``xt_gk`` is EXEMPT (below, M1). ``ctx.extras`` carries pressure_method + the oval/link radii.
    "add_xt_gk": (
        ObservabilityEntry("xt_gk_pressure", xt_gk_pressure_region),
        ObservabilityEntry("xt_gk_pev", xt_gk_pressure_region),
    ),
}
_OBSERVABILITY_EXEMPT: dict[str, str] = {
    # M1: the composite xt_gk is a region-dependent ``gamma*pev`` term ADDED to the GK-geometry /
    # completion base/rav/dzv terms; the region-dependent part is already covered by the
    # xt_gk_pressure / xt_gk_pev companions, and there is no honest SINGLE observed-fraction for the
    # whole composite (mixing a region-supported and a geometry-supported term).
    "xt_gk": (
        "composite of a region-dependent gamma*pev term and GK-geometry base/rav/dzv; the "
        "region-dependent part is covered by the xt_gk_pressure/xt_gk_pev companions; no honest "
        "single fraction (M1)."
    ),
}
# Task 8 (ADR-077): the ghost-keeper POSITION columns are the output of a LEARNED (ghost-GK) model,
# not a region integral. Their FOV dependence is the model's WHOLE-FRAME receptive field -- there is
# no single clean region-of-interest to integrate a visible-area fraction over, and a whole-pitch
# fraction would over-simplify (it would read the same for a keeper cropped near the goal as for one
# cropped at midfield). A bespoke ghost-observability model is deferred to a later cycle. Declared
# EXEMPT here (not companioned) so the Task-8 completeness gate accepts these two region_support
# columns as covered-by-exemption. ``ghost_gk_source`` is ``no_support`` and never enters `required`.
_OBSERVABILITY_EXEMPT["ghost_gk_x"] = _OBSERVABILITY_EXEMPT["ghost_gk_y"] = (
    "learned model; FOV dependence is its whole-frame receptive field, no single clean ROI; "
    "a whole-pitch fraction would over-simplify. Bespoke ghost-observability model is a later cycle."
)
#: Raw metric columns companioned by a NON-engine (custom) path -- declared at MODULE LOAD so the
#: Task 8 completeness gate reads a populated set WITHOUT running the aggregator. Task 6:
#: ``add_defensive_credit`` emits ONE per-action rollup companion
#: (``defensive_credit_observed_fraction`` / ``_observed_source``) for the whole credit family, so it
#: is a custom path. The raw columns it covers are exactly the three tagged ``region_support`` in the
#: SB360 audit (``tests/sb360/_entries/_offball.py``): net / minus / n. ``defensive_credit_plus`` is
#: ``no_support`` (measured 0 delta under the crop) and is deliberately NOT companioned.
_CUSTOM_COMPANION_COVERS: set[str] = {
    "defensive_credit_net",
    "defensive_credit_minus",
    "n_defensive_credits",
}

#: R1 (ADR-077): the hand-curated aggregate FOV-crop-sensitive bucket the perturbation probe CANNOT
#: reach. The SB360 audit's ``region_support`` tag is a SINGLE-PLAYER-PERTURBATION axis -- move one
#: player and see if the metric moves -- so a mean-over-many aggregate is robust to it (measured
#: ``no_support``: ``defensive_line_x`` / ``team_shape_centroid_*`` / the packing region counts) even
#: though it is genuinely biased by an FOV CROP (S1: an entire cluster of players cut out of view). The
#: crop axis is a DIFFERENT axis than the perturbation probe, and the probe is blind to it, so these
#: columns must be gate-forced INDEPENDENTLY. Like ADR-054's ``_GUARD_EXEMPT``, this is a
#: manual-discipline surface: a NEW aggregate/region metric must be ADDED here to be gate-forced -- the
#: Task-8 M3 non-vacuity plant proves the enforcement fires. ``off_ball_xt_team`` is ALSO
#: ``region_support`` (the union deduplicates); it is listed for completeness of the aggregate family.
_AGGREGATE_FOV_SENSITIVE: frozenset[str] = frozenset(
    {
        "defensive_line_x",
        "team_shape_centroid_x_attacking",
        "team_shape_centroid_y_attacking",
        "team_shape_centroid_x_defending",
        "team_shape_centroid_y_defending",
        "off_ball_xt_team",
        *_PACKING_REGION_COUNT_COLUMNS,
    }
)


def companioned_columns() -> set[str]:
    """Every RAW metric column that receives a companion (engine or custom) -- the gate's coverage set."""
    cols = {c for ents in OBSERVABILITY_REGISTRY.values() for e in ents for c in (e.covers or (e.column,))}
    return cols | _CUSTOM_COMPANION_COVERS
