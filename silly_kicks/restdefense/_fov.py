"""FOV-observability companions for the rest-defense count columns (TF-60, ADR-077, opt-in).

SB360 freeze-frames carry only the in-FOV players, so a count over a region is a LOWER BOUND when
the region is partly cropped. Each count column therefore ships an opt-in
``<col>_observed_fraction`` / ``<col>_observed_source`` companion: the observed AREA fraction of the
column's convex region of interest inside the action's ``visible_area`` polygon.

Two rules from ADR-077. (1) restdefense uses the PUBLIC seams ``classify_region_observation`` +
``REGION_OBSERVATION_SOURCE_VALUES`` + ``VISIBLE_AREA_UNLINKED`` and assembles the companions itself
in the canonical naming -- the one-engine ``append_observability_companions`` is PRIVATE to tracking
and off-limits (the durable fix, a public companion-assembly seam, is a later cycle). (2) The ROI is
a FIXED action-LTR zone keyed on the column's ROLE, NEVER a ``goal_map``: the ``visible_area`` polygon
is action-LTR (own goal at x=0), so the ROI is built from the goal-relative distances
``gr = abs(x - own_goal_x)`` (which ARE the action-LTR coordinates) -- a frame-coordinate ``goal_map``
end would land on the opposite end from the polygon for every away-possession action (the S1 silent
failure). The zones span the full pitch width, so the y-orientation is irrelevant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.id_compat import canonical_id
from silly_kicks.tracking import (
    VISIBLE_AREA_UNLINKED,
    classify_region_observation,
)

from ._columns import (
    RD_COMPACTNESS_X,
    RD_DEPTH,
    RD_GK_LINE_HEIGHT,
    RD_GK_TO_LINE_DISTANCE,
    RD_LINE_HEIGHT,
    RD_LINE_HEIGHT_RELATIVE,
    RD_NUM_SUPERIORITY,
    RD_NUM_SUPERIORITY_GK,
    RD_SHAPE_STAGGER,
    RD_WIDTH,
    RD_ZONE_OCCUPANCY,
)
from ._geometry import danger_zone_bounds

_PITCH_WIDTH = float(spadlconfig.field_width)

#: The FOV-sensitive count/region columns. ``rd_num_superiority`` and ``rd_num_superiority_gk`` share
#: the behind-the-ball BAND region (the keeper-inclusion changes the count, not the region);
#: ``rd_zone_occupancy`` uses the danger-zone region.
_BAND_COLUMNS = (RD_NUM_SUPERIORITY, RD_NUM_SUPERIORITY_GK)
_ZONE_COLUMNS = (RD_ZONE_OCCUPANCY,)
FOV_SENSITIVE_COLUMNS = (*_BAND_COLUMNS, *_ZONE_COLUMNS)

#: Layer-1 columns that receive NO FOV companion, each with a reason -- the ADR-077 companion model
#: is "observed AREA fraction of a convex region", which only the count/region columns above have.
#: The position/shape/label metrics are honest-NaN on a crop (a cropped keeper -> NaN gk metrics; a
#: cropped back line shifts the line) but have no single convex ROI to integrate; a bespoke
#: aggregate-position companion (the ADR-077 fixed-third class) is deferred for restdefense. The
#: completeness gate asserts FOV_SENSITIVE_COLUMNS + _OBSERVABILITY_EXEMPT partitions RD_LAYER1_COLUMNS
#: exactly, so a NEW count/region column must companion-or-exempt (the anti-rot property).
_OBSERVABILITY_EXEMPT: dict[str, str] = {
    RD_LINE_HEIGHT: "a POSITION (rearguard-line distance from goal), not a region count; no observed-area fraction",
    RD_LINE_HEIGHT_RELATIVE: "a POSITION difference (line vs ball height), not a region count",
    RD_COMPACTNESS_X: "a rearguard-shape SPREAD (x-range), not a region count",
    RD_WIDTH: "a team-shape SPREAD (width), not a region count",
    RD_DEPTH: "a team-shape SPREAD (length), not a region count",
    RD_SHAPE_STAGGER: "a categorical stagger LABEL, not a region count",
    RD_GK_LINE_HEIGHT: "a GK POSITION; honest-NaN when the keeper is cropped (via _gk_x), not a region count",
    RD_GK_TO_LINE_DISTANCE: "a GK-to-line GAP (position-derived); honest-NaN on a crop, not a region count",
}


def _band_polygon(far: float) -> np.ndarray:
    """A convex CCW rectangle ``x in [0, far]``, full pitch height. NaN/0 ``far`` -> degenerate region."""
    return np.array([[0.0, 0.0], [far, 0.0], [far, _PITCH_WIDTH], [0.0, _PITCH_WIDTH]])


def _num(v) -> float:
    """A pandas Scalar (from itertuples) -> float; NA -> NaN. Untyped arg avoids the Scalar->float
    typing error and the NaN policy keeps a degenerate ROI honest."""
    return float(v) if pd.notna(v) else float("nan")


def polygons_by_action(visible_area: pd.DataFrame | None) -> dict:
    """``canonical_id(action_id) -> polygon`` -- the ADR-019-safe visible-area join (a private-free
    re-implementation of tracking's ``_polygons_by_action``; the canonical key is load-bearing:
    a raw-id dict misses SILENTLY across dtypes, indistinguishable from a genuine absence)."""
    out: dict = {}
    if visible_area is not None and len(visible_area) > 0:
        for row in visible_area[["action_id", "polygon"]].itertuples(index=False):
            out[canonical_id(row.action_id)] = row.polygon
    return out


def append_fov_companions(samples: pd.DataFrame, keep: pd.DataFrame, *, visible_area, params) -> pd.DataFrame:
    """Emit the ``<col>_observed_fraction`` / ``_observed_source`` companions (``samples`` order == ``keep`` order).

    ``keep`` carries the per-sample geometry (``ball_x`` / ``own_goal_x`` / ``defensive_line_x`` /
    ``frame_id`` / ``action_id``). An unlinked action is ``unlinked``; a row whose ROI cannot be
    built (unresolved goal end -> NaN ``own_goal_x``) yields a NaN-vertex region that classifies as
    ``degenerate_region``; otherwise the region is scored against the action's polygon (absent ->
    ``no_polygon``). Never a fabricated 1.0 / 0.0 on a non-``observed`` row.
    """
    polys = polygons_by_action(visible_area)
    n = len(keep)
    fracs = {c: np.full(n, np.nan) for c in FOV_SENSITIVE_COLUMNS}
    sources = {c: [] for c in FOV_SENSITIVE_COLUMNS}
    for i, row in enumerate(keep.itertuples(index=False)):
        aid = canonical_id(row.action_id)
        poly = polys.get(aid)
        linked = pd.notna(row.frame_id)
        own = _num(row.own_goal_x)
        gr_ball = abs(_num(row.ball_x) - own)  # NaN if either is NA
        gr_line = abs(_num(row.defensive_line_x) - own)
        zone_far = (
            danger_zone_bounds(gr_line, 0.0, zone_depth_m=params.zone_depth_m)[1]
            if np.isfinite(gr_line)
            else float("nan")
        )
        far_by_col = {**dict.fromkeys(_BAND_COLUMNS, gr_ball), **dict.fromkeys(_ZONE_COLUMNS, zone_far)}
        for col in FOV_SENSITIVE_COLUMNS:
            if not linked:
                sources[col].append(VISIBLE_AREA_UNLINKED)
                continue
            frac, src = classify_region_observation(poly, _band_polygon(far_by_col[col]))
            fracs[col][i] = frac
            sources[col].append(src)

    out = samples.copy()
    for col in FOV_SENSITIVE_COLUMNS:
        out[f"{col}_observed_fraction"] = fracs[col]
        out[f"{col}_observed_source"] = sources[col]
    return out
