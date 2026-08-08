"""Observed-region primitives for partial-visibility providers (ADR-055).

A freeze-frame provider does not see the whole pitch. StatsBomb 360 ships a ``visible_area``
polygon per event; SkillCorner's broadcast tracking has the same shape. Everything downstream --
a defender count, a zone density, a nearest-defender distance -- silently treats "nobody there"
and "nobody VISIBLE there" as the same observation, and they are not.

This module ships the two primitives that let a consumer tell them apart, plus one aggregator
that answers the only question needing no consumer choice (how much of the PITCH was observed).
Wiring coverage INTO the count features is deliberately not here: it would change existing values
and decide, on the consumer's behalf, what a partial observation means (ADR-009).

Provider-agnostic on arrival: ``polygon_to_spadl`` already yields SPADL coordinates, so these
take SPADL vertices and know nothing about StatsBomb. They live in ``tracking/`` rather than
``providers/`` because a parse port must not gain an ``add_*`` -- that inverts the dependency
direction the ports exist to hold.

See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks._nan_safety import nan_safe_enrichment
from silly_kicks._polygon import (
    MIN_VERTICES,
    as_polygon,
    clip_to_convex,
    counter_clockwise,
    covered_fraction,
    point_in_polygon,
    shoelace_area,
)
from silly_kicks.id_compat import canonical_id
from silly_kicks.spadl import config as spadlconfig

#: Closed vocabulary for ``visible_area_source`` (the ``DAS_SOURCE_VALUES`` /
#: ``GHOST_GK_SOURCE_VALUES`` pattern). Each token is exported so a consumer enum pins to this
#: set rather than to string literals.
#: The action's 360 polygon is present and has enough vertices to bound an area.
VISIBLE_AREA_OBSERVED = "observed"
#: The action has no 360 record at all -- nothing was published about what the camera saw.
VISIBLE_AREA_NO_POLYGON = "no_polygon"
#: A polygon arrived but carries fewer than 3 vertices, so it bounds no region. Distinct from
#: ``no_polygon``: something WAS published and it is unusable, which is a different data-quality
#: finding from nothing having been published.
VISIBLE_AREA_DEGENERATE_POLYGON = "degenerate_polygon"
#: The action reached no frame. Representable only because the aggregator accepts ``links``;
#: ``ghost_gk_source`` carries this same token for the same reason.
VISIBLE_AREA_UNLINKED = "unlinked"
VISIBLE_AREA_SOURCE_VALUES: tuple[str, ...] = (
    VISIBLE_AREA_OBSERVED,
    VISIBLE_AREA_NO_POLYGON,
    VISIBLE_AREA_DEGENERATE_POLYGON,
    VISIBLE_AREA_UNLINKED,
)


def point_observed(polygon, x: float, y: float) -> bool | None:
    """Was ``(x, y)`` inside the observed region? ``None`` when that cannot be answered.

    ``False`` is a CLAIM -- "the camera did not see this point" -- and a missing or degenerate
    polygon supports no such claim. Returning ``False`` there would make an absence of data
    indistinguishable from a negative observation, which is the confusion this whole module
    exists to remove.

    Parameters
    ----------
    polygon : array-like or None
        ``(N, 2)`` SPADL vertices of the observed region, e.g. from
        :func:`silly_kicks.providers.statsbomb.polygon_to_spadl`. Need not be convex.
    x, y : float
        SPADL point.

    Returns
    -------
    bool or None
        ``None`` if the polygon is absent, has fewer than 3 vertices, or is non-finite; also
        ``None`` for a non-finite query point, for the same reason.

    Examples
    --------
    A point inside the observed half, one outside it, and the unanswerable case:

    >>> import numpy as np
    >>> from silly_kicks.tracking import point_observed
    >>> half = np.array([[0.0, 0.0], [52.5, 0.0], [52.5, 68.0], [0.0, 68.0]])
    >>> point_observed(half, 20.0, 30.0)
    True
    >>> point_observed(half, 90.0, 30.0)
    False
    >>> point_observed(None, 20.0, 30.0) is None
    True

    See NOTICE for full bibliographic citations.
    """
    poly = as_polygon(polygon)
    if len(poly) < MIN_VERTICES or not np.isfinite([x, y]).all():
        return None
    return point_in_polygon(poly, x, y)


def region_observed_fraction(polygon, region) -> float:
    """What share of ``region`` the observed ``polygon`` covers, in ``[0, 1]``.

    ``region`` is a POLYGON, never a bounding box. The motivating consumers ask about a triangle
    to goal, a radius around a receiver, and pitch-control cells; a bounding box can only
    OVER-report coverage for a triangle, i.e. fabricate observation -- the exact failure this
    seam exists to prevent.

    Parameters
    ----------
    polygon : array-like or None
        ``(N, 2)`` observed region. May be non-convex.
    region : array-like
        ``(M, 2)`` region of interest. **Must be convex** -- see Raises.

    Returns
    -------
    float
        ``area(region ∩ polygon) / area(region)``. ``nan`` when the polygon is absent or
        degenerate (nothing is known), and ``nan`` when the region itself has no area (the
        question has no denominator). Never silently ``0.0`` for either.

    Raises
    ------
    ValueError
        If ``region`` is not convex. Sutherland-Hodgman clips against a CONVEX region only, and
        a non-convex one yields a wrong area rather than an error -- so this refuses instead of
        returning a plausible number. Split a concave region and sum the parts.

    Examples
    --------
    A triangle covering the whole pitch, observed only on the left half, is exactly three
    quarters visible -- the clipped trapezoid is 2677.5 m² against the triangle's 3570 m²:

    >>> import numpy as np
    >>> from silly_kicks.tracking import region_observed_fraction
    >>> left_half = np.array([[0.0, 0.0], [52.5, 0.0], [52.5, 68.0], [0.0, 68.0]])
    >>> triangle = np.array([[0.0, 0.0], [105.0, 0.0], [0.0, 68.0]])
    >>> round(region_observed_fraction(left_half, triangle), 6)
    0.75

    See NOTICE for full bibliographic citations.
    """
    return covered_fraction(polygon, region)


@nan_safe_enrichment
def add_visible_area_coverage(
    actions: pd.DataFrame,
    *,
    visible_area: pd.DataFrame,
    links: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Enrich actions with how much of the PITCH the provider observed, plus provenance.

    Emits ``visible_area_fraction`` -- the observed polygon CLIPPED to the pitch, over the pitch
    area -- and ``visible_area_source`` over :data:`VISIBLE_AREA_SOURCE_VALUES`.

    The frame-level pitch fraction is shipped because it is the only coverage question that
    needs no consumer choice. "Which region?" -- the triangle to goal, a radius, the pass lane --
    is a different feature's question each time, so the library ships
    :func:`region_observed_fraction` for the consumer's own region instead of picking one
    (ADR-009).

    Parameters
    ----------
    actions : pd.DataFrame
        SPADL actions; only ``action_id`` is read.
    visible_area : pd.DataFrame
        ``action_id`` -> ``polygon`` ``(N, 2)`` SPADL vertices, as produced by
        ``silly_kicks.providers.statsbomb.shape_snapshots``.
    links : pd.DataFrame or None
        Optional ``link_actions_to_frames`` pointers. When given, an action with no linked frame
        is tagged ``unlinked`` rather than ``no_polygon`` -- two different facts with two
        different remedies.

    Returns
    -------
    pd.DataFrame
        A copy of ``actions`` with the two columns added. ``visible_area_fraction`` is NaN for
        every non-``observed`` token -- **never 1.0 and never 0.0**, both of which are
        measurements this function did not make.

    Examples
    --------
    Attach coverage to a match's actions::

        from silly_kicks.providers.statsbomb import shape_snapshots
        from silly_kicks.tracking import add_visible_area_coverage

        snapshots, visible_area, _report = shape_snapshots(records, actions)
        enriched = add_visible_area_coverage(actions, visible_area=visible_area)
        enriched[["action_id", "visible_area_fraction", "visible_area_source"]].head()

    See NOTICE for full bibliographic citations.
    """
    out = actions.copy()
    pitch = np.array(
        [
            [0.0, 0.0],
            [spadlconfig.field_length, 0.0],
            [spadlconfig.field_length, spadlconfig.field_width],
            [0.0, spadlconfig.field_width],
        ]
    )
    pitch_area = shoelace_area(pitch)

    # ADR-019: `action_id` joins THREE separately-sourced frames here -- the caller's `actions`,
    # the provider's `visible_area` (from `shape_snapshots`) and `links` -- so the keys must be
    # canonicalized. A plain dict keyed on the raw id MISSES SILENTLY across dtypes, and the miss
    # is indistinguishable from a genuine absence: measured, an int64 `actions.action_id` against
    # an object `visible_area.action_id` reported `no_polygon` for EVERY row while every polygon
    # had in fact been supplied. That is precisely the confusion this module exists to remove.
    by_action: dict = {}
    if visible_area is not None and len(visible_area) > 0:
        for row in visible_area[["action_id", "polygon"]].itertuples(index=False):
            by_action[canonical_id(row.action_id)] = row.polygon

    linked_ids: set | None = None
    if links is not None and len(links) > 0:
        ok = links[links["frame_id"].notna()] if "frame_id" in links.columns else links
        linked_ids = {canonical_id(a) for a in ok["action_id"]}

    fractions = np.full(len(out), np.nan)
    sources: list[str] = []
    for i, aid in enumerate(out["action_id"]):
        key = canonical_id(aid)
        if linked_ids is not None and key not in linked_ids:
            sources.append(VISIBLE_AREA_UNLINKED)
            continue
        if key not in by_action:
            sources.append(VISIBLE_AREA_NO_POLYGON)
            continue
        poly = as_polygon(by_action[key])
        if len(poly) < MIN_VERTICES:
            sources.append(VISIBLE_AREA_DEGENERATE_POLYGON)
            continue
        # The polygon is deliberately UNCLIPPED on arrival (a camera legitimately sees past the
        # touchline, ADR-054 D5), so the clip happens here -- where the quantity being reported
        # is a share OF THE PITCH and off-pitch area would inflate it above 1.
        clipped = clip_to_convex(poly, counter_clockwise(pitch))
        fractions[i] = min(1.0, shoelace_area(clipped) / pitch_area)
        sources.append(VISIBLE_AREA_OBSERVED)

    out["visible_area_fraction"] = fractions
    out["visible_area_source"] = sources

    # Closed-vocabulary post-condition (the DAS_SOURCE_VALUES / GHOST_GK_SOURCE_VALUES pattern).
    # A vocabulary described as closed but unenforced is a comment, not a contract.
    emitted = set(pd.unique(out["visible_area_source"].dropna()))
    if not emitted <= set(VISIBLE_AREA_SOURCE_VALUES):
        raise ValueError(
            f"visible_area_source emitted values outside its closed vocabulary: "
            f"{sorted(emitted - set(VISIBLE_AREA_SOURCE_VALUES))}"
        )
    return out
