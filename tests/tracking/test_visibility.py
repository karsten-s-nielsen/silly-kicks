"""The ADR-055 observed-region seam.

The property under test throughout is one distinction: **"nothing there" is not "nothing
VISIBLE there".** Every assertion below exists because collapsing those two is how a
partial-visibility provider turns missing data into a confident measurement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking import (
    REGION_OBSERVATION_SOURCE_VALUES,
    VISIBLE_AREA_DEGENERATE_POLYGON,
    VISIBLE_AREA_NO_POLYGON,
    VISIBLE_AREA_OBSERVED,
    VISIBLE_AREA_SOURCE_VALUES,
    VISIBLE_AREA_UNLINKED,
    add_visible_area_coverage,
    classify_region_observation,
    point_observed,
    region_observed_fraction,
)

LEFT_HALF = np.array([[0.0, 0.0], [52.5, 0.0], [52.5, 68.0], [0.0, 68.0]])
#: Spans the whole pitch: area 0.5 * 105 * 68 = 3570 m^2.
FULL_TRIANGLE = np.array([[0.0, 0.0], [105.0, 0.0], [0.0, 68.0]])


def _actions(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame({"action_id": list(range(n)), "team_id": [1] * n})


# ---------------------------------------------------------------------------
# point_observed -- the bool | None contract
# ---------------------------------------------------------------------------


def test_point_observed_distinguishes_not_seen_from_cannot_say():
    """``False`` is a CLAIM; a missing polygon supports no claim, so it must be ``None``.

    Asserting ``is None`` rather than falsiness is the whole point: ``None`` and ``False`` are
    both falsy, so a truthiness assertion would pass on the very confusion this contract removes.
    """
    assert point_observed(LEFT_HALF, 20.0, 30.0) is True
    assert point_observed(LEFT_HALF, 90.0, 30.0) is False
    assert point_observed(None, 20.0, 30.0) is None
    assert point_observed(np.empty((0, 2)), 20.0, 30.0) is None
    assert point_observed(np.array([[0.0, 0.0], [1.0, 1.0]]), 0.5, 0.5) is None, (
        "two vertices bound no region, so membership is unanswerable"
    )


def test_point_observed_is_None_for_a_non_finite_query():
    """A NaN coordinate cannot be inside or outside anything -- same rule, other operand."""
    assert point_observed(LEFT_HALF, float("nan"), 30.0) is None


def test_point_observed_handles_a_NON_CONVEX_polygon():
    """Broadcast polygons are not guaranteed convex, so membership uses ray casting.

    The notch is the discriminator: a convex-hull implementation would answer True for the point
    inside it, because the hull swallows the concavity.
    """
    notched = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [5.0, 1.0], [0.0, 10.0]])
    assert point_observed(notched, 1.0, 1.0) is True
    assert point_observed(notched, 5.0, 8.0) is False, "the notch must be OUTSIDE, not hull-filled"


# ---------------------------------------------------------------------------
# region_observed_fraction
# ---------------------------------------------------------------------------


def test_region_fraction_on_the_half_pitch_triangle_is_exactly_three_quarters():
    """Hand-checkable: the clipped trapezoid is 2677.5 m^2 against the triangle's 3570 m^2.

    Vertices (0,0), (52.5,0), (52.5,34), (0,68) -- the hypotenuse from (105,0) to (0,68) crosses
    x=52.5 at y=34. Chosen because the answer is a round number derived by hand rather than a
    value read back out of the implementation.
    """
    assert region_observed_fraction(LEFT_HALF, FULL_TRIANGLE) == pytest.approx(0.75, abs=1e-9)


def test_a_bounding_box_would_OVERSTATE_this_and_that_is_why_region_is_a_polygon():
    """Non-vacuity for the choice the spec makes: bbox coverage != polygon coverage.

    The triangle's bounding box is the whole pitch, so a bbox implementation would report
    52.5*68 / (105*68) = 0.5 -- a DIFFERENT number. The direction matters less than the fact that
    they disagree: if they agreed, 'region is a polygon, not a bbox' would be untestable here.
    """
    bbox = np.array([[0.0, 0.0], [105.0, 0.0], [105.0, 68.0], [0.0, 68.0]])
    as_bbox = region_observed_fraction(LEFT_HALF, bbox)
    as_polygon = region_observed_fraction(LEFT_HALF, FULL_TRIANGLE)
    assert as_bbox == pytest.approx(0.5, abs=1e-9)
    assert abs(as_bbox - as_polygon) > 0.2, "bbox and polygon agree here -- the fixture proves nothing"


def test_full_and_zero_coverage_are_both_reachable():
    """Both ends of the band, so a clamp-to-one or a stuck-at-zero is visible."""
    whole_pitch = np.array([[0.0, 0.0], [105.0, 0.0], [105.0, 68.0], [0.0, 68.0]])
    right_box = np.array([[80.0, 10.0], [100.0, 10.0], [100.0, 50.0], [80.0, 50.0]])
    assert region_observed_fraction(whole_pitch, right_box) == pytest.approx(1.0, abs=1e-9)
    assert region_observed_fraction(LEFT_HALF, right_box) == pytest.approx(0.0, abs=1e-9)


def test_an_unanswerable_fraction_is_NaN_never_zero():
    """``0.0`` would mean "observed, and none of it" -- a measurement nobody made."""
    assert np.isnan(region_observed_fraction(None, FULL_TRIANGLE))
    assert np.isnan(region_observed_fraction(np.array([[0.0, 0.0], [1.0, 1.0]]), FULL_TRIANGLE))


def test_a_zero_area_region_is_NaN_because_the_question_has_no_denominator():
    collinear = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    assert np.isnan(region_observed_fraction(LEFT_HALF, collinear))


def test_a_concave_region_is_REFUSED_not_silently_mis_clipped():
    """Sutherland-Hodgman needs a convex CLIP; a concave one returns a wrong area, not an error.

    So the seam raises. This is the same principle as the goal map's degeneracy guard: a
    plausible number from an inapplicable algorithm is worse than a refusal.
    """
    concave = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [5.0, 2.0], [0.0, 10.0]])
    with pytest.raises(ValueError, match="not convex"):
        region_observed_fraction(LEFT_HALF, concave)


def test_orientation_of_the_region_ring_does_not_change_the_answer():
    """A clockwise region must clip identically to a counter-clockwise one.

    Sign conventions in a hand-rolled clipper are exactly where an orientation assumption hides,
    and provider/consumer polygons carry no orientation guarantee.
    """
    ccw = region_observed_fraction(LEFT_HALF, FULL_TRIANGLE)
    cw = region_observed_fraction(LEFT_HALF, FULL_TRIANGLE[::-1])
    assert cw == pytest.approx(ccw, abs=1e-12)


def test_a_NON_CONVEX_observed_polygon_is_clipped_correctly():
    """The convexity requirement is on the REGION, never on the provider's polygon.

    That asymmetry is the reason the clipper takes the polygon as SUBJECT: broadcast shapes are
    outside our control, regions are not.
    """
    notched = np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [10.0, 2.0], [0.0, 20.0]])
    box = np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0]])
    frac = region_observed_fraction(notched, box)
    assert 0.0 < frac < 1.0
    # Area of the notched polygon by shoelace = 220; box = 400.
    assert frac == pytest.approx(220.0 / 400.0, abs=1e-9)


# ---------------------------------------------------------------------------
# add_visible_area_coverage
# ---------------------------------------------------------------------------


def test_aggregator_tokens_and_NaN_policy():
    """One action per token, and the fraction is NaN for every non-``observed`` one."""
    actions = _actions(4)
    visible = pd.DataFrame(
        {
            "action_id": [0, 1],
            "polygon": [LEFT_HALF, np.array([[0.0, 0.0], [1.0, 1.0]])],
        }
    )
    links = pd.DataFrame({"action_id": [0, 1, 2], "frame_id": [10, 11, 12]})

    out = add_visible_area_coverage(actions, visible_area=visible, links=links)
    src = dict(zip(out["action_id"], out["visible_area_source"], strict=True))
    assert src == {
        0: VISIBLE_AREA_OBSERVED,
        1: VISIBLE_AREA_DEGENERATE_POLYGON,
        2: VISIBLE_AREA_NO_POLYGON,
        3: VISIBLE_AREA_UNLINKED,
    }

    frac = dict(zip(out["action_id"], out["visible_area_fraction"], strict=True))
    assert frac[0] == pytest.approx(0.5, abs=1e-9), "the left half is half the pitch"
    for aid in (1, 2, 3):
        assert np.isnan(frac[aid]), f"action {aid} is not observed, so the fraction must be NaN"


def test_the_source_vocabulary_is_closed_and_every_token_is_reachable():
    """A vocabulary nothing can emit is a comment; one the aggregator can exceed is unenforced.

    The test above reaches four tokens on one call, so this pins the set itself and that the
    emitted values are a subset of it.
    """
    assert set(VISIBLE_AREA_SOURCE_VALUES) == {
        VISIBLE_AREA_OBSERVED,
        VISIBLE_AREA_NO_POLYGON,
        VISIBLE_AREA_DEGENERATE_POLYGON,
        VISIBLE_AREA_UNLINKED,
    }
    actions = _actions(4)
    visible = pd.DataFrame({"action_id": [0, 1], "polygon": [LEFT_HALF, np.array([[0.0, 0.0], [1.0, 1.0]])]})
    links = pd.DataFrame({"action_id": [0, 1, 2], "frame_id": [10, 11, 12]})
    out = add_visible_area_coverage(actions, visible_area=visible, links=links)
    assert set(out["visible_area_source"]) <= set(VISIBLE_AREA_SOURCE_VALUES)


def test_without_links_an_action_is_no_polygon_rather_than_unlinked():
    """``unlinked`` is only representable when the caller supplied links.

    Claiming it without them would assert a link failure nobody checked for.
    """
    out = add_visible_area_coverage(_actions(2), visible_area=pd.DataFrame(columns=["action_id", "polygon"]))
    assert set(out["visible_area_source"]) == {VISIBLE_AREA_NO_POLYGON}


def test_a_polygon_past_the_touchline_is_CLIPPED_not_reported_above_one():
    """The provider polygon is deliberately unclipped on arrival (ADR-054 D5); the share is not.

    Non-vacuity: the unclipped area really is larger than the pitch here (the polygon is
    130 x 88 = 11440 m^2 against the pitch's 7140), so a missing clip would report 1.60.
    """
    big = np.array([[-10.0, -10.0], [120.0, -10.0], [120.0, 78.0], [-10.0, 78.0]])
    visible = pd.DataFrame({"action_id": [0], "polygon": [big]})
    out = add_visible_area_coverage(_actions(1), visible_area=visible)
    assert out["visible_area_fraction"].iloc[0] == pytest.approx(1.0, abs=1e-9)


def test_aggregator_is_pure():
    """ADR-033: no mutation of the caller's frames, and a NEW object back."""
    actions = _actions(2)
    before = actions.copy(deep=True)
    visible = pd.DataFrame({"action_id": [0], "polygon": [LEFT_HALF]})
    poly_before = visible["polygon"].iloc[0].copy()

    out = add_visible_area_coverage(actions, visible_area=visible)

    pd.testing.assert_frame_equal(actions, before)
    np.testing.assert_array_equal(visible["polygon"].iloc[0], poly_before)
    assert out is not actions


@pytest.mark.parametrize(
    ("actions_dtype", "join_dtype"),
    [("int64", "string"), ("string", "int64"), ("float64", "int64"), ("int64", "float64")],
)
def test_the_action_id_JOIN_is_dtype_invariant(actions_dtype, join_dtype):
    """ADR-019. ``action_id`` joins THREE separately-sourced frames here.

    ``actions`` is the caller's, ``visible_area`` comes from the provider port's
    ``shape_snapshots``, and ``links`` from ``link_actions_to_frames`` -- nothing guarantees they
    agree on dtype. A plain dict keyed on the raw id misses silently, and the miss is reported as
    ``no_polygon``, i.e. indistinguishable from a genuine absence. That is the exact confusion
    this module exists to remove, so it is the one join that must not have it.

    MEASURED before the fix: int64 ``actions`` against object ``visible_area`` reported
    ``no_polygon`` for EVERY row while every polygon had been supplied.
    """
    n = 3
    actions = pd.DataFrame({"action_id": pd.Series(range(n)).astype(actions_dtype), "team_id": [1] * n})
    visible = pd.DataFrame({"action_id": pd.Series(range(n)).astype(join_dtype), "polygon": [LEFT_HALF] * n})
    links = pd.DataFrame({"action_id": pd.Series(range(n)).astype(join_dtype), "frame_id": list(range(n))})

    out = add_visible_area_coverage(actions, visible_area=visible, links=links)
    assert set(out["visible_area_source"]) == {VISIBLE_AREA_OBSERVED}, (
        f"{actions_dtype} actions vs {join_dtype} join: {list(out['visible_area_source'])} -- a "
        "dtype skew must not read as a missing polygon"
    )
    assert out["visible_area_fraction"].tolist() == pytest.approx([0.5] * n)


def test_a_genuinely_absent_polygon_still_reads_no_polygon():
    """Non-vacuity for the test above: the fix must not make everything 'observed'.

    Without this, canonicalizing the key could be masking real absences and the dtype test would
    still pass.
    """
    actions = pd.DataFrame({"action_id": [0, 1], "team_id": [1, 1]})
    visible = pd.DataFrame({"action_id": [0], "polygon": [LEFT_HALF]})
    out = add_visible_area_coverage(actions, visible_area=visible)
    assert list(out["visible_area_source"]) == [VISIBLE_AREA_OBSERVED, VISIBLE_AREA_NO_POLYGON]


# ---------------------------------------------------------------------------
# classify_region_observation (Task 3): (fraction, source) over the closed set
# {observed, no_polygon, degenerate_polygon, degenerate_region}. `degenerate_region`
# NEVER raises -- a zero-area region-of-interest is a missing denominator, not an error.
# ---------------------------------------------------------------------------

_TRI = np.array([[0.0, 0.0], [105.0, 0.0], [0.0, 68.0]])  # convex region of interest


def test_classify_fully_observed():
    f, s = classify_region_observation(_TRI, _TRI)  # region == polygon
    assert f == 1.0 and s == "observed"


def test_classify_partial():
    f, s = classify_region_observation(LEFT_HALF, _TRI)  # left half observed
    assert 0.0 < f < 1.0 and s == "observed"


def test_classify_no_polygon():
    f, s = classify_region_observation(None, _TRI)
    assert np.isnan(f) and s == "no_polygon"


def test_classify_degenerate_polygon():
    two_vertices = np.array([[0.0, 0.0], [10.0, 10.0]])  # present but bounds no area
    f, s = classify_region_observation(two_vertices, _TRI)
    assert np.isnan(f) and s == "degenerate_polygon"


def test_classify_degenerate_region_never_raises():
    zero = np.array([[10.0, 10.0], [10.0, 10.0], [10.0, 10.0]])  # zero-area region
    f, s = classify_region_observation(_TRI, zero)
    assert np.isnan(f) and s == "degenerate_region"  # NOT a ValueError


def test_classify_all_sources_are_in_the_closed_set():
    assert set(REGION_OBSERVATION_SOURCE_VALUES) == {
        "observed",
        "no_polygon",
        "degenerate_polygon",
        "degenerate_region",
    }
    # It reuses the polygon tokens but is NOT the pinned VISIBLE_AREA set (adds degenerate_region,
    # drops the action<->frame `unlinked`, which the caller overlays).
    assert "unlinked" not in REGION_OBSERVATION_SOURCE_VALUES
    assert set(REGION_OBSERVATION_SOURCE_VALUES) != set(VISIBLE_AREA_SOURCE_VALUES)
