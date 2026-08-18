"""Tests for add_action_context aggregator: provenance columns, NaN safety, dtypes."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def actions_and_frames_aggregator():
    """2 linked actions + 1 unlinked action (period 1, t=1000s --- no frame at 1000s)."""
    actions = pd.DataFrame(
        {
            "action_id": [101, 102, 999],
            "period_id": [1, 1, 1],
            "time_seconds": [10.0, 20.0, 1000.0],
            "team_id": [1, 1, 1],
            "player_id": [11, 11, 11],
            "start_x": [50.0, 60.0, 50.0],
            "start_y": [34.0, 30.0, 34.0],
            "end_x": [55.0, 65.0, 55.0],
            "end_y": [34.0, 30.0, 34.0],
        }
    )
    rows = []
    for fid, t in [(1000, 10.0), (2000, 20.0)]:
        rows.append(
            dict(
                game_id=1,
                period_id=1,
                frame_id=fid,
                time_seconds=t,
                frame_rate=25.0,
                player_id=11,
                team_id=1,
                is_ball=False,
                is_goalkeeper=False,
                x=50.0 if fid == 1000 else 60.0,
                y=34.0 if fid == 1000 else 30.0,
                z=float("nan"),
                speed=2.0,
                speed_source="native",
                ball_state="alive",
                team_attacking_direction="ltr",
                confidence=None,
                visibility=None,
                source_provider="test",
            )
        )
        rows.append(
            dict(
                game_id=1,
                period_id=1,
                frame_id=fid,
                time_seconds=t,
                frame_rate=25.0,
                player_id=22,
                team_id=2,
                is_ball=False,
                is_goalkeeper=False,
                x=(50.0 if fid == 1000 else 60.0) + 5.0,
                y=34.0 if fid == 1000 else 30.0,
                z=float("nan"),
                speed=1.0,
                speed_source="native",
                ball_state="alive",
                team_attacking_direction="ltr",
                confidence=None,
                visibility=None,
                source_provider="test",
            )
        )
    frames = pd.DataFrame(rows)
    return actions, frames


def test_add_action_context_returns_input_plus_8_columns(actions_and_frames_aggregator):
    from silly_kicks.tracking.features import add_action_context

    actions, frames = actions_and_frames_aggregator
    enriched = add_action_context(actions, frames)

    new_cols = set(enriched.columns) - set(actions.columns)
    expected = {
        "nearest_defender_distance",
        "actor_speed",
        "receiver_zone_density",
        "defenders_in_triangle_to_goal",
        "frame_id",
        "time_offset_seconds",
        "link_quality_score",
        "n_candidate_frames",
    }
    assert expected.issubset(new_cols)


def test_add_action_context_unlinked_action_has_nan_features(actions_and_frames_aggregator):
    from silly_kicks.tracking.features import add_action_context

    actions, frames = actions_and_frames_aggregator
    enriched = add_action_context(actions, frames)

    unlinked = enriched[enriched["action_id"] == 999].iloc[0]
    assert pd.isna(unlinked["nearest_defender_distance"])
    assert pd.isna(unlinked["actor_speed"])
    assert pd.isna(unlinked["frame_id"])


def test_add_action_context_linked_action_has_features(actions_and_frames_aggregator):
    from silly_kicks.tracking.features import add_action_context

    actions, frames = actions_and_frames_aggregator
    enriched = add_action_context(actions, frames)

    linked = enriched[enriched["action_id"] == 101].iloc[0]
    assert linked["nearest_defender_distance"] == 5.0
    assert linked["actor_speed"] == 2.0


def test_add_action_context_dtypes(actions_and_frames_aggregator):
    from silly_kicks.tracking.features import add_action_context

    actions, frames = actions_and_frames_aggregator
    enriched = add_action_context(actions, frames)

    assert enriched["nearest_defender_distance"].dtype == "float64"
    assert enriched["actor_speed"].dtype == "float64"
    assert enriched["receiver_zone_density"].dtype.name == "Int64"
    assert enriched["defenders_in_triangle_to_goal"].dtype.name == "Int64"


def test_add_action_context_is_nan_safe_decorated():
    """ADR-003 contract: add_action_context is in the auto-discovered NaN-safety registry."""
    from silly_kicks._nan_safety import is_nan_safe_enrichment
    from silly_kicks.tracking.features import add_action_context

    assert is_nan_safe_enrichment(add_action_context) is True


# ---------------------------------------------------------------------------
# Task 4: visibility companions (opt-in via visible_area=). Additive by construction --
# the three primary count columns are byte-identical with and without a polygon, and the six
# companion columns appear ONLY when visible_area is supplied.
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402

_WHOLE_PITCH = np.array([[0.0, 0.0], [105.0, 0.0], [105.0, 68.0], [0.0, 68.0]])
_LEFT_HALF = np.array([[0.0, 0.0], [52.5, 0.0], [52.5, 68.0], [0.0, 68.0]])

_PRIMARY = ["nearest_defender_distance", "actor_speed", "receiver_zone_density", "defenders_in_triangle_to_goal"]
_COMPANIONS = [
    "nearest_defender_distance_observed_fraction",
    "nearest_defender_distance_observed_source",
    "receiver_zone_density_observed_fraction",
    "receiver_zone_density_observed_source",
    "defenders_in_triangle_to_goal_observed_fraction",
    "defenders_in_triangle_to_goal_observed_source",
]

_VA_101_WHOLE_PITCH = pd.DataFrame({"action_id": [101], "polygon": [_WHOLE_PITCH]})


def _cell(out, aid, col):
    """Type-clean scalar extraction: pandas ``.loc[label, col]`` types as the ambiguous ``Scalar``
    (which includes ``complex``), so ``< float`` / ``float(...)`` / ``np.isnan(...)`` on it fail
    pyright. Round-tripping through numpy yields an ``Any`` the comparisons accept."""
    return out.loc[out.index == aid, col].to_numpy()[0]


def test_primary_columns_are_byte_identical_with_and_without_visible_area(actions_and_frames_aggregator):
    from silly_kicks.tracking.features import add_action_context

    actions, frames = actions_and_frames_aggregator
    base = add_action_context(actions, frames)
    withva = add_action_context(actions, frames, visible_area=_VA_101_WHOLE_PITCH)
    for c in _PRIMARY:
        pd.testing.assert_series_equal(base[c], withva[c])


def test_companions_appear_iff_visible_area_supplied(actions_and_frames_aggregator):
    from silly_kicks.tracking.features import add_action_context

    actions, frames = actions_and_frames_aggregator
    base = add_action_context(actions, frames)
    withva = add_action_context(actions, frames, visible_area=_VA_101_WHOLE_PITCH)
    assert not any(c in base.columns for c in _COMPANIONS), "companions must be absent without visible_area"
    assert all(c in withva.columns for c in _COMPANIONS), "companions must appear when visible_area supplied"


def test_companion_sources_are_in_the_closed_set(actions_and_frames_aggregator):
    from silly_kicks.tracking import REGION_OBSERVATION_SOURCE_VALUES, VISIBLE_AREA_UNLINKED
    from silly_kicks.tracking.features import add_action_context

    actions, frames = actions_and_frames_aggregator
    out = add_action_context(actions, frames, visible_area=_VA_101_WHOLE_PITCH)
    allowed = set(REGION_OBSERVATION_SOURCE_VALUES) | {VISIBLE_AREA_UNLINKED}
    for c in _COMPANIONS:
        if c.endswith("_source"):
            assert set(out[c]) <= allowed, f"{c} emitted outside the closed set: {set(out[c]) - allowed}"


def test_companion_tokens_observed_no_polygon_unlinked(actions_and_frames_aggregator):
    from silly_kicks.tracking.features import add_action_context

    actions, frames = actions_and_frames_aggregator
    out = add_action_context(actions, frames, visible_area=_VA_101_WHOLE_PITCH).set_index("action_id")
    # 101: linked, whole-pitch polygon -> observed with a real fraction in (0, 1].
    assert out.loc[101, "defenders_in_triangle_to_goal_observed_source"] == "observed"
    assert 0.0 < _cell(out, 101, "defenders_in_triangle_to_goal_observed_fraction") <= 1.0
    # 102: linked, NO polygon record -> no_polygon, NaN fraction.
    assert out.loc[102, "defenders_in_triangle_to_goal_observed_source"] == "no_polygon"
    assert np.isnan(_cell(out, 102, "defenders_in_triangle_to_goal_observed_fraction"))
    # 999: unlinked -> unlinked on every companion, NaN fraction.
    for c in _COMPANIONS:
        if c.endswith("_source"):
            assert out.loc[999, c] == "unlinked"
        else:
            assert np.isnan(_cell(out, 999, c))


def _single_actor_no_opponent():
    actions = pd.DataFrame(
        {
            "action_id": [1],
            "period_id": [1],
            "time_seconds": [10.0],
            "team_id": [1],
            "player_id": [11],
            "start_x": [50.0],
            "start_y": [34.0],
            "end_x": [55.0],
            "end_y": [34.0],
        }
    )
    frames = pd.DataFrame(
        [
            dict(
                game_id=1,
                period_id=1,
                frame_id=1000,
                time_seconds=10.0,
                frame_rate=25.0,
                player_id=11,
                team_id=1,
                is_ball=False,
                is_goalkeeper=False,
                x=50.0,
                y=34.0,
                z=float("nan"),
                speed=2.0,
                speed_source="native",
                ball_state="alive",
                team_attacking_direction="ltr",
                confidence=None,
                visibility=None,
                source_provider="test",
            )
        ]
    )
    return actions, frames


def test_nan_distance_nearest_defender_is_degenerate_region():
    """A linked action with NO opposing defender has a NaN nearest-defender distance, so its disk
    radius is undefined -> degenerate_region, never a fabricated fraction."""
    from silly_kicks.tracking.features import add_action_context

    actions, frames = _single_actor_no_opponent()
    va = pd.DataFrame({"action_id": [1], "polygon": [_WHOLE_PITCH]})
    out = add_action_context(actions, frames, visible_area=va).set_index("action_id")
    assert np.isnan(_cell(out, 1, "nearest_defender_distance"))  # precondition
    assert out.loc[1, "nearest_defender_distance_observed_source"] == "degenerate_region"
    assert np.isnan(_cell(out, 1, "nearest_defender_distance_observed_fraction"))


def test_partial_observation_yields_fraction_between_0_and_1(actions_and_frames_aggregator):
    """A left-half polygon over a triangle straddling midfield is partially observed."""
    from silly_kicks.tracking.features import add_action_context

    actions, frames = actions_and_frames_aggregator
    va = pd.DataFrame({"action_id": [101], "polygon": [_LEFT_HALF]})
    out = add_action_context(actions, frames, visible_area=va).set_index("action_id")
    frac = float(_cell(out, 101, "defenders_in_triangle_to_goal_observed_fraction"))
    assert 0.0 < frac < 1.0
    assert out.loc[101, "defenders_in_triangle_to_goal_observed_source"] == "observed"


def test_per_series_functions_and_default_xfns_untouched():
    """Calibration-path additive proof: per-Series functions keep their bare-Series signature and
    tracking_default_xfns is unchanged (no visible_area anywhere)."""
    from silly_kicks.tracking.features import (
        defenders_in_triangle_to_goal,
        nearest_defender_distance,
        receiver_zone_density,
        tracking_default_xfns,
    )

    actions, frames = _single_actor_no_opponent()
    assert isinstance(nearest_defender_distance(actions, frames), pd.Series)
    assert isinstance(receiver_zone_density(actions, frames), pd.Series)
    assert isinstance(defenders_in_triangle_to_goal(actions, frames), pd.Series)
    assert len(tracking_default_xfns) == 4  # unchanged; no companion transformer added


def test_inscribed_disk_under_reports_area_and_is_centered():
    """Inscribed => area < pi r^2 (the honesty invariant), and the vertices lie on the circle."""
    from silly_kicks._polygon import shoelace_area
    from silly_kicks.tracking._kernels import _inscribed_disk

    cx, cy, r, n = 10.0, 20.0, 3.0, 20
    disk = _inscribed_disk(cx, cy, r, n)
    assert disk.shape == (n, 2)
    # every vertex is exactly r from the centre
    radii = np.hypot(disk[:, 0] - cx, disk[:, 1] - cy)
    np.testing.assert_allclose(radii, r, atol=1e-12)
    # inscribed n-gon area = 0.5 n r^2 sin(2 pi / n), strictly below the circle's pi r^2
    area = abs(shoelace_area(disk))
    expected = 0.5 * n * r**2 * np.sin(2.0 * np.pi / n)
    np.testing.assert_allclose(area, expected, rtol=1e-9)
    assert area < np.pi * r**2
