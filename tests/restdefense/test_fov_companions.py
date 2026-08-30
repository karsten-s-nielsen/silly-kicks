"""FOV-observability companions for rest defense (TF-60, ADR-077) -- opt-in, additive."""

import numpy as np
import pandas as pd

from silly_kicks.restdefense._compute import compute_rest_defense
from silly_kicks.restdefense._fov import FOV_SENSITIVE_COLUMNS
from silly_kicks.tracking import REGION_OBSERVATION_SOURCE_VALUES, VISIBLE_AREA_UNLINKED
from tests.restdefense._fixtures import make_rest_defense_fixture

_WHOLE_PITCH = np.array([[0.0, 0.0], [105.0, 0.0], [105.0, 68.0], [0.0, 68.0]])
# Action-LTR crop that keeps only x >= 35 (the attacking half), cutting out the rest-defense zone.
_CROP_ATTACKING = np.array([[35.0, 0.0], [105.0, 0.0], [105.0, 68.0], [35.0, 68.0]])
_VALID_SOURCES = set(REGION_OBSERVATION_SOURCE_VALUES) | {VISIBLE_AREA_UNLINKED}


def _visible_area(polygon):
    return pd.DataFrame({"action_id": [0, 1, 2, 3], "polygon": [polygon] * 4})


def _companion_cols():
    out = []
    for c in FOV_SENSITIVE_COLUMNS:
        out += [f"{c}_observed_fraction", f"{c}_observed_source"]
    return out


def test_companions_absent_without_visible_area():
    actions, frames = make_rest_defense_fixture()
    samples, _ = compute_rest_defense(actions, frames)
    for c in _companion_cols():
        assert c not in samples.columns


def test_primary_columns_byte_identical_with_and_without_visible_area():
    actions, frames = make_rest_defense_fixture()
    plain, _ = compute_rest_defense(actions, frames)
    withva, _ = compute_rest_defense(actions, frames, visible_area=_visible_area(_WHOLE_PITCH))
    pd.testing.assert_frame_equal(withva[plain.columns], plain)
    for c in _companion_cols():
        assert c in withva.columns


def test_full_coverage_is_fully_observed():
    actions, frames = make_rest_defense_fixture()
    samples, _ = compute_rest_defense(actions, frames, visible_area=_visible_area(_WHOLE_PITCH))
    resolved = samples[samples["rd_geometry_source"] == "resolved"]
    for c in FOV_SENSITIVE_COLUMNS:
        assert (resolved[f"{c}_observed_fraction"] == 1.0).all()
        assert (resolved[f"{c}_observed_source"] == "observed").all()


def test_cropped_region_is_partially_observed():
    actions, frames = make_rest_defense_fixture()
    samples, _ = compute_rest_defense(actions, frames, visible_area=_visible_area(_CROP_ATTACKING))
    a0 = samples[samples["action_id"] == 0].iloc[0]
    # band [0, 70] intersect [35, 105] = [35, 70] -> 35/70 = 0.5
    assert a0["rd_num_superiority_observed_fraction"] == 0.5
    assert a0["rd_num_superiority_observed_source"] == "observed"
    # danger zone [0, 24] intersect [35, 105] = empty -> observed BUT 0.0 (fully cropped out)
    assert a0["rd_zone_occupancy_observed_fraction"] == 0.0
    assert a0["rd_zone_occupancy_observed_source"] == "observed"


def test_missing_polygon_is_no_polygon_not_a_fabricated_fraction():
    actions, frames = make_rest_defense_fixture()
    va = pd.DataFrame({"action_id": [0], "polygon": [_WHOLE_PITCH]})  # only action 0 has a polygon
    samples, _ = compute_rest_defense(actions, frames, visible_area=va)
    a2 = samples[samples["action_id"] == 2].iloc[0]
    assert a2["rd_num_superiority_observed_source"] == "no_polygon"
    assert pd.isna(a2["rd_num_superiority_observed_fraction"])


def test_all_sources_in_vocabulary():
    actions, frames = make_rest_defense_fixture()
    samples, _ = compute_rest_defense(actions, frames, visible_area=_visible_area(_CROP_ATTACKING))
    for c in FOV_SENSITIVE_COLUMNS:
        seen = set(samples[f"{c}_observed_source"].dropna().unique())
        assert seen <= _VALID_SOURCES, f"{c}: {seen - _VALID_SOURCES} not in vocabulary"
