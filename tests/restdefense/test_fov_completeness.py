"""FOV completeness gate for restdefense (TF-60, ADR-077, restdefense-LOCAL).

The shared tracking FOV completeness gate excludes non-``add_*`` / boundary surfaces (verified in
``tests/tracking/test_fov_companions.py``), so restdefense carries its own: every FOV-sensitive
(count/region) Layer-1 column must be companioned when ``visible_area`` is supplied, or be listed in
``_OBSERVABILITY_EXEMPT`` with a reason. The anti-rot meta pins the partition against
``RD_LAYER1_COLUMNS`` so a NEW count/region column cannot ship un-companioned and unnoticed.
"""

import numpy as np
import pandas as pd

from silly_kicks.restdefense import RD_METRIC_COLUMNS
from silly_kicks.restdefense._compute import compute_rest_defense
from silly_kicks.restdefense._fov import _OBSERVABILITY_EXEMPT, FOV_SENSITIVE_COLUMNS
from silly_kicks.tracking import REGION_OBSERVATION_SOURCE_VALUES, VISIBLE_AREA_UNLINKED
from tests.restdefense._fixtures import make_rest_defense_fixture

_WHOLE_PITCH = np.array([[0.0, 0.0], [105.0, 0.0], [105.0, 68.0], [0.0, 68.0]])
_VALID_SOURCES = set(REGION_OBSERVATION_SOURCE_VALUES) | {VISIBLE_AREA_UNLINKED}


def test_partition_is_exact_and_disjoint():
    """META (anti-rot): companioned + exempt == RD_LAYER1_COLUMNS, disjoint, every exempt has a reason."""
    sensitive = set(FOV_SENSITIVE_COLUMNS)
    exempt = set(_OBSERVABILITY_EXEMPT)
    assert sensitive.isdisjoint(exempt), sensitive & exempt
    assert sensitive | exempt == set(RD_METRIC_COLUMNS), (
        f"uncovered: {set(RD_METRIC_COLUMNS) - (sensitive | exempt)}; "
        f"extra: {(sensitive | exempt) - set(RD_METRIC_COLUMNS)}"
    )
    for col, reason in _OBSERVABILITY_EXEMPT.items():
        assert isinstance(reason, str) and reason.strip(), f"{col} exempt without a reason"


def test_every_sensitive_column_is_companioned_when_visible_area_supplied():
    actions, frames = make_rest_defense_fixture()
    va = pd.DataFrame({"action_id": [0, 1, 2, 3], "polygon": [_WHOLE_PITCH] * 4})
    samples, _ = compute_rest_defense(actions, frames, visible_area=va)
    for col in FOV_SENSITIVE_COLUMNS:
        assert f"{col}_observed_fraction" in samples.columns, f"{col} missing _observed_fraction"
        assert f"{col}_observed_source" in samples.columns, f"{col} missing _observed_source"


def test_exempt_columns_are_never_companioned():
    actions, frames = make_rest_defense_fixture()
    va = pd.DataFrame({"action_id": [0, 1, 2, 3], "polygon": [_WHOLE_PITCH] * 4})
    samples, _ = compute_rest_defense(actions, frames, visible_area=va)
    for col in _OBSERVABILITY_EXEMPT:
        assert f"{col}_observed_fraction" not in samples.columns
        assert f"{col}_observed_source" not in samples.columns


def test_all_observed_sources_are_in_the_closed_vocabulary():
    actions, frames = make_rest_defense_fixture()
    # only action 0 has a polygon -> exercises observed + no_polygon
    va = pd.DataFrame({"action_id": [0], "polygon": [_WHOLE_PITCH]})
    samples, _ = compute_rest_defense(actions, frames, visible_area=va)
    for col in FOV_SENSITIVE_COLUMNS:
        seen = set(samples[f"{col}_observed_source"].dropna().unique())
        assert seen <= _VALID_SOURCES, f"{col}: {seen - _VALID_SOURCES} not in the vocabulary"
