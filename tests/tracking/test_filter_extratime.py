"""Tests for the public calibration-labelled ``filter_extratime_frames`` helper.

ADR-010 §4 / spec §7: drops ET periods (3/4) for sampling/calibration ONLY,
with a UserWarning. Production must source ``home_team_start_left_extratime``
via ``require_et_direction`` instead of dropping ET.
"""

from __future__ import annotations

import pandas as pd
import pytest


def test_filter_extratime_drops_et_with_warning():
    from silly_kicks.tracking.utils import filter_extratime_frames

    f = pd.DataFrame({"period_id": [1, 1, 3, 4]})
    with pytest.warns(UserWarning, match="ET"):
        out = filter_extratime_frames(f, label="gs 1")
    assert set(out["period_id"]) == {1}


def test_filter_extratime_noop_without_et():
    from silly_kicks.tracking.utils import filter_extratime_frames

    f = pd.DataFrame({"period_id": [1, 2]})
    out = filter_extratime_frames(f, label="x")
    assert len(out) == 2


def test_filter_extratime_accepts_period_column_for_events():
    from silly_kicks.tracking.utils import filter_extratime_frames

    # Events-input shapes (Sportec/Metrica) use ``period`` not ``period_id``.
    f = pd.DataFrame({"period": [1, 2, 3], "x": [1.0, 2.0, 3.0]})
    with pytest.warns(UserWarning, match="ET"):
        out = filter_extratime_frames(f, label="sportec events")
    assert set(out["period"]) == {1, 2}


def test_filter_extratime_reexported_from_tracking():
    import silly_kicks.tracking as t

    assert hasattr(t, "filter_extratime_frames")
