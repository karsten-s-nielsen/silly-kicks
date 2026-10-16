"""TF-46 cluster_corner_routines: agglomerative routine grouping (raw structure)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.corners import RoutineParams, cluster_corner_routines


def _features():
    # two clearly-separated groups in 2-feature coefficient space + one NaN (short/unclear) row
    return pd.DataFrame(
        {
            "game_id": ["g1"] * 7,
            "corner_id": list(range(7)),
            "routine_feat_0": [5.0, 5.1, 4.9, 0.1, 0.0, 0.2, np.nan],
            "routine_feat_1": [0.1, 0.0, 0.2, 5.0, 5.1, 4.9, np.nan],
        }
    )


def test_two_groups_resolve_to_two_clusters():
    out = cluster_corner_routines(_features(), params=RoutineParams(n_clusters=2))
    valid = out[out["cluster_id"] != -1]
    assert valid["cluster_id"].nunique() == 2
    # the first three and the last three (valid) corners are in different clusters
    first_three = set(out.loc[0:2, "cluster_id"])
    next_three = set(out.loc[3:5, "cluster_id"])
    assert first_three.isdisjoint(next_three)


def test_nan_row_is_unclustered_kept():
    out = cluster_corner_routines(_features(), params=RoutineParams(n_clusters=2))
    assert out.iloc[6]["cluster_id"] == -1  # NaN-coeff corner kept, unclustered
    assert len(out) == 7


def test_columns_are_raw_structure_only():
    out = cluster_corner_routines(_features(), params=RoutineParams(n_clusters=2))
    assert list(out.columns) == ["game_id", "corner_id", "cluster_id"]  # no archetype name/rank columns


def test_pure_does_not_mutate():
    feats = _features()
    before = feats.copy(deep=True)
    cluster_corner_routines(feats, params=RoutineParams(n_clusters=2))
    pd.testing.assert_frame_equal(feats, before)
