"""Tests for ``silly_kicks.spadl.coverage_metrics`` (added in 1.10.0).

Mirrors the PR-S8 ``boundary_metrics`` test discipline. Coverage is measured
on a SPADL action stream, resolving ``type_id`` to action-type name via
``spadlconfig.actiontypes_df`` and reporting per-type counts plus any
``expected_action_types`` that produced zero rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl.utils import CoverageMetrics, coverage_metrics

_ACT = spadlconfig.actiontype_id


def _df_one(action_type: str = "pass") -> pd.DataFrame:
    """Single-row SPADL-shaped DataFrame with one action of the given type."""
    return pd.DataFrame({"type_id": [_ACT[action_type]]})


def _df_many(action_types: list[str]) -> pd.DataFrame:
    """SPADL-shaped DataFrame with one row per supplied action type name."""
    return pd.DataFrame({"type_id": [_ACT[t] for t in action_types]})


class TestCoverageMetricsContract:
    def test_returns_dict_with_required_keys(self):
        m = coverage_metrics(actions=_df_one("pass"))
        assert set(m.keys()) == {"counts", "missing", "total_actions"}

    def test_counts_value_type_is_int(self):
        m = coverage_metrics(actions=_df_many(["pass", "pass", "shot"]))
        for v in m["counts"].values():
            assert isinstance(v, int)

    def test_total_actions_is_int(self):
        m = coverage_metrics(actions=_df_many(["pass", "shot"]))
        assert isinstance(m["total_actions"], int)
        assert m["total_actions"] == 2

    def test_keyword_only_args_required(self):
        with pytest.raises(TypeError):
            coverage_metrics(_df_one("pass"))  # type: ignore[misc]

    def test_missing_type_id_column_raises_value_error(self):
        actions = pd.DataFrame({"team_id": [1, 2]})
        with pytest.raises(ValueError, match=r"type_id"):
            coverage_metrics(actions=actions)

    def test_returns_typeddict_shape(self):
        m: CoverageMetrics = coverage_metrics(actions=_df_one("pass"))
        # TypedDict access pattern
        assert m["counts"]["pass"] == 1


class TestCoverageMetricsCorrectness:
    def test_single_type_dataframe_counts_correctly(self):
        m = coverage_metrics(actions=_df_many(["pass", "pass", "pass"]))
        assert m["counts"] == {"pass": 3}
        assert m["total_actions"] == 3

    def test_multi_type_dataframe_counts_correctly(self):
        m = coverage_metrics(actions=_df_many(["pass", "pass", "shot", "tackle"]))
        assert m["counts"] == {"pass": 2, "shot": 1, "tackle": 1}
        assert m["total_actions"] == 4

    def test_expected_fully_present_missing_empty(self):
        m = coverage_metrics(
            actions=_df_many(["pass", "shot", "tackle"]),
            expected_action_types={"pass", "shot", "tackle"},
        )
        assert m["missing"] == []

    def test_expected_partially_absent_missing_sorted(self):
        m = coverage_metrics(
            actions=_df_many(["pass", "shot"]),
            expected_action_types={"pass", "shot", "tackle", "keeper_save"},
        )
        assert m["missing"] == ["keeper_save", "tackle"]

    def test_unknown_type_id_reported_as_unknown(self):
        # type_id = 999 is not in the spadlconfig.actiontype_id reverse map.
        actions = pd.DataFrame({"type_id": [_ACT["pass"], 999, 999]})
        m = coverage_metrics(actions=actions)
        assert m["counts"].get("pass") == 1
        assert m["counts"].get("unknown") == 2

    def test_counts_dict_is_deterministic(self):
        # Either alphabetical or first-seen ordering is acceptable, as long
        # as it is deterministic. Verify determinism by repeating.
        df = _df_many(["shot", "pass", "tackle"])
        m1 = coverage_metrics(actions=df)
        m2 = coverage_metrics(actions=df)
        assert list(m1["counts"].keys()) == list(m2["counts"].keys())


class TestCoverageMetricsDegenerate:
    def test_empty_dataframe_returns_zeros(self):
        actions = pd.DataFrame({"type_id": pd.Series([], dtype=np.int64)})
        m = coverage_metrics(actions=actions)
        assert m["counts"] == {}
        assert m["missing"] == []
        assert m["total_actions"] == 0

    def test_empty_with_expected_returns_all_missing_sorted(self):
        actions = pd.DataFrame({"type_id": pd.Series([], dtype=np.int64)})
        m = coverage_metrics(
            actions=actions,
            expected_action_types={"pass", "shot", "keeper_save"},
        )
        assert m["counts"] == {}
        assert m["missing"] == ["keeper_save", "pass", "shot"]
        assert m["total_actions"] == 0

    def test_expected_action_types_none_returns_empty_missing(self):
        m = coverage_metrics(
            actions=_df_many(["pass", "shot"]),
            expected_action_types=None,
        )
        assert m["missing"] == []

    def test_does_not_mutate_input(self):
        actions = _df_many(["pass", "shot"])
        cols_before = list(actions.columns)
        len_before = len(actions)
        coverage_metrics(actions=actions, expected_action_types={"pass"})
        assert list(actions.columns) == cols_before
        assert len(actions) == len_before
