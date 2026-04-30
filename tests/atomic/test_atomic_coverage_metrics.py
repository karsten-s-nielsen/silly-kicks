"""Tests for ``silly_kicks.atomic.spadl.coverage_metrics`` (added in 2.2.0).

Mirrors ``tests/spadl/test_coverage_metrics.py`` (PR-S6 / 1.10.0). Coverage
is measured on an atomic-SPADL action stream, resolving ``type_id`` to
action-type name via ``atomicspadl.actiontypes_df`` and reporting per-type
counts plus any ``expected_action_types`` that produced zero rows. The
``CoverageMetrics`` TypedDict is the standard one (single source of truth);
the atomic-side ``coverage_metrics`` function uses the atomic-vocabulary
33-type alphabet for resolution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.atomic.spadl import config as atomicspadl
from silly_kicks.atomic.spadl.utils import coverage_metrics
from silly_kicks.spadl.utils import CoverageMetrics

_ACT = atomicspadl.actiontype_id


def _df_one(action_type: str = "pass") -> pd.DataFrame:
    """Single-row atomic-SPADL-shaped DataFrame with one action of the given type."""
    return pd.DataFrame({"type_id": [_ACT[action_type]]})


def _df_many(action_types: list[str]) -> pd.DataFrame:
    """Atomic-SPADL-shaped DataFrame with one row per supplied action type name."""
    return pd.DataFrame({"type_id": [_ACT[t] for t in action_types]})


class TestAtomicCoverageMetricsContract:
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
        # The atomic coverage_metrics returns the standard CoverageMetrics
        # TypedDict (single source of truth — re-exported from
        # silly_kicks.spadl.utils).
        m: CoverageMetrics = coverage_metrics(actions=_df_one("pass"))
        assert m["counts"]["pass"] == 1


class TestAtomicCoverageMetricsCorrectness:
    def test_atomic_only_action_type_counted(self):
        # 'receival' is atomic-only — not present in standard SPADL vocabulary.
        m = coverage_metrics(actions=_df_many(["pass", "receival", "receival"]))
        assert m["counts"] == {"pass": 1, "receival": 2}
        assert m["total_actions"] == 3

    def test_collapsed_freekick_name_counted(self):
        # Atomic collapses freekick_short / freekick_crossed → 'freekick'.
        # Verify the post-collapse name resolves correctly.
        m = coverage_metrics(actions=_df_many(["freekick", "freekick", "shot"]))
        assert m["counts"] == {"freekick": 2, "shot": 1}

    def test_collapsed_corner_name_counted(self):
        # Atomic collapses corner_short / corner_crossed → 'corner'.
        m = coverage_metrics(actions=_df_many(["corner", "corner"]))
        assert m["counts"] == {"corner": 2}

    def test_expected_partially_absent_missing_sorted(self):
        m = coverage_metrics(
            actions=_df_many(["pass", "shot"]),
            expected_action_types={"pass", "shot", "tackle", "keeper_save"},
        )
        assert m["missing"] == ["keeper_save", "tackle"]

    def test_unknown_type_id_reported_as_unknown(self):
        # type_id = 999 is not in the atomic actiontype_id reverse map.
        actions = pd.DataFrame({"type_id": [_ACT["pass"], 999, 999]})
        m = coverage_metrics(actions=actions)
        assert m["counts"].get("pass") == 1
        assert m["counts"].get("unknown") == 2


class TestAtomicCoverageMetricsDegenerate:
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
            expected_action_types={"pass", "shot", "receival"},
        )
        assert m["counts"] == {}
        assert m["missing"] == ["pass", "receival", "shot"]
        assert m["total_actions"] == 0

    def test_does_not_mutate_input(self):
        actions = _df_many(["pass", "receival"])
        cols_before = list(actions.columns)
        len_before = len(actions)
        coverage_metrics(actions=actions, expected_action_types={"pass"})
        assert list(actions.columns) == cols_before
        assert len(actions) == len_before
