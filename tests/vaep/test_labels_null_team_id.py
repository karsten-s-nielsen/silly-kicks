"""VAEP labels must survive a NULL `team_id`, and must not invent an opponent when they see one.

ADR-027 is explicit that Gradient Sports emits null-actor rows (`OTB`+`CH`, `FOUL`+`FO`) whose
`team_id` is NULL, as nullable `Int64` carrying NA rather than a sentinel `0` -- precisely so a
non-NaN sentinel cannot bypass `pd.isna` routing downstream. The label functions compared ids with a
raw `==`, which ADR-019 forbids at consumer seams, and that produced THREE distinct failures
depending on the shape of the comparison:

1. **Series-vs-Series** (`_scores_action`, `_concedes_action`, and the xG variants): `team_id ==
   shifted_team` is nullable-boolean, so a goal falling inside the window of a NULL-team row yields
   a `pd.NA` LABEL. `.to_numpy()` then gives an object array, and the calibration harness's
   `np.unique(y_scores[train_idx])` raises `TypeError: boolean value of NA is ambiguous`. Found
   exactly that way, three hours into a TF-24 Stage-2 run.
2. **Scalar-vs-scalar in a loop** (the possession variants): `if same_team:` on a `pd.NA` raises
   immediately -- the ADR-027 `_line_breaking.py` defect, same shape, different module.
3. **numpy array** (the time variants): `np.asarray(team_id.values)` turns nullable `Int64` into
   `float64` with `nan`, so `nan == nan` is False and `nan != nan` is True. This one does NOT crash.
   It silently reads a NULL-team row as an OPPONENT, which in `_concedes_time` counts it as a
   concede.

The third is the reason `~ids_equal(...)` is the wrong fix and `ids_differ(...)` is the right one: a
row with no team is neither the same team nor an opponent, and only `ids_differ` (which requires
BOTH ids present) expresses that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import silly_kicks.spadl.config as spadl
from silly_kicks.vaep.labels import concedes, scores


def _actions(team_ids, *, results, types=None, dtype="Int64") -> pd.DataFrame:
    """A minimal SPADL frame: one goal at the end, and a caller-chosen `team_id` column."""
    n = len(team_ids)
    return pd.DataFrame(
        {
            "game_id": [1] * n,
            "period_id": [1] * n,
            "action_id": list(range(n)),
            "time_seconds": [float(i) for i in range(n)],
            "team_id": pd.Series(team_ids, dtype=dtype),
            "player_id": pd.Series([10] * n, dtype="Int64"),
            "type_name": types or (["pass"] * (n - 1) + ["shot"]),
            "result_id": results,
        }
    )


_GOAL = spadl.result_id["success"]
_FAIL = spadl.result_id["fail"]


@pytest.mark.parametrize("label_fn", [scores, concedes])
def test_a_null_team_id_does_not_produce_an_NA_label(label_fn):
    """The crash shape. A NA label is neither True nor False -- it cannot be trained on, and the
    harness discovers it only when something sorts the array."""
    a = _actions([1, pd.NA, 2, 2], results=[_FAIL, _FAIL, _FAIL, _GOAL])
    out = label_fn(a, nr_actions=3)
    col = out.columns[0]
    assert not out[col].isna().any(), f"{label_fn.__name__} produced a pd.NA label on a NULL team_id"
    arr = out[col].to_numpy()
    np.unique(arr)  # the calibration harness does exactly this; object+NA raises here


@pytest.mark.parametrize("label_fn", [scores, concedes])
def test_labels_are_a_real_boolean_dtype_not_object(label_fn):
    """`to_numpy()` on a nullable-boolean carrying NA yields OBJECT, which is what reaches
    `np.unique`. Pinning the dtype catches the defect one step earlier than the crash."""
    a = _actions([1, pd.NA, 2, 2], results=[_FAIL, _FAIL, _FAIL, _GOAL])
    arr = label_fn(a, nr_actions=3)[lambda d: d.columns[0]].to_numpy()
    assert arr.dtype == bool, f"expected a bool array, got {arr.dtype} (NA leaked into the label)"


def test_a_null_team_row_is_not_treated_as_an_OPPONENT():
    """The silent shape, and the reason `ids_differ` is required rather than `~ids_equal`.

    A goal by a known team must not be charged as a CONCEDE to a row whose team is unknown. With
    `nan != nan -> True` (or `~ids_equal` treating NA as different), it is.
    """
    a = _actions([pd.NA, 1, 1], results=[_FAIL, _FAIL, _GOAL])
    out = concedes(a, nr_actions=3)
    assert not bool(out["concedes"].iloc[0]), "a NULL-team row was charged with the opponent's goal"


def test_a_null_team_row_is_not_credited_with_a_GOAL_either():
    """The mirror of the test above: unknown team means unknown, in both directions. Without both,
    a fix could satisfy one by flipping the default rather than by routing NA."""
    a = _actions([pd.NA, 1, 1], results=[_FAIL, _FAIL, _GOAL])
    out = scores(a, nr_actions=3)
    assert not bool(out["scores"].iloc[0]), "a NULL-team row was credited with another team's goal"


@pytest.mark.parametrize("label_fn", [scores, concedes])
def test_clean_int64_labels_are_UNCHANGED_by_the_na_routing(label_fn):
    """Non-vacuity + blast-radius bound in one: on a corpus with no NULL team the labels must be
    byte-identical to the plain-int64 result, or this is a retrain trigger for every provider
    rather than only for those carrying null-actor rows."""
    ids = [1, 1, 2, 2, 1]
    results = [_FAIL, _FAIL, _FAIL, _FAIL, _GOAL]
    plain = label_fn(_actions(ids, results=results, dtype="int64"), nr_actions=3)
    nullable = label_fn(_actions(ids, results=results, dtype="Int64"), nr_actions=3)
    pd.testing.assert_frame_equal(plain.astype(bool), nullable.astype(bool))


# --------------------------------------------------------------------------------------------
# The possession and time windows reach DIFFERENT comparison shapes (scalar-in-a-loop, and a
# numpy array), so the action-window tests above cannot speak for them.


@pytest.mark.parametrize("label_fn", [scores, concedes])
def test_possession_window_survives_a_null_team_id(label_fn):
    """Shape 2: `if same_team:` on a `pd.NA` raises outright -- the ADR-027 `_line_breaking.py`
    defect in a different module."""
    a = _actions([1, pd.NA, 2, 2], results=[_FAIL, _FAIL, _FAIL, _GOAL])
    a["possession_id"] = [1, 1, 1, 1]
    out = label_fn(a, nr_actions=10, window="possession")
    assert not out[out.columns[0]].isna().any()


@pytest.mark.parametrize("label_fn", [scores, concedes])
def test_time_window_survives_a_null_team_id(label_fn):
    a = _actions([1, pd.NA, 2, 2], results=[_FAIL, _FAIL, _FAIL, _GOAL])
    out = label_fn(a, nr_actions=10, window="time", window_seconds=15.0)
    assert not out[out.columns[0]].isna().any()


def test_time_window_does_not_charge_a_null_team_row_with_a_concede():
    """Shape 3, the SILENT one: `np.asarray` on nullable Int64 gives float64+nan, and `nan != nan`
    is True -- so a NULL-team row reads as an opponent and is charged with the goal. No exception,
    just a wrong label."""
    a = _actions([pd.NA, 1, 1], results=[_FAIL, _FAIL, _GOAL])
    out = concedes(a, nr_actions=10, window="time", window_seconds=15.0)
    assert not bool(out["concedes"].iloc[0]), "NULL-team row charged with another team's goal"


# --------------------------------------------------------------------------------------------
# `ids_equal`/`ids_differ` are POSITIONAL (fresh RangeIndex). Combining their output with a
# label-indexed Series silently changes LENGTH. This repo has been bitten three times.


@pytest.mark.parametrize("label_fn", [scores, concedes])
def test_a_non_rangeindex_frame_keeps_its_length_and_index(label_fn):
    """The regression this fix introduced and this test now pins: `ids_equal` returns a fresh
    RangeIndex, so `result | ids_equal(...)` LABEL-aligns and yields a UNION -- 410 rows out of a
    400-row frame. Any filtered or sliced caller has a non-0..n-1 index, which is the normal case
    downstream, not an exotic one."""
    a = _actions([1, 1, 2, 2, 1], results=[_FAIL, _FAIL, _FAIL, _FAIL, _GOAL])
    a.index = [100, 201, 302, 403, 504]  # non-monotonic, non-RangeIndex
    out = label_fn(a, nr_actions=3)
    assert len(out) == len(a), f"length changed: {len(out)} vs {len(a)} -- positional/label mismatch"
    assert list(out.index) == list(a.index), "the original index was not reattached"


@pytest.mark.parametrize("label_fn", [scores, concedes])
def test_a_sliced_frame_keeps_its_length(label_fn):
    """The skip-row shape the memory requires: >= 2 resolvable rows AFTER a filter."""
    full = _actions([1, 1, 2, 2, 1, 1], results=[_FAIL] * 5 + [_GOAL])
    sliced = full[full["team_id"].notna()].iloc[1:]
    out = label_fn(sliced, nr_actions=3)
    assert len(out) == len(sliced)
