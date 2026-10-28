"""Spec r3 D10: the ghost_gk `importance` stage splits by column and stays byte-identical to the single call.

sklearn's `permutation_importance` draws ONE `random_seed` and permutes every column with a fresh
`RandomState(random_seed)`, so each column is an independent unit. Each unit calls sklearn's PRIVATE
`_calculate_permutation_scores` with the seed, scorer and `max_samples` derived as the public function derives
them; a `baseline` unit scores the unpermuted rows; the reduce applies sklearn's `baseline - permuted` arithmetic
over the columns in order. The compared artifact is the importance table (feature, mean, std).
"""

from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import pytest

from scripts import train_ghost_gk as T
from silly_kicks.tracking._ghost_gk import GHOST_GK_FEATURE_NAMES, GHOST_GK_FEATURE_NAMES_POSITION_ONLY, GhostGkModel


def test_permutation_private_api_signature_is_pinned():
    """The tripwire for the accepted private-API dependency (A plan review PLAN-R3-01, B review R3-07): a sklearn
    change to `_calculate_permutation_scores` fails HERE, before any numeric identity gate runs."""
    from sklearn.inspection._permutation_importance import _calculate_permutation_scores

    assert list(inspect.signature(_calculate_permutation_scores).parameters) == [
        "estimator",
        "X",
        "y",
        "sample_weight",
        "col_idx",
        "random_state",
        "n_repeats",
        "scorer",
        "max_samples",
    ]


@pytest.fixture(scope="module")
def fitted():
    """A small fitted ghost-GK model and its importance inputs (NaNs included, as in real corpora)."""
    rng = np.random.default_rng(3)
    n = 400
    features = pd.DataFrame(rng.standard_normal((n, len(GHOST_GK_FEATURE_NAMES))), columns=GHOST_GK_FEATURE_NAMES)
    features["phase"] = rng.integers(0, 3, n).astype(float)
    features.iloc[::17, 2] = np.nan
    labels = pd.DataFrame({"gk_x": rng.uniform(2, 20, n), "gk_y": rng.uniform(25, 45, n)})
    model = GhostGkModel(n_estimators=15).fit(features, labels)
    return model, features.columns, features.values, labels["gk_x"].values


def _reference_table(model, columns, x, y) -> list[list]:
    """The single-call table: sklearn's public `permutation_importance`, exactly as the stage called it."""
    from sklearn.inspection import permutation_importance

    pi = permutation_importance(
        T._importance_estimator(model, columns),
        x,
        y,
        scoring="neg_mean_absolute_error",
        n_repeats=5,
        random_state=42,
        n_jobs=1,
    )
    rows = sorted(
        zip(columns, pi.importances_mean, pi.importances_std, strict=True),  # type: ignore[reportAttributeAccessIssue]
        key=lambda r: -r[1],
    )
    return [[str(n), float(m), float(s)] for n, m, s in rows]


def _canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def test_importance_split_table_is_byte_identical_to_the_single_unit_table(fitted):
    model, columns, X, y = fitted
    est = T._importance_estimator(model, columns)
    baseline = T._importance_baseline(est, X, y)
    scores = [T._importance_column_scores(est, X, y, col) for col in range(X.shape[1])]
    split = T._importance_table(baseline, scores, [str(c) for c in columns])
    assert _canon(split) == _canon(_reference_table(model, columns, X, y))
    assert any(row[1] != 0.0 for row in split)  # non-vacuity: some column matters


def test_the_reduce_of_persisted_units_rebuilds_the_identical_table(fitted):
    """The units persist JSON (`_canon_json`); the reduce reads them back. Python's float repr round-trips, so the
    reduce over the persisted unit results gives the single-call table byte for byte."""
    from scripts._stages import _canon_json, int_key

    model, columns, X, y = fitted
    est = T._importance_estimator(model, columns)
    results = {T._IMPORTANCE_BASELINE: json.loads(_canon_json({"baseline": T._importance_baseline(est, X, y)}))}
    for col in reversed(range(X.shape[1])):  # completion order must not matter
        unit = {"column": str(columns[col]), "scores": T._importance_column_scores(est, X, y, col)}
        results[int_key(col)] = json.loads(_canon_json(unit))
    out = T._importance_reduce(results, ctx=None)
    assert out["skipped"] is False
    assert _canon(out["top"]) == _canon(_reference_table(model, columns, X, y))


@pytest.mark.parametrize(
    ("feature_set", "names"),
    [("faithful", GHOST_GK_FEATURE_NAMES), ("position_only", GHOST_GK_FEATURE_NAMES_POSITION_ONLY)],
)
def test_importance_has_a_baseline_unit_and_one_unit_per_corpus_column(feature_set, names):
    """The extraction width follows `feature_set` (26 faithful, 21 position_only), so the plan derives the units
    without reading the corpus."""
    import types

    from scripts._stages import int_key

    ctx = types.SimpleNamespace(params={"feature_set": feature_set, "skip_permutation_importance": False})
    assert T._importance_units(ctx) == [T._IMPORTANCE_BASELINE, *(int_key(i) for i in range(len(names)))]


def test_a_skipped_importance_is_one_unit_and_an_empty_table():
    import types

    ctx = types.SimpleNamespace(params={"feature_set": "faithful", "skip_permutation_importance": True})
    assert T._importance_units(ctx) == [T._IMPORTANCE_SKIPPED]
    assert T._importance_reduce({T._IMPORTANCE_SKIPPED: {"skipped": True}}, ctx=None) == {"skipped": True, "top": []}


def test_a_corpus_wider_than_the_feature_set_fails_loud():
    """A per-column unit indexes the corpus by position; a corpus whose width disagrees with the derived unit count
    (an extractor that widened) must refuse, never permute the wrong column or silently skip one."""
    with pytest.raises(RuntimeError, match="columns"):
        T._check_importance_width([*GHOST_GK_FEATURE_NAMES, "extra"], "faithful")
    T._check_importance_width(list(GHOST_GK_FEATURE_NAMES), "faithful")  # the exact width passes
