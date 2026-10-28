"""Permutation importance split into units (spec r3 D10; HFU plan R3-8 S3 item 8 (c)).

sklearn's `permutation_importance` draws ONE `random_seed` from `check_random_state(random_state)` and permutes
every column with a fresh `RandomState(random_seed)` (`_permutation_importance.py`, sklearn 1.7.2 and 1.9.0), so
each column is an independent unit: `column_scores` calls sklearn's PRIVATE `_calculate_permutation_scores` with
the seed, scorer and `max_samples` derived exactly as the public function derives them; `baseline` scores the
unpermuted rows; `importances` is `_create_importances_bunch`'s arithmetic. The units' results combine to the
public call's numbers byte for byte (`test_ghost_importance_split.py`, `test_xcross_tf19_units.py`); the private
signature is pinned (`test_permutation_private_api_signature_is_pinned`), a maintenance cost the owner's review
accepted. Users: `train_ghost_gk` `importance` and `train_xcross_attempt` `tf19_perm`.
"""

from __future__ import annotations

import numpy as np


def prepared(estimator, x, scoring: str):
    """``(x, scorer)`` exactly as `permutation_importance` prepares them for an ndarray ``x``."""
    from sklearn.metrics import check_scoring
    from sklearn.utils import check_array

    # sklearn's inline types say bool / str; its runtime takes "allow-nan" and None (permutation_importance does)
    x = check_array(x, ensure_all_finite="allow-nan", dtype=None)  # type: ignore[arg-type]
    scorer = check_scoring(estimator, scoring=scoring)
    if scorer is None:
        raise RuntimeError(f"permutation importance: sklearn returned no scorer for {scoring!r}")
    return x, scorer


def baseline(estimator, x, y, *, scoring: str) -> float:
    """The unpermuted score (`permutation_importance`'s ``baseline_score``; no sample weights)."""
    x, scorer = prepared(estimator, x, scoring)
    score = scorer(estimator, x, y)
    if isinstance(score, dict):  # a multi-metric scorer; the units use ONE metric
        raise TypeError(f"permutation importance: {scoring!r} returned several scores")
    return float(score)


def column_scores(estimator, x, y, col: int, *, scoring: str, random_state: int, n_repeats: int) -> list[float]:
    """Column ``col``'s permuted scores, one per repeat, exactly as `permutation_importance` computes them."""
    from sklearn.inspection._permutation_importance import _calculate_permutation_scores
    from sklearn.utils import check_random_state

    x, scorer = prepared(estimator, x, scoring)
    random_seed = check_random_state(random_state).randint(np.iinfo(np.int32).max + 1)
    max_samples = int(1.0 * x.shape[0])  # the public default max_samples=1.0
    scores = _calculate_permutation_scores(estimator, x, y, None, col, random_seed, n_repeats, scorer, max_samples)
    return [float(s) for s in scores]


def importances(baseline_score: float, scores) -> np.ndarray:
    """`_create_importances_bunch`'s ``baseline - permuted``: one row per column (in column order), one value per
    repeat. Its ``importances_mean`` is ``np.mean(..., axis=1)``."""
    return baseline_score - np.array(scores)
