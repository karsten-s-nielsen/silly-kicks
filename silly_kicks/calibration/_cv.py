"""Match-stratified cross-validation for the TF-24 calibration harness.

A single count-driven threshold (spec §2): GroupKFold(5) for >7 matches, leave-one-match-out
for <=7. Random-action splits are forbidden — they leak match structure into the held-out fold.

Examples
--------
Split a provider's actions into match-stratified folds::

    import numpy as np
    from silly_kicks.calibration._cv import match_cv_splits

    match_ids = np.array(["a", "a", "b", "b", "c"])
    for train_idx, test_idx in match_cv_splits(match_ids):
        ...  # no match appears in both train and test
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

_GROUPKFOLD_THRESHOLD = 7  # > this many matches => GroupKFold(5); else leave-one-match-out
_N_SPLITS = 5


def cv_scheme_for(n_matches: int) -> Literal["groupkfold", "lomo"]:
    """Return the CV scheme name for a provider with ``n_matches`` distinct matches.

    Examples
    --------
    >>> from silly_kicks.calibration._cv import cv_scheme_for
    >>> cv_scheme_for(7), cv_scheme_for(10)
    ('lomo', 'groupkfold')
    """
    return "groupkfold" if n_matches > _GROUPKFOLD_THRESHOLD else "lomo"


def match_cv_splits(match_ids: npt.NDArray) -> list[tuple[npt.NDArray, npt.NDArray]]:
    """Return (train_idx, test_idx) folds grouped by match (no match in both sides).

    GroupKFold(5) when the number of distinct matches exceeds the threshold, otherwise
    leave-one-match-out (one fold per match).

    Examples
    --------
    >>> import numpy as np
    >>> from silly_kicks.calibration._cv import match_cv_splits
    >>> folds = match_cv_splits(np.array(["a", "a", "b", "b", "c"]))
    >>> len(folds)
    3
    """
    n_matches = len(np.unique(match_ids))
    x = np.zeros((len(match_ids), 1))  # GroupKFold ignores X values, only needs shape
    if cv_scheme_for(n_matches) == "groupkfold":
        splitter: GroupKFold | LeaveOneGroupOut = GroupKFold(n_splits=_N_SPLITS)
    else:
        splitter = LeaveOneGroupOut()
    return [(tr, te) for tr, te in splitter.split(x, groups=match_ids)]


def cv_standard_error(fold_metrics: list[float]) -> float:
    """Standard error of the mean across CV folds: ``std(ddof=1) / sqrt(n_folds)``.

    Returns ``nan`` for a single fold (SE undefined).

    Examples
    --------
    >>> from silly_kicks.calibration._cv import cv_standard_error
    >>> round(cv_standard_error([0.04, 0.05, 0.06]), 6)
    0.005774
    """
    arr = np.asarray(fold_metrics, dtype=float)
    if len(arr) < 2:
        return float("nan")
    return float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
