"""The ghost-GK common keeper domain and its folds (spec 4.3). Pure; no I/O."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DomainReport:
    n_domain_keepers: int
    n_excluded_keepers: int
    underpowered: bool


def common_keeper_domain(
    keepers: np.ndarray, *, expansion_keepers: set[str], n_splits: int = 5
) -> tuple[np.ndarray, DomainReport]:
    """Baseline keepers MINUS anyone appearing in the 98 (spec 4.3).

    The two corpora have DIFFERENT keeper populations, so there is no shared domain to hold fixed
    unless we construct one. Courtois is in the WC2022 Gradient Sports corpus AND in 45 of the 98
    -- so a keeper the expanded model trained on could otherwise land in the baseline's TEST fold.

    Raises when ``expansion_keepers`` is empty: an inert exclusion is worse than none, because every
    downstream assertion still passes while the guard does nothing.
    """
    if not expansion_keepers:
        raise ValueError(
            "common_keeper_domain: expansion_keepers is empty -- the exclusion would be inert. "
            "Pass --expansion-keepers from the Stage-B run, or state explicitly that the corpora "
            "share no keepers (they do share at least Courtois)."
        )
    domain = np.array([str(k) not in expansion_keepers for k in keepers], dtype=bool)
    n_dom = len(set(keepers[domain].tolist()))
    n_exc = len(set(keepers.tolist())) - n_dom
    return domain, DomainReport(n_domain_keepers=n_dom, n_excluded_keepers=n_exc, underpowered=n_dom < n_splits * 2)


def keeper_folds(
    keepers: np.ndarray, domain: np.ndarray, *, n_splits: int = 5
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """GroupKFold by KEEPER -- not by match. The target IS keeper positioning, and half the new
    cohort's keeper-slots are three Real Madrid keepers, so match folds would let Courtois appear
    in both train and test."""
    from sklearn.model_selection import GroupKFold

    idx = np.flatnonzero(domain)
    cv = GroupKFold(n_splits=n_splits)
    for tr, te in cv.split(idx, groups=keepers[idx]):
        yield idx[tr], idx[te]
