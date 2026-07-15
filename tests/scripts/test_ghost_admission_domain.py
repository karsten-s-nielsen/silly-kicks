"""The four registered ghost-GK rules (spec 4.3). Each test names its kill-line."""

import numpy as np
import pytest
from _ghost_domain import common_keeper_domain, keeper_folds


def test_expansion_keepers_are_excluded_from_the_evaluation_domain():
    """KILL-LINE: delete the `k not in expansion` filter -> this MUST fail."""
    keepers = np.array(["alisson", "courtois", "neuer", "courtois"])
    domain, report = common_keeper_domain(keepers, expansion_keepers={"courtois"})
    assert list(domain) == [True, False, True, False]
    assert report.n_excluded_keepers == 1


def test_the_exclusion_is_NON_VACUOUS_on_the_real_overlap():
    """META-ASSERTION. The exclusion only matters if it actually removes someone. If a future
    refactor made `expansion_keepers` empty, every test above would still pass while the guard did
    nothing. This is the test that notices."""
    keepers = np.array(["alisson", "courtois", "neuer"])
    _, report = common_keeper_domain(keepers, expansion_keepers={"courtois"})
    assert report.n_excluded_keepers > 0, "the domain exclusion removed nobody -- it is inert"
    with pytest.raises(ValueError, match="empty"):
        common_keeper_domain(keepers, expansion_keepers=set())


def test_no_keeper_appears_in_both_train_and_test_folds():
    """KILL-LINE: swap GroupKFold for KFold -> this MUST fail.

    Keepers are INTERLEAVED (`i % 10`), not contiguous blocks (`i // 20`), on purpose: with
    contiguous blocks KFold's contiguous 40-row test folds align exactly with the 20-row keeper
    boundaries, so plain KFold is group-disjoint BY ACCIDENT and the kill-line would be inert
    (verified: contiguous -> KFold overlap=False; interleaved -> KFold overlap=True). Interleaving
    forces each keeper across the contiguous fold boundary so only a real GroupKFold keeps folds
    keeper-disjoint. Still 10 keepers, 20 rows each -- only the row ORDER changed.
    """
    keepers = np.array([f"gk{i % 10}" for i in range(200)])  # 10 keepers, 20 rows each, interleaved
    domain = np.ones(len(keepers), bool)
    for tr, te in keeper_folds(keepers, domain, n_splits=5):
        assert not (set(keepers[tr]) & set(keepers[te]))


def test_underpowered_domain_is_reported_not_interpreted():
    keepers = np.array(["a", "b", "c"])  # 3 keepers, 5 folds -> underpowered
    _, report = common_keeper_domain(keepers, expansion_keepers={"z"}, n_splits=5)
    assert report.underpowered is True
