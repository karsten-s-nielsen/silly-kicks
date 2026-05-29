import numpy as np
import pytest

from silly_kicks.calibration._cv import cv_scheme_for, cv_standard_error, match_cv_splits


def test_cv_scheme_threshold():
    assert cv_scheme_for(7) == "lomo"  # IDSSE
    assert cv_scheme_for(8) == "groupkfold"
    assert cv_scheme_for(10) == "groupkfold"  # SkillCorner
    assert cv_scheme_for(64) == "groupkfold"  # Gradient Sports


def test_lomo_one_held_out_match_per_fold():
    match_ids = np.array(["a", "a", "b", "b", "c"])
    splits = match_cv_splits(match_ids)
    assert len(splits) == 3  # one fold per match
    for train_idx, test_idx in splits:
        held_out = set(match_ids[test_idx])
        assert len(held_out) == 1  # exactly one match held out
        assert held_out.isdisjoint(set(match_ids[train_idx]))  # no leakage


def test_groupkfold_5_for_many_matches():
    match_ids = np.array([f"m{i}" for i in range(10) for _ in range(3)])
    splits = match_cv_splits(match_ids)
    assert len(splits) == 5
    for train_idx, test_idx in splits:
        assert set(match_ids[test_idx]).isdisjoint(set(match_ids[train_idx]))


def test_standard_error_of_fold_means():
    # SE = std(fold_briers, ddof=1) / sqrt(n_folds)
    briers = [0.04, 0.05, 0.06]
    se = cv_standard_error(briers)
    assert se == pytest.approx(np.std(briers, ddof=1) / np.sqrt(3))


def test_standard_error_single_fold_is_nan():
    assert np.isnan(cv_standard_error([0.04]))
