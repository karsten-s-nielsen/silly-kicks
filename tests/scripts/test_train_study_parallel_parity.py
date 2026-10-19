"""Task 6 guard: an F1b study run on the shared-mmap corpus == the in-memory study, byte-identical.

ADR-128: the serial-vs-parallel ASSEMBLE equality (xshot and xcross) moved to
tests/scripts/test_parity_xshot_occurrence.py and tests/scripts/test_parity_xcross_attempt.py -- the 5c
study fan-out is gone, and every stage output is compared at W2N2 there.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")
pytest.importorskip("ruthless")

from scripts import train_xshot_occurrence as tr
from scripts._corpus_mmap import load_design_matrix, persist_design_matrix


@pytest.mark.slow
def test_study_from_shared_corpus_matches_in_memory_booster_bytes(tmp_path):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((400, 3)).astype(np.float64), columns=["a", "b", "c"])
    y = (X["a"] > 0).to_numpy().astype(np.int8)
    groups = np.arange(400) % 6

    s_params, s_booster = tr._fit_study_for_test(X, y, groups, "full_f0", n_trials=5, seed=42)

    p = tmp_path / "dm"
    persist_design_matrix(X, y, groups, p)
    Xs, ys, gs = load_design_matrix(p)
    p_params, p_booster = tr._fit_study_for_test(Xs, ys, gs, "full_f0", n_trials=5, seed=42)

    assert s_params == p_params
    assert s_booster.save_raw("json") == p_booster.save_raw("json")  # byte-identical model
    assert s_booster.feature_names == p_booster.feature_names == ["a", "b", "c"]
