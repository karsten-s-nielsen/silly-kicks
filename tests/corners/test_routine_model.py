"""TF-46 CornerRoutineModel (NMF): fit/transform, round-trip, no-sklearn load, elbow."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from silly_kicks.corners._routines import (
    CornerRoutineModel,
    IntegrityError,
    _routine_feature_names,
    compute_corner_routine_features,
)

_BASIS = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])  # two run-combinations over 4 runs


def _term_matrix(n=60, seed=0):
    rng = np.random.default_rng(seed)
    coeffs = rng.random((n, 2)) * 2.0
    return coeffs @ _BASIS


def _toy_model(rank=2) -> CornerRoutineModel:
    # alpha=0.0 is the DEFAULT (plain Frobenius NMF); passed explicitly here only to pin the mechanics
    # fixtures against a future default change. The default-path regression guard is
    # test_default_alpha_reconstruction_is_non_degenerate below.
    return CornerRoutineModel().fit(_term_matrix(), rank=rank, alpha=0.0)


def test_fit_reconstructs_rank2_structure():
    m = _toy_model(rank=2)
    x = _term_matrix()
    w = m.transform(x)
    rel = np.linalg.norm(x - w @ m.basis) / np.linalg.norm(x)
    assert rel < 0.1  # rank-2 basis reconstructs rank-2 data well
    assert w.shape == (len(x), 2)


def test_round_trip_transforms_identically(tmp_path):
    m = _toy_model()
    x = _term_matrix(seed=1)
    w0 = m.transform(x)
    m.save(tmp_path)
    w1 = CornerRoutineModel.load(tmp_path).transform(x)
    np.testing.assert_allclose(w0, w1)


def test_missing_sha_raises(tmp_path):
    with pytest.raises(IntegrityError):
        CornerRoutineModel.load(tmp_path)


def test_sha_tamper_raises(tmp_path):
    m = _toy_model()
    m.save(tmp_path)
    npz = tmp_path / "model.npz"
    npz.write_bytes(npz.read_bytes() + b"\x00")
    with pytest.raises(IntegrityError):
        CornerRoutineModel.load(tmp_path)


def test_load_imports_no_sklearn(tmp_path, monkeypatch):
    m0 = _toy_model()
    x = _term_matrix(seed=2)
    expected = m0.transform(x)
    m0.save(tmp_path)
    monkeypatch.setitem(sys.modules, "sklearn", None)
    monkeypatch.setitem(sys.modules, "sklearn.decomposition", None)
    m = CornerRoutineModel.load(tmp_path)
    np.testing.assert_allclose(m.transform(x), expected)


def test_elbow_selects_small_rank_for_low_rank_data():
    m = CornerRoutineModel().fit(_term_matrix(), rank=None, rank_range=(2, 4), alpha=0.0)
    assert m.rank is not None and 2 <= m.rank <= 4  # planted rank-2 data -> elbow at a small rank
    assert "reconstruction" in m.selected


def test_default_alpha_reconstruction_is_non_degenerate():
    # Regression guard (TF46-IMPL-05): the DEFAULT alpha must give a non-degenerate NMF. The old default
    # alpha=1.0 with l1_ratio=1.0 crushed W,H -> 0 on an occupancy term matrix, so the reconstruction
    # error equalled ||X||_F CONSTANT across every rank (a dead rank-2 basis -- the ADR-032 / PR-S118
    # class). Fit a non-trivial (full-rank) matrix with the DEFAULT alpha (NOT an explicit 0.0) over a
    # rank range; the curve must be non-constant and net-decreasing. This RED-greens the guard:
    # alpha=1.0 -> constant -> both asserts fail; alpha=0.0 -> decreasing -> both pass.
    rng = np.random.default_rng(0)
    term = rng.random((40, 12))  # full-rank, non-negative -> error must fall as rank grows
    m = CornerRoutineModel().fit(term, rank_range=(2, 6))  # DEFAULT alpha
    vals = [m.selected["reconstruction"][k] for k in sorted(m.selected["reconstruction"], key=int)]
    assert len(set(round(v, 6) for v in vals)) > 1, "reconstruction constant across ranks (dead basis)"
    assert vals[0] > vals[-1], "reconstruction did not decrease with rank (over-regularised collapse)"


def test_compute_routine_features_handles_nan_rows():
    m = _toy_model()
    enc = pd.DataFrame(
        {
            "game_id": ["g1", "g1"],
            "corner_id": [0, 1],
            "run_1a": [1.0, np.nan],
            "run_1b": [0.0, np.nan],
            "run_2a": [1.0, np.nan],
            "run_2b": [0.0, np.nan],
        }
    )
    out = compute_corner_routine_features(enc, model=m)
    feat_cols = _routine_feature_names(m)
    assert list(out.columns) == ["game_id", "corner_id", *feat_cols]
    assert out.iloc[0][feat_cols].notna().all()  # valid row transformed
    assert out.iloc[1][feat_cols].isna().all()  # NaN run row -> NaN coeffs kept
