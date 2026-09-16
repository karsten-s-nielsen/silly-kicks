"""Rung-3a Dixon-Coles dependence + fail-closed rho load (TF-53 Task 4)."""

from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np
import pytest

from silly_kicks.match_outcome import (
    DependenceModel,
    MatchOutcomeIntegrityError,
    MatchOutcomeParams,
    _dependence,
    apply_dependence,
    compute_match_outcome,
    dixon_coles_tau,
    goal_count_pmf,
)
from tests.match_outcome._helpers import SHOT, make_actions, rows_by


def _write_artifact(d: pathlib.Path, *, rho=0.05, tc="abc1234", tamper=False) -> None:
    (d / "model.json").write_text(
        json.dumps({"rho": rho, "training_commit": tc, "corpus": "statsbomb-open"}), encoding="utf-8"
    )
    sha = hashlib.sha256((d / "model.json").read_bytes()).hexdigest()
    if tamper:
        sha = "0" * 64
    (d / "SHA256SUMS").write_text(f"{sha}  model.json\n", encoding="utf-8")


def test_tau_matches_dixon_coles_reference():
    lam, mu, rho = 1.5, 1.2, 0.08
    assert dixon_coles_tau(0, 0, lam, mu, rho) == 1.0 - lam * mu * rho
    assert dixon_coles_tau(0, 1, lam, mu, rho) == 1.0 + lam * rho
    assert dixon_coles_tau(1, 0, lam, mu, rho) == 1.0 + mu * rho
    assert dixon_coles_tau(1, 1, lam, mu, rho) == 1.0 - rho
    assert dixon_coles_tau(2, 3, lam, mu, rho) == 1.0  # outside the low block
    for i in range(3):
        for j in range(3):
            assert dixon_coles_tau(i, j, lam, mu, 0.0) == 1.0  # rho=0 -> no correction


def test_apply_dependence_rho_zero_is_independent():
    home, away = goal_count_pmf([0.5, 0.3]), goal_count_pmf([0.4])
    np.testing.assert_allclose(apply_dependence(home, away, rho=0.0), np.outer(home, away), atol=1e-12)


def test_apply_dependence_sums_to_one():
    home, away = goal_count_pmf([0.6, 0.4, 0.2]), goal_count_pmf([0.5, 0.3])
    assert abs(apply_dependence(home, away, rho=0.1).sum() - 1.0) < 1e-12


def test_load_good_artifact(tmp_path):
    _write_artifact(tmp_path)
    m = DependenceModel.load(tmp_path)
    assert m.rho == 0.05 and m.training_commit == "abc1234"


def test_load_tampered_sha_raises(tmp_path):
    _write_artifact(tmp_path, tamper=True)
    with pytest.raises(MatchOutcomeIntegrityError, match="SHA-256"):
        DependenceModel.load(tmp_path)


def test_load_out_of_range_rho_raises(tmp_path):
    _write_artifact(tmp_path, rho=1.5)
    with pytest.raises(MatchOutcomeIntegrityError, match="range"):
        DependenceModel.load(tmp_path)


def test_load_missing_training_commit_raises(tmp_path):
    (tmp_path / "model.json").write_text(json.dumps({"rho": 0.05}), encoding="utf-8")
    sha = hashlib.sha256((tmp_path / "model.json").read_bytes()).hexdigest()
    (tmp_path / "SHA256SUMS").write_text(f"{sha}  model.json\n", encoding="utf-8")
    with pytest.raises(MatchOutcomeIntegrityError, match="training_commit"):
        DependenceModel.load(tmp_path)


def test_load_missing_files_raises(tmp_path):
    with pytest.raises(MatchOutcomeIntegrityError, match="no dependence artifact"):
        DependenceModel.load(tmp_path)


def _two_team():
    recs = [{"team_id": 10, "type_id": SHOT, "xg": x} for x in (0.5, 0.3)]
    recs += [{"team_id": 20, "type_id": SHOT, "xg": x} for x in (0.4, 0.2)]
    return make_actions(recs)


def test_compute_dixon_coles_fails_closed_without_bundled_weights():
    # no bundled artifact under weights/ -> dixon_coles must RAISE, never silently fall back to independent
    p = MatchOutcomeParams(team_dependence="dixon_coles")
    if (pathlib.Path(_dependence.__file__).resolve().parent / "weights" / "model.json").exists():
        pytest.skip("bundled weights present (commit 2); fail-closed-absent path not exercisable")
    with pytest.raises(MatchOutcomeIntegrityError):
        compute_match_outcome(_two_team(), xg_column="xg", params=p)


def test_compute_dixon_coles_and_collapse_compose(monkeypatch):
    monkeypatch.setattr(_dependence, "resolve_rho", lambda params: 0.1)
    indep = rows_by(compute_match_outcome(_two_team(), xg_column="xg")[0])
    dc = rows_by(
        compute_match_outcome(_two_team(), xg_column="xg", params=MatchOutcomeParams(team_dependence="dixon_coles"))[0]
    )
    both = rows_by(
        compute_match_outcome(
            _two_team(),
            xg_column="xg",
            params=MatchOutcomeParams(same_possession="collapse", team_dependence="dixon_coles"),
        )[0]
    )
    assert abs(dc[10]["p_draw"] - indep[10]["p_draw"]) > 1e-4  # dependence moves the draw
    for r in (dc, both):  # both corrections yield a valid simplex
        r10 = r[10]
        assert abs(r10["p_win"] + r10["p_draw"] + r10["p_loss"] - 1.0) < 1e-12


@pytest.mark.skipif(
    not (pathlib.Path(_dependence.__file__).resolve().parent / "weights" / "model.json").exists(),
    reason="bundled rho artifact not present until commit 2",
)
def test_bundled_dependence_serves_when_present():
    m = DependenceModel.bundled()
    assert abs(m.rho) < 1.0 and m.training_commit
    s, _ = compute_match_outcome(_two_team(), xg_column="xg", params=MatchOutcomeParams(team_dependence="dixon_coles"))
    r10 = rows_by(s)[10]
    for col in ("p_win", "p_draw", "p_loss"):
        assert 0.0 <= r10[col] <= 1.0
