"""TF-53 Rung-3a training kernels (scipy-MLE rho recovery + realized-goal extraction)."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from scripts.train_match_outcome_dependence import _match_tuples, fit_rho
from silly_kicks.match_outcome import apply_dependence, goal_count_pmf
from tests.match_outcome._helpers import BAD_TOUCH, FAIL, OWNGOAL, PASS, SHOT, SUCCESS, make_actions


def _fake_corpus(n_games: int = 6):
    """A tiny synthetic open-data corpus: ``(provider, match_id, actions, frames, home)`` tuples."""
    rng = np.random.default_rng(0)
    out = []
    for k in range(n_games):
        recs = []
        for xg in rng.uniform(0.1, 0.5, size=4):
            recs.append({"team_id": 10, "type_id": SHOT, "xg": float(xg), "result_id": SUCCESS if xg > 0.4 else FAIL})
        for xg in rng.uniform(0.1, 0.5, size=3):
            recs.append({"team_id": 20, "type_id": SHOT, "xg": float(xg), "result_id": SUCCESS if xg > 0.45 else FAIL})
        recs.append({"team_id": 10, "type_id": PASS})
        actions = make_actions(recs, game_id=k)
        out.append(("statsbomb", f"g{k}", actions, pd.DataFrame(), 10))
    return out


def _patch_open_corpus(monkeypatch, matches):
    """Redirect the public open-data loader to a synthetic corpus (no network)."""
    import scripts._sb_open_data as sod

    monkeypatch.setattr(sod, "all_open_competitions", lambda: [(1, 1)])

    def _fake_load(*, competition_id, season_id, match_ids=None, max_matches=None, preserve_native=()):
        yield from matches

    monkeypatch.setattr(sod, "load_open_data_matches", _fake_load)


def _simulate(rho_true: float, *, n: int = 1000, seed: int = 0) -> list[tuple]:
    rng = np.random.default_rng(seed)
    matches = []
    for _ in range(n):
        hxg = rng.uniform(0.05, 0.4, size=int(rng.integers(3, 10))).tolist()
        axg = rng.uniform(0.05, 0.4, size=int(rng.integers(3, 10))).tolist()
        joint = apply_dependence(goal_count_pmf(hxg), goal_count_pmf(axg), rho=rho_true)
        flat = joint.flatten()
        i, j = np.unravel_index(int(rng.choice(flat.size, p=flat / flat.sum())), joint.shape)
        matches.append((hxg, axg, int(i), int(j)))
    return matches


def test_fit_recovers_known_rho():
    assert abs(fit_rho(_simulate(0.1)) - 0.1) < 0.06


def test_fit_rho_zero_data():
    assert abs(fit_rho(_simulate(0.0))) < 0.06


def test_match_tuples_extracts_goals_and_xgs():
    fx = make_actions(
        [
            {"team_id": 10, "type_id": SHOT, "xg": 0.5, "result_id": SUCCESS},  # team10 goal
            {"team_id": 10, "type_id": SHOT, "xg": 0.2, "result_id": FAIL},  # team10 shot, no goal
            {"team_id": 20, "type_id": BAD_TOUCH, "result_id": OWNGOAL},  # team20 own goal -> team10 +1
            {"team_id": 20, "type_id": PASS},
        ]
    )
    (hxg, axg, hg, ag) = _match_tuples(fx, "xg")[0]
    assert sorted(hxg) == [0.2, 0.5]  # team10 (listed first) shot xgs
    assert list(axg) == []  # team20 has no shot
    assert hg == 2  # one scored shot + one opponent own goal
    assert ag == 0


@pytest.mark.slow
def test_train_main_writes_loadable_artifact(tmp_path, monkeypatch):
    """Does-it-run smoke for main() end-to-end on a synthetic corpus (no network; --allow-dirty)."""
    from scripts import train_match_outcome_dependence as train
    from silly_kicks.match_outcome import DependenceModel

    _patch_open_corpus(monkeypatch, _fake_corpus())
    out = tmp_path / "weights"
    monkeypatch.setattr(sys, "argv", ["train_match_outcome_dependence.py", "--out", str(out), "--allow-dirty"])
    train.main()

    m = DependenceModel.load(out)  # fail-closed load must accept the freshly written artifact
    assert abs(m.rho) < 1.0 and m.training_commit
    assert (out / "model.json").exists() and (out / "SHA256SUMS").exists()
