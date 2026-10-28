"""TF-53 Rung-3a training kernels (scipy-MLE rho recovery + realized-goal extraction)."""

from __future__ import annotations

import os

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
    """Redirect the public open-data SOURCE (Task 8.5 refs+load factory) to a synthetic corpus (no
    network): ``open_data_source`` returns one ref per synthetic match + a loader serving it. Shared:
    ``test_match_outcome_calibration`` drives its validator with it."""
    from _fake_corpus import make_ref

    import scripts._sb_open_data as sod

    monkeypatch.setattr(sod, "all_open_competitions", lambda: [(1, 1)])

    def _source(comps, *, match_ids=None, max_matches=None, preserve_native=(), cache_dir=None):
        by_id = {m[1]: m for m in matches}
        refs = [make_ref(m[0], m[1]) for m in matches]
        return refs, lambda ref: by_id[ref.match_id]

    monkeypatch.setattr(sod, "open_data_source", _source)


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
def test_pipeline_writes_a_loadable_artifact(tmp_path):
    """Does-it-run smoke for the pipeline end to end on the offline open-data tree (no network): the
    fail-closed ``DependenceModel.load`` accepts the bundle ``--run-all --out`` writes. The pipeline's units run
    in worker processes, so the corpus is the served toy tree, not an in-process patch; statsbombpy reads it,
    so this runs in the ``.venv-hfjobs`` legs and skips in CI."""
    pytest.importorskip("statsbombpy")
    from scripts._toy_corpora import build_open_data
    from silly_kicks.match_outcome import DependenceModel
    from tests.scripts._open_data_server import serve_open_data
    from tests.scripts._parity import run_serial

    out = tmp_path / "weights"
    with serve_open_data(build_open_data(tmp_path / "od")) as url:
        env = {**os.environ, "SILLY_KICKS_OPEN_DATA_BASE": url}
        run_serial("scripts.train_match_outcome_dependence", tmp_path / "s", tmp_path / "data", [], out, env=env)

    m = DependenceModel.load(out)  # fail-closed load must accept the freshly written artifact
    assert abs(m.rho) < 1.0 and m.training_commit
    assert sorted(p.name for p in out.iterdir()) == ["SHA256SUMS", "model.json"]


def test_reduce_over_shards_equals_in_memory_fit():
    """ADR-052 U-shape: the per-match slice + whole-corpus reduce fits the SAME rho as the
    pre-migration in-memory pass, and the fit is order-independent (a resumed/partitioned run
    reduces its shards in any order)."""
    from scripts.train_match_outcome_dependence import extract_match_slice, matches_from_shards

    corpus = _fake_corpus(8)
    in_memory: list[tuple] = []
    for _p, _m, actions, _f, _h in corpus:
        in_memory.extend(_match_tuples(actions, "xg"))

    shards = [extract_match_slice(actions, "xg") for _p, _m, actions, _f, _h in corpus]
    assert abs(fit_rho(matches_from_shards(shards)) - fit_rho(in_memory)) < 1e-9
    # order-independent: a reversed shard order fits the same rho
    assert abs(fit_rho(matches_from_shards(list(reversed(shards)))) - fit_rho(in_memory)) < 1e-9
