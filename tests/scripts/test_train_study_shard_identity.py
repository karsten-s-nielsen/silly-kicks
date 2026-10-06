"""D21 x 5c: a study-shard is resumed ONLY under the identity its sqlite store would resume under.

The parallel study path caches each study's frozen params to ``<tag>.study.json`` (upstream 5c). D21
keys every HPO store on ``objective_id`` -- objective class + commit + declared inputs + per-fold tag,
with a per-call nonce on a dirty tree -- and the trial budget decides what a resumed store returns. A
shard IS that store's result, so it must obey the same resume rule: otherwise a shard left by another
run (other code / corpus / ``--n-trials``) is served stale, and a dirty tree resumes across code it
cannot describe (C4). ``OptunaStrategy`` is replaced by a counting fake, so "HPO ran" is observable
without a real sweep.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("ruthless")
pytest.importorskip("xgboost")

from scripts import train_xcross_attempt as trx
from scripts import train_xshot_occurrence as trs

_CLEAN = {"commit": "abc123", "dirty": False, "tree_state": "clean"}
_DIRTY = {"commit": "abc123", "dirty": True, "tree_state": "dirty"}
_INPUTS = {"driver": "test-shard-identity", "match_ids": ["m0", "m1"]}
_TAG = "full_f0"

TRAINERS = pytest.mark.parametrize("tr", [trs, trx], ids=["xshot", "xcross"])


class _CountingStrategy:
    """Stands in for ``OptunaStrategy``: counts runs and returns params tagged with the run number."""

    runs = 0

    def __init__(self, cfg, seed):
        self.cfg = cfg

    def run(self, obj, backend):
        type(self).runs += 1
        params = {"max_depth": 2.0 + type(self).runs, "store_objective_id": self.cfg.store.objective_id}
        return SimpleNamespace(best=SimpleNamespace(candidate=SimpleNamespace(params=params)))


@pytest.fixture
def hpo(monkeypatch):
    import ruthless.strategies.optuna_ as optuna_mod

    _CountingStrategy.runs = 0
    monkeypatch.setattr(optuna_mod, "OptunaStrategy", _CountingStrategy)
    return _CountingStrategy


def _hpo(tr, root, *, prov=_CLEAN, n_trials=5, inputs=_INPUTS):
    X = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0]})
    y = np.array([0, 1, 0, 1])
    groups = np.array(["g0", "g0", "g1", "g1"])
    return tr._hpo_once(X, y, groups, root, _TAG, n_trials, objective_inputs=inputs, prov=prov, study_shard_dir=root)


@TRAINERS
def test_shard_resumes_under_the_same_identity(tr, hpo, tmp_path):
    first = _hpo(tr, tmp_path)
    again = _hpo(tr, tmp_path)
    assert hpo.runs == 1  # the second call was served from the shard
    assert again == first


@TRAINERS
def test_shard_records_the_identity_it_was_computed_under(tr, hpo, tmp_path):
    params = _hpo(tr, tmp_path, n_trials=7)
    shard = json.loads((tmp_path / f"{_TAG}.study.json").read_text(encoding="utf-8"))
    assert shard["objective_id"] == params["store_objective_id"]  # the id the sqlite store was keyed on
    assert shard["n_trials"] == 7
    assert shard["params"] == params


@TRAINERS
def test_shard_from_other_inputs_is_recomputed(tr, hpo, tmp_path):
    _hpo(tr, tmp_path)
    other = _hpo(tr, tmp_path, inputs={**_INPUTS, "match_ids": ["m0", "m1", "m2"]})  # a different corpus
    assert hpo.runs == 2
    assert other["max_depth"] == 4.0  # the recomputed params, not the stale shard's


@TRAINERS
def test_shard_from_other_commit_is_recomputed(tr, hpo, tmp_path):
    _hpo(tr, tmp_path)
    _hpo(tr, tmp_path, prov={**_CLEAN, "commit": "def456"})
    assert hpo.runs == 2


@TRAINERS
def test_shard_from_other_trial_budget_is_recomputed(tr, hpo, tmp_path):
    _hpo(tr, tmp_path, n_trials=5)
    _hpo(tr, tmp_path, n_trials=50)  # a larger --n-trials must not be answered by the 5-trial shard
    assert hpo.runs == 2


@TRAINERS
def test_dirty_tree_never_resumes_from_a_shard(tr, hpo, tmp_path):
    _hpo(tr, tmp_path, prov=_DIRTY)
    _hpo(tr, tmp_path, prov=_DIRTY)
    assert hpo.runs == 2  # each dirty call carries its own nonce (C4)


@TRAINERS
def test_identityless_shard_is_recomputed(tr, hpo, tmp_path):
    """A shard in the pre-D21 format (tag + params only) carries no identity, so it is never trusted."""
    (tmp_path / f"{_TAG}.study.json").write_text(
        json.dumps({"tag": _TAG, "params": {"max_depth": 99.0}}), encoding="utf-8"
    )
    params = _hpo(tr, tmp_path)
    assert hpo.runs == 1
    assert params["max_depth"] == 3.0


@pytest.mark.parametrize("tr", [trs, trx], ids=["xshot", "xcross"])
def test_dirty_tree_studies_never_reopen_each_others_store_real_strategy(tr, tmp_path):
    # The REAL ruthless store guard (no fake): two dirty-tree studies of the same tag in the same folder -- a dirty
    # parallel run's worker and its reduce -- must each open a fresh store. Reopening the worker's store under the
    # reduce's new nonce would raise "written for a different objective".
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((80, 3)), columns=["a", "b", "c"])
    y = (X["a"] > 0).to_numpy().astype(int)
    groups = np.arange(80) % 4
    for _ in range(2):
        tr._hpo_once(X, y, groups, tmp_path, _TAG, 1, objective_inputs=_INPUTS, prov=_DIRTY)
    assert len(list(tmp_path.glob(f"study_{_TAG}*.db"))) == 2  # non-vacuity: two distinct stores were written
