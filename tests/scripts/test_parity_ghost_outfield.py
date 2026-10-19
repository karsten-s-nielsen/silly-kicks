"""ADR-128 W1N1-vs-W2N2 parity for `train_ghost_outfield` (extract fan-out; cv folds at W = 2)."""

from __future__ import annotations

import pytest

from scripts._toy_corpora import build_outfield_corpus
from scripts._train_guard import sklearn_supports_training
from tests.scripts._parity import (
    artifact,
    completion_out_of_order,
    outputs,
    plan,
    ran_concurrently,
    run_fanned,
    run_serial,
)

pytestmark = pytest.mark.skipif(
    not sklearn_supports_training(), reason="bundled-weights fit needs scikit-learn in [1.9, 2)"
)

MODULE = "scripts.train_ghost_outfield"
ARGS = ["--dataset", "toyof", "--variant", "default", "--n-estimators", "10", "--max-depth", "3", "--cv-folds", "4"]


@pytest.mark.slow  # slow: measured 114 s on the Windows dev box (ADR-023)
def test_w1n1_equals_w2n2(tmp_path):
    build_outfield_corpus(tmp_path / "data" / "toyof")
    data = tmp_path / "data"
    run_serial(MODULE, tmp_path / "s1", data, ARGS, tmp_path / "o1")
    run_fanned(MODULE, tmp_path / "s2", data, ARGS, tmp_path / "o2")
    assert outputs(tmp_path / "s1") == outputs(tmp_path / "s2")
    assert artifact(tmp_path / "o1") == artifact(tmp_path / "o2")
    assert set(artifact(tmp_path / "o1")) == {"model.npz", "metadata.json", "SHA256SUMS", "metrics.json"}
    gens = {s["name"]: s["generation"] for s in plan(MODULE, tmp_path / "s2", data, ARGS)["stages"]}
    assert completion_out_of_order(tmp_path / "s2", gens["extract"])
    assert ran_concurrently(tmp_path / "s2", gens["cv"])  # 4 folds: each job ran 2 folds on 2 workers
