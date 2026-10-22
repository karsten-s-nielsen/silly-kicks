"""ADR-128 W1N1-vs-W2N2 parity for `train_xshot_occurrence`, both arm shapes.

``single`` = an all-public payload (one 4-trial study); ``paired`` = the three-arm payload (15 nested
studies + the cv of every arm). Non-vacuity: the extract stage completes out of key order inside a job
(the heavy first game), and inside a study unit the InnerPool completes a trial's folds out of key order
(ruling 4: the folds really ran in parallel and were still combined by sorted key).
"""

from __future__ import annotations

import pytest

from scripts._toy_corpora import build_xshot_corpus, xshot_arms
from tests.scripts._parity import (
    artifact,
    completion_out_of_order,
    inner_out_of_order,
    outputs,
    plan,
    run_fanned,
    run_serial,
)

pytest.importorskip("xgboost")
pytest.importorskip("ruthless")

MODULE = "scripts.train_xshot_occurrence"


@pytest.mark.slow  # slow: measured 262 s (single) + 303 s (paired) on the Windows dev box (ADR-023)
@pytest.mark.parametrize("paired", [False, True], ids=["single", "paired"])
def test_w1n1_equals_w2n2(tmp_path, paired):
    build_xshot_corpus(tmp_path / "data" / "toyxs")
    data = tmp_path / "data"
    args = ["--dataset", "toyxs", "--arms-json", xshot_arms(data / "toyxs", paired=paired), "--n-trials", "4"]
    run_serial(MODULE, tmp_path / "s1", data, args, tmp_path / "o1")
    run_fanned(MODULE, tmp_path / "s2", data, args, tmp_path / "o2")
    assert outputs(tmp_path / "s1") == outputs(tmp_path / "s2")
    assert artifact(tmp_path / "o1") == artifact(tmp_path / "o2")
    assert set(artifact(tmp_path / "o1")) == {"model.json", "metadata.json", "SHA256SUMS", "metrics.json"}
    gens = {s["name"]: s["generation"] for s in plan(MODULE, tmp_path / "s2", data, args)["stages"]}
    assert completion_out_of_order(tmp_path / "s2", gens["extract"])
    assert inner_out_of_order(tmp_path / "s2", gens["study"])
