"""ADR-128 W1N1-vs-W2N2 parity for `train_xsuccess` (public source; the HPO folds on the InnerPool, HFU spec
15.5a, 15.7; the `fit` stage as 36 units, HFU fit-units spec 3.2): the fanned run gives the serial result byte
for byte, on a corpus that holds a row with a non-finite feature and matches of unequal row counts."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("statsbombpy")
pytest.importorskip("xgboost")

from scripts._toy_corpora import XSUCCESS_NONFINITE_MATCH_ID, build_open_data
from tests.scripts._open_data_server import serve_open_data
from tests.scripts._parity import (
    artifact,
    completion_out_of_order,
    inner_out_of_order,
    outputs,
    plan,
    run_fanned,
    run_serial,
)

MODULE = "scripts.train_xsuccess"
ARGS = ["--competition-id", "43", "--season-id", "106", "--hpo-trials", "3"]


@pytest.mark.slow  # slow: the toy open-data corpus plus a 3-trial study (ADR-023)
def test_w1n1_equals_w2n2(tmp_path):
    root = build_open_data(tmp_path / "od", nonfinite_in=XSUCCESS_NONFINITE_MATCH_ID)
    with serve_open_data(root) as url:
        env = {**os.environ, "SILLY_KICKS_OPEN_DATA_BASE": url}
        run_serial(MODULE, tmp_path / "s1", tmp_path / "data", ARGS, tmp_path / "o1", env=env)
        run_fanned(MODULE, tmp_path / "s2", tmp_path / "data", ARGS, tmp_path / "o2", env=env)
        gens = {
            s["name"]: s["generation"]
            for s in plan(MODULE, tmp_path / "s2", tmp_path / "data", ARGS, env=env)["stages"]
        }
    assert outputs(tmp_path / "s1") == outputs(tmp_path / "s2")
    assert artifact(tmp_path / "o1") == artifact(tmp_path / "o2")
    assert {"model.json", "SHA256SUMS", "metadata.json", "metrics.json", "MODEL_CARD.md"} <= set(
        artifact(tmp_path / "o1")
    )
    assert completion_out_of_order(tmp_path / "s2", gens["extract"])
    assert inner_out_of_order(tmp_path / "s2", gens["hpo"])  # the folds really ran on the pool, out of order
    # The `fit` stage is 36 units over 2 jobs x 2 workers (HFU fit-units spec 3.2).
    assert completion_out_of_order(tmp_path / "s2", gens["fit"])
    # The corpus really holds what the fit must handle (spec 3.5, A-R2): a row with a non-finite feature, and
    # matches of unequal row counts.
    import numpy as np
    import pandas as pd

    from silly_kicks.xsuccess._features import xsuccess_features

    pooled = pd.read_parquet(tmp_path / "s1" / gens["extract"] / "_output.parquet")
    assert (~np.isfinite(xsuccess_features(pooled)).all(axis=1)).sum() >= 1
    assert pooled["game_id"].value_counts().nunique() > 1
