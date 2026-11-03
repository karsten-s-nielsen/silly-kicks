import math
import pathlib

import pytest

from scripts.hf_jobs import registry as reg
from scripts.hf_jobs.headtohead import (
    SCORERS,
    EvalDataUnavailableError,
    build_eval,
    stamp,
    synthetic_bundle,
)

REPO = pathlib.Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("name", sorted(reg.REGISTRY))
def test_every_trainer_has_a_feasible_scorer(name):
    # Scoring PATH proof: load the COMMITTED model as BOTH operands on an injected synthetic bundle ->
    # finite, equal paired scalar. Proves load + predict + metric + fold is feasible per trainer at build
    # time, with no corpus / statsbombpy. (The REAL bundle comes from build_eval at P1; see below.)
    spec = reg.REGISTRY[name]
    committed = REPO / "silly_kicks" / spec.weights_paths[0]
    out = SCORERS[name](committed, committed, synthetic_bundle(name, committed))
    m = spec.primary_metric
    c, f = out[f"committed_{m}"], out[f"fresh_{m}"]
    assert math.isfinite(c) and math.isfinite(f) and c == pytest.approx(f)


def test_build_eval_is_fail_loud_without_data(monkeypatch):
    # The job must NEVER score the head-to-head on synthetic data (that would make the floor vacuous).
    # A public trainer whose source is unreachable and an owner trainer with no corpus must both RAISE,
    # not fall back. We FORCE the public unreachable condition (statsbombpy absent, as on CI) rather than
    # leaning on the ambient env: if statsbombpy happens to be installed with network, the real builder
    # would fetch live open data, so this would otherwise pass on CI yet fail (with a download) locally.
    from scripts.hf_jobs import headtohead as hh

    def _unreachable() -> None:
        raise EvalDataUnavailableError("statsbombpy unavailable (simulated)")

    monkeypatch.setattr(hh, "_require_statsbombpy", _unreachable)
    with pytest.raises(EvalDataUnavailableError):
        build_eval("win_probability", None, certification="public")
    # Owner trainer with no staged corpus -> raise (deterministic, offline).
    with pytest.raises(EvalDataUnavailableError):
        build_eval("ghost_gk", None, certification="owner")


def test_cli_e2e_writes_pair_plus_stamp_and_floor_admits(tmp_path, monkeypatch):
    # End-to-end of the IN-JOB step: main() -> build_eval -> scorer -> merge pair + stamp into metrics.json
    # -> floor ADMITS (committed == fresh). build_eval is monkeypatched to inject a synthetic bundle so the
    # merge/stamp/floor chain runs locally (the real build_eval needs statsbombpy/corpus; P1).
    import json
    import sys

    from scripts.hf_jobs import floor, headtohead

    spec = reg.REGISTRY["win_probability"]
    committed = str(REPO / "silly_kicks" / spec.weights_paths[0])
    monkeypatch.setattr(
        headtohead,
        "build_eval",
        lambda name, corpus_dir, **_kw: synthetic_bundle(name, pathlib.Path(committed)),
    )
    mj = tmp_path / "metrics.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "headtohead",
            "win_probability",
            "--committed-dir",
            committed,
            "--fresh-dir",
            committed,
            "--metrics-json",
            str(mj),
            "--certification",
            "public",
        ],
    )
    headtohead.main()
    data = json.loads(mj.read_text())
    assert "training_platform" in data and "sklearn_version" in data
    assert f"committed_{spec.primary_metric}" in data and f"fresh_{spec.primary_metric}" in data
    ok, _ = floor.evaluate_spec(spec, data)
    assert ok is True  # committed == fresh -> no regression -> admits


def test_stamp_has_platform_and_versions():
    s = stamp()
    assert "training_platform" in s and "sklearn_version" in s


def test_flat_tc3_public_eval_reads_eval_rows(tmp_path, monkeypatch):
    """HFU spec 14.4: an xshot/xcross public head-to-head scores the extract rows of the run itself
    (``--eval-rows``): no corpus directory, no re-extraction."""
    import numpy as np
    import pandas as pd

    from scripts.hf_jobs import headtohead as H
    from scripts.train_xshot_occurrence import _feature_names

    names = _feature_names("position_only")
    rows = pd.DataFrame({n: np.arange(3, dtype=float) for n in names}).assign(_y=[0, 1, 0], _group=["g1", "g1", "g2"])
    path = tmp_path / "rows.parquet"
    rows.to_parquet(path)

    def _no_corpus(*_a, **_k):
        raise AssertionError("no corpus read")

    monkeypatch.setattr(H, "_owner_shards", _no_corpus)
    b = H.build_eval("xshot_occurrence", None, certification="public", eval_rows=path, feature_set="position_only")
    assert list(b["features"].columns) == names
    assert b["y"].tolist() == [0, 1, 0] and b["groups"].tolist() == ["g1", "g1", "g2"]
    with pytest.raises(H.EvalDataUnavailableError):
        H.build_eval("xshot_occurrence", None, certification="public", eval_rows=None)
