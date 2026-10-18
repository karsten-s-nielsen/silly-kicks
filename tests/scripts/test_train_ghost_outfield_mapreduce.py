"""ADR-123 map/reduce determinism for ghost_outfield: fanned extract -> fold -> assemble ships
BYTE-IDENTICAL weights + CV metrics to a plain serial run.

@slow + skipif, like the sibling ghost_outfield smoke: the bundled-weights fit needs scikit-learn in
[1.9, 2) (VSSOT-IMPL-03), so this SKIPS on the local py3.10 .venv and runs on CI's py3.12 leg.
"""

from __future__ import annotations

import json

import pytest

from scripts._train_guard import sklearn_supports_training
from scripts.train_ghost_outfield import main
from tests.scripts.test_train_ghost_outfield import _write_tiny_corpus

pytestmark = pytest.mark.skipif(
    not sklearn_supports_training(),
    reason="bundled-weights fit needs scikit-learn in [1.9, 2)",
)


def _args(data, out, *extra):
    return [
        "--data-dir",
        str(data),
        "--output-dir",
        str(out),
        "--variant",
        "default",
        "--n-estimators",
        "20",
        "--max-depth",
        "3",
        "--cv-folds",
        "2",
        "--allow-dirty",
        *extra,
    ]


@pytest.mark.slow
def test_fanned_pipeline_is_byte_identical_to_serial(tmp_path):
    # Toy corpus: 4 tc3 games == 4 parquet shards (>=2 shards) -> 2-fold CV (>=2 folds); an N=1 fan-out
    # would pass vacuously (HFP-SPEC-01 / spec Global Constraints).
    data = tmp_path / "data"
    _write_tiny_corpus(data, n_games=4)

    serial = tmp_path / "serial"
    main(_args(data, serial))

    fanned = tmp_path / "fanned"
    for i in range(2):  # extract fan-out: disjoint file subsets -> shared shard_root
        main(_args(data, fanned, "--shard-index", f"{i}/2"))
    for k in range(2):  # fold fan-out: one CV fold each (rebuilds `data` from the resumed shards)
        main(_args(data, fanned, "--fold", str(k)))
    main(_args(data, fanned, "--assemble"))

    s, f = serial / "default", fanned / "default"
    assert (s / "model.npz").read_bytes() == (f / "model.npz").read_bytes()
    sm, fm = json.loads((s / "metrics.json").read_text()), json.loads((f / "metrics.json").read_text())
    for key in ("cv_mae", "cv_mae_by_slot", "cv_mae_by_provider", "cv_mae_by_possession", "n_games"):
        assert sm[key] == fm[key], f"{key}: serial {sm[key]} != fanned {fm[key]}"
