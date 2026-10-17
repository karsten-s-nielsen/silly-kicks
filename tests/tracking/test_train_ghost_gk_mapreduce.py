"""ADR-123 map/reduce determinism: the fanned extract -> build-cache -> fold -> assemble pipeline ships
BYTE-IDENTICAL weights + CV metrics to a plain serial run.

@slow + skipif: the ghost trainer refuses to fit under scikit-learn outside [1.9, 2) (VSSOT-IMPL-03), so
this SKIPS on the local py3.10 .venv (sklearn < 1.9) and runs on CI's py3.12 leg -- the same gating the
other train_ghost_gk CLI smokes use.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts._train_guard import sklearn_supports_training
from tests.tracking.test_train_ghost_gk_cli import _tiny_cache

pytestmark = pytest.mark.skipif(
    not sklearn_supports_training(),
    reason="ghost trainer requires scikit-learn in [1.9, 2) (VSSOT-IMPL-03 fit-time guard)",
)


def _invoke(data: Path, out: Path, *extra: str) -> subprocess.CompletedProcess:
    # Module form (python -m) -- the file form raises ModuleNotFoundError: no scripts/__init__ (registry docstring).
    return subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "scripts.train_ghost_gk",
            "--data-dir",
            str(data),
            "--output-dir",
            str(out),
            "--n-estimators",
            "10",
            "--cv-folds",
            "2",
            "--allow-dirty",
            "--skip-permutation-importance",
            *extra,
        ],
        capture_output=True,
        text=True,
    )


@pytest.mark.slow
def test_fanned_pipeline_is_byte_identical_to_serial(tmp_path):
    # Toy corpus: 2 tc3 games == 2 parquet shards (>=2 shards) -> 2-fold CV (>=2 folds); an N=1 fan-out
    # would pass vacuously (HFP-SPEC-01 / spec Global Constraints).
    data = tmp_path / "cache"
    data.mkdir()
    _tiny_cache(data, n_games=2)

    serial_out = tmp_path / "serial"
    assert _invoke(data, serial_out).returncode == 0, "serial run failed"

    fanned_out = tmp_path / "fanned"
    for i in range(2):  # extract fan-out: disjoint file subsets -> shared shard_root
        r = _invoke(data, fanned_out, "--shard-index", f"{i}/2")
        assert r.returncode == 0, f"extract {i}/2 failed: {r.stderr}"
    assert _invoke(data, fanned_out, "--build-cache").returncode == 0, "build-cache failed"
    for k in range(2):  # fold fan-out: one CV fold each from the race-free cache
        r = _invoke(data, fanned_out, "--fold", str(k))
        assert r.returncode == 0, f"fold {k} failed: {r.stderr}"
    assert _invoke(data, fanned_out, "--assemble").returncode == 0, "assemble failed"

    s, f = serial_out / "ghost_gk_v1", fanned_out / "ghost_gk_v1"
    # The shipped model weights are byte-identical (same fit inputs -> same HGBR trees).
    assert (s / "rfcde_weights.npz").read_bytes() == (f / "rfcde_weights.npz").read_bytes()
    # CV aggregate matches (fold JSON float round-trip is exact).
    sm, fm = json.loads((s / "metrics.json").read_text()), json.loads((f / "metrics.json").read_text())
    for key in ("cv_mae_euclidean_mean", "cv_mae_euclidean_std", "cv_mae_x_mean", "cv_mae_y_mean", "n_games"):
        assert sm[key] == fm[key], f"{key}: serial {sm[key]} != fanned {fm[key]}"
