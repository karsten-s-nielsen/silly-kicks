"""Task 7: ``validate_corpus_visibility`` is a shared module and the trainer re-imports it.

``scripts/`` has no ``__init__.py``; the trainer is loaded by file path, so the re-import identity is the
load-bearing contract (spec 7.1 item 5) -- the trainer's ``validate_corpus_visibility`` MUST be the one
object in ``scripts._corpus_visibility``, not a private copy.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import _corpus_visibility

REPO = Path(__file__).resolve().parents[2]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None, f"could not load scripts/{name}.py"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_train_ghost_gk_reimports_validate_corpus_visibility():
    t = _load("train_ghost_gk")
    assert t.validate_corpus_visibility is _corpus_visibility.validate_corpus_visibility


def _write_shard(path: Path, *, visibility) -> Path:
    tbl = pa.table({"visibility": pa.array(visibility, type=pa.bool_())})
    pq.write_table(tbl, path)
    return path


def test_discarded_flag_raises_shared_remedy(tmp_path):
    shard = _write_shard(tmp_path / "skillcorner__m1.parquet", visibility=[None, None])
    with pytest.raises(ValueError, match=r"tracking\.skillcorner"):
        _corpus_visibility.validate_corpus_visibility({shard: "skillcorner"})


def test_native_skillcorner_ok(tmp_path):
    shard = _write_shard(tmp_path / "skillcorner__m2.parquet", visibility=[True, False, True])
    assert _corpus_visibility.validate_corpus_visibility({shard: "skillcorner"}) is None


def test_missing_visibility_column_raises(tmp_path):
    path = tmp_path / "skillcorner__m3.parquet"
    pq.write_table(pa.table({"x": pa.array([1, 2, 3])}), path)
    with pytest.raises(ValueError, match=r"NO\s+`visibility` column|no .*visibility"):
        _corpus_visibility.validate_corpus_visibility({path: "skillcorner"})


def test_fully_observed_provider_is_noop(tmp_path):
    # A non-detection-aware provider is skipped entirely -- even an all-null visibility column is fine.
    shard = _write_shard(tmp_path / "sportec__m1.parquet", visibility=[None, None])
    assert _corpus_visibility.validate_corpus_visibility({shard: "sportec"}) is None


@pytest.mark.parametrize(("visibility", "raises"), [([None, None], True), ([True, None], False)])
def test_a_shard_without_statistics_is_decided_from_a_whole_file_read(tmp_path, monkeypatch, visibility, raises):
    """The rare no-statistics fallback reads the column from the WHOLE file under the read gate, never a
    range read on the (possibly mounted) path (ADR-128, the dataset read gate)."""
    from scripts import _dataset_reads

    path = tmp_path / "skillcorner__m4.parquet"
    pq.write_table(pa.table({"visibility": pa.array(visibility, type=pa.bool_())}), path, write_statistics=False)
    assert pq.ParquetFile(path).metadata.row_group(0).column(0).statistics is None  # the fallback's precondition
    whole: list[Path] = []
    real = _dataset_reads.read_bytes
    monkeypatch.setattr(_dataset_reads, "read_bytes", lambda p: whole.append(Path(p)) or real(p))
    if raises:
        with pytest.raises(ValueError, match=r"tracking\.skillcorner"):
            _corpus_visibility.validate_corpus_visibility({path: "skillcorner"})
    else:
        assert _corpus_visibility.validate_corpus_visibility({path: "skillcorner"}) is None
    assert whole == [path]
