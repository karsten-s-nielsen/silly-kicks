"""A stale _feature_cache/ must be a MISS (spec 3.2). The 2026-07 owner runs already populated one
on the DGX with no cache_meta.json / visibility taxonomy -- reusing it would silently re-introduce
the provider-name arm split."""

import json

from _cache import cache_is_valid, write_cache_meta


def test_absent_meta_is_a_miss(tmp_path):
    (tmp_path / "features.parquet").write_bytes(b"x")  # the OLD predicate would say "hit"
    assert cache_is_valid(tmp_path, fingerprint="abc") is False


def test_schema_version_mismatch_is_a_miss(tmp_path):
    (tmp_path / "features.parquet").write_bytes(b"x")  # payload present -> the schema check decides
    write_cache_meta(tmp_path, fingerprint="abc")
    meta = json.loads((tmp_path / "cache_meta.json").read_text())
    meta["schema_version"] = 0
    (tmp_path / "cache_meta.json").write_text(json.dumps(meta))
    assert cache_is_valid(tmp_path, fingerprint="abc") is False


def test_corpus_fingerprint_mismatch_is_a_miss(tmp_path):
    (tmp_path / "features.parquet").write_bytes(b"x")  # payload present -> the fingerprint check decides
    write_cache_meta(tmp_path, fingerprint="abc")
    assert cache_is_valid(tmp_path, fingerprint="DIFFERENT") is False


def test_matching_meta_is_a_hit(tmp_path):
    (tmp_path / "features.parquet").write_bytes(b"x")
    write_cache_meta(tmp_path, fingerprint="abc")
    assert cache_is_valid(tmp_path, fingerprint="abc") is True
