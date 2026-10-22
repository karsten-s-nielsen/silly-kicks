"""ADR-128 / spec 5.3 item 4: a resumed for_each never trusts a shard whose checksum fails."""

from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from scripts import _driver as d


def _gen(tmp_path):
    return d.generation_dir(tmp_path / "root", token_inputs={"t": 1})


def test_a_new_generation_is_marked_checksummed(tmp_path):
    gen = _gen(tmp_path)
    assert (gen / d._CHECKSUMMED).is_file()


def test_write_shard_writes_a_matching_sidecar(tmp_path):
    gen = _gen(tmp_path)
    p = d.shard_path(gen, "m1")
    d.write_shard(p, pd.DataFrame({"a": [1, 2]}), tag="w0")
    side = d.checksum_path(gen, "m1")
    assert side.read_text(encoding="utf-8").strip() == hashlib.sha256(p.read_bytes()).hexdigest()
    assert d.already_done(gen, "m1")


def test_a_corrupted_shard_is_not_done_and_is_removed(tmp_path):
    gen = _gen(tmp_path)
    p = d.shard_path(gen, "m2")
    d.write_shard(p, pd.DataFrame({"a": [1]}), tag="w0")
    p.write_bytes(p.read_bytes()[:-7])  # truncated / poisoned on the bucket
    with pytest.warns(UserWarning, match="checksum mismatch"):
        assert not d.already_done(gen, "m2")
    assert not p.exists() and not d.checksum_path(gen, "m2").exists()


def test_a_marked_generation_without_a_sidecar_is_not_done(tmp_path):
    gen = _gen(tmp_path)
    pd.DataFrame({"a": [1]}).to_parquet(d.shard_path(gen, "m3"), index=False)  # written by nothing we trust
    with pytest.warns(UserWarning, match="untrusted shard"):
        assert not d.already_done(gen, "m3")


def test_a_legacy_generation_keeps_file_exists_means_done(tmp_path):
    legacy = tmp_path / "legacy" / "0123456789abcdef"
    legacy.mkdir(parents=True)  # pre-ADR-128: no .checksummed marker
    pd.DataFrame({"a": [1]}).to_parquet(d.shard_path(legacy, "m4"), index=False)
    assert d.already_done(legacy, "m4")


def test_for_each_redoes_a_poisoned_shard_on_resume(tmp_path):
    calls = []

    def work(item):
        calls.append(item)
        return pd.DataFrame({"v": [item]})

    kw = {"key": str, "work": work, "shard_root": tmp_path / "r", "token_inputs": {"x": 1}}
    res = d.for_each(["a", "b"], **kw)
    d.shard_path(res.shard_dir, "a").write_bytes(b"not parquet")
    calls.clear()
    with pytest.warns(UserWarning, match="checksum mismatch"):
        d.for_each(["a", "b"], **kw)
    assert calls == ["a"]


def test_no_temp_files_survive(tmp_path):
    gen = _gen(tmp_path)
    d.write_shard(d.shard_path(gen, "m5"), pd.DataFrame({"a": [1]}), tag="w0")
    assert not [p for p in gen.iterdir() if p.name.endswith(".part")]
