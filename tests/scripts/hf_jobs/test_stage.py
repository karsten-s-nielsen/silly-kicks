from scripts.hf_jobs.stage import corpus_manifest, stage_corpus


def test_manifest_is_content_stable(tmp_path):
    (tmp_path / "a.parquet").write_bytes(b"xyz")
    m1 = corpus_manifest(tmp_path)
    m2 = corpus_manifest(tmp_path)
    assert m1 == m2 and m1["n_files"] == 1 and m1["bytes"] == 3


def test_stage_skips_when_manifest_unchanged(tmp_path):
    (tmp_path / "a.parquet").write_bytes(b"xyz")
    uploads = []
    m = stage_corpus(tmp_path, "ns/tc3", uploader=lambda *a, **k: uploads.append(a))
    assert len(uploads) == 1
    # second call with the prior manifest => no re-upload
    stage_corpus(tmp_path, "ns/tc3", uploader=lambda *a, **k: uploads.append(a), already=m)
    assert len(uploads) == 1


# NOTE (ADR-123 Amendment B): write_offline_manifest was removed -- the in-job trainer reads a NEUTRAL
# staged parquet via a plain --data-dir; no pining manifest / SILLY_KICKS_PINING_MANIFEST_DIR reaches HF.
