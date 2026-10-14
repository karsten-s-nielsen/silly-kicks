import json

from scripts.hf_jobs.stage import corpus_manifest, stage_corpus, write_offline_manifest


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


def test_write_offline_manifest_captures_each_providers_matches(tmp_path):
    # Stage-time (DGX, token-bearing): snapshot the /{provider}/matches response so the air-gapped HF Job
    # enumerates offline via SILLY_KICKS_PINING_MANIFEST_DIR. `lister` is injected (no network in the test).
    served = {"statsbomb": [{"id": "3986784", "visibility": "private", "artifacts": {}}]}
    out = write_offline_manifest(["statsbomb"], tmp_path, lister=lambda p: served[p])
    assert out == [tmp_path / "statsbomb" / "matches.json"]
    doc = json.loads((tmp_path / "statsbomb" / "matches.json").read_text())
    assert doc == {"matches": served["statsbomb"]}  # exactly the API shape _list_matches reads back
