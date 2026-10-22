"""Artifact identity (ADR-128 decision P3): same commit = every file; across commits = payload files plus
the JSON files with the provenance keys removed."""

import hashlib
import json

import pytest

from scripts.hf_jobs.identity import PROVENANCE_KEYS, artifact_identity, compare


def _write(d, meta_commit, weights=b"W"):
    d.mkdir(parents=True)
    (d / "model.json").write_bytes(weights)
    (d / "metadata.json").write_text(json.dumps({"training_commit": meta_commit, "feature_names": ["a"]}))
    (d / "metrics.json").write_text(json.dumps({"run_commit": meta_commit, "pr_auc": 0.08}))


def _sums(d):
    lines = [f"{hashlib.sha256((d / n).read_bytes()).hexdigest()}  {n}" for n in ("model.json", "metadata.json")]
    (d / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def test_cross_commit_compare_ignores_only_provenance(tmp_path):
    _write(tmp_path / "a", "c1")
    _write(tmp_path / "b", "c2")
    assert compare(artifact_identity(tmp_path / "a"), artifact_identity(tmp_path / "b"), same_commit=False) == []
    assert compare(artifact_identity(tmp_path / "a"), artifact_identity(tmp_path / "b"), same_commit=True)


def test_a_weight_or_metric_change_is_always_reported(tmp_path):
    _write(tmp_path / "a", "c1")
    _write(tmp_path / "b", "c2", weights=b"X")
    diffs = compare(artifact_identity(tmp_path / "a"), artifact_identity(tmp_path / "b"), same_commit=False)
    assert any("model.json" in d for d in diffs)
    (tmp_path / "b" / "model.json").write_bytes(b"W")
    (tmp_path / "b" / "metrics.json").write_text(json.dumps({"run_commit": "c2", "pr_auc": 0.09}))
    diffs = compare(artifact_identity(tmp_path / "a"), artifact_identity(tmp_path / "b"), same_commit=False)
    assert any("metrics.json" in d for d in diffs)


def test_provenance_keys_are_exactly_the_declared_set():
    assert "sklearn_version" not in PROVENANCE_KEYS and "training_commit" in PROVENANCE_KEYS
    assert set(PROVENANCE_KEYS) == {
        "training_commit",
        "training_platform",
        "run_commit",
        "run_tree_dirty",
        "run_tree_state",
        "reproducibility_note",
        "generation",
    }


def test_sha256sums_compares_on_its_payload_lines_across_commits(tmp_path):
    """SHA256SUMS lists metadata.json too, so across commits only its payload lines count -- but a payload
    checksum change is still reported."""
    _write(tmp_path / "a", "c1")
    _write(tmp_path / "b", "c2")
    _sums(tmp_path / "a")
    _sums(tmp_path / "b")
    a, b = artifact_identity(tmp_path / "a"), artifact_identity(tmp_path / "b")
    assert a["files"]["SHA256SUMS"] != b["files"]["SHA256SUMS"]
    assert compare(a, b, same_commit=False) == []
    (tmp_path / "b" / "model.json").write_bytes(b"X")
    _sums(tmp_path / "b")
    diffs = compare(a, artifact_identity(tmp_path / "b"), same_commit=False)
    assert any("SHA256SUMS" in d for d in diffs) and any("model.json" in d for d in diffs)


def test_a_driver_report_is_normalized_when_named(tmp_path):
    for name, commit in (("a", "c1"), ("b", "c2")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "census.json").write_text(json.dumps({"run_commit": commit, "icc": 0.4}))
    a = artifact_identity(tmp_path / "a", json_files=("census.json",))
    b = artifact_identity(tmp_path / "b", json_files=("census.json",))
    assert compare(a, b, same_commit=False) == []
    assert compare(artifact_identity(tmp_path / "a"), artifact_identity(tmp_path / "b"), same_commit=False)


def test_an_extra_file_and_a_nested_dir_are_reported(tmp_path):
    _write(tmp_path / "a", "c1")
    _write(tmp_path / "b", "c1")
    (tmp_path / "b" / "extra.bin").write_bytes(b"x")
    assert any(
        "extra.bin" in d
        for d in compare(artifact_identity(tmp_path / "a"), artifact_identity(tmp_path / "b"), same_commit=True)
    )
    (tmp_path / "b" / "sub").mkdir()
    with pytest.raises(ValueError, match="nested"):
        artifact_identity(tmp_path / "b")
