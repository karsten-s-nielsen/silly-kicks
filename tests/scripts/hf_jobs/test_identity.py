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
        "artifact_size_bytes",
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


# ---- the cross-commit rule of `compare` (HFU fit-units spec 6 step 3) --------------------------------

C1, C2 = "a" * 40, "b" * 40


def _bundle(d, commit, *, weights=b"W", auc=0.9, platform="Linux-6.12.94", size=100, card_word="model"):
    """A toy artifact: a payload, metadata.json and metrics.json with provenance, a card and SHA256SUMS."""
    d.mkdir(parents=True)
    (d / "model.npz").write_bytes(weights)
    (d / "metadata.json").write_text(
        json.dumps({"training_commit": commit, "training_platform": platform, "feature_names": ["a"]}, indent=2)
    )
    (d / "metrics.json").write_text(
        json.dumps({"run_commit": commit, "auc": auc, "nested": {"artifact_size_bytes": size, "n": 3}})
    )
    (d / "MODEL_CARD.md").write_text(f"# A {card_word}\n\nTrained at `{commit}`.\n")
    lines = [f"{hashlib.sha256((d / n).read_bytes()).hexdigest()}  {n}" for n in ("model.npz", "metadata.json")]
    (d / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    return d


def _match_outcome(d, commit, *, rho=-0.1):
    """The match_outcome shape: the commit INSIDE model.json, and one SHA256SUMS line for model.json."""
    d.mkdir(parents=True)
    (d / "model.json").write_text(json.dumps({"rho": rho, "training_commit": commit}, indent=2))
    (d / "SHA256SUMS").write_text(f"{hashlib.sha256((d / 'model.json').read_bytes()).hexdigest()}  model.json\n")
    return d


def test_identical_folders_have_no_difference(tmp_path):
    from scripts.hf_jobs.identity import cross_commit_differences

    assert cross_commit_differences(_bundle(tmp_path / "a", C1), _bundle(tmp_path / "b", C1)) == []


def test_a_payload_byte_is_a_difference(tmp_path):
    from scripts.hf_jobs.identity import cross_commit_differences

    diffs = cross_commit_differences(_bundle(tmp_path / "a", C1), _bundle(tmp_path / "b", C2, weights=b"X"))
    assert any(d.startswith("model.npz") for d in diffs)


def test_provenance_only_json_is_identical(tmp_path):
    from scripts.hf_jobs.identity import cross_commit_differences

    a = _bundle(tmp_path / "a", C1, platform="Linux-6.12.94-123")
    b = _bundle(tmp_path / "b", C2, platform="Linux-6.12.113-138.209")
    assert (a / "metadata.json").read_bytes() != (b / "metadata.json").read_bytes()
    assert cross_commit_differences(a, b) == []


def test_artifact_size_bytes_is_a_provenance_key(tmp_path):
    from scripts.hf_jobs.identity import cross_commit_differences

    assert cross_commit_differences(_bundle(tmp_path / "a", C1, size=100), _bundle(tmp_path / "b", C1, size=101)) == []


def test_a_non_provenance_leaf_is_a_difference(tmp_path):
    from scripts.hf_jobs.identity import cross_commit_differences

    diffs = cross_commit_differences(_bundle(tmp_path / "a", C1, auc=0.9), _bundle(tmp_path / "b", C2, auc=0.91))
    assert diffs == ["metrics.json: differs after removing the provenance keys"]


def test_sums_lines_of_normalized_files_are_dropped(tmp_path):
    from scripts.hf_jobs.identity import cross_commit_differences

    a, b = _bundle(tmp_path / "a", C1), _bundle(tmp_path / "b", C2)
    assert (a / "SHA256SUMS").read_bytes() != (b / "SHA256SUMS").read_bytes()  # the metadata.json line differs
    assert cross_commit_differences(a, b) == []


def test_match_outcome_shape(tmp_path):
    from scripts.hf_jobs.identity import cross_commit_differences

    a, b = _match_outcome(tmp_path / "a", C1), _match_outcome(tmp_path / "b", C2)
    assert cross_commit_differences(a, b, commit_file="model.json") == []
    c = _match_outcome(tmp_path / "c", C2, rho=-0.2)
    assert cross_commit_differences(a, c, commit_file="model.json") == [
        "model.json: differs after removing the provenance keys"
    ]


def test_match_outcome_shape_without_commit_file_differs(tmp_path):
    from scripts.hf_jobs.identity import cross_commit_differences

    a, b = _match_outcome(tmp_path / "a", C1), _match_outcome(tmp_path / "b", C2)
    diffs = cross_commit_differences(a, b)  # non-vacuity: the per-artifact commit file is what makes them equal
    assert any(d.startswith("model.json") for d in diffs) and any(d.startswith("SHA256SUMS") for d in diffs)


def test_card_commit_is_normalized(tmp_path):
    from scripts.hf_jobs.identity import cross_commit_differences

    assert cross_commit_differences(_bundle(tmp_path / "a", C1), _bundle(tmp_path / "b", C2)) == []
    diffs = cross_commit_differences(tmp_path / "a", _bundle(tmp_path / "c", C2, card_word="thing"))
    assert diffs == ["MODEL_CARD.md: differs after replacing the commit"]


def test_a_file_in_one_folder_is_a_difference(tmp_path):
    from scripts.hf_jobs.identity import cross_commit_differences

    a, b = _bundle(tmp_path / "a", C1), _bundle(tmp_path / "b", C1)
    (b / "extra.bin").write_bytes(b"x")
    assert cross_commit_differences(a, b) == ["extra.bin: present in only one artifact"]
