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


def _shipped_bundles():
    import pathlib
    import subprocess

    out = subprocess.run(
        ["git", "ls-files", "silly_kicks/**/SHA256SUMS"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    )
    return sorted({str(pathlib.PurePosixPath(p).parent) for p in out.stdout.split()})


#: Shipped bundles whose writer does NOT follow `sums_names`, each with the reason. None passes through a stage
#: pipeline (`commit_free` / `stamp_provenance`), so its own `save()` stays the only writer.
_SUMS_CONVENTION_EXCEPTIONS = {
    "silly_kicks/xsuccess/weights": "XSuccessModel.save covers model.json only; xsuccess is not a stage pipeline",
}


@pytest.mark.parametrize("bundle", _shipped_bundles())
def test_write_sums_reproduces_every_shipped_bundle_byte_for_byte(tmp_path, bundle):
    """S5.10 'the same writer the trainer uses today': a stamped bundle's SHA256SUMS must equal what the
    trainer's own `save()` shipped -- payload first, then metadata.json, never metrics.json (B: the receiver
    loader reads only the first token, so an all-files sorted manifest made an adopted bundle unloadable)."""
    import pathlib
    import shutil
    import subprocess

    from scripts.hf_jobs import identity

    if bundle in _SUMS_CONVENTION_EXCEPTIONS:
        pytest.skip(_SUMS_CONVENTION_EXCEPTIONS[bundle])
    tracked = subprocess.run(  # noqa: S603
        ["git", "ls-files", bundle],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    work = tmp_path / "b"
    work.mkdir()
    for t in tracked:
        p = pathlib.Path(t)
        if p.parent.as_posix() == bundle and p.name != identity.SUMS:
            shutil.copyfile(p, work / p.name)
    identity.write_sums(work)
    shipped = pathlib.Path(bundle, identity.SUMS).read_bytes().replace(b"\r\n", b"\n")
    assert (work / identity.SUMS).read_bytes() == shipped
    assert identity.verify_sums(work)


def test_the_sums_exceptions_name_real_bundles():
    assert set(_SUMS_CONVENTION_EXCEPTIONS) <= set(_shipped_bundles())
