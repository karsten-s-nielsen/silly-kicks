"""The power driver must not stamp a clean SHA on numbers derived from a dirty upstream table.

CLAUDE.md states the rule directly: an artifact whose inputs came from another driver needs
provenance on BOTH, or the clean SHA on the downstream metrics launders the dirty upstream input.
The spells table and the arm-values table are both produced by OTHER drivers, at other times.
"""

from __future__ import annotations

import json

import pytest

import scripts.run_signoff_power as mod  # bare import: tests/scripts/ has NO __init__.py


def _table_with_manifest(tmp_path, **manifest):
    (tmp_path / "table.parquet").write_bytes(b"not-really-parquet")  # never read by the guard
    (tmp_path / "thing_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return str(tmp_path / "table.parquet")


def test_a_CLEAN_upstream_is_recorded_and_accepted(tmp_path):
    path = _table_with_manifest(tmp_path, run_commit="abc123", run_tree_dirty=False)
    got = mod._upstream_provenance(path, allow_dirty=False)
    assert got["run_commit"] == "abc123"
    assert got["run_tree_dirty"] is False


def test_a_DIRTY_upstream_is_REFUSED(tmp_path):
    """The other side of the test above -- without it the guard passes identically when it does
    nothing at all."""
    path = _table_with_manifest(tmp_path, run_commit="abc123", run_tree_dirty=True)
    with pytest.raises(SystemExit) as e:
        mod._upstream_provenance(path, allow_dirty=False)
    assert "not certified clean" in str(e.value)


def test_allow_dirty_permits_the_dirty_upstream_but_still_RECORDS_it(tmp_path):
    """The escape hatch must never launder the fact -- same contract as `require_clean_tree`."""
    path = _table_with_manifest(tmp_path, run_commit="abc123", run_tree_dirty=True)
    got = mod._upstream_provenance(path, allow_dirty=True)
    assert got["run_tree_dirty"] is True


def test_a_table_with_NO_manifest_is_unknown_not_assumed_clean(tmp_path):
    """Fail-closed, the same principle as an absent git checkout counting as dirty."""
    (tmp_path / "table.parquet").write_bytes(b"x")
    with pytest.raises(SystemExit):
        mod._upstream_provenance(str(tmp_path / "table.parquet"), allow_dirty=False)
    got = mod._upstream_provenance(str(tmp_path / "table.parquet"), allow_dirty=True)
    assert got["run_commit"] == "unknown" and got["run_tree_dirty"] is True


def test_a_MIXED_COMMIT_upstream_is_refused_even_when_clean(tmp_path):
    """Every worker had a clean tree, but not the SAME tree: the table is a blend of two code
    versions while every individual provenance record looks impeccable."""
    path = _table_with_manifest(
        tmp_path, run_commit=["abc123", "def456"], run_tree_dirty=False, commit_consistent=False
    )
    with pytest.raises(SystemExit) as e:
        mod._upstream_provenance(path, allow_dirty=False)
    assert "DIFFERENT commits" in str(e.value)


def _worker_manifest(tmp_path, name, **payload):
    (tmp_path / f"manifest_{name}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_EVERY_manifest_is_read_not_just_the_first_one_alphabetically(tmp_path):
    """`sorted(...)[0]` certified whichever manifest sorted first. Here a CLEAN corpus manifest
    sorts ahead of a DIRTY partition -- exactly the arrangement that would let one bad worker
    through while the artifact claims corpus scope."""
    _table_with_manifest(tmp_path, run_commit="abc123", run_tree_dirty=False)  # "thing_manifest"
    _worker_manifest(tmp_path, "p0", run_commit="abc123", run_tree_dirty=True)
    with pytest.raises(SystemExit) as e:
        mod._upstream_provenance(str(tmp_path / "table.parquet"), allow_dirty=False)
    assert "not certified clean" in str(e.value)


def test_mixed_commits_are_DERIVED_from_the_worker_records(tmp_path):
    """No manifest self-reports `commit_consistent` here -- the disagreement is inferred from the
    commits actually recorded, so a pre-4.64.0 artifact is judged on evidence rather than
    fail-opening on a field that did not exist when it was written."""
    _worker_manifest(tmp_path, "p0", run_commit="abc123", run_tree_dirty=False)
    _worker_manifest(tmp_path, "p1", run_commit="def456", run_tree_dirty=False)
    (tmp_path / "table.parquet").write_bytes(b"x")
    with pytest.raises(SystemExit) as e:
        mod._upstream_provenance(str(tmp_path / "table.parquet"), allow_dirty=False)
    assert "DIFFERENT commits" in str(e.value)


def test_a_legacy_manifest_without_the_consistency_FIELD_still_passes_when_it_agrees(tmp_path):
    """The other side: the real 64-match arm-values table predates `commit_consistent` and its
    eight worker manifests all record one SHA. It must validate, not be refused for a missing key."""
    for i in range(8):
        _worker_manifest(tmp_path, f"p{i}", run_commit="93ac3ba", run_tree_dirty=False, n_matches=8)
    (tmp_path / "table.parquet").write_bytes(b"x")
    got = mod._upstream_provenance(str(tmp_path / "table.parquet"), allow_dirty=False)
    assert got["run_commit"] == "93ac3ba"
    assert got["commit_consistent"] is True
    assert len(got["manifests"]) == 8
