"""Structural guard: the slow-decouple CONSERVATION invariant lives ONLY in CI runtime.

The proof that decoupling the slow suite dropped nothing -- ``non-slow(primary) disjoint-union slow ==
not-e2e`` -- exists only in the ``shard-reconcile`` job body plus the ``slow`` job's node-ID uploads.
No local test exercises it, so its only other verification would be "green CI", which proves nothing
ran. A future ``ci.yml`` edit that dropped a slow upload or the conservation proof would leave every
other guard green and silently reopen the silent-drop hole. This pins it the pandas-span way: parsed-
YAML topology plus exactly ONE text-grep of the reconcile ``run`` scalar for a required sentinel.

See docs/superpowers/specs/2026-09-25-ci-runtime-slow-decouple-design.md (CI-SPEC-01).
"""

from __future__ import annotations

import pathlib

import yaml

_REPO = pathlib.Path(__file__).resolve().parent.parent
_CI = yaml.safe_load((_REPO / ".github/workflows/ci.yml").read_text(encoding="utf-8"))


def _upload_names(job: dict) -> set[str]:
    return {
        str(s["with"]["name"])
        for s in job["steps"]
        if "upload-artifact" in str(s.get("uses", "")) and isinstance(s.get("with"), dict) and "name" in s["with"]
    }


def test_slow_job_uploads_reconcile_artifacts() -> None:
    slow = _CI["jobs"]["slow"]
    names = _upload_names(slow)
    # the per-shard set carries a ${{ matrix.slow-shard }} suffix -> match by prefix; the two single
    # artifacts are exact names.
    assert any(n.startswith("slow-shard-nodeids") for n in names), f"slow job must upload slow-shard-nodeids: {names}"
    assert "slow-full-nodeids" in names, f"slow job must upload slow-full-nodeids, got {names}"
    assert "combined-full-nodeids" in names, f"slow job must upload combined-full-nodeids, got {names}"


def test_reconcile_consumes_slow_and_combined() -> None:
    job = _CI["jobs"]["shard-reconcile"]
    needs = job["needs"] if isinstance(job["needs"], list) else [job["needs"]]
    assert "slow" in needs, f"shard-reconcile must need slow, got {needs}"
    patterns = {
        str(s["with"]["pattern"])
        for s in job["steps"]
        if "download-artifact" in str(s.get("uses", "")) and isinstance(s.get("with"), dict) and "pattern" in s["with"]
    }
    # slow shards carry a -<shard> suffix (glob); the two single artifacts are matched by their exact
    # name as a literal glob pattern (a `-*` suffix would NOT match a suffix-less name).
    for required in ("slow-shard-nodeids-*", "slow-full-nodeids", "combined-full-nodeids"):
        assert required in patterns, f"reconcile must download {required} (proof #2/#3 inputs), got {patterns}"


def test_reconcile_body_has_conservation_sentinel() -> None:
    job = _CI["jobs"]["shard-reconcile"]
    bodies = "\n".join(s["run"] for s in job["steps"] if "run" in s)
    # the single text check: a load-bearing sentinel the conservation proof must carry, so this guard
    # is non-vacuous (numba-cache idiom). Removing proof #3 removes the marker -> RED.
    assert "# CONSERVATION:" in bodies, (
        "reconcile body must carry the '# CONSERVATION:' proof (non-slow union slow == not-e2e)"
    )
