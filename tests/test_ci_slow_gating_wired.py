"""Structural guard: the slow suite runs in a DEDICATED job, not inline on a special matrix leg.

Under the slow-decouple topology (spec 2026-09-25-ci-runtime-slow-decouple), every matrix leg runs the same
``not e2e and not slow`` selection -- so no leg is structurally heavier -- and the platform-/interpreter-
INVARIANT ``@pytest.mark.slow`` tail runs once in a dedicated ``slow`` job on the primary interpreter. Since
the asymmetric-sharding refactor the matrix is split into ``test-ubuntu`` + ``test-windows``; both must carry
exactly one unconditional non-slow bulk step. This guard pins that on the parsed YAML (not string presence):
no ``matrix.primary`` anywhere, one unconditional non-slow bulk step per matrix job, and a ``slow`` job
selecting ``slow and not e2e`` on ubuntu/3.12.

See ADR-023.
"""

from __future__ import annotations

import pathlib

import yaml

_REPO = pathlib.Path(__file__).resolve().parent.parent
_CI = _REPO / ".github" / "workflows" / "ci.yml"

_MATRIX_JOBS = ("test-ubuntu", "test-windows")


def test_slow_is_a_dedicated_job_not_inline() -> None:
    ci = yaml.safe_load(_CI.read_text(encoding="utf-8"))

    for job_name in _MATRIX_JOBS:
        job = ci["jobs"][job_name]
        # (a) matrix.primary is gone -- slow no longer runs inline on a special leg
        include = job["strategy"]["matrix"].get("include", [])
        assert not any(e.get("primary") for e in include), f"{job_name}: matrix.primary must be removed, got {include}"
        assert "matrix.primary" not in str(job["steps"]), f"{job_name}: no matrix.primary token anywhere in steps"

        # (b) exactly one UNCONDITIONAL bulk step, excluding slow
        bulk = [
            s for s in job["steps"] if "run" in s and "pytest tests/" in s["run"] and "--benchmark-skip" in s["run"]
        ]
        assert len(bulk) == 1, (
            f"{job_name}: expected one unconditional bulk step, got {len(bulk)}: {[s.get('run') for s in bulk]}"
        )
        assert "if" not in bulk[0], f"{job_name}: bulk step must be unconditional (no per-leg gate)"
        assert "not e2e and not slow" in bulk[0]["run"], f"{job_name}: bulk step must exclude slow"

    # (c) a dedicated `slow` job runs the invariant tail once on the primary interpreter
    slow = ci["jobs"]["slow"]
    assert str(slow["runs-on"]).startswith("ubuntu"), f"slow job must run on ubuntu, got {slow['runs-on']}"
    setup = [s for s in slow["steps"] if "setup-python" in str(s.get("uses", ""))]
    assert setup and str(setup[0]["with"]["python-version"]) == "3.12", (
        "slow job must use python 3.12 (ADR-023 invariance)"
    )
    sbulk = [s for s in slow["steps"] if "run" in s and "pytest tests/" in s["run"] and "--benchmark-skip" in s["run"]]
    assert len(sbulk) == 1, f"expected one slow bulk step, got {len(sbulk)}"
    assert "slow and not e2e" in sbulk[0]["run"], "slow job must select 'slow and not e2e'"


def test_slow_marker_set_is_non_empty() -> None:
    hits = sum(1 for p in (_REPO / "tests").rglob("*.py") if "pytest.mark.slow" in p.read_text(encoding="utf-8"))
    assert hits >= 1, "no tests carry @pytest.mark.slow; the gating would be a no-op"
