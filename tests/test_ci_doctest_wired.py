"""Structural guard: CI executes doctests on the PUBLIC surface, on every leg (PR-S124).

The ``--doctest-modules`` step is the enforcement that public-API examples stay runnable. If the step is
dropped, or its private-module ``--ignore-glob`` drifts, public examples silently stop being checked (or
private modules start failing CI). Since the asymmetric-sharding refactor the matrix is split into
``test-ubuntu`` + ``test-windows``; each must carry exactly one doctest step, shard-1-gated. We assert the
SEMANTIC wiring -- step exists per matrix job, targets ``silly_kicks/``, ignores single-underscore privates
while KEEPING dunder ``__init__``, runs on every leg (no ``matrix.primary``) once per leg (shard 1) -- not
mere string presence. See the AGENTS.md Testing note.
"""

from __future__ import annotations

import fnmatch
import pathlib
import re

import yaml

_REPO = pathlib.Path(__file__).resolve().parent.parent
_CI = _REPO / ".github" / "workflows" / "ci.yml"

_MATRIX_JOBS = ("test-ubuntu", "test-windows")


def _doctest_steps_by_job() -> dict[str, list[dict]]:
    ci = yaml.safe_load(_CI.read_text(encoding="utf-8"))
    return {
        job: [s for s in ci["jobs"][job]["steps"] if "run" in s and "--doctest-modules" in s["run"]]
        for job in _MATRIX_JOBS
    }


def _guard(expr: object) -> str:
    """Normalize a step ``if:`` to its inner expression (mirrors test_ci_slow_gating_wired)."""
    s = "".join(str(expr).split())
    if s.startswith("${{") and s.endswith("}}"):
        s = s[3:-2]
    return s


def test_public_doctest_step_is_wired_on_every_matrix_job() -> None:
    for job, steps in _doctest_steps_by_job().items():
        assert len(steps) == 1, (
            f"{job}: expected exactly one --doctest-modules step, got {len(steps)}: {[s.get('run') for s in steps]}"
        )
        run = steps[0]["run"]
        assert "silly_kicks/" in run, f"{job}: the doctest step must target the silly_kicks package"
        assert "--ignore-glob" in run, f"{job}: the doctest step must scope to the public surface via --ignore-glob"
        # EVERY leg: the step must NOT be gated on matrix.primary (public examples are enforced on all
        # OS/interpreter axes because doctest output is version-sensitive).
        guard = _guard(steps[0].get("if", ""))
        assert "matrix.primary" not in guard, (
            f"{job}: the public-doctest step must run on EVERY leg, not be gated by {guard!r}"
        )


def test_doctest_runs_once_per_leg_on_shard_1() -> None:
    """Under sharding, doctest is a separate MODULE invocation (not sharded), so it must run on each leg's
    shard 1 ONLY -- once per leg, not N times. Pins the shard-1 gating so it can't drift to every shard
    (wasteful) or off a leg."""
    for job, steps in _doctest_steps_by_job().items():
        guard = _guard(steps[0].get("if", "")).replace("'", "").replace('"', "")
        assert "matrix.shard==1" in guard, f"{job}: doctest must be gated on shard 1 (once per leg), got {guard!r}"


def test_ignore_glob_excludes_privates_but_keeps_dunder_and_public() -> None:
    """Semantic check on the glob itself (fnmatch, exactly how pytest's --ignore-glob matches): it must skip
    single-underscore private modules while KEEPING dunder ``__init__.py`` and public (non-underscore)
    modules -- else the enforced surface is silently wrong. Checked on every matrix job's step."""
    for job, steps in _doctest_steps_by_job().items():
        run = steps[0]["run"]
        m = re.search(r"--ignore-glob=(['\"]?)([^'\"\s]+)\1", run)
        assert m, f"{job}: could not parse --ignore-glob value from: {run}"
        glob = m.group(2)
        # a single-underscore private IS ignored (not enforced)...
        assert fnmatch.fnmatch("silly_kicks/tracking/pitch_control/_surface.py", glob), (
            f"{job}: glob {glob!r} must ignore single-underscore private modules"
        )
        # ...but dunder __init__ and public modules are NOT ignored (they stay enforced).
        assert not fnmatch.fnmatch("silly_kicks/tracking/__init__.py", glob), (
            f"{job}: glob {glob!r} must NOT ignore dunder __init__.py (public package modules)"
        )
        assert not fnmatch.fnmatch("silly_kicks/tracking/features.py", glob), (
            f"{job}: glob {glob!r} must NOT ignore public (non-underscore) modules"
        )
