"""Structural guard: the sharded CI matrix is internally consistent and deterministic.

Sharding the suite with ``pytest-split`` is only a valid PARTITION if (a) each shard axis is contiguous
``1..N``, (b) ``--splits N`` matches that axis, (c) collection order is pinned (no shuffle plugin) so every
shard collects identically, and (d) no test silently runs in zero shards. Since the asymmetric-sharding
refactor there are TWO matrix jobs -- ``test-ubuntu`` (``--splits 6``) and ``test-windows`` (``--splits 16``,
windows runs ~2.4x ubuntu) -- plus the ``slow`` job (``--splits 4``); each is checked independently. This is
the pre-flight complement to the runtime ``shard-reconcile`` job (which proves the node-ID partition per leg).
Same idiom as ``test_ci_slow_gating_wired`` / ``test_ci_pandas_span_wired``. See the CI-parallelization ADR.
"""

from __future__ import annotations

import ast
import pathlib
import re

import yaml

_REPO = pathlib.Path(__file__).resolve().parent.parent
_CI = yaml.safe_load((_REPO / ".github/workflows/ci.yml").read_text(encoding="utf-8"))

#: the three duration-sharded jobs and the matrix axis each shards over.
_SHARDED = [("test-ubuntu", "shard"), ("test-windows", "shard"), ("slow", "slow-shard")]


def _defines_njit(src: str) -> bool:
    """True iff ``src`` applies numba ``njit`` -- as a ``@njit(...)`` / ``@numba.njit(...)`` DECORATOR
    or as the ``njit(...)(fn)`` CALL form (``_turnover.py``). AST, not a regex, so a docstring mention
    of ``@njit`` is never a false positive (ADR-056: read code, not prose)."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    njit_names = {"njit", "_njit"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                f = dec.func if isinstance(dec, ast.Call) else dec
                name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
                if name in njit_names:
                    return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Call):  # njit(...)(fn)
            inner = node.func.func
            name = inner.attr if isinstance(inner, ast.Attribute) else getattr(inner, "id", None)
            if name in njit_names:
                return True
    return False


def _sharded_cmds() -> list[str]:
    return [
        s["run"]
        for job_name, _ in _SHARDED
        for s in _CI["jobs"][job_name]["steps"]
        if "run" in s and "--splits" in s["run"] and "pytest tests/" in s["run"]
    ]


def test_shard_axis_is_contiguous_1_to_N() -> None:
    for job_name, axis in _SHARDED:
        shards = _CI["jobs"][job_name]["strategy"]["matrix"][axis]
        assert shards == list(range(1, len(shards) + 1)), (
            f"{job_name} {axis} axis must be 1..N contiguous, got {shards}"
        )


def test_splits_value_matches_shard_count() -> None:
    for job_name, axis in _SHARDED:
        job = _CI["jobs"][job_name]
        n = len(job["strategy"]["matrix"][axis])
        group_var = f"matrix.{axis}"
        cmds = [s["run"] for s in job["steps"] if "run" in s and "--splits" in s["run"] and "pytest tests/" in s["run"]]
        assert cmds, f"no sharded pytest commands found in {job_name}"
        for cmd in cmds:
            m = re.search(r"--splits\s+(\d+)", cmd)
            assert m and int(m.group(1)) == n, f"{job_name}: --splits must equal shard count {n}: {cmd}"
            assert f"--group ${{{{ {group_var} }}}}" in cmd, f"{job_name}: missing per-shard --group {group_var}: {cmd}"


def test_every_sharded_command_pins_collection_order() -> None:
    for cmd in _sharded_cmds():
        assert "-p no:randomly" in cmd, f"sharded command must pin collection order: {cmd}"


def test_no_collection_shuffling_plugin_in_test_extra() -> None:
    # pytest-randomly (or any shuffle plugin) auto-activates and would break the shard partition; keep
    # it out entirely. Plain text scan on purpose -- `tomllib` is py3.11+ and ABSENT on the CI 3.10 leg
    # (this test itself broke that leg once); a whole-file scan is also stricter (the plugin must not
    # appear ANYWHERE, not just in [test]).
    text = (_REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert "pytest-randomly" not in text, (
        "pytest-randomly auto-activates and would break the shard partition; keep it out of pyproject.toml"
    )


def test_benchmark_is_a_standalone_job_not_on_a_shard() -> None:
    assert "benchmark" in _CI["jobs"], "benchmark must be its own parallel job (spec N1)"
    for job_name in ("test-ubuntu", "test-windows"):
        runs = " ".join(s.get("run", "") for s in _CI["jobs"][job_name]["steps"])
        assert "--benchmark-only" not in runs, f"benchmark-only must NOT sit on a sharded {job_name} step"


def test_shard_reconcile_job_exists_and_needs_the_sharded_jobs() -> None:
    job = _CI["jobs"].get("shard-reconcile")
    assert job is not None, "shard-reconcile job must exist"
    needs = job["needs"] if isinstance(job["needs"], list) else [job["needs"]]
    for required in ("test-ubuntu", "test-windows", "slow"):
        assert required in needs, f"shard-reconcile must need {required}, got {needs}"


def test_numba_cache_key_covers_all_njit_files() -> None:
    """Every @njit file (decorator OR call form) must be in the numba actions/cache hashFiles(), or its
    recompiled blob is never saved (cache HIT => no upload) and Lever C silently no-ops for it. The
    ``*_numba*`` naming is NOT relied on (``_turnover.py`` breaks it via the call form); this pins coverage
    so a NEW @njit file fails CI until the key is extended (P5/P10)."""
    njit_files = sorted(
        str(p.relative_to(_REPO)).replace("\\", "/")
        for p in (_REPO / "silly_kicks").rglob("*.py")
        if _defines_njit(p.read_text(encoding="utf-8"))
    )
    # non-vacuity (ADR-056): the detector must still find the awkward call-form file it exists for.
    assert "silly_kicks/xtgk/_turnover.py" in njit_files, (
        "detector no longer finds the call-form njit file (_turnover.py) -- it has drifted"
    )
    # EVERY job that compiles the kernels (both matrix legs + the dedicated slow job) must cover every @njit
    # file, or Lever C silently no-ops for the uncovered ones in that job.
    for job_name in ("test-ubuntu", "test-windows", "slow"):
        cache = [s for s in _CI["jobs"][job_name]["steps"] if "actions/cache" in str(s.get("uses", ""))]
        assert cache, f"no numba actions/cache step in the {job_name} job"
        patterns = re.findall(r"'([^']+)'", str(cache[0]["with"]["key"]))
        # Path.glob DOES treat ** as zero-or-more dirs, matching GitHub hashFiles -- fnmatch does NOT.
        covered: set[str] = set()
        for pat in patterns:
            covered |= {str(p.relative_to(_REPO)).replace("\\", "/") for p in _REPO.glob(pat)}
        missing = set(njit_files) - covered
        assert not missing, f"@njit files not covered by the {job_name} numba cache key: {sorted(missing)}"
