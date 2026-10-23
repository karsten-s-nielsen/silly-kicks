"""Fail-closed runtime checks for a stage process (ADR-129, S5.9): modules, tracked-file opens, environment.

Every process that executes a stage callable -- the unit worker, the reduce command and the inner-pool children --
installs these at start (a fresh interpreter, on HF and in every CI toy run). A violation raises
``ClosureViolationError`` and the unit writes no result.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

from scripts.hf_jobs.closures import ClosureViolationError

_STATE: dict = {}
#: The repo dirs the closure governs (like ``closures._module_source``): a tracked file outside these is not a
#: repo closure module, so the import audit never governs a ``tests/`` or ``docs/`` import.
_REPO_DIRS = ("silly_kicks", "scripts")
#: Harness modules that reach EVERY stage closure through the runner's function-level imports (none defines a
#: stage callable; `test_stage_modules` proves that). They are always allowed in the module check.
_HARNESS = (
    "scripts/hf_jobs/executor.py",
    "scripts/hf_jobs/stage_checks.py",
    "scripts/hf_jobs/env_contract.py",
    "scripts/hf_jobs/closures.py",
    "scripts/hf_jobs/phases.py",
    "scripts/_thread_pin.py",
    "scripts/_mem_cap.py",
    "scripts/_parallel_launch.py",
    "scripts/_dataset_reads.py",
    "scripts/__init__.py",
    "scripts/hf_jobs/__init__.py",
)


def _tracked(root: pathlib.Path) -> frozenset[str]:
    out = subprocess.run(
        ["git", "ls-files"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
        cwd=root,
    )
    return frozenset(out.stdout.splitlines())


def install_stage_checks(stage_key: str, *, root: pathlib.Path, entry=None, tracked=None) -> None:
    from scripts.hf_jobs import env_contract
    from scripts.hf_jobs.closures import load_lock

    entry = entry if entry is not None else load_lock()[stage_key]
    env_contract.check_job_env(os.environ)
    if os.environ.get("HF_HUB_OFFLINE") != "1":  # set by the PROCESS ENVIRONMENT, before any import (r1 B PLAN-08)
        raise ClosureViolationError(f"{stage_key}: a stage process must start with HF_HUB_OFFLINE=1")
    # uv.lock and the dataset lock are infrastructure in EVERY token (the image digest / the dataset identity),
    # so a stage may always open them (like uv.lock, S5.9).
    allowed_open = set(entry["modules"]) | set(entry["data"]) | {"uv.lock", "scripts/hf_jobs/datasets.lock.json"}
    _STATE["root"] = root.resolve()  # `_rel` below (in the data_roots comprehension) needs it set FIRST
    _STATE.update(
        root=root.resolve(),
        key=stage_key,
        modules=set(entry["modules"]) | set(_HARNESS),
        allowed=allowed_open,
        tracked=frozenset(tracked) if tracked is not None else _tracked(root),
        # repo-relative, like every path the hook compares (an absolute ctx.data_root normalizes here;
        # a data root outside the repo has no tracked files to guard; r3 B R3-02)
        data_roots=tuple(
            r for d in (os.environ.get("SILLY_KICKS_DATA_ROOT"),) if d for r in (_rel(d),) if r and r != "."
        ),
    )
    sys.addaudithook(_audit)
    check_modules(stage_key)


def _rel(path) -> str | None:
    try:
        return pathlib.Path(path).resolve().relative_to(_STATE["root"]).as_posix()
    except (ValueError, OSError, TypeError, KeyError):
        return None


def _audit(event: str, args) -> None:
    if not _STATE:
        return
    if event == "import":
        # Module governance (S5.9): an import of a CLOSURE-DIR module (silly_kicks/ or scripts/) outside the
        # closure fails CLOSED DURING the unit (before any result is written) and in every inner-pool child. It
        # fires for cached (.pyc) imports too, which the open hook cannot see; the after-unit `check_modules` is
        # a backstop for pre-installed modules. Scoped to the closure dirs (like `_module_source`): a `tests/`
        # import is outside the governed domain, never a repo closure module.
        name = args[0] if args and isinstance(args[0], str) else ""
        for cand in (name.replace(".", "/") + ".py", name.replace(".", "/") + "/__init__.py"):
            if cand.split("/", 1)[0] in _REPO_DIRS and cand in _STATE["tracked"] and cand not in _STATE["modules"]:
                raise ClosureViolationError(f"{_STATE['key']}: imported repo module {name!r} outside the closure")
        return
    if event != "open":
        return
    rel = _rel(args[0]) if args and isinstance(args[0], (str, bytes, os.PathLike)) else None
    if rel is None or rel not in _STATE["tracked"] or rel in _STATE["allowed"]:
        return
    if rel.endswith((".py", ".pyc")):
        # Module source/bytecode: governed by the `import` event (CCTP-CORE-03). The open hook must NOT flag it
        # -- importing a task function unpickled in a spawn child opens its parents' `__init__.py` (e.g.
        # `tests/__init__.py`), and a non-closure repo-module IMPORT is already refused above, cleanly.
        return
    if any(rel.startswith(r.rstrip("/") + "/") for r in _STATE["data_roots"]):
        return
    # A tracked NON-.py file outside the closure, the data membership and the dataset roots fails closed.
    raise ClosureViolationError(
        f"{_STATE['key']}: opened tracked repo file {rel!r} outside its closure and data members"
    )


def _module_source(path: str) -> str | None:
    """The repo-relative SOURCE path of a loaded module, for files under ``silly_kicks/`` or ``scripts/`` only (a
    venv inside the repo root is never a repo module, r1 B CCTP-PLAN-04). A sourceless ``.pyc`` maps to the source
    it was compiled from."""
    import importlib.util

    if path.endswith(".pyc"):
        try:
            path = importlib.util.source_from_cache(path)
        except ValueError:  # a sourceless .pyc outside __pycache__: map by name
            path = path[:-1]
    rel = _rel(path)
    if rel is None or not rel.endswith(".py") or rel.split("/", 1)[0] not in ("silly_kicks", "scripts"):
        return None
    return rel


def check_modules(stage_key: str) -> None:
    bad = sorted(
        rel
        for m in list(sys.modules.values())
        if (rel := _module_source(getattr(m, "__file__", None) or "")) and rel not in _STATE["modules"]
    )
    if bad:
        raise ClosureViolationError(f"{stage_key}: loaded repo module(s) outside the closure: {bad[:5]}")
