"""The locked closure of every pipeline stage (ADR-129, S5.9).

The CODE closure of a stage = the static import closure of its stage module plus the runner modules, restricted to
repo files (``silly_kicks/**``, ``scripts/**``). It follows every ``import`` / ``from ... import`` in EVERY code
object (an ``ast.walk`` over the whole module, so imports inside functions count -- ``modulefinder`` semantics), and
it adds every parent package ``__init__.py`` an import executes. Resolution is a stdlib AST resolver, NOT
``modulefinder``: ``scripts`` is a namespace package (no ``scripts/__init__.py``), and ``modulefinder`` crashes on it
(``AttributeError: 'NoneType' object has no attribute 'is_package'``, r2 B R2-03). Names resolve against the job's
``sys.path`` order: the repo root, then ``scripts/`` (the census bare ``from _crossed_icc import ...``). A name that
resolves to no repo file is third-party (``uv.lock`` and the image digest cover it). Files de-duplicate by path.
The DATA membership = the stage's declared ``data_files`` prefixes, expanded with ``git ls-files``. The lock records
MEMBERSHIP only (paths, never digests): digests come from HEAD at plan time.
"""

from __future__ import annotations

import ast
import functools
import json
import pathlib
import subprocess
import sys
from collections.abc import Sequence

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
LOCK = pathlib.Path(__file__).with_name("closures.lock.json")
RUNNER_MODULES = ("scripts._stages", "scripts._driver")
_REPO_DIRS = ("silly_kicks", "scripts")


class ClosureViolationError(RuntimeError):
    """A stage process loaded a repo module, opened a tracked repo file, or saw an environment variable outside its
    locked closure / data membership / registry. The unit writes no result."""


def _bases(root: pathlib.Path) -> tuple[pathlib.Path, ...]:
    return (root, root / "scripts")  # the job's sys.path order (`_stages.main` runs with both)


def _file_of(name: str, root: pathlib.Path) -> pathlib.Path | None:
    """The repo file of module ``name`` (``a.b`` -> ``a/b.py`` or ``a/b/__init__.py``), or None (third-party, or a
    namespace package with no ``__init__.py``)."""
    parts = name.split(".")
    for base in _bases(root):
        stem = base.joinpath(*parts)
        for cand in (stem.with_suffix(".py"), stem / "__init__.py"):
            if cand.is_file() and _inside(cand, root):
                return cand
    return None


def _package_inits(name: str, root: pathlib.Path) -> list[pathlib.Path]:
    """Every ``__init__.py`` of the parents of ``name`` that an import executes (``a.b.c`` -> ``a/__init__.py``,
    ``a/b/__init__.py``), when they exist (a namespace package has none)."""
    out = []
    parts = name.split(".")
    for i in range(1, len(parts)):
        f = _file_of(".".join(parts[:i]), root)
        if f is not None and f.name == "__init__.py":
            out.append(f)
    return out


def _module_name(path: pathlib.Path, root: pathlib.Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    parts = list(rel.parts[:-1]) if rel.name == "__init__" else list(rel.parts)
    return ".".join(parts)


def _imports(path: pathlib.Path, root: pathlib.Path) -> set[str]:
    """Every module name ``path`` imports, anywhere in the file; relative imports made absolute."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    me = _module_name(path, root)
    pkg = me if path.name == "__init__.py" else me.rpartition(".")[0]
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                anchor = pkg.split(".")[: len(pkg.split(".")) - node.level + 1] if pkg else []
                base = ".".join([*anchor, *(node.module.split(".") if node.module else [])])
            else:
                base = node.module or ""
            if base:
                names.add(base)
            names |= {f"{base}.{a.name}" if base else a.name for a in node.names if a.name != "*"}  # submodules
    return names


def _inside(path, root: pathlib.Path) -> bool:
    try:
        rel = pathlib.Path(path).resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return bool(rel.parts) and rel.parts[0] in _REPO_DIRS


@functools.cache
def module_closure(module: str, *, root: pathlib.Path = REPO_ROOT) -> tuple[str, ...]:
    start = _file_of(module, root)
    if start is None:
        raise ValueError(f"{module}: not a repo module")
    seen: set[pathlib.Path] = set()
    todo = [start, *_package_inits(module, root)]
    while todo:
        f = todo.pop().resolve()
        if f in seen:
            continue
        seen.add(f)
        for name in _imports(f, root):
            hit = _file_of(name, root)
            if hit is not None:
                todo.append(hit)
                todo.extend(_package_inits(name, root))
    return tuple(sorted(p.relative_to(root.resolve()).as_posix() for p in seen))


def _ls_files(prefixes: Sequence[str], root: pathlib.Path) -> tuple[str, ...]:
    if not prefixes:
        return ()
    out = subprocess.run(  # noqa: S603
        ["git", "ls-files", "--", *prefixes],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
        cwd=root,
    )
    return tuple(sorted(line for line in out.stdout.splitlines() if line))


def stage_module(st) -> str:
    return st.work.__module__


def stage_membership(st, *, root: pathlib.Path = REPO_ROOT) -> dict[str, list[str]]:
    mods = set(module_closure(stage_module(st), root=root))
    for r in RUNNER_MODULES:
        mods |= set(module_closure(r, root=root))
    return {"modules": sorted(mods), "data": list(_ls_files(tuple(st.data_files), root))}


def _population() -> list[str]:
    """The compliant pipeline modules, single-sourced in production code (`scripts/_pipeline_population.py`)."""
    from scripts._pipeline_population import compliant

    return [f"scripts.{name}" for name in compliant()]


def record(*, write: bool = True, root: pathlib.Path = REPO_ROOT) -> dict:
    import importlib

    lock: dict[str, dict] = {}
    for mod in _population():
        pipe = importlib.import_module(mod).PIPELINE
        for st in pipe.stages:
            lock[f"{pipe.name}/{st.name}"] = stage_membership(st, root=root)
    if write:
        LOCK.write_text(json.dumps(lock, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return lock


def load_lock() -> dict:
    return json.loads(LOCK.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args != ["record"]:
        raise SystemExit("usage: python -m scripts.hf_jobs closures record")
    record()
    return 0
