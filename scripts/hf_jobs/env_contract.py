"""The environment-variable registry and its AST gate (ADR-129, S5.9).

Every ``SILLY_KICKS_*`` variable is registered in ONE class: ``neutral`` (orchestration / provenance / caches /
diagnostics; never changes a result byte), ``output`` (can change bytes; its value or absence enters the token) or
``forbidden_in_job`` (selects a data source, a credential or a model outside the token; set in a job it fails the job
before any unit runs). A fixed set of platform names needs no class. The AST gate proves every environment READ in
``silly_kicks/`` and ``scripts/`` resolves to a registered name; the job gate (``check_job_env``) refuses any
unregistered or forbidden name in a running job.
"""

from __future__ import annotations

import ast
import pathlib
from collections.abc import Iterable

from scripts.hf_jobs.closures import ClosureViolationError

#: The sentinel ``env_reads`` returns for a key expression it cannot resolve to a literal name.
UNRESOLVED = object()

NEUTRAL = frozenset(
    f"SILLY_KICKS_{n}"
    for n in (
        "RUN_KEY",
        "JOB_LABEL",
        "ATTEMPT_ID",
        "ACTION_KIND",
        "PHASE_FILE",
        "IMAGE",
        "IN_HF_JOB",
        "DATASET_READ_CAP",
        "DATASET_READ_GATE_DIR",
        "STORE_READ_CAP",
        "NUMBA_CACHE",
        "CORPUS_CACHE_DIR",
        "IMPORTANCE_PID_LOG",
        "DEBUG_MODULES",
        "STAGE_CHECKS",
        "DATA_ROOT",
    )
)
OUTPUT = frozenset(
    f"SILLY_KICKS_{n}"
    for n in (
        "GHOST_FORCE_NUMPY",
        "DAS_FORCE_NUMPY",
        "COORDINATION_FORCE_NUMPY",
        "COORDINATION_REFERENCE_NUMERICS",
        "ASSERT_INVARIANTS",
        "OPEN_DATA_REVISION",
    )
)
FORBIDDEN_IN_JOB = frozenset(
    {f"SILLY_KICKS_{n}" for n in ("OPEN_DATA_BASE", "PINING_MANIFEST_DIR", "GHOST_GK_PATH", "GHOST_OUTFIELD_PATH")}
    | {"PINING_API_URL", "PINING_FOR_THE_DATA_TOKEN", "SB_USERNAME", "SB_PASSWORD"}
)
FORBIDDEN_PREFIXES = ("DATABRICKS_", "SK_DAS_")
PLATFORM_NAMES = frozenset(
    {
        "PATH",
        "HOME",
        "TMPDIR",
        "HF_TOKEN",
        "HF_HOME",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN",
        "HF_HUB_OFFLINE",
        "NUMBA_CACHE_DIR",
        "NUMBA_NUM_THREADS",
        *__import__("scripts._thread_pin", fromlist=["DETERMINISTIC_VARS"]).DETERMINISTIC_VARS,
    }
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_DEFAULT_SEARCH = (_REPO_ROOT, _REPO_ROOT / "scripts")


def registered(name: str) -> bool:
    """True when ``name`` has a class (a job may hold it, modulo the forbidden/job rules of ``check_job_env``)."""
    return name in NEUTRAL or name in OUTPUT or name in FORBIDDEN_IN_JOB or name.startswith(FORBIDDEN_PREFIXES)


def output_values(environ) -> dict[str, str | None]:
    """The token's environment slice: every ``output`` variable's value (or None when unset), name-sorted."""
    return {n: environ.get(n) for n in sorted(OUTPUT)}


#: Output switches the controller must NOT carry into its ``--plan`` subprocess: the job never gets them
#: (``launch.job_env`` sets none), so an ambient value would move the controller token away from the job
#: token and the content-addressed generation would never match. ``OPEN_DATA_REVISION`` is left in place: it
#: is excluded from the token (the dataset identity keys it) and the launcher sets it in the job.
_PLAN_STRIP = OUTPUT - frozenset({"SILLY_KICKS_OPEN_DATA_REVISION"})


def controller_plan_env(environ) -> dict[str, str]:
    """The environment for the controller's ``--plan`` subprocess (CCTP-IMPL-26): the ambient environment with
    every ``output`` switch removed, so the token's output slice equals the job's whatever the operator's shell
    holds. The controller therefore NEVER keys the token on an output switch's presence; the job must not set one
    either (``launch.job_env`` refuses it, CCTP-IMPL-27), or the two sides would diverge. ``OPEN_DATA_REVISION``
    is kept here: it is excluded from the token and the ``--plan`` subprocess may read it."""
    return {k: v for k, v in environ.items() if k not in _PLAN_STRIP}


def check_job_env(environ) -> None:
    """In a job (``SILLY_KICKS_IN_HF_JOB`` set), refuse any ``SILLY_KICKS_*`` not registered, any
    ``forbidden_in_job`` name and any name under a forbidden prefix (cause ``infra``: the job fails before any unit)."""
    if not environ.get("SILLY_KICKS_IN_HF_JOB"):
        return
    for name in environ:
        if name in FORBIDDEN_IN_JOB or name.startswith(FORBIDDEN_PREFIXES):
            raise ClosureViolationError(f"forbidden environment variable in a job: {name} (S5.9 forbidden_in_job)")
        if name.startswith("SILLY_KICKS_") and not registered(name):
            raise ClosureViolationError(
                f"unregistered SILLY_KICKS_* variable in a job: {name} (register it in env_contract)"
            )


# ---- the AST gate -----------------------------------------------------------------------------------


def _const_names(tree: ast.Module) -> dict[str, str]:
    """Module-level ``NAME = "literal"`` bindings (the indirect-read case: ``_ENV_VAR = "SILLY_KICKS_..."``)."""
    out: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out[t.id] = node.value.value
    return out


def _import_sources(tree: ast.Module) -> dict[str, tuple[str, str]]:
    """``local name -> (module, original name)`` for ``from <module> import <name>`` of a repo module (so the
    gate can follow an imported env-name constant to its definition)."""
    out: dict[str, tuple[str, str]] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for a in node.names:
                if a.name != "*":
                    out[a.asname or a.name] = (node.module, a.name)
    return out


def _module_aliases(tree: ast.Module) -> dict[str, str]:
    """``local name -> dotted module path`` for module imports, so an attribute key ``mod.CONST`` resolves:
    ``import a.b.c as z`` -> ``z``; ``from a.b import sub`` -> ``sub`` (when ``a.b.sub`` is a submodule)."""
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.asname:
                    out[a.asname] = a.name
                else:
                    out[a.name.split(".")[0]] = a.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for a in node.names:
                if a.name != "*":
                    out[a.asname or a.name] = f"{node.module}.{a.name}"  # a submodule, if it resolves to a file
    return out


def _has_os_environ_import(tree: ast.Module) -> bool:
    return any(
        isinstance(n, ast.ImportFrom) and n.module == "os" and any(a.name == "environ" for a in n.names)
        for n in ast.walk(tree)
    )


def _module_file(module: str, search: Iterable[pathlib.Path]) -> pathlib.Path | None:
    parts = module.split(".")
    for base in search:
        for cand in (base.joinpath(*parts).with_suffix(".py"), base.joinpath(*parts) / "__init__.py"):
            if cand.is_file():
                return cand
    return None


def _resolve_name(name: str, consts: dict[str, str], imports: dict[str, tuple[str, str]], search):
    """Resolve a ``Name`` env key: a module-level literal, or an imported literal (read the source module once)."""
    if name in consts:
        return consts[name]
    if name in imports:
        module, orig = imports[name]
        f = _module_file(module, search)
        if f is not None:
            src_consts = _const_names(ast.parse(f.read_text(encoding="utf-8")))
            if orig in src_consts:
                return src_consts[orig]
    return UNRESOLVED


def _is_env_read(call: ast.Call, *, environ_imported: bool) -> bool:
    """``os.environ[...]`` is a Subscript, handled separately; here: ``os.environ.get``, ``os.getenv`` and --
    only when the module did ``from os import environ`` -- a bare ``environ.get`` (so a local parameter named
    ``environ`` is NOT a false match)."""
    f = call.func
    if isinstance(f, ast.Attribute):
        if f.attr == "get" and isinstance(f.value, ast.Attribute) and f.value.attr == "environ":
            return True
        if environ_imported and f.attr == "get" and isinstance(f.value, ast.Name) and f.value.id == "environ":
            return True
        if f.attr == "getenv" and isinstance(f.value, ast.Name) and f.value.id == "os":
            return True
    return False


def _resolve_attr(expr: ast.Attribute, aliases: dict[str, str], search) -> object:
    """Resolve an attribute key ``mod.CONST`` (e.g. ``phases.ACTION_KIND_ENV``) by reading ``mod``'s source."""
    if isinstance(expr.value, ast.Name) and expr.value.id in aliases:
        f = _module_file(aliases[expr.value.id], search)
        if f is not None:
            src_consts = _const_names(ast.parse(f.read_text(encoding="utf-8")))
            if expr.attr in src_consts:
                return src_consts[expr.attr]
    return UNRESOLVED


def _key_name(expr: ast.expr, consts, imports, aliases, search):
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return expr.value
    if isinstance(expr, ast.Name):
        return _resolve_name(expr.id, consts, imports, search)
    if isinstance(expr, ast.Attribute):
        return _resolve_attr(expr, aliases, search)
    return UNRESOLVED


def env_reads(tree: ast.Module, *, module_path: pathlib.Path, search: Iterable[pathlib.Path] | None = None) -> list:
    """Every environment-variable name ``tree`` reads: a resolved string, or ``UNRESOLVED`` for a key the gate
    cannot reduce to a literal. Detects ``os.environ[X]``, ``os.environ.get(X)``, ``os.getenv(X)`` and
    ``environ.get(X)`` (after ``from os import environ``). ``X`` is a literal, a module-level name bound to a
    literal, a name imported from a repo module, or ``mod.CONST`` of an imported repo module (resolved by
    reading that module)."""
    bases = tuple(search) if search is not None else _DEFAULT_SEARCH
    consts = _const_names(tree)
    imports = _import_sources(tree)
    aliases = _module_aliases(tree)
    environ_imported = _has_os_environ_import(tree)
    out: list = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "environ"
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "os"
        ):
            out.append(_key_name(node.slice, consts, imports, aliases, bases))
        elif isinstance(node, ast.Call) and _is_env_read(node, environ_imported=environ_imported) and node.args:
            out.append(_key_name(node.args[0], consts, imports, aliases, bases))
    return out
