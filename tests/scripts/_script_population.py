"""One AST walk over `scripts/*.py`, shared by every gate that derives a driver population.

ADR-052's corpus-driver gate and Cycle B's artifact-driver gate need the same scaffolding --
glob, skip private, parse -- and differ only in their predicate. Two independent walkers would
drift with nothing relating them, which is the defect class this cycle exists to remove. The
reconciliation is structural: single-source the UNIVERSE, let the predicates differ.
"""

from __future__ import annotations

import ast
import functools
import pathlib

SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "scripts"


@functools.cache
def iter_scripts() -> dict[str, ast.AST]:
    """Every non-private script in `scripts/`, parsed once per session, keyed by stem.

    Cached because both gates call this from inside PARAMETRISED test bodies, not just at
    collection. Measured: 97.6 ms to parse the 48 scripts, and ADR-052's population gate alone
    reaches it 23 times for its 22 drivers -- 2.2 s from one module, before the artifact-driver
    gate's own calls. `@functools.cache` is the established idiom here (the `spadlconfig`
    DataFrames use it).

    The cache assumes `scripts/` does not change mid-session, which holds: a test run never
    writes a driver. Callers must not mutate the returned dict -- it is shared.
    """
    return {
        p.stem: ast.parse(p.read_text(encoding="utf-8"))
        for p in sorted(SCRIPTS.glob("*.py"))
        if not p.name.startswith("_")
    }


def _docstring_ids(tree: ast.AST) -> set[int]:
    """Identity of every leading Expr(Constant(str)) that IS a docstring."""
    out: set[int] = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(n, "body", None)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                out.add(id(body[0].value))
    return out


def string_literals(tree: ast.AST) -> set[str]:
    """Every string literal EXCEPT docstrings.

    Docstrings must be excluded or every content predicate fires on PROSE. Measured: three
    separate source-text scans were fooled during this cycle's review --
    `make_ghost_gk_golden` matched a `_weights` rule solely because its module docstring
    mentions `test_weights_bundle_golden.py`; `render_sb360_matrix` matched a `_provenance`
    scan through the sentence "No provenance guard, deliberately."; and
    `regenerate_gs_et_native_gk` matched a corpus-loader scan through a docstring saying it
    MIRRORS the loader. None of the three carried the literal in code.

    `tests/scripts/test_provenance_wiring.py` already learned this once -- its
    `_shells_out_to_rev_parse` is AST-matched on CALLS with the comment "they cannot tell a
    described defect from a committed one". This is that same lesson, in the shared seam so
    both populations inherit it.
    """
    skip = _docstring_ids(tree)
    return {
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in skip
    }


def called_names(tree: ast.AST) -> set[str]:
    """Every called function/method name -- `f(...)` and `x.f(...)` alike."""
    return {
        (getattr(n.func, "id", "") or getattr(n.func, "attr", "")) for n in ast.walk(tree) if isinstance(n, ast.Call)
    }
