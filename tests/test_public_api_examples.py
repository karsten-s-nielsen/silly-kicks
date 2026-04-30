"""Enforce: every public function / class / method docstring includes an Examples section.

Closes D-8 (PR-S13). Backstops the discipline by failing CI when a future
PR adds a public symbol without an Examples section.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Files containing public API. Excludes underscore-prefixed modules,
# tests, and scripts.
_PUBLIC_MODULE_FILES = (
    "silly_kicks/spadl/utils.py",
    "silly_kicks/spadl/statsbomb.py",
    "silly_kicks/spadl/opta.py",
    "silly_kicks/spadl/wyscout.py",
    "silly_kicks/spadl/sportec.py",
    "silly_kicks/spadl/metrica.py",
    "silly_kicks/spadl/kloppy.py",
    "silly_kicks/atomic/spadl/utils.py",
    "silly_kicks/atomic/spadl/base.py",
    "silly_kicks/vaep/base.py",
    "silly_kicks/vaep/hybrid.py",
    "silly_kicks/atomic/vaep/base.py",
    "silly_kicks/xthreat.py",
    "silly_kicks/vaep/labels.py",
    "silly_kicks/vaep/formula.py",
    "silly_kicks/atomic/vaep/features.py",
    "silly_kicks/atomic/vaep/labels.py",
    "silly_kicks/atomic/vaep/formula.py",
    "silly_kicks/vaep/features/core.py",
    "silly_kicks/vaep/features/actiontype.py",
    "silly_kicks/vaep/features/result.py",
    "silly_kicks/vaep/features/bodypart.py",
    "silly_kicks/vaep/features/spatial.py",
    "silly_kicks/vaep/features/temporal.py",
    "silly_kicks/vaep/features/context.py",
    "silly_kicks/vaep/features/specialty.py",
)

# Pure-type symbols that don't fit the illustrative-example pattern.
# Adding a new entry here is a deliberate documentation-policy decision —
# the additive-only nature is a forcing function.
_SKIP_SYMBOLS = frozenset(
    {
        "BoundaryMetrics",  # TypedDict — fields are the documentation
        "CoverageMetrics",  # TypedDict
        "ConversionReport",  # TypedDict
    }
)


def _has_examples_section(docstring: str | None) -> bool:
    """True if the docstring contains a NumPy-style Examples section or doctest."""
    if not docstring:
        return False
    if "Examples\n    --------" in docstring or "Examples\n--------" in docstring:
        return True
    return ">>> " in docstring


def _walk_public_definitions(tree: ast.AST) -> list[tuple[str, int, str, ast.AST]]:
    """Yield (kind, lineno, qualified_name, node) for top-level public defs + public methods."""
    out: list[tuple[str, int, str, ast.AST]] = []
    for node in tree.body:  # type: ignore[attr-defined]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_") or node.name in _SKIP_SYMBOLS:
                continue
            out.append(("function", node.lineno, node.name, node))
        elif isinstance(node, ast.ClassDef):
            if node.name.startswith("_") or node.name in _SKIP_SYMBOLS:
                continue
            out.append(("class", node.lineno, node.name, node))
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if child.name.startswith("_") or child.name in _SKIP_SYMBOLS:
                        continue
                    out.append(("method", child.lineno, f"{node.name}.{child.name}", child))
    return out


@pytest.mark.parametrize("file_path", _PUBLIC_MODULE_FILES)
def test_public_definitions_have_examples_section(file_path: str):
    """Every public function / class / method in *file_path* has an Examples section.

    See ``silly_kicks.spadl.add_possessions`` and ``silly_kicks.spadl.boundary_metrics``
    for canonical illustrative-style examples. Add a 3-7 line example showing typical
    usage; no doctest verification is required.
    """
    abs_path = REPO_ROOT / file_path
    assert abs_path.exists(), f"public-API module file does not exist: {file_path}"

    tree = ast.parse(abs_path.read_text(encoding="utf-8"))
    missing: list[str] = []
    for kind, lineno, name, node in _walk_public_definitions(tree):
        doc = ast.get_docstring(node)
        if not _has_examples_section(doc):
            missing.append(f"  {file_path}:{lineno}  {kind}  {name}")

    assert not missing, (
        f"Public symbols in {file_path} missing 'Examples' section in docstring:\n"
        + "\n".join(missing)
        + "\n\nAdd a 3-7 line illustrative Examples section. See "
        "`silly_kicks.spadl.add_possessions` or `silly_kicks.spadl.boundary_metrics` "
        "for the canonical style. Pure-type symbols (TypedDict / dataclass) that don't "
        "fit the example pattern can be added to `_SKIP_SYMBOLS` in this test file — "
        "but only with a clear documentation-policy justification."
    )
