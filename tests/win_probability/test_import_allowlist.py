"""AST import-allowlist for ``win_probability`` (both directions).

- The package imports NO ``tracking``, NO ``vaep``, NO ``match_outcome`` (self-contained, event-only).
- Nothing in ``silly_kicks`` imports ``win_probability`` except ``vaep``, and ``vaep``'s import must be
  function-local (a module-level ``import ...win_probability`` in ``vaep`` fails the gate).
"""

from __future__ import annotations

import ast
import pathlib

import silly_kicks

_PKG_ROOT = pathlib.Path(silly_kicks.__file__).resolve().parent
_WP_ROOT = _PKG_ROOT / "win_probability"
_FORBIDDEN = ("silly_kicks.tracking", "silly_kicks.vaep", "silly_kicks.match_outcome")


def _imported_modules(node: ast.AST) -> list[str]:
    mods: list[str] = []
    if isinstance(node, ast.Import):
        mods += [a.name for a in node.names]
    elif isinstance(node, ast.ImportFrom) and node.module:
        mods.append(node.module)
    return mods


def test_package_imports_no_tracking_vaep_or_match_outcome():
    for path in _WP_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            for mod in _imported_modules(node):
                for bad in _FORBIDDEN:
                    assert not (mod == bad or mod.startswith(bad + ".")), f"{path.name} imports {mod}"


def test_only_vaep_imports_win_probability_and_only_lazily():
    for path in _PKG_ROOT.rglob("*.py"):
        if _WP_ROOT in path.parents or path.parent == _WP_ROOT:
            continue
        rel = path.relative_to(_PKG_ROOT)
        in_vaep = rel.parts and rel.parts[0] == "vaep"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            for mod in _imported_modules(node):
                if mod == "silly_kicks.win_probability" or mod.startswith("silly_kicks.win_probability."):
                    assert in_vaep, f"{rel} imports win_probability but is not vaep"
                    col_offset = getattr(node, "col_offset", 0)  # ast.AST base lacks it; Import/ImportFrom carry it
                    assert col_offset > 0, f"{rel}: win_probability import must be function-local (module-level)"
