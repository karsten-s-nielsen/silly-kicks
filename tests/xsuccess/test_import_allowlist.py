"""TF-61: `silly_kicks.xsuccess` is event-only — it may import only the allowlist and NEVER
`silly_kicks.tracking`. Mirrors `tests/expected_passing/test_import_allowlist.py` (AST, module-level
imports only; function-local `xgboost`/`sklearn` are invisible to this gate by design)."""

import ast
import pathlib

_PKG = pathlib.Path(__file__).parents[2] / "silly_kicks" / "xsuccess"
_ALLOWED_SILLY = {"silly_kicks.spadl", "silly_kicks.id_compat"}
_ALLOWED_TOP = {"numpy", "pandas"}
_ALLOWED_STDLIB = {"__future__", "importlib", "hashlib", "json", "warnings", "pathlib", "dataclasses"}
_FORBIDDEN_SUBSTR = ("silly_kicks.tracking",)


def _is_module_level(tree: ast.Module, node: ast.AST) -> bool:
    return any(node is child for child in ast.iter_child_nodes(tree))


def _module_level_imports(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)) or not _is_module_level(tree, node):
            continue
        if isinstance(node, ast.Import):
            mods |= {a.name for a in node.names}
        elif node.module is not None and node.level == 0:  # skip relative (._features etc.)
            mods.add(node.module)
    return mods


def test_xsuccess_imports_only_allowlist():
    files = list(_PKG.glob("*.py"))
    assert files, "xsuccess package has no modules"
    for py in files:
        for mod in _module_level_imports(py):
            assert not any(s in mod for s in _FORBIDDEN_SUBSTR), f"{py.name} imports forbidden {mod}"
            if mod.startswith("silly_kicks"):
                assert any(mod == a or mod.startswith(a + ".") for a in _ALLOWED_SILLY), f"{py.name}: {mod}"
            else:
                assert mod.split(".")[0] in (_ALLOWED_TOP | _ALLOWED_STDLIB), f"{py.name}: {mod}"
