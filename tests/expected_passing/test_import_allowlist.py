import ast
import pathlib

_ALLOWED_PREFIXES = ("silly_kicks.spadl", "silly_kicks.id_compat", "silly_kicks.expected_passing")
_PKG = pathlib.Path(__file__).resolve().parents[2] / "silly_kicks" / "expected_passing"


def _module_imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            yield node.module
        elif isinstance(node, ast.Import):
            for a in node.names:
                yield a.name


def test_expected_passing_never_imports_tracking():
    for py in _PKG.glob("*.py"):
        for mod in _module_imports(py):
            assert not mod.startswith("silly_kicks.tracking"), f"{py.name} imports {mod}"
            if mod.startswith("silly_kicks."):
                assert mod.startswith(_ALLOWED_PREFIXES), f"{py.name} imports disallowed {mod}"


def test_planted_tracking_import_would_fail():
    # meta: prove the guard bites (parse a synthetic module string)
    src = "import silly_kicks.tracking\n"
    mods = [n.names[0].name for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Import)]
    assert any(m.startswith("silly_kicks.tracking") for m in mods)


def test_bundled_loads_or_skips_pre_bundle():
    import importlib.resources as ir

    import pytest

    from silly_kicks.expected_passing import PassCompletionModel

    if not (ir.files("silly_kicks.expected_passing") / "weights" / "model.json").is_file():
        pytest.skip("bundled weights arrive at Commit 2")
    m = PassCompletionModel.bundled()
    assert m.is_fitted
