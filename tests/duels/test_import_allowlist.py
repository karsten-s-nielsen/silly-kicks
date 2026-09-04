"""duels is EVENT-ONLY: never imports silly_kicks.tracking; nothing imports duels.

AST module-level check (mirrors tests/territory/test_import_allowlist.py). Allowed silly-kicks deps:
silly_kicks.spadl (config), silly_kicks.id_compat. Each detector carries a planted-violation meta-test.
"""

from __future__ import annotations

import ast
import pathlib

import silly_kicks
import silly_kicks.duels  # must import cleanly

ROOT = pathlib.Path(silly_kicks.__file__).resolve().parent
DUELS = ROOT / "duels"
_BANNED_PREFIX = "silly_kicks.tracking"


def _imported_modules(path: pathlib.Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            mods.append(node.module)
        elif isinstance(node, ast.Import):
            mods.extend(a.name for a in node.names)
    return mods


def _is_banned(m: str) -> bool:
    return m == _BANNED_PREFIX or m.startswith(_BANNED_PREFIX + ".")


def _imports_duels(path: pathlib.Path) -> bool:
    return any(m == "silly_kicks.duels" or m.startswith("silly_kicks.duels.") for m in _imported_modules(path))


def test_duels_never_imports_tracking():
    offenders = {
        py.relative_to(DUELS).as_posix(): [m for m in _imported_modules(py) if _is_banned(m)]
        for py in sorted(DUELS.rglob("*.py"))
    }
    offenders = {k: v for k, v in offenders.items() if v}
    assert not offenders, f"{offenders}: duels is event-only -- must NEVER import silly_kicks.tracking."


def test_nothing_imports_duels():
    offenders = [
        py.relative_to(ROOT).as_posix()
        for py in sorted(ROOT.rglob("*.py"))
        if not py.is_relative_to(DUELS) and _imports_duels(py)
    ]
    assert not offenders, f"{offenders}: nothing in silly_kicks should import duels (a leaf metric)."


def test_public_surface_exists():
    from silly_kicks.duels import compute_duel_ratings, update_glicko  # noqa: F401


def test_banned_detector_fires_on_planted_violation(tmp_path):
    planted = tmp_path / "_p.py"
    planted.write_text("from silly_kicks.tracking import add_das\n", encoding="utf-8")
    assert any(_is_banned(m) for m in _imported_modules(planted))
    planted.write_text("from silly_kicks.id_compat import canonical_id\n", encoding="utf-8")
    assert not any(_is_banned(m) for m in _imported_modules(planted))
