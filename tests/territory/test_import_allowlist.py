"""territory is EVENT-ONLY: never imports silly_kicks.tracking; nothing imports territory.

AST module-level check (mirrors tests/shot_stopping/test_import_allowlist.py). Allowed silly-kicks deps:
silly_kicks.spadl (config), silly_kicks.id_compat, silly_kicks.xthreat (the injected model type). Each
detector carries a planted-violation meta-test.
"""

from __future__ import annotations

import ast
import pathlib

import silly_kicks
import silly_kicks.territory  # must import cleanly

ROOT = pathlib.Path(silly_kicks.__file__).resolve().parent
TERRITORY = ROOT / "territory"
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


def _imports_territory(path: pathlib.Path) -> bool:
    return any(m == "silly_kicks.territory" or m.startswith("silly_kicks.territory.") for m in _imported_modules(path))


def test_territory_never_imports_tracking():
    offenders = {
        py.relative_to(TERRITORY).as_posix(): [m for m in _imported_modules(py) if _is_banned(m)]
        for py in sorted(TERRITORY.rglob("*.py"))
    }
    offenders = {k: v for k, v in offenders.items() if v}
    assert not offenders, f"{offenders}: territory is event-only -- must NEVER import silly_kicks.tracking."


def test_nothing_imports_territory():
    offenders = [
        py.relative_to(ROOT).as_posix()
        for py in sorted(ROOT.rglob("*.py"))
        if not py.is_relative_to(TERRITORY) and _imports_territory(py)
    ]
    assert not offenders, f"{offenders}: nothing in silly_kicks should import territory (a leaf metric)."


def test_public_surface_exists():
    from silly_kicks.territory import compute_territorial_dominance  # noqa: F401


def test_banned_detector_fires_on_planted_violation(tmp_path):
    planted = tmp_path / "_p.py"
    planted.write_text("from silly_kicks.tracking import add_das\n", encoding="utf-8")
    assert any(_is_banned(m) for m in _imported_modules(planted))
    planted.write_text("from silly_kicks.xthreat import values_at_points\n", encoding="utf-8")
    assert not any(_is_banned(m) for m in _imported_modules(planted))
