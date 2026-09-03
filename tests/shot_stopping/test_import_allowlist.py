"""shot_stopping is EVENT-ONLY: never imports silly_kicks.tracking; nothing imports shot_stopping.

Mirrors tests/providers/test_appearances_import_allowlist.py (AST module-level). Allowed silly-kicks
deps: silly_kicks.spadl (config), silly_kicks.id_compat, silly_kicks.keeper_identity (top-level, not
tracking). Each detector carries a planted-violation meta-test.
"""

from __future__ import annotations

import ast
import pathlib

import silly_kicks
import silly_kicks.shot_stopping  # must import cleanly

ROOT = pathlib.Path(silly_kicks.__file__).resolve().parent
SHOT_STOPPING = ROOT / "shot_stopping"
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


def _imports_shot_stopping(path: pathlib.Path) -> bool:
    return any(
        m == "silly_kicks.shot_stopping" or m.startswith("silly_kicks.shot_stopping.") for m in _imported_modules(path)
    )


def test_shot_stopping_never_imports_tracking():
    offenders = {
        py.relative_to(SHOT_STOPPING).as_posix(): [m for m in _imported_modules(py) if _is_banned(m)]
        for py in sorted(SHOT_STOPPING.rglob("*.py"))
    }
    offenders = {k: v for k, v in offenders.items() if v}
    assert not offenders, f"{offenders}: shot_stopping is event-only -- must NEVER import silly_kicks.tracking."


def test_nothing_imports_shot_stopping():
    offenders = [
        py.relative_to(ROOT).as_posix()
        for py in sorted(ROOT.rglob("*.py"))
        if not py.is_relative_to(SHOT_STOPPING) and _imports_shot_stopping(py)
    ]
    assert not offenders, f"{offenders}: nothing in silly_kicks should import shot_stopping (a leaf metric)."


def test_public_surface_exists():
    from silly_kicks.shot_stopping import compute_shot_stopping  # noqa: F401


def test_banned_detector_fires_on_planted_violation(tmp_path):
    planted = tmp_path / "_p.py"
    planted.write_text("from silly_kicks.tracking import add_das\n", encoding="utf-8")
    assert any(_is_banned(m) for m in _imported_modules(planted))
    planted.write_text("from silly_kicks.keeper_identity import add_defending_gk_player_id\n", encoding="utf-8")
    assert not any(_is_banned(m) for m in _imported_modules(planted))
