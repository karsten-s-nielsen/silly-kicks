"""match_outcome is EVENT-ONLY: imports spadl / id_compat only, NEVER silly_kicks.tracking (public or
private); and nothing in silly_kicks imports match_outcome (it is a leaf metric).

Mirrors ``tests/territory/`` / ``tests/duels/`` / ``tests/shot_stopping/`` (event-only allowlist -- the
whole ``silly_kicks.tracking`` subtree is banned, not just its privates). Each detector carries a
planted-violation meta-test (both directions).
"""

from __future__ import annotations

import ast
import pathlib

import silly_kicks
import silly_kicks.match_outcome  # must import cleanly

ROOT = pathlib.Path(silly_kicks.__file__).resolve().parent
PKG = ROOT / "match_outcome"
_BANNED_PREFIX = "silly_kicks.tracking"


def _mods(path: pathlib.Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: list[str] = []
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module:
            out.append(n.module)
        elif isinstance(n, ast.Import):
            out.extend(a.name for a in n.names)
    return out


def _is_banned(m: str) -> bool:
    return m == _BANNED_PREFIX or m.startswith(_BANNED_PREFIX + ".")


def _imports_pkg(path: pathlib.Path) -> bool:
    return any(m == "silly_kicks.match_outcome" or m.startswith("silly_kicks.match_outcome.") for m in _mods(path))


def test_match_outcome_never_imports_tracking():
    offenders = {
        py.relative_to(PKG).as_posix(): [m for m in _mods(py) if _is_banned(m)] for py in sorted(PKG.rglob("*.py"))
    }
    offenders = {k: v for k, v in offenders.items() if v}
    assert not offenders, f"{offenders}: match_outcome is event-only; it may NEVER import silly_kicks.tracking."


def test_nothing_in_silly_kicks_imports_match_outcome():
    offenders = [
        py.relative_to(ROOT).as_posix()
        for py in sorted(ROOT.rglob("*.py"))
        if not py.is_relative_to(PKG) and _imports_pkg(py)
    ]
    assert not offenders, f"{offenders}: nothing in silly_kicks may import match_outcome (it is a leaf metric)."


def test_import_ban_fires_on_planted_violation(tmp_path):
    planted = tmp_path / "_p.py"
    planted.write_text("from silly_kicks.tracking import compute_packing_metrics\n", encoding="utf-8")
    assert any(_is_banned(m) for m in _mods(planted))  # any tracking import FIRES
    planted.write_text("from silly_kicks.tracking._das import get_das\n", encoding="utf-8")
    assert any(_is_banned(m) for m in _mods(planted))  # a private too
    planted.write_text("from silly_kicks.spadl import config\n", encoding="utf-8")
    assert not any(_is_banned(m) for m in _mods(planted))  # spadl does NOT
