"""gk_decision imports only silly_kicks.tracking PUBLIC seams (never a tracking._* private); nothing
imports gk_decision.

Phase 1's native tier needed no tracking; Phase 2's reconstruction tier (:class:`ReconstructedOptionSet`)
consumes tracking PUBLIC seams (``compute_packing_metrics`` / ``action_ltr_goal_map`` / linking) +
``keeper_identity`` + ``reflection``, so this gate (mirroring ``tests/territorial_defense/`` and
``tests/gkdv/``) allows the tracking PUBLIC surface and bans only ``silly_kicks.tracking._*`` privates.
Each detector carries a planted-violation meta-test (both directions).
"""

from __future__ import annotations

import ast
import pathlib

import silly_kicks
import silly_kicks.gk_decision  # must import cleanly

ROOT = pathlib.Path(silly_kicks.__file__).resolve().parent
PKG = ROOT / "gk_decision"
_BANNED_PRIVATE_PREFIX = "silly_kicks.tracking._"


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
    return m.startswith(_BANNED_PRIVATE_PREFIX)


def _imports_pkg(path: pathlib.Path) -> bool:
    return any(m == "silly_kicks.gk_decision" or m.startswith("silly_kicks.gk_decision.") for m in _mods(path))


def test_gk_decision_never_imports_a_tracking_private():
    offenders = {
        py.relative_to(PKG).as_posix(): [m for m in _mods(py) if _is_banned(m)] for py in sorted(PKG.rglob("*.py"))
    }
    offenders = {k: v for k, v in offenders.items() if v}
    assert not offenders, (
        f"{offenders}: gk_decision may import tracking PUBLIC seams only, never a tracking._* private."
    )


def test_nothing_in_silly_kicks_imports_gk_decision():
    offenders = [
        py.relative_to(ROOT).as_posix()
        for py in sorted(ROOT.rglob("*.py"))
        if not py.is_relative_to(PKG) and _imports_pkg(py)
    ]
    assert not offenders, f"{offenders}: nothing in silly_kicks may import gk_decision (it is a leaf metric)."


def test_public_surface_exists():
    from silly_kicks.gk_decision import (  # noqa: F401
        ReconstructedOptionSet,
        SkillCornerGIOptionSet,
        compute_gk_decision_value,
        summarize_gk_decision,
    )


def test_private_ban_fires_on_planted_violation(tmp_path):
    planted = tmp_path / "_p.py"
    planted.write_text("from silly_kicks.tracking._das import get_das\n", encoding="utf-8")
    assert any(_is_banned(m) for m in _mods(planted))  # a tracking._* private FIRES the ban
    planted.write_text("from silly_kicks.tracking import compute_packing_metrics\n", encoding="utf-8")
    assert not any(_is_banned(m) for m in _mods(planted))  # a tracking PUBLIC seam does NOT
