"""corners is a LEAF: nothing in silly_kicks imports it.

Unlike event-only shot_stopping, corners is TRACKING-CONSUMING, so it MAY import silly_kicks.tracking
(frames + the TF-79 ball-contact primitive) -- that is not banned. The invariant pinned here is the
leaf discipline: nothing else in silly_kicks may import corners. AST module-level; the detector carries
a planted-violation meta-test.
"""

from __future__ import annotations

import ast
import pathlib

import silly_kicks
import silly_kicks.corners  # must import cleanly

ROOT = pathlib.Path(silly_kicks.__file__).resolve().parent
CORNERS = ROOT / "corners"


def _imported_modules(path: pathlib.Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            mods.append(node.module)
        elif isinstance(node, ast.Import):
            mods.extend(a.name for a in node.names)
    return mods


def _imports_corners(path: pathlib.Path) -> bool:
    return any(m == "silly_kicks.corners" or m.startswith("silly_kicks.corners.") for m in _imported_modules(path))


def test_nothing_imports_corners():
    offenders = [
        py.relative_to(ROOT).as_posix()
        for py in sorted(ROOT.rglob("*.py"))
        if not py.is_relative_to(CORNERS) and _imports_corners(py)
    ]
    assert not offenders, f"{offenders}: nothing in silly_kicks should import corners (a leaf metric family)."


def test_public_surface_exists():
    from silly_kicks.corners import CornerWindowParams  # noqa: F401


def test_leaf_detector_fires_on_planted_violation(tmp_path):
    planted = tmp_path / "_p.py"
    planted.write_text("from silly_kicks.corners import identify_corners\n", encoding="utf-8")
    assert _imports_corners(planted)
    planted.write_text("from silly_kicks.tracking import add_das\n", encoding="utf-8")
    assert not _imports_corners(planted)
