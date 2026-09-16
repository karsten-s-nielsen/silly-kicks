"""team_metrics is EVENT-ONLY: never imports silly_kicks.tracking; nothing imports team_metrics.

Mirrors tests/shot_stopping/test_import_allowlist.py (AST module-level). Allowed silly-kicks deps:
silly_kicks.spadl (config, add_possessions, reflection helpers), silly_kicks.id_compat,
silly_kicks.reflection, silly_kicks._frame_index. Banned: silly_kicks.tracking.
"""

from __future__ import annotations

import ast
import pathlib

import silly_kicks
import silly_kicks.team_metrics  # must import cleanly

ROOT = pathlib.Path(silly_kicks.__file__).resolve().parent
TEAM_METRICS = ROOT / "team_metrics"
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


def _imports_team_metrics(path: pathlib.Path) -> bool:
    return any(
        m == "silly_kicks.team_metrics" or m.startswith("silly_kicks.team_metrics.") for m in _imported_modules(path)
    )


def test_team_metrics_never_imports_tracking():
    offenders = {
        py.relative_to(TEAM_METRICS).as_posix(): [m for m in _imported_modules(py) if _is_banned(m)]
        for py in sorted(TEAM_METRICS.rglob("*.py"))
    }
    offenders = {k: v for k, v in offenders.items() if v}
    assert not offenders, f"{offenders}: team_metrics is event-only -- must NEVER import silly_kicks.tracking."


def test_nothing_imports_team_metrics():
    offenders = [
        py.relative_to(ROOT).as_posix()
        for py in sorted(ROOT.rglob("*.py"))
        if not py.is_relative_to(TEAM_METRICS) and _imports_team_metrics(py)
    ]
    assert not offenders, f"{offenders}: nothing in silly_kicks should import team_metrics (a leaf metric)."


def test_public_surface_exists():
    from silly_kicks.team_metrics import TeamKpiParams, compute_team_kpis  # noqa: F401


def test_banned_detector_fires_on_planted_violation(tmp_path):
    planted = tmp_path / "_p.py"
    planted.write_text("from silly_kicks.tracking import add_das\n", encoding="utf-8")
    assert any(_is_banned(m) for m in _imported_modules(planted))
    planted.write_text("from silly_kicks.reflection import reflect_columns\n", encoding="utf-8")
    assert not any(_is_banned(m) for m in _imported_modules(planted))
