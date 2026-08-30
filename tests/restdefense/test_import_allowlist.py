"""restdefense -> tracking/gkdv: PUBLIC SEAMS ONLY, and never the reverse (TF-60, ADR-080).

Mirrors ``tests/gkdv/test_import_allowlist.py`` (ADR-037): a composite metric package may
import ``silly_kicks.tracking`` / ``silly_kicks.gkdv`` PUBLIC seams, but neither may reach into
the other's private (``._foo``) submodules, and ``tracking`` must NEVER import ``restdefense``.
Each direction carries a planted-violation meta-test so the gate cannot pass vacuously.
"""

from __future__ import annotations

import ast
import pathlib

import silly_kicks
import silly_kicks.restdefense  # must import cleanly

ROOT = pathlib.Path(silly_kicks.__file__).resolve().parent
RESTDEFENSE = ROOT / "restdefense"
TRACKING = ROOT / "tracking"

# restdefense may import PUBLIC silly_kicks.tracking / silly_kicks.gkdv ONLY -- never their
# private (`._foo`) submodules. Empty allowlist: the Layer-1 impl uses only public seams
# (classify_region_observation, REGION_OBSERVATION_SOURCE_VALUES, compute_defensive_line, ...).
# `group_rows` is `silly_kicks._frame_index` (a package-level util recorded in
# PRIVATE_CONSUMERS.md), NOT a tracking/gkdv private, so it is not caught here.
_PRIVATE_IMPORT_ALLOWLIST: set[tuple[str, str]] = set()


def _imported_modules(path: pathlib.Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            mods.append(node.module)
        elif isinstance(node, ast.Import):
            mods.extend(a.name for a in node.names)
    return mods


def _private_tracking_or_gkdv_hits(path: pathlib.Path) -> list[str]:
    """Private ``tracking.``/``gkdv.`` submodules imported by *path*, minus the allowlist."""
    hits: list[str] = []
    for m in _imported_modules(path):
        for pkg in ("silly_kicks.tracking.", "silly_kicks.gkdv."):
            if m.startswith(pkg) and m.rsplit(".", 1)[-1].startswith("_"):
                if (path.stem, m) not in _PRIVATE_IMPORT_ALLOWLIST:
                    hits.append(m)
    return hits


def _imports_restdefense(path: pathlib.Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("silly_kicks.restdefense"):
            return True
        if isinstance(node, ast.Import):
            if any(a.name.startswith("silly_kicks.restdefense") for a in node.names):
                return True
    return False


def test_restdefense_public_surface_exists():
    from silly_kicks.restdefense import RD_SAMPLE_KEYS

    assert RD_SAMPLE_KEYS == ["game_id", "period_id", "team_id", "action_id"]


def test_tracking_never_imports_restdefense():
    # rglob, NOT glob: a flat glob silently stops scanning the moment tracking grows a
    # subpackage, and an unscanned module is indistinguishable from a compliant one.
    offenders = [
        py.relative_to(TRACKING).as_posix() for py in sorted(TRACKING.rglob("*.py")) if _imports_restdefense(py)
    ]
    assert not offenders, (
        f"{offenders}: tracking/ must NEVER import restdefense/ -- restdefense consumes "
        "tracking public seams, never the reverse (ADR-080 layering)."
    )


def test_restdefense_imports_only_public_tracking_and_gkdv_seams():
    offenders = {
        py.relative_to(RESTDEFENSE).as_posix(): hits
        for py in sorted(RESTDEFENSE.rglob("*.py"))
        if (hits := _private_tracking_or_gkdv_hits(py))
    }
    assert not offenders, (
        f"{offenders}: restdefense imports PRIVATE tracking/gkdv seam(s); import the public "
        "seam (silly_kicks.tracking.<name> / silly_kicks.gkdv.<name>) or add an "
        "_PRIVATE_IMPORT_ALLOWLIST entry with a reason."
    )


def test_restdefense_package_is_non_empty():
    """META: pins the gate's surface -- an empty package would make the scans vacuous."""
    modules = sorted(p.relative_to(RESTDEFENSE).as_posix() for p in RESTDEFENSE.rglob("*.py"))
    assert len(modules) >= 4, f"expected the restdefense module set, found {modules}"


def test_restdefense_import_detector_fires_on_planted_violation(tmp_path):
    """META: the tracking->restdefense detector must actually detect."""
    planted = tmp_path / "_planted.py"
    planted.write_text("from silly_kicks.restdefense import compute_rest_defense\n", encoding="utf-8")
    assert _imports_restdefense(planted)
    planted.write_text("import silly_kicks.restdefense._compute\n", encoding="utf-8")
    assert _imports_restdefense(planted)
    planted.write_text("from silly_kicks.tracking import compute_defensive_line\n", encoding="utf-8")
    assert not _imports_restdefense(planted)


def test_private_seam_detector_fires_on_planted_violation(tmp_path):
    """META: the private-seam detector must flag a private import and pass a public one."""
    planted = tmp_path / "_planted.py"
    planted.write_text("from silly_kicks.tracking._ghost_gk import GhostGkModel\n", encoding="utf-8")
    assert _private_tracking_or_gkdv_hits(planted) == ["silly_kicks.tracking._ghost_gk"]
    planted.write_text("from silly_kicks.gkdv._arms import delta_das_batch\n", encoding="utf-8")
    assert _private_tracking_or_gkdv_hits(planted) == ["silly_kicks.gkdv._arms"]
    planted.write_text(
        "from silly_kicks.tracking import compute_defensive_line, resolve_defended_goals\n",
        encoding="utf-8",
    )
    assert _private_tracking_or_gkdv_hits(planted) == []
