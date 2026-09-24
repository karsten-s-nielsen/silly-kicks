"""positioning -> tracking/id_compat/reflection: PUBLIC SEAMS ONLY; NOTHING imports it (TF-56).

Mirrors ``tests/territorial_defense/test_import_allowlist.py`` (ADR-037): a tracking-consuming
metric package may import ``silly_kicks.tracking`` PUBLIC seams + ``silly_kicks.id_compat`` /
``silly_kicks.reflection`` / ``silly_kicks.spadl.config``, but must NOT reach into a private
(``._foo``) tracking submodule, must NOT import ``silly_kicks.xthreat`` at all (``xt`` is INJECTED,
the port pattern), and NOTHING in ``silly_kicks/`` may import this package. Each direction carries a
planted-violation meta-test so the gate cannot pass vacuously.
"""

from __future__ import annotations

import ast
import pathlib

import silly_kicks
import silly_kicks.positioning  # must import cleanly

ROOT = pathlib.Path(silly_kicks.__file__).resolve().parent
PKG_DIR = ROOT / "positioning"
TRACKING = ROOT / "tracking"

_PKG_TAIL = "positioning"  # unique package basename

# Public seams only. Empty allowlist: the impl uses only public seams.
_PRIVATE_IMPORT_ALLOWLIST: set[tuple[str, str]] = set()

# Public packages positioning may import; a `._private` submodule of any of these is a violation.
_GUARDED_PUBLIC_PACKAGES = ("silly_kicks.tracking.",)

# Modules positioning must NOT import at all (xt is INJECTED -- never imported for weights).
_FORBIDDEN_MODULES = ("silly_kicks.xthreat",)


def _imported_modules(path: pathlib.Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            mods.append(node.module)
        elif isinstance(node, ast.Import):
            mods.extend(a.name for a in node.names)
    return mods


def _private_seam_hits(path: pathlib.Path) -> list[str]:
    """Private ``tracking.`` submodules imported by *path*, minus the allowlist."""
    hits: list[str] = []
    for m in _imported_modules(path):
        for pkg in _GUARDED_PUBLIC_PACKAGES:
            if (
                m.startswith(pkg)
                and m.rsplit(".", 1)[-1].startswith("_")
                and (path.stem, m) not in _PRIVATE_IMPORT_ALLOWLIST
            ):
                hits.append(m)
    return hits


def _forbidden_hits(path: pathlib.Path) -> list[str]:
    """Forbidden modules (``silly_kicks.xthreat``) imported by *path*."""
    return [m for m in _imported_modules(path) if any(m == f or m.startswith(f + ".") for f in _FORBIDDEN_MODULES)]


def _imports_positioning(path: pathlib.Path) -> bool:
    """True iff *path* imports silly_kicks.positioning -- absolute OR relative."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if _PKG_TAIL in (node.module or "").split("."):
                return True
            if node.level > 0 and any(a.name == _PKG_TAIL for a in node.names):
                return True
        if isinstance(node, ast.Import) and any(_PKG_TAIL in a.name.split(".") for a in node.names):
            return True
    return False


def test_positioning_imports_only_public_seams():
    offenders = {
        py.relative_to(PKG_DIR).as_posix(): hits
        for py in sorted(PKG_DIR.rglob("*.py"))
        if (hits := _private_seam_hits(py))
    }
    assert not offenders, (
        f"{offenders}: positioning imports a PRIVATE tracking seam; import the public seam or add an "
        "_PRIVATE_IMPORT_ALLOWLIST entry with a reason."
    )


def test_positioning_never_imports_xthreat():
    offenders = {
        py.relative_to(PKG_DIR).as_posix(): hits
        for py in sorted(PKG_DIR.rglob("*.py"))
        if (hits := _forbidden_hits(py))
    }
    assert not offenders, (
        f"{offenders}: positioning must NEVER import silly_kicks.xthreat -- xt is INJECTED (port pattern)."
    )


def test_tracking_never_imports_positioning():
    offenders = [
        py.relative_to(TRACKING).as_posix() for py in sorted(TRACKING.rglob("*.py")) if _imports_positioning(py)
    ]
    assert not offenders, f"{offenders}: tracking/ must NEVER import positioning/ (ADR-037 layering)."


def test_nothing_in_silly_kicks_imports_positioning():
    offenders = [
        py.relative_to(ROOT).as_posix()
        for py in sorted(ROOT.rglob("*.py"))
        if PKG_DIR not in py.parents and _imports_positioning(py)
    ]
    assert not offenders, f"{offenders}: NOTHING in silly_kicks/ may import positioning/ -- it is a leaf consumer."


def test_positioning_package_is_non_empty():
    """META: pins the gate's surface -- an empty package would make the scans vacuous."""
    modules = sorted(p.relative_to(PKG_DIR).as_posix() for p in PKG_DIR.rglob("*.py"))
    assert len(modules) >= 2, f"expected the positioning module set, found {modules}"


def test_private_seam_detector_fires_on_planted_violation(tmp_path):
    """META: the private-seam detector must flag a private import and pass public ones."""
    planted = tmp_path / "_planted.py"
    planted.write_text("from silly_kicks.tracking._cover_shadows import compute_threat_pc\n", encoding="utf-8")
    assert _private_seam_hits(planted) == ["silly_kicks.tracking._cover_shadows"]
    planted.write_text(
        "from silly_kicks.tracking import compute_threat_pc, resolve_defended_goals\n"
        "from silly_kicks.tracking.pitch_control import SpearmanParams\n",
        encoding="utf-8",
    )
    assert _private_seam_hits(planted) == []


def test_forbidden_detector_fires_on_planted_violation(tmp_path):
    """META: the xthreat detector must flag any xthreat import and pass tracking."""
    planted = tmp_path / "_planted.py"
    planted.write_text("from silly_kicks.xthreat import ExpectedThreat\n", encoding="utf-8")
    assert _forbidden_hits(planted) == ["silly_kicks.xthreat"]
    planted.write_text("import silly_kicks.xthreat._physical\n", encoding="utf-8")
    assert _forbidden_hits(planted) == ["silly_kicks.xthreat._physical"]
    planted.write_text("from silly_kicks.tracking import compute_threat_pc\n", encoding="utf-8")
    assert _forbidden_hits(planted) == []


def test_reverse_detector_fires_on_planted_violation(tmp_path):
    """META: the reverse detector must catch absolute AND relative imports, pass a non-import."""
    planted = tmp_path / "_planted.py"
    planted.write_text("from silly_kicks.positioning import optimise_positions\n", encoding="utf-8")
    assert _imports_positioning(planted)
    planted.write_text("import silly_kicks.positioning._solve\n", encoding="utf-8")
    assert _imports_positioning(planted)
    planted.write_text("from ..positioning import optimise_positions\n", encoding="utf-8")
    assert _imports_positioning(planted)
    planted.write_text("from . import positioning\n", encoding="utf-8")
    assert _imports_positioning(planted)
    planted.write_text("from silly_kicks.tracking import compute_threat_pc\n", encoding="utf-8")
    assert not _imports_positioning(planted)
