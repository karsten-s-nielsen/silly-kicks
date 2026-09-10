"""territorial_defense -> tracking/keeper_identity/territory: PUBLIC SEAMS ONLY, and NOTHING imports
it (TF-54b).

Mirrors ``tests/restdefense/test_import_allowlist.py`` (ADR-037/ADR-080): a tracking-consuming
metric package may import ``silly_kicks.tracking`` / ``silly_kicks.keeper_identity`` /
``silly_kicks.territory`` PUBLIC seams, but must not reach into their private (``._foo``) submodules,
and NOTHING in ``silly_kicks/`` may import this package. Each direction carries a planted-violation
meta-test so the gate cannot pass vacuously.

NOTE: restdefense's reverse test is TRACKING-scoped only. Per the spec, the whole-tree
``nothing-imports-territorial_defense`` sweep is authored FRESH here (``ROOT.rglob`` over all of
``silly_kicks/``, excluding this package's own modules), and the detector catches RELATIVE imports
too (``from ..territorial_defense import`` / ``from . import territorial_defense``), not just absolute.
"""

from __future__ import annotations

import ast
import pathlib

import silly_kicks
import silly_kicks.territorial_defense  # must import cleanly

ROOT = pathlib.Path(silly_kicks.__file__).resolve().parent
PKG_DIR = ROOT / "territorial_defense"
TRACKING = ROOT / "tracking"

_PKG_TAIL = "territorial_defense"  # unique package basename

# Public seams only. Empty allowlist: the impl uses only public seams (compute_threat_pc,
# resolve_defended_goals, region_observed_fraction, zero_velocity_if_unavailable from tracking;
# apply_actor_identities_to_frames from keeper_identity; build_trimmed_hull/Hull from territory).
_PRIVATE_IMPORT_ALLOWLIST: set[tuple[str, str]] = set()

# Public packages territorial_defense may import; a `._private` submodule of any of these is a
# violation. keeper_identity is a flat module (no private submodules), so it needs no entry.
_GUARDED_PUBLIC_PACKAGES = (
    "silly_kicks.tracking.",
    "silly_kicks.territory.",
    "silly_kicks.xthreat.",
)


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
    """Private ``tracking.``/``territory.``/``xthreat.`` submodules imported by *path*, minus allowlist."""
    hits: list[str] = []
    for m in _imported_modules(path):
        for pkg in _GUARDED_PUBLIC_PACKAGES:
            if m.startswith(pkg) and m.rsplit(".", 1)[-1].startswith("_"):
                if (path.stem, m) not in _PRIVATE_IMPORT_ALLOWLIST:
                    hits.append(m)
    return hits


def _imports_territorial_defense(path: pathlib.Path) -> bool:
    """True iff *path* imports silly_kicks.territorial_defense -- absolute OR relative."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            # absolute `from silly_kicks.territorial_defense import x`
            # AND relative `from ..territorial_defense import x` (module="territorial_defense")
            if _PKG_TAIL in (node.module or "").split("."):
                return True
            # relative `from . import territorial_defense`
            if node.level > 0 and any(a.name == _PKG_TAIL for a in node.names):
                return True
        if isinstance(node, ast.Import) and any(_PKG_TAIL in a.name.split(".") for a in node.names):
            return True
    return False


def test_territorial_defense_public_surface_exists():
    from silly_kicks.territorial_defense import TD_SAMPLE_KEYS

    assert TD_SAMPLE_KEYS == ["game_id", "player_id"]


def test_tracking_never_imports_territorial_defense():
    # rglob, NOT glob: a flat glob silently stops scanning the moment tracking grows a subpackage.
    offenders = [
        py.relative_to(TRACKING).as_posix() for py in sorted(TRACKING.rglob("*.py")) if _imports_territorial_defense(py)
    ]
    assert not offenders, (
        f"{offenders}: tracking/ must NEVER import territorial_defense/ -- the package consumes "
        "tracking public seams, never the reverse (ADR-037 layering)."
    )


def test_nothing_in_silly_kicks_imports_territorial_defense():
    """FRESH whole-tree sweep (authored here -- restdefense's reverse test is tracking-scoped only)."""
    offenders = [
        py.relative_to(ROOT).as_posix()
        for py in sorted(ROOT.rglob("*.py"))
        if PKG_DIR not in py.parents and _imports_territorial_defense(py)
    ]
    assert not offenders, (
        f"{offenders}: NOTHING in silly_kicks/ may import territorial_defense/ -- it is a leaf "
        "consumer (nothing depends on it)."
    )


def test_territorial_defense_imports_only_public_seams():
    offenders = {
        py.relative_to(PKG_DIR).as_posix(): hits
        for py in sorted(PKG_DIR.rglob("*.py"))
        if (hits := _private_seam_hits(py))
    }
    assert not offenders, (
        f"{offenders}: territorial_defense imports a PRIVATE tracking/territory/xthreat seam; import "
        "the public seam or add an _PRIVATE_IMPORT_ALLOWLIST entry with a reason."
    )


def test_territorial_defense_package_is_non_empty():
    """META: pins the gate's surface -- an empty package would make the scans vacuous."""
    modules = sorted(p.relative_to(PKG_DIR).as_posix() for p in PKG_DIR.rglob("*.py"))
    assert len(modules) >= 4, f"expected the territorial_defense module set, found {modules}"


def test_import_detector_fires_on_planted_violation(tmp_path):
    """META: the reverse detector must catch absolute AND relative imports, and pass a non-import."""
    planted = tmp_path / "_planted.py"
    planted.write_text("from silly_kicks.territorial_defense import compute_territorial_defense\n", encoding="utf-8")
    assert _imports_territorial_defense(planted)
    planted.write_text("import silly_kicks.territorial_defense._arms\n", encoding="utf-8")
    assert _imports_territorial_defense(planted)
    planted.write_text("from ..territorial_defense import compute_territorial_defense\n", encoding="utf-8")
    assert _imports_territorial_defense(planted)  # RELATIVE form (PLAN-11)
    planted.write_text("from . import territorial_defense\n", encoding="utf-8")
    assert _imports_territorial_defense(planted)  # relative bare-name form
    planted.write_text("from silly_kicks.tracking import compute_threat_pc\n", encoding="utf-8")
    assert not _imports_territorial_defense(planted)


def test_private_seam_detector_fires_on_planted_violation(tmp_path):
    """META: the private-seam detector must flag a private import and pass a public one."""
    planted = tmp_path / "_planted.py"
    planted.write_text("from silly_kicks.tracking._cover_shadows import compute_threat_pc\n", encoding="utf-8")
    assert _private_seam_hits(planted) == ["silly_kicks.tracking._cover_shadows"]
    planted.write_text("from silly_kicks.territory._hull import build_trimmed_hull\n", encoding="utf-8")
    assert _private_seam_hits(planted) == ["silly_kicks.territory._hull"]
    planted.write_text(
        "from silly_kicks.tracking import compute_threat_pc, resolve_defended_goals\n"
        "from silly_kicks.territory import build_trimmed_hull\n",
        encoding="utf-8",
    )
    assert _private_seam_hits(planted) == []
