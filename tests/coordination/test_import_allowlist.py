"""coordination isolation gate (spec §7.6, C10): kernels are numpy/scipy/stdlib/numba only; the package
consumes PUBLIC tracking seams plus a small allowlist of tracking privates; nothing else imports it.

Mirrors ``tests/restdefense/test_import_allowlist.py``. Every direction carries a planted-violation
meta-test so the gate cannot pass vacuously (ADR-051 anti-rot).
"""

from __future__ import annotations

import ast
import pathlib
import sys

import silly_kicks
import silly_kicks.coordination  # must import cleanly

ROOT = pathlib.Path(silly_kicks.__file__).resolve().parent
COORD = ROOT / "coordination"
KERNELS = COORD / "_kernels"

_STDLIB = set(sys.stdlib_module_names)
_KERNEL_ALLOWED_TOP = {"numpy", "scipy", "numba"}

# coordination MAY reach into these tracking privates (public seam does not exist / it is an array kernel).
# Each entry is (importing module stem, imported private module) with a reason (C10; PRIVATE_CONSUMERS.md).
_PRIVATE_IMPORT_ALLOWLIST: set[tuple[str, str]] = {
    ("_signals", "silly_kicks.tracking._collective"),  # back_line_batch / compact_rows array kernels (not public)
    ("_signals", "silly_kicks.tracking._geometry"),  # to_goal_relative_{x,y}_array twins (ADR-051)
    ("_signals", "silly_kicks.tracking._provider_visibility"),  # _DETECTION_AWARE_PROVIDERS + detection-aware checks
    ("_windows", "silly_kicks.tracking._provider_visibility"),  # dead_ball_observed (ADR-069)
    # winter_correction / butterworth_min_length; grid_span (the ONE run-edge rule, F7) and butterworth_lowpass_rows
    # (x and y in one pass, ADR-111 ruling C) -- array kernels with no public seam
    ("_signals", "silly_kicks.tracking.preprocess._butterworth"),
}


def _imported_modules(path: pathlib.Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.append(node.module)
        elif isinstance(node, ast.Import):
            mods.extend(a.name for a in node.names)
    return mods


def _kernel_violations(path: pathlib.Path) -> list[str]:
    """Imports in a kernel module that are NOT numpy/scipy/numba/stdlib or a sibling ``_kernels`` module."""
    bad: list[str] = []
    for m in _imported_modules(path):
        top = m.split(".", 1)[0]
        if top in _KERNEL_ALLOWED_TOP or top in _STDLIB:
            continue
        if m == "silly_kicks.coordination._kernels" or m.startswith("silly_kicks.coordination._kernels."):
            continue  # sibling kernels only -- the subpackage composes internally; it never reaches OUT (spec §7.6)
        bad.append(m)
    return bad


def _private_tracking_hits(path: pathlib.Path) -> list[str]:
    """Private ``tracking.`` submodules imported by *path*, minus the allowlist."""
    hits: list[str] = []
    for m in _imported_modules(path):
        if m.startswith("silly_kicks.tracking.") and m.rsplit(".", 1)[-1].startswith("_"):
            if (path.stem, m) not in _PRIVATE_IMPORT_ALLOWLIST:
                hits.append(m)
    return hits


def _imports_coordination(path: pathlib.Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("silly_kicks.coordination"):
            return True
        if isinstance(node, ast.Import) and any(a.name.startswith("silly_kicks.coordination") for a in node.names):
            return True
    return False


def test_coordination_public_surface_exists():
    from silly_kicks.coordination import CoordinationParams, compute_team_coordination  # noqa: F401


def test_kernels_import_only_numpy_scipy_stdlib_numba():
    # rglob, NOT glob: a flat glob silently stops scanning the moment _kernels grows a subpackage.
    offenders = {
        py.relative_to(KERNELS).as_posix(): hits
        for py in sorted(KERNELS.rglob("*.py"))
        if (hits := _kernel_violations(py))
    }
    assert not offenders, (
        f"{offenders}: coordination/_kernels/ modules import only numpy, scipy, the standard library and numba "
        "(plus sibling _kernels modules) -- NO pandas, NO other silly-kicks (spec §7.6)."
    )


def test_coordination_private_imports_are_allowlisted():
    offenders = {
        py.relative_to(COORD).as_posix(): hits
        for py in sorted(COORD.rglob("*.py"))
        if (hits := _private_tracking_hits(py))
    }
    assert not offenders, (
        f"{offenders}: coordination imports a PRIVATE tracking seam; import the public seam "
        "(silly_kicks.tracking.<name>) or add a reasoned _PRIVATE_IMPORT_ALLOWLIST entry (C10)."
    )


def test_nothing_imports_coordination():
    offenders = [
        py.relative_to(ROOT).as_posix()
        for py in sorted(ROOT.rglob("*.py"))
        if COORD not in py.parents and _imports_coordination(py)
    ]
    assert not offenders, (
        f"{offenders}: nothing in silly_kicks outside coordination/ may import it -- coordination is a "
        "LEAF consumer of tracking, never a dependency (spec §7.6)."
    )


def test_coordination_package_is_non_empty():
    """META: pins the gate's surface -- an empty package would make the scans vacuous."""
    modules = sorted(p.relative_to(COORD).as_posix() for p in COORD.rglob("*.py"))
    assert len(modules) >= 12, f"expected the coordination module set, found {modules}"
    kernels = sorted(p.relative_to(KERNELS).as_posix() for p in KERNELS.rglob("*.py"))
    assert len(kernels) >= 8, f"expected the coordination kernel set, found {kernels}"


def test_kernel_detector_fires_on_planted_violation(tmp_path):
    """META: the kernel purity detector flags a silly-kicks/pandas import and passes numpy/scipy/numba/stdlib."""
    planted = tmp_path / "_planted.py"
    planted.write_text("from silly_kicks.id_compat import canonical_id\n", encoding="utf-8")
    assert _kernel_violations(planted) == ["silly_kicks.id_compat"]
    planted.write_text("import pandas as pd\n", encoding="utf-8")
    assert _kernel_violations(planted) == ["pandas"]
    planted.write_text(
        "import numpy as np\nimport scipy.signal\nimport hashlib\n"
        "from silly_kicks.coordination._kernels._numba import use_numba\n",
        encoding="utf-8",
    )
    assert _kernel_violations(planted) == []


def test_private_seam_detector_fires_on_planted_violation(tmp_path):
    """META: the private-seam detector flags a private tracking import and passes a public one."""
    planted = tmp_path / "_planted.py"
    planted.write_text("from silly_kicks.tracking._ghost_gk import GhostGkModel\n", encoding="utf-8")
    assert _private_tracking_hits(planted) == ["silly_kicks.tracking._ghost_gk"]
    planted.write_text("from silly_kicks.tracking import GoalMap, resolve_defended_goals\n", encoding="utf-8")
    assert _private_tracking_hits(planted) == []
    planted.write_text("from silly_kicks.tracking.preprocess import butterworth_lowpass\n", encoding="utf-8")
    assert _private_tracking_hits(planted) == []


def test_reverse_import_detector_fires_on_planted_violation(tmp_path):
    """META: the nothing-imports-coordination detector must actually detect."""
    planted = tmp_path / "_planted.py"
    planted.write_text("from silly_kicks.coordination import compute_team_coordination\n", encoding="utf-8")
    assert _imports_coordination(planted)
    planted.write_text("import silly_kicks.coordination._compute\n", encoding="utf-8")
    assert _imports_coordination(planted)
    planted.write_text("from silly_kicks.tracking import compute_defensive_line\n", encoding="utf-8")
    assert not _imports_coordination(planted)
