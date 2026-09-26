"""F1b (ADR-106) — frame-coordinate reads upcast to float64 at the compute boundary.

Tracking-frame coordinates are stored float32 (memory), but every compute path upcasts the coord slice
to float64 at its kernel boundary so the ONLY numeric drift is the deterministic storage-rounding
(~1e-5 m) and the numba `@njit` kernels keep their float64 signatures (the ADR-076 bit-identity
contract; separately guarded by the numba-parity gates). This gate pins the pattern forward: a
coordinate column read into numpy via ``.to_numpy()`` / ``.values`` inside a tracking kernel must
declare ``dtype=...float64`` (or be exempt-with-reason), so a future kernel cannot silently compute in
float32.

Scope + known limit: the gate covers the ``df[[...coord...]].to_numpy()`` / ``.values`` read idiom (the
class that admits a silent float32 compute). It does NOT prove exhaustiveness over every conceivable
float32-compute path (e.g. arithmetic on a float32 Series before ``to_numpy``); the numba-parity gates
(ADR-076) are the correctness backstop for the boundary that actually matters. Cite this gate as covering
its idiom, not the whole defect class (the AGENTS.md completeness-by-enumeration lesson).
"""

import ast
from pathlib import Path

_COORD_LITERALS = {"x", "y", "z", "x_smoothed", "y_smoothed"}
_KERNEL_DIRS = (
    Path("silly_kicks/tracking"),
    Path("silly_kicks/tracking/pitch_control"),
)

# (module_path, reason). A coord read that legitimately need not upcast (e.g. it is immediately
# astype'd, or the values are non-numeric). Empty today; grows only with a stated reason.
_UPCAST_EXEMPT: dict[str, str] = {}


def _kernel_files() -> list[Path]:
    out: list[Path] = []
    for d in _KERNEL_DIRS:
        out.extend(p for p in d.glob("*.py") if not p.name.startswith("__"))
    # dedupe (pitch_control is nested under tracking's glob is *.py non-recursive, so no overlap)
    return sorted(set(out))


def _slice_is_coord(node: ast.AST) -> bool:
    """True iff the subscript slice references a coord column literal (a List of str Constants)."""
    sl = node.slice if isinstance(node, ast.Subscript) else None
    if isinstance(sl, ast.List):
        names = {e.value for e in sl.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)}
        return bool(names & _COORD_LITERALS)
    if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
        return sl.value in _COORD_LITERALS
    return False


def _bare_coord_reads(tree: ast.AST) -> list[int]:
    """Line numbers of coord `.to_numpy()`(no dtype) / `.values` reads on a coord subscript."""
    bad: list[int] = []
    for node in ast.walk(tree):
        # `<coord-subscript>.to_numpy()` with NO dtype -- neither a positional dtype
        # (`.to_numpy(float)`) nor a `dtype=` keyword. A bare `.to_numpy()` yields the float32 storage
        # dtype and computes in float32.
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "to_numpy"
            and isinstance(node.func.value, ast.Subscript)
            and _slice_is_coord(node.func.value)
            and not node.args  # a positional first arg IS the dtype (`.to_numpy(float)`)
            and not any(kw.arg == "dtype" for kw in node.keywords)
        ):
            bad.append(node.lineno)
        # `<coord-subscript>.values`
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "values"
            and isinstance(node.value, ast.Subscript)
            and _slice_is_coord(node.value)
        ):
            bad.append(node.lineno)
    return bad


def test_frame_coord_reads_upcast_to_float64():
    violations: dict[str, list[int]] = {}
    for path in _kernel_files():
        rel = path.as_posix()
        if rel in _UPCAST_EXEMPT:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        lines = _bare_coord_reads(tree)
        if lines:
            violations[rel] = lines
    assert not violations, (
        "float32 frame coords read into numpy without a float64 upcast "
        f"(add dtype='float64' to .to_numpy(), or use .to_numpy(dtype='float64') instead of .values): "
        f"{violations}"
    )


def test_upcast_exempt_paths_exist():
    # anti-rot: an exemption for a deleted/renamed file is a silent hole.
    for rel in _UPCAST_EXEMPT:
        assert Path(rel).is_file(), rel
