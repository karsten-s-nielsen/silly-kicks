"""Static backstop for the id-dtype contract (ADR-019).

Boundary-focused: flags the comparison shapes that actually cross the action/frame/scalar-arg
boundary and therefore break for a string-id caller -- NOT every id comparison (frame-vs-frame
comparisons of two columns from the same frames DataFrame are dtype-consistent by construction
and safe). The behavioral gate (test_id_dtype_invariance.py) is the primary guard; this catches a
boundary comparison introduced in a new primitive before it is wired into an aggregator.

Two flagged shapes:
  1. `x == home_team_id` / `x != home_team_id` -- the caller's scalar arg crosses the boundary.
  2. `df["a_action"] == df["a_frame"]` -- a cross-source (suffix) comparison after an action<->frame
     merge. Use the _id_compat helpers (same_id / ids_match / ids_equal / ids_differ) instead.
"""

import ast
import pathlib

import pytest

TRACKING = pathlib.Path(__file__).resolve().parents[2] / "silly_kicks" / "tracking"

# Modules where a raw `== home_team_id` is the converter's OWN arg in the provider id space
# (ADR-001, not the ADR-019 feature contract), plus the helper module itself.
ALLOW_MODULES = {"_id_compat.py", "sportec.py", "gradientsports.py", "kloppy.py"}

_CROSS_SUFFIXES = ("_action", "_frame", "_gk", "_dl")


def _is_home_team_id(node) -> bool:
    return isinstance(node, ast.Name) and node.id == "home_team_id"


def _subscript_suffix(node):
    """Return the cross-source suffix of a `df["col_action"]`-style subscript, else None."""
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
        key = node.slice.value
        if isinstance(key, str):
            for suf in _CROSS_SUFFIXES:
                if key.endswith(suf):
                    return suf
    return None


def _boundary_compares(path: pathlib.Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits = []
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Compare) and isinstance(n.ops[0], (ast.Eq, ast.NotEq))):
            continue
        operands = [n.left, *n.comparators]
        # shape 1: against the home_team_id scalar arg
        if any(_is_home_team_id(o) for o in operands):
            hits.append((n.lineno, "== home_team_id"))
            continue
        # shape 2: cross-source suffix comparison (e.g. team_id_action vs team_id_frame)
        suffixes = {s for s in (_subscript_suffix(o) for o in operands) if s is not None}
        if len(suffixes) >= 2:
            hits.append((n.lineno, "cross-source suffix"))
    return hits


@pytest.mark.parametrize("path", sorted(TRACKING.glob("*.py")), ids=lambda p: p.name)
def test_no_boundary_id_comparisons(path):
    if path.name in ALLOW_MODULES:
        pytest.skip("converter / helper module (ADR-001 or the helpers themselves)")
    hits = _boundary_compares(path)
    assert not hits, (
        f"{path.name}: raw boundary id comparison(s) {hits}; route through _id_compat "
        "(same_id / ids_match / ids_equal / ids_differ)"
    )
