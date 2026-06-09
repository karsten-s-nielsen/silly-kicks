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

Scope includes the converter-adapter orientation seam (ADR-019 amendment, 4.21.1). A tracking
converter's ``convert_to_frames`` derives ``team_attacking_direction`` from
``team_id == home_team_id``, where ``home_team_id`` is a caller-supplied arg of uncontrolled dtype
-- the same boundary as a feature seam. This is shape 1, so the adapters are covered; only modules
with a load-bearing, documented exemption stay in ALLOW_MODULES (see below).
"""

import ast
import pathlib

import pytest

TRACKING = pathlib.Path(__file__).resolve().parents[2] / "silly_kicks" / "tracking"

# Only `_id_compat.py` is exempt -- it defines and tests the helpers themselves (the canonical
# `== home_team_id` lives there). EVERY tracking module, including the converter adapters, routes its
# id comparisons through the helpers (ADR-019 amendment, 4.21.1).
#
# The converter-adapter orientation seam used to be blanket-skipped, which hid BUG-4 (4.20.1): in
# gradientsports.py / sportec.py, convert_to_frames took a caller-supplied `home_team_id` of
# uncontrolled dtype and compared it against an object/Int64 frame `team_id` via a raw
# `team_id == home_team_id` -- silently matching zero players for an int arg vs string frames,
# mislabeling team_attacking_direction, and double-flipping the frame in play_left_to_right (the
# structural_sgm away-team blow-up root cause). gradientsports/sportec now use `ids_match`; kloppy's
# orientation seam (str-vs-str internally, no caller-dtype boundary) routes through `same_id` for
# consistency. The lint guards every adapter against reintroducing a raw boundary comparison.
ALLOW_MODULES = {"_id_compat.py"}

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


def test_detector_fires_on_bug4_orientation_shape(tmp_path):
    """Discriminating proof (NOT a fixture gap): the BUG-4 shape `out["team_id"] == home_team_id`
    -- a convert_to_frames orientation comparison against the caller's scalar arg -- IS flagged by
    the detector. This is the regression class that un-skipping gradientsports.py / sportec.py now
    guards: a green run on the live (already-fixed) adapters could otherwise be a detector that
    never fires for this shape rather than genuinely-clean code."""
    p = tmp_path / "fake_adapter.py"
    p.write_text('is_home = out["team_id"] == home_team_id\n', encoding="utf-8")
    hits = _boundary_compares(p)
    assert [kind for _, kind in hits] == ["== home_team_id"]


def test_only_id_compat_is_exempt():
    """Anti-regression lock (ADR-019 amendment, 4.21.1): `_id_compat.py` (which defines the helpers)
    is the SOLE exempt module. The converter adapters -- gradientsports.py / sportec.py (caller-supplied
    home_team_id) and kloppy.py (orientation seam) -- must never be re-added to ALLOW_MODULES; a blanket
    file-skip is exactly what hid BUG-4. Pinning the whole set keeps every adapter under the lint."""
    assert ALLOW_MODULES == {"_id_compat.py"}
