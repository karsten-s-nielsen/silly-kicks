"""ADR-065 §3d completeness gate: no order-sensitive action-mart consumer may order by an
action_id-PRIMARY sort.

A persisted mart may carry a non-chronological ``action_id`` (mart reads bypass the
``_finalize_output`` guard), so a consumer that scans / neighbours / windows an action frame
must establish order by the ROBUST ``(time_seconds, action_id)`` key -- via
``_sort_actions_chronological_or_action_id`` or an inline time-primary ``sort_values`` -- NOT
``action_id`` alone / first. ``secured_reception`` was exactly this miss (4.119.1): an
``action_id``-alone scan that silently mislabeled AND crashed on a non-chronological mart,
caught only by a downstream crash, never by the hand-maintained §3d retrofit list.

This gate enumerates EVERY ``action_id``-primary ``sort_values`` in the package (AST) and requires
each to be an ``_EXEMPT`` non-action-scan site with a written reason. The exemption -- not a
smarter predicate -- is the mechanism ON PURPOSE: a safe ``action_id`` sort and an unsafe one are
the IDENTICAL AST (only provenance separates them), which is why the ADR-019 name/AST lint was
DELETED (ADR-043); the reviewed ``_EXEMPT`` bucket (ADR-056 three-bucket) is what a heuristic
cannot be. A new ``action_id``-primary sort fails CI until it is fixed to the robust key or
exempted with a reason.
"""

from __future__ import annotations

import ast
import pathlib

import silly_kicks

_PKG = pathlib.Path(silly_kicks.__file__).parent

#: "<relpath>::<enclosing function>" -> why an action_id-PRIMARY sort is correct here
#: (i.e. NOT an order-sensitive action-mart chronological scan).
_EXEMPT: dict[str, str] = {
    "silly_kicks/tracking/_kernels.py::_actor_pre_window_kernel": (
        "frame-grouping, not an action-mart scan: sort_values(['action_id','time_seconds']) "
        "co-locates each action's FRAMES and orders them by time WITHIN groupby('action_id'); the "
        "within-group chronology is the time_seconds SECONDARY key, so a non-chronological action_id "
        "across actions is immaterial (action_id here is a frame<->action merge key, not a scan order)."
    ),
}


def _first_sort_key(call: ast.Call) -> str | None:
    """The first ``by`` key of a ``.sort_values(...)`` call (positional or ``by=``), if a str literal."""
    arg: ast.expr | None = call.args[0] if call.args else None
    for kw in call.keywords:
        if kw.arg == "by":
            arg = kw.value
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return arg.value
    if isinstance(arg, (ast.List, ast.Tuple)) and arg.elts:
        head = arg.elts[0]
        if isinstance(head, ast.Constant) and isinstance(head.value, str):
            return head.value
    return None


class _Scan(ast.NodeVisitor):
    def __init__(self, rel: str) -> None:
        self.rel = rel
        self._fn = ["<module>"]
        self.hits: set[str] = set()

    def _enter_fn(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._fn.append(node.name)
        self.generic_visit(node)
        self._fn.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._enter_fn(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._enter_fn(node)

    def visit_Call(self, node: ast.Call) -> None:
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr == "sort_values" and _first_sort_key(node) == "action_id":
            self.hits.add(f"{self.rel}::{self._fn[-1]}")
        self.generic_visit(node)


def _action_id_primary_sorts() -> set[str]:
    flagged: set[str] = set()
    for py in sorted(_PKG.rglob("*.py")):
        rel = py.relative_to(_PKG.parent).as_posix()
        scan = _Scan(rel)
        scan.visit(ast.parse(py.read_text(encoding="utf-8")))
        flagged |= scan.hits
    return flagged


def test_no_unexempted_action_id_primary_sort():
    flagged = _action_id_primary_sorts()
    assert flagged, "scanner found NO action_id sort_values at all -- broken scan (non-vacuity guard)"
    unexempted = sorted(flagged - set(_EXEMPT))
    assert not unexempted, (
        "action_id-PRIMARY sort_values in an order-sensitive-suspect site (ADR-065 §3d): order by the "
        "ROBUST (time_seconds, action_id) key (via _sort_actions_chronological_or_action_id or an inline "
        f"time-primary sort), or add it to _EXEMPT with a reason: {unexempted}"
    )


def test_no_stale_exemptions():
    flagged = _action_id_primary_sorts()
    stale = sorted(set(_EXEMPT) - flagged)
    assert not stale, f"_EXEMPT names a site that no longer sorts action_id-primary -- remove it: {stale}"


def test_fixed_consumers_are_not_action_id_primary():
    """The 4.119.1 fix + its anchor establish order time-primary, so neither is flagged -- and the
    gate WOULD have caught the original secured_reception miss (which sorted action_id-alone)."""
    flagged = _action_id_primary_sorts()
    assert not any("secured_reception" in k for k in flagged)
    assert not any("_resolve_next_touch_positions" in k for k in flagged)
