"""The set of default xfn lists the leakage guards sweep is pinned EXACTLY (Cycle B).

Three leakage guards each carried their own copy of the discovery rule behind
`assert len(lists) >= 10`, against a real population of 19. That floor cannot detect an omission,
which is the whole argument of this cycle -- and the thing omitted here would be a default list
that NO leakage guard sweeps, leaving a leaky factory free to be opted into it.
"""

from __future__ import annotations

from tests.tracking._xfn_default_lists import SWEPT, default_lists


def test_the_swept_default_lists_are_EXACT():
    """Fails BOTH ways -- the property `assert len(lists) >= 10` never had against 19."""
    found = set(default_lists())
    assert found == set(SWEPT), (
        f"new and unswept: {sorted(found - SWEPT)}; registered but gone: {sorted(SWEPT - found)}"
    )


def test_the_registry_is_not_silently_empty():
    """Meta-assertion: a broken discovery would make the gate above pass only if SWEPT were also
    emptied, but an import failure inside `default_lists` would silently shrink BOTH sides of a
    naive comparison written against a live re-derivation."""
    assert len(SWEPT) >= 15, f"SWEPT looks truncated: {len(SWEPT)}"
    assert len(default_lists()) >= 15, "default-list discovery looks broken"
