"""NaN-safety contract decorator for enrichment helpers (ADR-003).

Decorated functions claim that they tolerate NaN in caller-supplied
identifier columns: NaN identifiers route to the documented per-row
default rather than crashing.

The CI gates at ``tests/test_enrichment_nan_safety.py`` and
``tests/test_enrichment_provider_e2e.py`` auto-discover decorated
helpers via the ``_nan_safe`` attribute set by this decorator.
"""

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T", bound=Callable)


def nan_safe_enrichment(fn: T) -> T:
    """Marker decorator declaring fn satisfies the NaN-safety contract.

    See ADR-003 for the contract definition (caller-supplied NaN
    identifiers route to per-row default; helper does not crash).

    Examples
    --------
    Mark an enrichment helper as NaN-safe::

        from silly_kicks._nan_safety import nan_safe_enrichment

        @nan_safe_enrichment
        def my_enrichment(actions):
            return enriched_actions
    """
    fn._nan_safe = True  # type: ignore[attr-defined]
    return fn
