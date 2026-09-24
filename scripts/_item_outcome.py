"""A deterministic admission outcome for a corpus item (ADR-052 D13).

A LEAF module: it imports nothing from ``scripts/``, so ``scripts/_driver.py`` (which recognises the
exception and persists it as a marker) and the loader modules (which raise it) can both import it
without a dependency cycle -- the loader, an adapter, never reaches for the orchestration seam just
for an exception type.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping


class ItemExcluded(Exception):  # noqa: N818 -- named for the OUTCOME (an exclusion), not an error condition
    """A corpus item is not in the pass, for a deterministic ``reason``.

    Deterministic means a pure function of the item's artifact bytes and the code (the SkillCorner S1
    geometry gate is one), so it is safe to persist: ``for_each`` writes it as a ``<key>.excluded.json``
    marker and skips the item on resume. Time-varying *availability* (a manifest that does not list an
    artifact yet) must NOT be raised here -- it is resolved at ref-listing time instead.

    ``details`` is a JSON-serializable map recorded beside ``reason`` in the marker, so a later consumer
    (e.g. Task 0 reading an S1 exclusion's measured off-pitch rates) reads structured values instead of
    parsing them out of the human-readable string. Both are validated at construction: a marker with an
    empty reason or unserializable details would otherwise fail only at write time, mid-corpus.
    """

    def __init__(self, reason: str, *, details: Mapping[str, object] | None = None) -> None:
        if not reason:
            raise ValueError("ItemExcluded needs a non-empty reason")
        resolved = dict(details or {})
        try:
            json.dumps(resolved)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"ItemExcluded details must be JSON-serializable: {exc}") from exc
        super().__init__(reason)
        self.reason = reason
        self.details = resolved


# Path-independent identity. This leaf is imported both as ``scripts._item_outcome`` (the package path
# ``_driver`` and the loaders use) and bare ``_item_outcome`` (the tests' sys.path convention, and the
# drivers that do ``sys.path.insert(0, "scripts")``). Two module OBJECTS would mean two distinct
# ``ItemExcluded`` classes, so ``except ItemExcluded`` in ``_driver`` would MISS a ``MatchExcluded`` a
# loader raised under the other path -- an exclusion would read as a failure. Aliasing both names to
# THIS object gives the class ONE identity whichever path imports it first. Measured: `_driver` imports
# ``scripts._item_outcome`` while the tests + `validate_xshot_causal` raise via bare imports.
for _alias in ("scripts._item_outcome", "_item_outcome"):
    sys.modules.setdefault(_alias, sys.modules[__name__])
