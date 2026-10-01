"""Fail-loud read-only match-load seam for the Phase 2 MCP tools (ADR-052 D14).

Resolves a ``match_ref`` string to a loaded match via the existing ``scripts/_loader_pining.py``
path, and RAISES (never returns a verdict) on a tokenless / failed / empty-degenerate resolution — a
tokenless pining run can return empty refs with exit 0, and a tool MUST NOT report ``OK`` on a bad
load. It writes no artifact and runs no clean-tree gate.

The loader lives in ``scripts/`` (not a package); this opt-in server module adds it to ``sys.path``
the same way ``scripts/measure_rc4_orientation.py`` does. ``list_match_refs`` and ``_pining_load_match``
are module-level names so tests can substitute a fake corpus without a network.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = str(Path(__file__).resolve().parents[2] / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from _loader_pining import LoadedMatch, list_match_refs  # noqa: E402  (scripts path shim above)
from _loader_pining import load_match as _pining_load_match  # noqa: E402

#: Providers searched when the caller does not name one (public via the built-in token; owner
#: providers resolve to nothing without ``PINING_FOR_THE_DATA_TOKEN`` — an empty resolution raises).
_DEFAULT_PROVIDERS: tuple[str, ...] = ("skillcorner", "idsse", "gradientsports")


def load_match(match_ref: str, provider: str | None = None) -> LoadedMatch:
    """Resolve + load ONE match, read-only, fail-loud. Returns a full ``LoadedMatch`` (frames present)."""
    providers = [provider] if provider else list(_DEFAULT_PROVIDERS)
    refs = list_match_refs(providers=providers, match_ids={p: [str(match_ref)] for p in providers})
    if not refs:
        raise RuntimeError(
            f"no match resolved for match_ref={match_ref!r} provider={provider!r} — tokenless/empty "
            "resolution or unknown ref. Refusing to return a verdict (read-only tripwire, ADR-052 D14)."
        )
    loaded = _pining_load_match(refs[0], events_only=False)
    if loaded.frames is None or len(loaded.frames) == 0:
        raise RuntimeError(f"empty/degenerate frames for match_ref={match_ref!r} — refusing to return a verdict.")
    return loaded
