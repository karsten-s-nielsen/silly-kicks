"""Per-group ``MirrorEntry`` registrations, auto-discovered by ``_mirror_registry``.

One module per aggregator group rather than one big file: the groups were populated
independently, and a single file would have serialised that work behind edit conflicts for no
benefit. Auto-discovery (``pkgutil.iter_modules``) is also anti-rot in its own right -- a new
group module is picked up without editing a manifest that could go stale.

Each module exposes ``register()``, which calls ``_mirror_registry._entry(...)`` once per
aggregator it owns.
"""

from __future__ import annotations
