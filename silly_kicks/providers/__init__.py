"""Per-provider raw-data parse ports (bytes -> provider-canonical bronze rows).

A parse port is the faithful ``bytes -> bronze`` boundary for a provider's native
files; a separate ``shape_*`` composable maps bronze -> a silly-kicks converter's
input.

Extras are PER PORT, not package-wide:

* ``sportec`` -- the IDSSE/DFL XML parser, behind the ``[parse-dfl]`` extra.
  See ADR-031 (T3) + ``docs/superpowers/specs/2026-06-16-dfl-parse-port-design.md``.
* ``statsbomb`` -- SB360 freeze-frames to the tracking-snapshot contract. **No extra
  and no new runtime dependency**: it shapes already-loaded payloads and never fetches,
  so ``statsbombpy`` is not required (it is a script dependency, imported lazily by
  ``scripts/build_sb360_coverage.py``).
"""
