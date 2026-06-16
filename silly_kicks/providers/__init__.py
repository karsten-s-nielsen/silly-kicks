"""Per-provider raw-data parse ports (bytes -> provider-canonical bronze rows).

A parse port is the faithful ``bytes -> bronze`` boundary for a provider's native
files; a separate ``shape_*`` composable maps bronze -> a silly-kicks converter's
input. See ADR-031 (T3) +
``docs/superpowers/specs/2026-06-16-dfl-parse-port-design.md``. Behind the
``[parse-dfl]`` extra.
"""
