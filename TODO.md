# TODO

Open items tracked for future work. Closed items live in
[CHANGELOG.md](CHANGELOG.md) and git history.

## Documentation

(none currently queued — D-8 closed in silly-kicks 2.1.1)

## Architecture

| # | Size | Item | Context |
|---|------|------|---------|
| A9 | Low | `atomic/vaep/features.py` per-concern coupling to `vaep/features` (12 symbols across 4 submodules — was monolith) | Partially addressed in 2.3.0 via vaep/features decomposition. Full decoupling deferred until atomic features genuinely need to diverge independently — extracting truly-shared framework into a cross-package module is the trigger condition. |

## Open PRs

(none currently queued — PR-S14 shipped in silly-kicks 2.2.0)

## Tech Debt

(none currently queued — A19 / O-M1 / O-M6 reviewed and closed in 2.3.0 as stale-or-by-design; D-9 closed in 2.1.1; C-1 closed in 2.2.0)

