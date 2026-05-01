# TODO

Open items tracked for future work. Closed items live in
[CHANGELOG.md](CHANGELOG.md) and git history.

## Deferred tracking-aware features

Closed by silly-kicks 2.7.0: `silly_kicks.tracking` namespace primitive
layer (PR-S19; ADR-004). The following deferred items are scheduled for
follow-up PR cycles, in priority order:

1. `action_context()` --- tracking-aware VAEP/xG features (PR-S20, target 2.8.0).
2. `pressure_on_carrier()` --- pressure on the ball-carrier feature.
3. `infer_ball_carrier()` --- heuristic per-frame ball-carrier inference (lakehouse session addition).
4. `sync_score()` --- per-action tracking <-> events sync-quality score (lakehouse session addition).
5. Pitch-control models (Spearman / Voronoi).
6. Smoothing primitives (Savitzky-Golay, EMA).
7. Multi-frame interpolation / gap filling.
8. ReSpo.Vision tracking adapter (licensing-gated, not engineering-gated).

See `docs/superpowers/adrs/ADR-004-tracking-namespace-charter.md` invariant 9.

## Documentation

(none currently queued — D-8 closed in silly-kicks 2.1.1)

## Architecture

(none currently queued — A9 closed in silly-kicks 2.4.0 via `silly_kicks.vaep.feature_framework` extraction; see ADR-002)

## Open PRs

(none currently queued — PR-S14 shipped in silly-kicks 2.2.0)

## Tech Debt

(none currently queued — A19 / O-M1 / O-M6 reviewed and closed in 2.3.0 as stale-or-by-design; D-9 closed in 2.1.1; C-1 closed in 2.2.0)
