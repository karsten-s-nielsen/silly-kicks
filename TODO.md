# TODO

Open items tracked for future work. Closed items live in
[CHANGELOG.md](CHANGELOG.md) and git history.

## Documentation

(none currently queued — D-8 closed in silly-kicks 2.1.1)

## Architecture

| # | Size | Item | Context |
|---|------|------|---------|
| A9 | Medium | Reduce `atomic/vaep/features.py` coupling to `vaep/features` (12 imports) | Legitimate delegation today, but tight coupling will fight if atomic features need to diverge independently |
| — | Medium | Decompose `vaep/features.py` (809 lines) | Natural split: spatial features, temporal features, categorical features. Do when next adding features to this file |

## Open PRs

(none currently queued — PR-S14 shipped in silly-kicks 2.2.0)

## Tech Debt

| # | Sev | Item | Context |
|---|-----|------|---------|
| A19 | Low | Default hyperparameters scattered across 3 learner functions | Extracted to named constants in `learners.py`; could centralize further but low impact. Audit-source: DEFERRED.md (Phase 2 architecture audit, migrated 1.9.0). |
| O-M1 | Low | Full `events.copy()` at top of StatsBomb `convert_to_actions` (`spadl/statsbomb.py:78`) | Defensive copy — could shrink on demand. Audit-source: DEFERRED.md (migrated 1.9.0). |
| O-M6 | Low | Temporary n×3 DataFrame for StatsBomb fidelity version check (`spadl/statsbomb.py:171`) | Audit-source: DEFERRED.md (migrated 1.9.0). |

