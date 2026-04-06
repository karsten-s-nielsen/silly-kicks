# Deferred Findings

Items identified during Phase 2 audits that are intentionally deferred. Each has a rationale
for why it's deferred and a trigger for when to revisit.

## Phase 3c: Converter Rewrite (2026-04-06)

- **O-2** (Wyscout 3x apply): RESOLVED — replaced with np.select
- **O-2b** (StatsBomb apply): RESOLVED — replaced with np.select
- **O-2c** (Opta 3x apply): RESOLVED — replaced with np.select
- **O-3** (Wyscout position extraction): RESOLVED — vectorized list indexing
- **O-4** (Wyscout 55x tag apply): RESOLVED — batch DataFrame constructor
- **O-5b** (StatsBomb dispatch dict per row): RESOLVED — module-level np.select
- **O-6** (StatsBomb coordinate for-loop): RESOLVED — numpy vectorized
- **O-8** (gamestates groupby.apply): RESOLVED — vectorized shift + boundary detection
- **O-12** (kloppy dispatch): RECLASSIFIED — kloppy EventDataset API, not actionable
- **O-13** (atomic 3x concat+sort): RESOLVED — single deferred sort
- **O-15** (config DataFrame caching): RESOLVED — @lru_cache
- **A1** (Wyscout god module): RESOLVED — split into 3 files

## Phase 3b: I/O Contract (2026-04-06)

- **A14** (pandera dependency): RESOLVED — replaced with plain schema constants (`SPADL_COLUMNS`, `ATOMIC_SPADL_COLUMNS`)
- **A18** (frozen schema validators): RESOLVED — no longer applicable (pandera removed)
- numpy upper bound removed (was `<2.0` due to pandera `np.string_` usage)
- `multimethod<2.0` dependency removed (was pandera transitive)

## Architecture Audit (2026-04-06)

| # | Sev | Finding | Rationale | Revisit When |
|---|-----|---------|-----------|-------------|
| A1 | Med | `wyscout.py` still ~900 lines (god module size) | Internals fixed (underscores, magic ints, config lookup). Structural decomposition belongs in Phase 3 — the "Nothing Left Behind" mapping registry will naturally restructure this file | Phase 3 |
| A9 | Med | `atomic/vaep/features.py` imports 12 names from `vaep.features` | Legitimate delegation — atomic features reuse standard ones. Tight but correct coupling | Only if atomic module is restructured |
| A14 | Med | ~~`pandera` pervasive in all domain function signatures~~ | **RESOLVED in Phase 3b** — pandera removed, replaced with plain schema constants | ~~Standalone decision~~ |
| A15 | Med | kloppy converter signature differs from others (LSP) | By design — kloppy uses `EventDataset`, not `pd.DataFrame`. Low ROI to change | Only if a Protocol for converters is introduced |
| A16 | Med | No plugin registry for converters (OCP) | YAGNI — 4 converters that change rarely | Only if >6 converters exist |
| A17 | Med | 3 `_fit_*` functions still tightly coupled to VAEP (SRP partial) | Dispatch fixed (registry), implementations extracted to `learners.py`. Could further decouple but diminishing returns | Phase 3 API evolution |
| A18 | Med | ~~Schema validators frozen at class-definition time~~ | **RESOLVED in Phase 3b** — pandera removed, no more class-time validation | ~~With A14~~ |
| A19 | Low | Default hyperparameters scattered across 3 learner functions | Extracted to named constants in `learners.py`. Could centralize further but low impact | Phase 3 API evolution |

## Security Audit (2026-04-06)

| # | Sev | Finding | Rationale | Revisit When |
|---|-----|---------|-----------|-------------|
| S5 | Low | Optional ML deps have no upper bounds | Standard for libraries — upper bounds cause resolver conflicts for downstream users. Lockfile for CI is the better answer | When CI reproducibility becomes an issue |
| S6 | Info | Unseeded `np.random.permutation` in `VAEP.fit()` | Reproducibility, not security. Adding `random_state` parameter is a public API change | Phase 3 API evolution |

## Optimization Audit (2026-04-06)

### Vectorization — converter hot paths (all High, medium effort)

These are the biggest remaining performance wins. All involve replacing `df.apply(axis=1)` row-wise
Python dispatch with vectorized operations. Deferred to Phase 3 because the converters are being
rewritten anyway for the "Nothing Left Behind" mapping registry.

| # | Sev | Finding | File:Lines | Revisit When |
|---|-----|---------|------------|-------------|
| O-2 | High | 3x `df.apply(axis=1)` for Wyscout row classification | `spadl/wyscout.py:633-635` | Phase 3 converter rewrite |
| O-2b | High | `apply(axis=1)` for StatsBomb event parsing + end location | `spadl/statsbomb.py:90,109-111,152` | Phase 3 converter rewrite |
| O-2c | High | 3x `apply(axis=1)` for Opta type/result/bodypart | `spadl/opta.py:57-63` | Phase 3 converter rewrite |
| O-8 | High | `groupby().apply()` per shift step in `gamestates` | `vaep/features.py:91-96` | Phase 3 — vectorize with masked shifts |

### Medium-effort improvements

| # | Sev | Finding | File:Lines | Revisit When |
|---|-----|---------|------------|-------------|
| O-3 | Med | `apply(axis=1)` + redundant merge for Wyscout position extraction | `spadl/wyscout.py:224-228` | Phase 3 converter rewrite |
| O-4 | Med | 55 separate `apply()` passes for Wyscout tag columns | `spadl/wyscout.py:122-126` | Phase 3 converter rewrite |
| O-6 | Med | Python for-loop filling NumPy array in StatsBomb coordinate converter | `spadl/statsbomb.py:199-217` | Phase 3 converter rewrite |
| O-7 | Med | Nested loop adding 27 columns in labels; chained index assignment | `vaep/labels.py:39-49,82-92` | Phase 3 API evolution |
| O-12 | Med | kloppy dispatch dict rebuilt per row | `spadl/kloppy.py:181-208` | Phase 3 converter rewrite |
| O-13 | Med | 3x concat+sort in atomic conversion instead of 1 deferred sort | `atomic/spadl/base.py:110-197` | Phase 3 |
| O-15 | Med | `actiontypes_df()`/`results_df()`/`bodyparts_df()` reconstruct constant DataFrames per call | `vaep/features.py`, `spadl/utils.py` | Phase 3 — cache at module level |
| O-5b | Med | StatsBomb `_parse_event` dispatch dict rebuilt per row (now using dict lookups but dict itself still inside function) | `spadl/statsbomb.py:228-248` | Phase 3 converter rewrite |

### Low-priority items

| # | Sev | Finding | File:Lines | Revisit When |
|---|-----|---------|------------|-------------|
| O-14 | Low | `feature_column_names` not cached, called redundantly in fit/rate cycle | `vaep/base.py:178,201` | Phase 3 API evolution |
| O-16 | Low | 138-iteration loop for type×result onehot cross-product | `vaep/features.py:270-273` | Phase 3 |
| O-M1 | Low | Full `events.copy()` at top of StatsBomb convert_to_actions | `spadl/statsbomb.py:78` | Phase 3 converter rewrite |
| O-M6 | Low | Temporary n×3 DataFrame for StatsBomb fidelity version check | `spadl/statsbomb.py:171` | Phase 3 converter rewrite |

## Documentation Audit (2026-04-06)

### Repository governance (Phase 3 territory — project maturity)

| # | Sev | Finding | Rationale | Revisit When |
|---|-----|---------|-----------|-------------|
| D-1 | Med | Missing CONTRIBUTING.md | Not needed until community contributors arrive. Write before making repo public | Before public release |
| D-2 | Med | Missing SECURITY.md | No vulnerability disclosure process. Write before making repo public | Before public release |
| D-3 | Low | Missing CODE_OF_CONDUCT.md | Standard community file | Before public release |
| D-4 | Med | Missing CHANGELOG.md | Not needed until first PyPI release | Before v0.1.0 PyPI publish |
| D-5 | Low | Missing GitHub issue/PR templates | Nice-to-have for structured contributions | Before public release |

### Documentation content (Phase 3 territory — API may evolve)

| # | Sev | Finding | Rationale | Revisit When |
|---|-----|---------|-----------|-------------|
| D-6 | High | README missing VAEP workflow example | The "V" in the library's value proposition is undemonstrated. Deferred because API may change in Phase 3 | After Phase 3 API stabilizes |
| D-7 | Med | README missing multi-provider examples (Wyscout, Opta, kloppy) | Listed in Features but never shown | After Phase 3 |
| D-8 | Low | No docstring `Examples` sections on any public function | Systematic gap — 49 functions, zero examples. Valuable for Sphinx/doctest but not blocking | After Phase 3 API stabilizes |
| D-9 | Low | 5 xthreat module-level functions not underscore-prefixed but not re-exported | `scoring_prob`, `get_move_actions`, etc. are implementation helpers but technically public | Phase 3 API cleanup |
