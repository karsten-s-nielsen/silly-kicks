# Deferred Findings

Items identified during Phase 2 audits that are intentionally deferred. Each has a rationale
for why it's deferred and a trigger for when to revisit.

## Architecture Audit (2026-04-06)

| # | Sev | Finding | Rationale | Revisit When |
|---|-----|---------|-----------|-------------|
| A1 | Med | `wyscout.py` still ~900 lines (god module size) | Internals fixed (underscores, magic ints, config lookup). Structural decomposition belongs in Phase 3 — the "Nothing Left Behind" mapping registry will naturally restructure this file | Phase 3 |
| A9 | Med | `atomic/vaep/features.py` imports 12 names from `vaep.features` | Legitimate delegation — atomic features reuse standard ones. Tight but correct coupling | Only if atomic module is restructured |
| A14 | Med | `pandera` pervasive in all domain function signatures | Biggest deferred item. Dropping pandera changes public API, unblocks `numpy>=2.0`, removes `multimethod<2.0` pin. Tracked as TODO in `pyproject.toml` | Standalone decision — after all audits complete |
| A15 | Med | kloppy converter signature differs from others (LSP) | By design — kloppy uses `EventDataset`, not `pd.DataFrame`. Low ROI to change | Only if a Protocol for converters is introduced |
| A16 | Med | No plugin registry for converters (OCP) | YAGNI — 4 converters that change rarely | Only if >6 converters exist |
| A17 | Med | 3 `_fit_*` functions still tightly coupled to VAEP (SRP partial) | Dispatch fixed (registry), implementations extracted to `learners.py`. Could further decouple but diminishing returns | Phase 3 API evolution |
| A18 | Med | Schema validators frozen at class-definition time | Pandera-specific — resolves naturally if/when pandera is dropped (A14) | With A14 |
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
