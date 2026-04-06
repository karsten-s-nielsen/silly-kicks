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
