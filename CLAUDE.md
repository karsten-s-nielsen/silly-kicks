# silly-kicks

Maintained fork of socceraction — SPADL event conversion + VAEP action valuation for soccer analytics.

## Architecture

- **Hexagonal**: All core functions are pure (pandas in, pandas out). Zero I/O, zero global state mutation.
- **Converters** (`spadl/`): Each provider (StatsBomb, Opta, Wyscout, Sportec, Metrica, PFF, plus the Kloppy gateway) has its own module. All return `tuple[pd.DataFrame, ConversionReport]` with guaranteed columns/dtypes via `_finalize_output()`.
- **VAEP** (`vaep/`): Feature engineering + gradient boosting binary classifiers. `HybridVAEP` removes result leakage from a0 features. xG-targeted labels available via `xg_column` parameter.
- **Atomic-SPADL** (`atomic/`): Variant where actions are decomposed into atomic sub-actions (receival, interception, out, etc.).

## Key conventions

- No pandera — schemas are plain Python dicts (`SPADL_COLUMNS`, `ATOMIC_SPADL_COLUMNS`).
- Config DataFrames (`actiontypes_df()`, etc.) are cached with `@functools.cache`.
- Vectorized dispatch: converters use `np.select` over pre-flattened columns, not `apply(axis=1)`.
- All `warnings.warn()` calls include `stacklevel=2`.
- ML naming conventions (uppercase `X`, `Y`, `Pscores`) are allowed in `vaep/` and `xthreat.py` per ruff per-file-ignores.
- **Converter identifier conventions are sacred.** SPADL DataFrame converters never override the caller's `team_id` / `player_id` columns from provider-specific qualifiers. Qualifier-derived facts surface as dedicated output columns (see `tackle_winner_*` / `tackle_loser_*` on sportec). Decision: ADR-001.
- **Public enrichment helpers (post-conversion `add_*` family) tolerate NaN in caller-supplied identifier columns.** NaN identifiers route to the documented per-row default (typically NaN-output / False / 0); helpers never crash on NaN input. Decoration with `@nan_safe_enrichment` from `silly_kicks._nan_safety` is the formal opt-in; the CI gate at `tests/test_enrichment_nan_safety.py` auto-discovers decorated helpers. Decision: ADR-003.

## Testing

```bash
python -m pytest tests/ -m "not e2e" -v --tb=short
```

e2e tests require dataset fixtures not committed to the repo. Tests with
fixtures committed to the repo should not be marked e2e — they run in
the regular suite.

## Open Items

See [TODO.md](TODO.md) for tracked work.

## Dependencies

- Runtime: pandas, numpy, scikit-learn (no pandera, no multimethod)
- Optional: kloppy, xgboost, catboost, lightgbm
- numpy>=2.0 compatible
