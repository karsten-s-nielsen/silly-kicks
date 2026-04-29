# TODO

Open items tracked for future work. Closed items live in
[CHANGELOG.md](CHANGELOG.md) and git history.

## Documentation

| # | Size | Item | Context |
|---|------|------|---------|
| D-8 | Large | Add docstring `Examples` sections to public functions | 49 public functions, zero examples. Start with the 10 most-used: `convert_to_actions` (×4 providers), `VAEP.fit`, `VAEP.rate`, `gamestates`, `add_names`, `validate_spadl`, `HybridVAEP` |

## Architecture

| # | Size | Item | Context |
|---|------|------|---------|
| A9 | Medium | Reduce `atomic/vaep/features.py` coupling to `vaep/features` (12 imports) | Legitimate delegation today, but tight coupling will fight if atomic features need to diverge independently |
| — | Medium | Decompose `vaep/features.py` (809 lines) | Natural split: spatial features, temporal features, categorical features. Do when next adding features to this file |

## Open PRs

| # | Size | Item | Context |
|---|------|------|---------|
| PR-S9 | Medium | e2e prediction tests in CI via WorldCup HDF5 generation | Generate `tests/datasets/statsbomb/spadl-WorldCup-2018.h5` from open-data raw events (~64 matches × ~3 MB). Output structure: `games` table + `actions/game_<id>` per match (see `tests/vaep/test_vaep.py:48` for shape contract). Drop `@pytest.mark.e2e` on the 5 `test_predict*` cases (`tests/vaep/test_vaep.py:72,82`, `tests/test_xthreat.py:219,229`, `tests/atomic/test_atomic_vaep.py:24`). Conversion script committed at `scripts/build_worldcup_fixture.py`. Estimated 1-2 hours. See `docs/superpowers/specs/2026-04-29-recall-based-add-possessions-validation-design.md` § 10 for design notes. |
| PR-S10 | Medium-Large | `add_possessions` algorithmic precision improvement | Close the precision gap from ~42% toward 60-70% via brief-opposing-action merge rule, defensive-action class, and/or spatial continuity check. Plus re-measure `max_gap_seconds` parameter sweep using the new `boundary_metrics` utility before changing the default. New parameters likely: `merge_brief_opposing_actions`, `brief_action_window_seconds`. Atomic-SPADL counterpart must mirror any semantic change. Best done AFTER PR-S9 — 64-match WorldCup fixture is more reliable than PR-S8's 3-match set for parameter-tuning. |

