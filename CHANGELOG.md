# Changelog

All notable changes to silly-kicks will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-04-07

### Added
- DEBUG logging for kloppy silent event drops (aerial duels, unrecognized GK subtypes)
- `.github/CODEOWNERS` for code owner review enforcement

### Fixed
- StatsBomb converter now accepts both `"goalkeeper"` and `"goal_keeper"` keys in the
  extra dict — adapters that snake-case the event type name no longer silently lose all
  keeper actions

### Improved
- `ConversionReport` docstring: full Attributes section, usage example, provider-specific
  key type note
- `add_names()` docstring: explicit guarantee that caller-added columns are preserved
- `_finalize_output()` docstring: guarantee that all SPADL_COLUMNS are present
- `config.py` docstring: `actiontype_id`, `result_id`, `bodypart_id` reverse dicts documented
- Wyscout `convert_to_actions()`: Returns section now documents `ConversionReport`;
  `goalkeeper_ids` notes `None` ≡ empty set equivalence

### Removed
- `docs/plans/` and `docs/specs/` — internal development artifacts with local paths

### Changed
- Version bump: 0.1.0 → 1.0.0 (Production/Stable)
- C4 diagram genericized (removed project-specific references)

## [0.1.0] — 2026-04-06

### Added
- Initial release as maintained successor to socceraction v1.5.3
- SPADL converters: StatsBomb, Opta, Wyscout, Kloppy
- VAEP and Atomic-VAEP frameworks
- HybridVAEP — result-leakage-free action valuation
- xG-targeted labels via `xg_column` parameter
- Expected Saves (xS) label via `save_from_shot()`
- Expected Claims (xC) label via `claim_from_cross()`
- Cross zone feature (Gelade 2017 four-zone classification)
- Assist type feature (through ball, cutback, cross, set piece, progressive pass)
- Wyscout `goalkeeper_ids` parameter for GK aerial duel routing (#37)
- `ConversionReport` audit trail for every conversion
- `validate_spadl()` utility for DataFrame validation
- Input validation with clear error messages per provider
- "Nothing Left Behind" mapping registries (mapped/excluded/unrecognized events)
- Reproducible training via `random_state` parameter

### Changed (from socceraction v1.5.3)
- Dropped pandera dependency — schemas are plain Python constants
- Dropped multimethod dependency
- Removed numpy<2.0 upper bound
- All converters return `tuple[pd.DataFrame, ConversionReport]`
- All `apply(axis=1)` hot paths replaced with `np.select` vectorization
- Wyscout module decomposed into 3 files
- Gamestates uses vectorized shift instead of `groupby().apply()`
- Config DataFrame factories cached with `@functools.cache`
- Labels vectorized (shift-based accumulation replaces 27-column loop)
- `actiontype_result_onehot` uses numpy broadcasting

### Fixed
- Bug #507: Empty game crash in `gamestates()`
- Bug #950: `actiontype` feature wrong for Atomic-SPADL
- Bug #784: Opta converter silently drops card events
- Bug #831: Atomic-SPADL missing "out" for blocked/saved shots
- Bug #37/D44: Wyscout keeper_claim/punch differentiation
- Bug #946: pandas 3.0 `fillna(inplace=True)` deprecation
- pandas 3.0 `groupby().apply(as_index=False)` key column drop
