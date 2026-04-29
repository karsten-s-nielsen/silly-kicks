# Changelog

All notable changes to silly-kicks will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.10.0] — 2026-04-29

### Added
- **Public `silly_kicks.spadl.coverage_metrics(*, actions, expected_action_types)` utility**
  for computing per-action-type coverage on a SPADL action stream. Returns
  a `CoverageMetrics` TypedDict (also re-exported from `silly_kicks.spadl`).
  Keyword-only arguments. Resolves `type_id` to action-type name via
  `spadlconfig.actiontypes_df`; reports any expected action types that
  produced zero rows under `missing`. Out-of-vocab `type_id` values are
  reported as `"unknown"` rather than raising. Mirrors the PR-S8
  `boundary_metrics` shape and discipline.
- **`goalkeeper_ids: set[str] | None = None` parameter on
  `silly_kicks.spadl.sportec.convert_to_actions`** as a supplementary
  signal: when provided, Play events whose `player_id` is in the set
  AND which have NO explicit `play_goal_keeper_action` qualifier are
  routed to the keeper_pick_up + pass 2-action synthesis. The
  qualifier-driven mapping remains the primary contract.
- **`goalkeeper_ids: set[str] | None = None` parameter on
  `silly_kicks.spadl.metrica.convert_to_actions`** as the PRIMARY
  mechanism for surfacing GK actions. Metrica's source format lacks
  native GK markers; with `goalkeeper_ids`, conservative routing applies
  (PASS by GK → synth, RECOVERY by GK → keeper_pick_up, CHALLENGE
  AERIAL-WON by GK → keeper_claim). Without it: 0 keeper_* actions
  (1.9.0 default behaviour preserved — no breaking change).
- **`goalkeeper_ids` no-op acceptance on `statsbomb.convert_to_actions`
  and `opta.convert_to_actions`** for cross-provider API symmetry. Both
  source formats natively mark GK actions; the parameter is silently
  accepted with byte-for-byte identical output.
- **DFL distribution qualifiers `throwOut` and `punt` now produce SPADL
  actions** (sportec converter). Each source row synthesizes TWO
  actions: `keeper_pick_up + pass` (bodypart=other) for `throwOut`,
  `keeper_pick_up + goalkick` (bodypart=foot) for `punt`. Both rows
  inherit the source's `(player_id, team, period, time, x, y)`.
  `preserve_native` columns propagate to both. Action_ids renumbered
  dense after synthesis.
- **Production-shape vendored fixtures** under
  `tests/datasets/idsse/sample_match.parquet` (~166 KB; 308-row subset
  of `soccer_analytics.bronze.idsse_events` match `idsse_J03WMX`,
  includes throwOut + punt rows) and
  `tests/datasets/metrica/sample_match.parquet` (~20 KB; 300-event
  subset of Metrica Sample Game 2). Build script at
  `scripts/extract_provider_fixtures.py` (Databricks pull for IDSSE,
  offline kloppy-fixture subset for Metrica). Attribution READMEs
  alongside.
- **Cross-provider parity meta-test** at
  `tests/spadl/test_cross_provider_parity.py`. Parametrized over all 5
  DataFrame converters (statsbomb, opta, wyscout, sportec, metrica);
  asserts each emits at least one `keeper_*` action when given a
  fixture exercising GK paths. This is the regression gate that would
  have caught Bugs 1-3 in 1.7.0 if it had existed.
- **`pyarrow>=14.0.0` added to `[test]` extras** to back parquet I/O
  for the new fixtures (`pd.read_parquet` / `pd.DataFrame.to_parquet`).

### Fixed
- **Sportec converter no longer drops all DFL `Play` events to
  non_action.** The pre-1.10.0 dispatch checked `et == "Pass"` for
  pass-class events, but DFL bronze never emits `"Pass"` — the actual
  event_type is `"Play"`. Net effect since 1.7.0: all IDSSE matches in
  production lost ~60-80% of their actions (every pass, cross, and head
  pass) to silent non_action drop. Fix restructures the dispatch so
  `Play` events with no GK qualifier route to `pass` / `cross` (with
  optional head bodypart) and `Play` events with a recognized GK
  qualifier route to `keeper_*` actions. Defensive: `Play` events with
  an unrecognized non-empty qualifier still drop to `non_action`.
  ``"Pass"`` is removed from the recognized event-type vocabulary so
  legacy callers (if any) surface in `unrecognized_counts` (loud)
  rather than silently mapping to non_action.
- **Sportec converter no longer drops `throwOut` and `punt` GK
  distribution events to non_action.** These DFL qualifier values
  represent GK distribution actions (throwing or kicking the ball to
  a teammate); pre-1.10.0 they were unmapped. Fix synthesizes 2
  SPADL actions per source event (see Added section).
- **Metrica converter now produces non-zero GK coverage when
  `goalkeeper_ids` is supplied.** Pre-1.10.0 the converter had no
  mechanism to surface GK actions, leaving downstream `add_gk_role` /
  `add_pre_shot_gk_context` enrichments at 100% NULL on every Metrica
  match in production.

### Notes
- This release closes the upstream gap that surfaced during
  luxury-lakehouse PR-LL2 production deploy (2026-04-29): post-deploy
  validation found 100% NULL `gk_role` and `defending_gk_player_id` on
  IDSSE (2,522 rows) and Metrica (5,839 rows) sources. With silly-kicks
  1.10.0, downstream lakehouse can re-run `apply_spadl_enrichments`
  against IDSSE + Metrica with non-NULL GK coverage (handled by
  separate lakehouse PR-LL3).
- Behaviour change for IDSSE consumers: bronze.spadl_actions row count
  per IDSSE match will increase materially (every Play event now
  surfaces as a SPADL pass, plus throwOut/punt rows now produce 2
  actions each). This is the intended fix; downstream aggregation may
  need to re-baseline.
- Wyscout converter unchanged — `goalkeeper_ids` was already present
  from 1.0.0.
- Atomic-SPADL `coverage_metrics` parity is queued as tech debt
  (atomic uses 33 action types vs standard's 23; deferred until a
  consumer asks). Tracked in `TODO.md ## Tech Debt`.

## [1.9.0] — 2026-04-29

### Added
- **Vendored `tests/datasets/statsbomb/spadl-WorldCup-2018.h5`** — committed
  HDF5 fixture for the FIFA World Cup 2018 (64 matches, 128,484 SPADL
  actions, 5.9 MB on disk with zlib compression). All 5 prediction
  pipeline tests in `tests/vaep/`, `tests/test_xthreat.py`, and
  `tests/atomic/` now run on every PR + push. Pre-1.9.0 these tests
  silently skipped in CI and locally because the fixture was never
  committed. Net: ~9 release cycles of zero coverage on the prediction
  pipeline (VAEP fit/rate, xT fit/rate, atomic VAEP fit/rate) is now
  closed.
- **`scripts/build_worldcup_fixture.py`** — reproducible HDF5 generator.
  Downloads StatsBomb open-data WorldCup-2018 raw events (cached at
  `tests/datasets/statsbomb/raw/.cache/`, gitignored), converts each via
  `silly_kicks.spadl.statsbomb.convert_to_actions`, writes the multi-key
  HDFStore. CLI: `--output`, `--cache-dir`, `--no-cache`, `--verbose`,
  `--quiet`. Cold-cache run on broadband: ~30-60 sec. Warm-cache re-run:
  ~5 sec. No new dependencies (stdlib + pandas + already-present
  pytables).
- **`scripts/` is now linted in CI** — `.github/workflows/ci.yml` runs
  `ruff check` and `ruff format --check` on `silly_kicks/`, `tests/`,
  AND `scripts/`. Pyright include stays `silly_kicks/` only — build
  scripts aren't worth full type-checking.

### Changed
- **`tests/conftest.py::sb_worldcup_data` calls `pytest.fail` instead of
  `pytest.skip` when the HDF5 is absent.** Matches the PR-S8 pattern for
  committed fixtures: once a fixture is committed, "missing" is a
  packaging error worth surfacing prominently — not a silent skip that
  lets CI quietly regress. Failure message points at the build script
  for regeneration.
- The 5 `test_predict*` cases (`tests/vaep/test_vaep.py::test_predict`,
  `tests/vaep/test_vaep.py::test_predict_with_missing_features`,
  `tests/test_xthreat.py::test_predict`,
  `tests/test_xthreat.py::test_predict_with_interpolation`,
  `tests/atomic/test_atomic_vaep.py::test_predict`) no longer carry the
  `@pytest.mark.e2e` marker. They run in the regular suite on every CI
  matrix slot (4 slots, ~5-15 sec overhead per slot — negligible).

### Fixed
- **`silly_kicks.xthreat.ExpectedThreat.interpolator()` is no longer
  broken on SciPy 1.14+.** The wrapper used `scipy.interpolate.interp2d`
  which was removed in SciPy 1.14.0 (the import succeeds but the call
  raises `NotImplementedError`). The bug was latent since 1.0.0 because
  `tests/test_xthreat.py::test_predict_with_interpolation` was the only
  consumer and it was `@pytest.mark.e2e`-marked + skipping silently.
  Surfaced precisely when this PR dropped the marker. Replaced with
  `scipy.interpolate.RectBivariateSpline` — the SciPy-recommended
  bug-for-bug compatible replacement for regular grids — wrapped to
  preserve the legacy `interp(xs, ys) -> (W, L)` calling convention so
  callers downstream of `interpolator()` need no changes. Output shape
  and indexing semantics unchanged.
- The `test_interpolate_xt_grid_no_scipy` regression test that mocks
  the missing-scipy path now mocks `RectBivariateSpline` instead of the
  removed `interp2d`.

### Documentation
- **`docs/DEFERRED.md` deleted; live items migrated to a new `## Tech
  Debt` section in `TODO.md`.** Per the National Park Principle —
  bundle the cleanup of the rotting parallel doc into this cycle since
  we're already touching `TODO.md` anyway. Audit history preserved in
  `git log -- docs/DEFERRED.md`. Migrated items: A19 (default
  hyperparameters scattered), D-9 (5 xthreat module-level functions
  naming), O-M1 (StatsBomb `events.copy()`), O-M6 (StatsBomb fidelity
  version check temporary DataFrame). Items judged "by design / accept"
  and not migrated: A15 (kloppy LSP differs by design), A16 (no plugin
  registry — YAGNI for 4 converters), A17 (`_fit_*` coupling — partial
  refactor done, diminishing returns), S5 (optional ML deps no upper
  bounds — librarian convention).
- `CLAUDE.md` no longer references `docs/DEFERRED.md` (file removed).

### Notes
- WorldCup HDF5 file size: 5.9 MB on disk (well under GitHub's 50 MB soft
  warn / 100 MB hard reject thresholds — no Git LFS needed). Total wheel
  size unchanged (test fixtures live under `tests/`, excluded from
  `[tool.hatch.build.targets.wheel] packages = ["silly_kicks"]`).
- The `tests/datasets/statsbomb/raw/.cache/` directory is gitignored —
  raw event JSONs (~192 MB total) are downloaded on demand by the build
  script and never committed.

## [1.8.0] — 2026-04-29

### Added
- **Public `silly_kicks.spadl.boundary_metrics(*, heuristic, native)` utility**
  for computing precision / recall / F1 between two possession-id sequences.
  Returns a `BoundaryMetrics` TypedDict (also re-exported from
  `silly_kicks.spadl`). Keyword-only arguments — the metric is asymmetric
  (precision and recall swap when inputs swap), so positional usage is a
  silent footgun the API surface eliminates. Returns `0.0` for any metric
  whose denominator is zero (empty / single-row / constant sequences).
  Length-mismatched inputs raise `ValueError`.
- 3 vendored StatsBomb open-data fixtures under
  `tests/datasets/statsbomb/raw/events/` (matches 7298, 7584, 3754058 —
  Women's World Cup, Champions League, Premier League; ~9 MB total).
  License attribution in `tests/datasets/statsbomb/README.md`. Used by
  the new parametrized regression gate.

### Changed
- **`add_possessions` docstring is now honest about empirical performance.**
  The previous "boundary-F1 ~0.90" claim was 30+ percentage points above
  the actual measurement on StatsBomb open-data. New text reports
  recall ~0.93, precision ~0.42, F1 ~0.58 (peak ~0.605 at
  `max_gap_seconds=10.0`) and explains why precision is the way it is
  (intrinsic to the team-change-with-carve-outs algorithm class, not a
  defect — StatsBomb's proprietary annotation merges brief opposing-
  team actions back into the containing possession; the heuristic
  cannot replicate that structurally).
- **e2e validation gate replaces F1 ≥ 0.80 with recall ≥ 0.85 AND
  precision ≥ 0.30 per match.** Recall enforces the helper's primary
  contract (catching every real boundary). Precision floor catches the
  "boundary cardinality halved or doubled" regression class that affects
  per-possession aggregation downstream. F1 stays in the assert message
  for diagnostics only — gating on F1 would re-introduce the
  misrepresentation problem this PR is fixing.
- **Test class renamed** `TestBoundaryF1AgainstStatsBombNative` →
  `TestBoundaryAgainstStatsBombNative`. Parametrized over the 3 vendored
  fixtures with per-match independent gates.

### Fixed
- **e2e regression coverage now actually runs in CI.** The previous
  `TestBoundaryF1AgainstStatsBombNative::test_boundary_f1_against_native_possession_id`
  was `@pytest.mark.e2e` and silently skipped on every CI run since
  1.2.0 because the fixture wasn't committed. It was also skipping
  locally (the fixture was never on the user's only development
  machine). Net: ~6 release cycles of zero coverage on this test. PR-S8
  vendors the fixtures and drops the marker so the test runs on every
  PR + push.

### Notes
- Empirical baselines verified locally on the committed fixtures:
  recall {0.9425, 0.9268, 0.9259}, precision {0.4484, 0.4306, 0.3855},
  F1 {0.6077, 0.5880, 0.5443} for matches 7298 / 7584 / 3754058
  respectively. All comfortably above the gate thresholds; tightest
  margin is precision on 3754058 (8.55pp above floor).
- The 5 `test_predict*` cases in `tests/vaep/`, `tests/test_xthreat.py`,
  and `tests/atomic/` continue to skip in CI (and locally) because they
  depend on the un-committed `tests/datasets/statsbomb/spadl-WorldCup-2018.h5`
  fixture. Closing that gap is queued as PR-S9 (generate the HDF5 from
  open-data raw events; commit + drop e2e markers). Tracked in
  `TODO.md`.
- Algorithmic precision improvement for `add_possessions` is queued as
  PR-S10 (look-ahead merge rules for brief opposing-team actions;
  re-measure `max_gap_seconds` defaults using the new
  `boundary_metrics` utility).

## [1.7.0] — 2026-04-29

### Added
- **Dedicated DataFrame SPADL converters for Sportec and Metrica.** New
  modules `silly_kicks.spadl.sportec` and `silly_kicks.spadl.metrica`
  expose `convert_to_actions(events_df, home_team_id, *,
  preserve_native=None) -> tuple[pd.DataFrame, ConversionReport]`,
  matching the established `statsbomb` / `wyscout` / `opta` shape.
  Designed for consumers who already have normalized event data in
  pandas form (lakehouse bronze layers, ETL pipelines, research
  notebooks) and don't want to reconstruct a kloppy `EventDataset` from
  flat rows. Existing kloppy-path consumers continue to use
  `silly_kicks.spadl.kloppy` — both paths produce equivalent SPADL output
  (empirically verified by cross-path consistency tests under
  `tests/spadl/test_sportec.py::TestSportecCrossPathConsistency` and
  `tests/spadl/test_metrica.py::TestMetricaCrossPathConsistency`).
- ~120 recognized DFL qualifier columns surfaced via Sportec converter,
  covering pass / shot / tackle / foul / set-piece / play / cross /
  cards / substitution / penalty / VAR / chance / specialised /
  tracking-derived qualifier groups.
- Metrica set-piece-then-shot composition rule: `SET PIECE` (FREE KICK)
  immediately followed (≤ 5s, same player, same period) by `SHOT`
  upgrades the shot to SPADL `shot_freekick` and drops the SET PIECE
  row.

### Changed
- **`silly_kicks.spadl.kloppy.convert_to_actions` now applies
  `_fix_direction_of_play` automatically** (extracting home team from
  `dataset.metadata.teams[0].team_id`). Pre-1.7.0 the kloppy converter
  was the lone outlier among silly-kicks SPADL converters — it stayed
  in kloppy's `Orientation.HOME_AWAY` (home plays LTR, away plays RTL)
  while StatsBomb / Wyscout / Opta all flipped away-team coords for
  canonical "all-actions-LTR" SPADL convention. 1.7.0 unifies the
  convention across all 6 converters
  (`statsbomb` / `wyscout` / `opta` / `kloppy` / new `sportec` / new
  `metrica`) so all converters emit semantically equivalent SPADL output
  for the same source event stream. Hyrum's Law disclaimer: zero current
  consumers built against 1.6.0's HOME_AWAY-oriented kloppy output (per
  user confirmation during brainstorming).

### Notes
- Cross-path consistency proof: dedicated DataFrame converters and the
  kloppy gateway path produce equivalent SPADL DataFrames when given
  the same source data bridged through test helpers.
- New shared pytest conftest at `tests/spadl/conftest.py` provides
  module-scoped `sportec_dataset` and `metrica_dataset` fixtures
  reusable across `test_kloppy.py`, `test_sportec.py`, and
  `test_metrica.py`.

## [1.6.0] — 2026-04-28

### Added
- **Kloppy converter: Sportec and Metrica support.** `Provider.SPORTEC`
  (Sportec Solutions / IDSSE Bundesliga event format) and `Provider.METRICA`
  (Metrica Sports) are now first-class whitelisted providers in
  `silly_kicks.spadl.kloppy.convert_to_actions`. Empirical verification on
  real fixture data confirms zero new event-type mappings are required —
  both providers' kloppy serializers emit only event types already covered
  by the existing `_MAPPED_EVENT_TYPES` ∪ `_EXCLUDED_EVENT_TYPES` sets.
  `preserve_native` works transparently for both (their `raw_event` is a
  `dict`).
- Real-fixture end-to-end test suites for Sportec and Metrica under
  `tests/spadl/test_kloppy.py`, plus a parametrized coordinate-clamping
  test and a per-provider `ConversionReport` shape test. Test fixtures
  vendored from kloppy's BSD-3-Clause-licensed test files into
  `tests/datasets/kloppy/`.

### Fixed
- **`_SoccerActionCoordinateSystem` was unusable on real datasets.** The
  class definition omitted `__init__`, but `convert_to_actions()`
  instantiated it with `pitch_length=` / `pitch_width=` kwargs. On any
  kloppy version with the current `CoordinateSystem` ABC signature
  (kloppy 3.15+), this raised `TypeError` the moment a real
  `EventDataset` reached `dataset.transform()`. Latent since 1.0.0
  because pre-existing `tests/spadl/test_kloppy.py` was pure mocks
  that never reached the transform call. Affected **all** kloppy-based
  conversion including the previously-whitelisted StatsBomb path.
- 2 pyright errors in `silly_kicks/xthreat.py:402` surfaced by newer
  pandas-stubs / numpy-stubs versions: explicit `dtype=np.float64` added
  to two `np.linspace` calls so the inferred `NDArray[float64]` matches
  the `interp(...)` callable signature.

### Changed
- **Kloppy converter now clamps output coordinates to
  `[0, field_length] × [0, field_width]` (105 × 68 m).** This aligns the
  kloppy converter with the established silly-kicks convention — StatsBomb
  / Wyscout / Opta converters all clamp; kloppy was the lone outlier.
  Empirically Metrica events emit slight off-pitch coords (observed
  `x ∈ [-1.62, 104.63]` on the sample game) within source-recording-noise
  tolerance. Downstream consumers depending on raw off-pitch coordinates
  from the kloppy path specifically should re-verify (no such consumer
  documented).

## [1.5.0] — 2026-04-27

### Added
- **Atomic-SPADL parity for the 1.1.0 → 1.4.0 helper family.** The five
  helpers shipped on standard SPADL (`preserve_native` primitive,
  `add_possessions`, `add_gk_role`, `add_gk_distribution_metrics`,
  `add_pre_shot_gk_context`) plus a new defensive `validate_atomic_spadl`
  helper now have first-class atomic counterparts under
  `silly_kicks.atomic.spadl`:
  - `convert_to_atomic(actions, *, preserve_native=...)` — surfaces
    caller-attached columns from the input SPADL dataframe alongside the
    canonical 13 atomic columns. Synthetic atomic rows generated by the
    conversion (`receival` / `interception` / `out` / `offside` / `goal`
    / `owngoal` / `yellow_card` / `red_card`) receive `NaN` in the
    preserved columns — same behaviour as the standard converters'
    `preserve_native` for synthetic dribble rows.
  - `add_possessions(actions)` — atomic counterpart with two atomic-
    specific adaptations: (a) set-piece restart names match the post-
    collapse atomic types (`corner` / `freekick` / `throw_in` /
    `goalkick`); (b) `yellow_card` / `red_card` synthetic rows are
    transparent to boundary detection — they never trigger a possession
    boundary on their own and inherit the surrounding state via
    forward-fill within `game_id`.
  - `add_gk_role(actions)` — atomic counterpart; reads `x` (NOT
    `start_x`) for the penalty-area threshold check. Same five
    categories.
  - `add_gk_distribution_metrics(actions, xt_grid=None)` — atomic
    counterpart with three atomic-specific adaptations: (a) length is
    `sqrt(dx² + dy²)` from atomic's `(dx, dy)` columns; (b) xT delta is
    from `(x, y)` to `(x + dx, y + dy)`; (c) pass success is detected
    from the FOLLOWING atomic action by row index (`receival` =
    success; `interception` / `out` / `offside` = failure; no following
    action = conservative failure with `gk_xt_delta = NaN`). Atomic
    launch types collapse `{pass, goalkick, freekick_short,
    freekick_crossed}` into `{pass, goalkick, freekick}` (where
    `freekick` is the post-collapse name).
  - `add_pre_shot_gk_context(actions)` — atomic counterpart; recognises
    only `shot` and `shot_penalty` as shot rows. (Standard SPADL's
    `shot_freekick` is collapsed into atomic's `freekick`, mixing
    pass-class and shot-class freekicks; the helper does not attempt to
    disambiguate.)
  - `validate_atomic_spadl(df)` — defensive schema validator. Returns
    input unchanged for chaining; warns on dtype mismatches; raises on
    missing columns.

  All five helpers are vectorised on numpy/pandas; sub-50ms per 1500-
  action match (CI hard bound 200ms; benchmark assertions in
  `tests/test_benchmark.py`). 174 new atomic tests including a
  cross-validation suite asserting algorithmic equivalence between the
  standard and atomic helpers when applied to a SPADL stream and its
  atomic projection.

### Fixed
- Test infra: `tables>=3.9.0` (pytables) added to the `[test]` extras —
  required by `pd.HDFStore` for the `sb_worldcup_data` fixture in
  `tests/conftest.py`. Without it, the 5 `test_predict*` cases (vaep /
  xthreat / atomic vaep) errored at collection time with
  `ImportError("Missing optional dependency 'pytables'")`.
- Test infra: the `sb_worldcup_data` fixture now `pytest.skip(...)`s
  when the `spadl-WorldCup-2018.h5` dataset is not present locally,
  rather than erroring with `FileNotFoundError`. Aligns with the
  `@pytest.mark.e2e` semantics ("requires downloaded datasets") for the
  5 affected tests.

### Notes
- Atomic-SPADL parity TODO is now closed.

## [1.4.0] — 2026-04-27

### Added
- **GK analytics suite v1** — three composable post-conversion enrichments
  for SPADL action streams, mirroring the public-helper shape of
  `add_names()` and `add_possessions()`:
  - `add_gk_role(actions)` — tags each action with the goalkeeper's role
    context: `shot_stopping` / `cross_collection` / `sweeping` / `pick_up` /
    `distribution` (or `None` for non-GK actions). Sweeping is a
    position-based override for `keeper_*` actions taken outside the
    penalty area; in clean event data only `keeper_save` realistically
    appears outside the box (sweeper-style rush-out save). The other
    three keeper types outside the box are illegal handball offences and
    effectively non-existent in regulation play.
  - `add_gk_distribution_metrics(actions, xt_grid=None)` — adds
    `gk_pass_length_m`, `gk_pass_length_class` (short/medium/long),
    `is_launch`, and `gk_xt_delta` to GK distribution actions. Auto-calls
    `add_gk_role` when `gk_role` column is absent. xT delta only computed
    for successful distributions when an xT grid is provided. `is_launch`
    requires both length > `long_threshold` and a deliberate-distribution
    pass type (`pass`, `goalkick`, `freekick_short`, `freekick_crossed`).
  - `add_pre_shot_gk_context(actions)` — for every shot, looks back up to
    `lookback_actions` rows or `lookback_seconds` seconds (smaller wins)
    in the same `(game_id, period_id)` and tags the defending GK's recent
    activity: `gk_was_distributing`, `gk_was_engaged`,
    `gk_actions_in_possession`, `defending_gk_player_id`. Genuinely novel
    — no published OSS / academic equivalent surfaces a goalkeeper's
    pre-shot activity context as explicit per-shot features.

  All three are vectorised on numpy/pandas; sub-50ms per 1500-action match.
  References cited in docstrings: Yam (MIT Sloan), Lamberts GVM (2025),
  Butcher et al. xGOT (2025).

### Notes
- Atomic-SPADL parity for the GK analytics suite is deferred (TODO under
  `## Architecture`). Same disposition as `add_possessions`.

## [1.3.0] — 2026-04-27

### Added
- `pandas-stubs>=2.2.0` pinned in the `[dev]` extras and the CI lint job.
  Without `pandas-stubs`, pyright's bundled pandas typings under-report
  Series / DataFrame types (e.g. arithmetic on ``.values`` collapses to
  the union ``np_1darray | ExtensionArray | Categorical``), masking real
  type issues in CI while spuriously failing locally on certain method
  chains. With `pandas-stubs` in the dev path, pyright reports a
  consistent set of issues across all environments.

### Fixed
- 15 type errors that surfaced once `pandas-stubs` was installed:
  - `vaep/features.py` and `atomic/vaep/features.py` — replaced
    `Series.values` with `Series.to_numpy()` in polar-coordinate
    arithmetic so the return type is `np.ndarray` instead of the
    ``np_1darray | ExtensionArray | Categorical`` union (which doesn't
    support `**` / `/` / `-`).
  - `spadl/opta.py` — same `.values` → `.to_numpy()` swap in
    ``_fix_owngoals`` arithmetic.
  - `spadl/statsbomb.py` — synthetic interception-event `extra` payload
    now built as an explicit ``pd.Series([..], dtype=object)`` instead
    of `[dict] * n`, matching pandas-stubs's accepted setitem value types.
  - `spadl/utils.py` `_finalize_output()` — schema dtype string passed
    through `np.dtype(...)` so it narrows to ``DtypeObj`` for the
    `astype` overload set.
- Removed two `cast(pd.DataFrame, ...)` workarounds in
  `add_possessions` (introduced in 1.2.0). With `pandas-stubs`,
  non-inplace ``sort_values()`` / ``drop()`` correctly return
  `DataFrame`, making the casts redundant.

## [1.2.0] — 2026-04-27

### Added
- `silly_kicks.spadl.utils.add_possessions(actions, *, max_gap_seconds=5.0,
  retain_on_set_pieces=True)` — provider-agnostic possession-sequence
  reconstruction for any SPADL action stream. Adds a `possession_id: int64`
  column via a team-change-with-carve-outs heuristic: boundaries on team
  change, period change (within a game), or time gap >= `max_gap_seconds`,
  with a foul→opposing-team-set-piece carve-out that retains the previous
  possession (the team that won the foul resumes its sequence). Counter
  resets to 0 at each new `game_id`. Mirrors the public-enrichment shape
  of `add_names()` (post-conversion, returns a copy with the new column).
  Vectorised on numpy/pandas; ~1ms per 1500-action match, sub-3ms on 10k.
- Performance benchmarks for `add_possessions` (1500-action and 10k-action
  scenarios) added to `tests/test_benchmark.py` with hard CI bounds
  (200ms / 2s respectively) catching accidental quadratic regressions.
- e2e-marked boundary-F1 validation test against StatsBomb's native
  `possession` field (using `preserve_native=['possession']` from 1.1.0
  to surface the native truth alongside the heuristic). Skips when the
  raw StatsBomb fixture is absent; documents the validation procedure
  for downstream consumers wanting to re-measure the agreement rate
  against their own data.

### Notes
- Atomic-SPADL parity for `add_possessions` is deferred (TODO under
  `## Architecture`). Apply the same passthrough mechanism when there's
  a concrete consumer asking for it.

## [1.1.0] — 2026-04-27

### Added
- `preserve_native` parameter on `convert_to_actions` for all four SPADL
  converters (`statsbomb`, `wyscout`, `opta`, `kloppy`). Surfaces provider-
  native event fields alongside the canonical SPADL output as extra columns
  on the returned DataFrame — useful for surfacing fields that the canonical
  SPADL schema doesn't carry (e.g. StatsBomb's native `possession` sequence
  number, `possession_team`, `play_pattern`; Wyscout bronze passthroughs;
  Opta competition metadata). Each `preserve_native` field must be present
  on the input and must not overlap with the SPADL schema; both conditions
  raise `ValueError` early. Synthetic actions inserted by `_add_dribbles`
  get NaN in preserved columns (no source event to inherit from).
- `extra_columns` parameter on internal `silly_kicks.spadl.utils._finalize_output()`
  that powers the public `preserve_native` feature.
- `_validate_preserve_native()` helper in `silly_kicks.spadl.utils` for
  shared upfront validation across providers (input-column presence +
  schema-overlap check).
- Kloppy `preserve_native` requires kloppy >= 3.15 with raw-event
  preservation. Each preserved field is read from `event.raw_event[field]`.

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
