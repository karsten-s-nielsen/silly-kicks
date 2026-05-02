# Changelog

All notable changes to silly-kicks will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.1] — 2026-05-02

### Breaking-correctness fix (PR-S23) — Sportec + Metrica per-period direction-of-play

`silly_kicks.spadl.sportec.convert_to_actions` and
`silly_kicks.spadl.metrica.convert_to_actions` now correctly handle
per-period-absolute bronze events (teams switching ends after halftime).
silly-kicks 3.0.0 declared these converters as `ABSOLUTE_FRAME_HOME_RIGHT`,
producing wrong-end SPADL output for half of every match. ADR-006 erratum
documents the corrected per-converter declaration table.

Callers must now pass per-period direction info via one of two paths
(otherwise `ValueError` with migration guidance):

```python
# Path A -- bool pair (preferred; matches PFF events + tracking-Sportec API)
actions, report = sportec.convert_to_actions(
    events,
    home_team_id="DFL-CLU-XXXXX",
    home_team_start_left=True,                     # from DFL MatchInformation.xml
    home_team_start_left_extratime=False,          # only when ET periods present
)

# Path B -- explicit mapping (escape hatch for arbitrary periods)
actions, report = metrica.convert_to_actions(
    events,
    home_team_id="Home",
    home_attacks_right_per_period={1: True, 2: False},
)
```

Trained VAEP / HybridVAEP / xT models on Sportec or Metrica data from
silly-kicks 3.0.0 must be re-trained on 3.0.1 output.

### Test infrastructure

- New per-period orientation fixtures committed at
  `tests/datasets/idsse/per_period_match.parquet` (Bassek et al. CC-BY 4.0)
  and `tests/datasets/metrica/per_period_match.parquet` (CC-BY-NC-4.0;
  same precedent as existing Metrica Sample Game 2 fixture). Both are
  excluded from the published wheel.
- New `test_per_team_per_period_shots_attack_high_x` parametrized over
  both new fixtures in `tests/invariants/test_direction_of_play.py`.
  Closes the invariant-density gap that let PR-S22's bug ship.
- 5 new `TestSportecPerPeriodKwargContract` + 5 new
  `TestMetricaPerPeriodKwargContract` negative-path tests for kwarg
  resolution policy.

### Detector hardening (TF-22)

`silly_kicks.spadl.orientation.detect_input_convention` no longer
false-positives `ABSOLUTE_FRAME_HOME_RIGHT` on sparse-shot
per-period-absolute matches. New guard: when no team has reliable shots
in ≥ 2 distinct periods, returns `convention=None, confidence="low"`.
Validator re-enabled at sportec / metrica / pff converter call sites
declaring `PER_PERIOD_ABSOLUTE`.

### Atomic-SPADL pathway

Smoke test added at `tests/atomic/test_atomic_orientation.py` verifying
the SPADL → atomic-SPADL composition preserves canonical-LTR. No
converter changes (atomic has no native sportec/metrica converter).

### Other

- `silly_kicks/__init__.py` `__version__` bumped from "1.0.2" (stale
  since at least 2.0.0) to "3.0.1" so it now matches `pyproject.toml`.
- `scripts/extract_provider_fixtures.py` gains `--variant {default, per_period}`
  flag for regenerating either fixture variant. Per-period extraction
  pulls from `bronze.idsse_events` / `bronze.metrica_events` on Databricks
  (env-var auth).
- `NOTICE` "Test Data Sources" section attributes the new IDSSE +
  Metrica Sample Game 1 fixtures.

## [3.0.0] — 2026-05-02

### Breaking — Correctness (PR-S22)

**Direction-of-play handling refactor.** The dual-mirror inversion that has
been present since v0.1.0 is fixed. SPADL canonical convention is "all teams
attack left-to-right" -- every team's actions at high-x in their own frame.
Every silly-kicks SPADL converter now produces this convention directly via
the new :func:`silly_kicks.spadl.to_spadl_ltr` dispatcher. Decision: ADR-006.

**Code-side regression window.** The bug was present in the native StatsBomb,
Wyscout, and Opta converters AND in `vaep.base.VAEP.compute_features` since
the v0.1.0 fork (verified `git show 0b29178`). The kloppy gateway acquired
the same code path in 1.7.0 but routed correctly because kloppy's transform
already normalised to absolute-frame-home-right.

**Consumer-artifact impact depends on which converter path each artifact's
data went through.** Categorically affected:

- Cached SPADL action tables derived from native ``silly_kicks.spadl.statsbomb``
  / ``wyscout`` / ``opta`` -- away-team ``(x, y)`` were mirrored to the wrong
  end of the pitch.
- Trained VAEP / HybridVAEP models built on Sportec / Metrica / kloppy-gateway
  / PFF SPADL -- VAEP feature engineering (now correctly free of the second
  mirror) inverted away-team rows in gamestates.
- Trained xG / xT models that consume polar / spatial features.
- Pre-computed xT grids derived from broken SPADL inputs (U-shaped instead of
  goal-monotonic).
- Tracking-aware features: ``add_action_context`` (PR-S20),
  ``add_pre_shot_gk_context`` (PR-S21).
- Any downstream model trained on action-coord features.
- Any test baseline / golden value calibrated on the prior pipeline.
- Any dataset published from silly-kicks output that mirrors SPADL or VAEP.

Per-consumer migration is the consumer's responsibility; this CHANGELOG enumerates
the categorical impact rather than specific consumer artifacts.

### Added

- **`silly_kicks.spadl.orientation`** (NEW module) — canonical direction-of-play
  primitives:
  - ``InputConvention`` enum: ``POSSESSION_PERSPECTIVE`` (StatsBomb, Wyscout),
    ``ABSOLUTE_FRAME_HOME_RIGHT`` (Sportec, Metrica, Opta, kloppy gateway),
    ``PER_PERIOD_ABSOLUTE`` (PFF).
  - ``to_spadl_ltr(actions, *, input_convention, home_team_id, ...)`` —
    single canonical normalizer; each converter calls it exactly once.
  - ``detect_input_convention(events, *, match_col, x_max, ...)`` — heuristic
    detector; tiered confidence (≥10 shots/group = high, 5-9 = medium, <5 =
    ambiguous defer).
  - ``validate_input_convention(events, declared, *, on_mismatch)`` — wired
    into every converter; warn by default, raise under
    ``SILLY_KICKS_ASSERT_INVARIANTS=1``. Surfaces upstream loader regressions.
- **`silly_kicks.vaep.base.VAEP.compute_features(..., frames_convention="absolute_frame")`**
  — explicit kwarg controlling tracking-frame normalisation.
- **`silly_kicks.tracking.{sportec,pff,kloppy}.convert_to_frames(..., output_convention=…)`**
  — opt-in ``"ltr"`` mode for callers wanting SPADL LTR tracking output
  directly. Default behaviour preserved (absolute_frame); ``None`` (legacy
  unspecified) emits ``DeprecationWarning`` recommending callers be explicit.
- **`tests/invariants/`** (NEW directory) — physical-invariant test layer
  parametrised across providers with real fixtures:
  - ``test_direction_of_play.py`` — per-team shots cluster at high-x,
    parametrised × ``xy_fidelity_version ∈ {1, 2}`` for StatsBomb.
  - ``test_vaep_geometric_sanity.py`` — VAEP shot dist < 50m AND xT
    goal-monotonic.
  - ``test_gk_position.py`` — GK actions cluster at defended (low-x) goal.
  - ``test_input_convention_detector.py`` — detector + validator semantics
    against real fixtures.

### Changed

- **`silly_kicks.spadl.statsbomb`, `wyscout`, `opta`, `sportec`, `metrica`,
  `kloppy`, `pff`** — every ``convert_to_actions`` now routes the
  direction-of-play step through ``to_spadl_ltr(input_convention=…)`` and
  emits canonical SPADL LTR. The ``input_convention`` declared by each
  converter is the load-bearing contract; ``validate_input_convention``
  surfaces violations.
- **`silly_kicks.spadl.opta.convert_to_actions`** — docstring contract
  added: the converter expects loader-pre-normalised absolute-frame data
  with NO per-period switching. Raw Opta f24 ships per-period switching;
  callers must pre-normalise upstream.
- **`silly_kicks.vaep.base.VAEP.compute_features`** — removed the inline
  ``play_left_to_right`` call (the dual-mirror that this CHANGELOG fixes).
  Converter output is already canonical SPADL LTR.
- **`silly_kicks.spadl.utils._finalize_output`** — debug-mode invariant
  assertion gated on ``SILLY_KICKS_ASSERT_INVARIANTS=1``: per-team shot mean
  start_x must be > field_length/2.
- **`silly_kicks.spadl.play_left_to_right`** + atomic-SPADL,
  ``silly_kicks.vaep.features.play_left_to_right`` + atomic-VAEP equivalents
  — docstrings updated. Functions are retained as public boundary helpers
  (absolute-frame → SPADL LTR) but no longer called by silly-kicks itself.

### Removed

- **`silly_kicks.spadl.base._fix_direction_of_play`** (private symbol) —
  replaced by ``silly_kicks.spadl.to_spadl_ltr``. Was only ever called by
  the converters themselves; no public API impact.

### Migration

Re-derive any cached artifact whose path went through an affected converter.
Specifically: re-derive SPADL action tables from raw events; re-train VAEP /
HybridVAEP models; re-compute xT grids; re-baseline empirical golden values;
re-publish any silly-kicks-derived datasets. The new validator surfaces input
convention mismatches as warnings; set ``SILLY_KICKS_ASSERT_INVARIANTS=1`` in
CI to promote them to failures.

## [2.9.0] — 2026-05-01

### Added — Pre-shot GK position + baselines backfill (PR-S21)

- **`silly_kicks.tracking.features`** — 4 GK-position helpers: `pre_shot_gk_x`,
  `pre_shot_gk_y`, `pre_shot_gk_distance_to_goal`, `pre_shot_gk_distance_to_shot`.
  Plus aggregator `add_pre_shot_gk_position(actions, frames) -> pd.DataFrame`
  that emits the 4 GK columns + 4 linkage-provenance columns. Decorated with
  `@nan_safe_enrichment` per ADR-003. Plus `pre_shot_gk_default_xfns` (4
  `lift_to_states` wrappers) for HybridVAEP integration.
- **`silly_kicks.atomic.tracking.features`** — atomic-SPADL parity with the
  same public surface (`atomic_pre_shot_gk_default_xfns`). Mirrors the standard
  surface with atomic-shaped column reads (`x, y`) and atomic shot type ids
  (`{shot, shot_penalty}` — atomic does not recognize `shot_freekick`).
- **`silly_kicks.spadl.utils.add_pre_shot_gk_context(*, frames=None)`** — additive
  optional `frames` kwarg. When supplied, emits 4 GK-position columns + 4
  provenance columns by lazy-importing the canonical compute (preserves
  ADR-005 §5 no-cycle invariant). When `frames=None` (default), behavior is
  bit-identical to silly-kicks 2.8.0 — no frames-related columns appear.
  Backward-compat pinned by golden-fixture test.
- **`silly_kicks.atomic.spadl.utils.add_pre_shot_gk_context`** — atomic mirror
  of the same `frames=None` extension.
- **`silly_kicks.tracking._kernels._pre_shot_gk_position`** (private) —
  schema-agnostic compute kernel shared between standard and atomic surfaces.
- **`silly_kicks.tracking.feature_framework.ActionFrameContext`** gains
  `defending_gk_rows: pd.DataFrame` field (default-factory empty DataFrame —
  preserves direct construction backward-compat).
- **`scripts/regenerate_action_context_baselines.py`** — one-shot regenerator
  for `*_expected.parquet` files + `empirical_action_context_baselines.json`.
- **`tests/datasets/tracking/action_context_slim/{provider}_expected.parquet`**
  — per-provider expected output committed for the bit-exact per-row
  regression gate (4 providers).
- **`tests/tracking/_provider_inputs.py`** — shared loader/synthesizer for the
  regenerator and CI gate; keeps both in sync.
- **`tests/tracking/test_action_context_expected_output.py`** — bit-exact
  per-row regression gate (4 providers).
- **`tests/tracking/test_empirical_action_context_baselines.py`** — JSON shape
  gate + JSON-vs-parquet consistency gate.

### Changed

- **`silly_kicks.spadl.utils.add_pre_shot_gk_context`** + atomic mirror —
  bug-fix: `defending_gk_player_id` output column now preserves the input
  `player_id` dtype. Numeric `player_id` (canonical SPADL_COLUMNS:
  PFF / StatsBomb / Opta / Wyscout / Metrica) → `float64` NaN-coded (unchanged).
  Object/string `player_id` (`KLOPPY_SPADL_COLUMNS` / `SPORTEC_SPADL_COLUMNS`
  schema) → `object` dtype with `None` for unidentified rows. Previous
  unconditional `int(gk_id_raw)` cast crashed on string Sportec player_ids;
  surfaced by PR-S21's TF-11 regression-gate exercising real-shot rows on
  Sportec data.
- **`tests/datasets/tracking/empirical_action_context_baselines.json`** —
  all 256 percentile slots backfilled (4 percentiles × 8 features × 4 providers).
  Per-row gate exercises real GK-position computation on at least one shot
  per provider (synthesizer in `tests/tracking/_provider_inputs.py` stamps a
  synthetic keeper_save → shot pair anchored on real frame goalkeeper data
  so the events-side helper populates `defending_gk_player_id` and the
  tracking aggregator emits non-NaN GK position).
- **`NOTICE`** — Anzer & Bauer (2021) entry description expanded to enumerate
  defending-GK-position alongside player_speed and distance-to-defender.
- **`TODO.md`** — TF-1 + TF-11 marked SHIPPED. PR-S21 active-cycle entry.
  Bundled National Park additions: TF-12 (`pre_shot_gk_angle_*`), TF-13
  (frame-based GK identification fallback), TF-14 (defensive-line features).

### Removed

- **4 vestigial `test_placeholder` stubs** (National Park cleanup): the
  `TestKloppyE2E.test_placeholder` (`test_kloppy.py`),
  `TestSpadlConvertorE2E.test_placeholder` (`test_opta.py`, `test_wyscout.py`),
  and `TestSpadlConvertor.test_placeholder` (`test_statsbomb.py`) classes
  were inert `pytest.skip()` calls inherited from the v0.1.0 socceraction
  fork (the original DataLoader classes — `OptaLoader` / `StatsBombLoader` /
  `PublicWyscoutLoader` / `KloppyLoader` — were removed at fork time but the
  e2e test scaffolds were left behind as no-op skip stubs). Plus the
  unreferenced `pytestmark_e2e` module attribute in `test_opta.py`. Net
  effect: `pytest -m e2e` now runs 12 PASSED / 0 SKIPPED instead of
  12 PASSED / 4 SKIPPED — the SKIPPED column is no longer a hiding place
  for genuine missing-fixture failures.

### Notes

- No breaking changes. PR-S21 ships entirely within ADR-005's locked
  architecture; no new ADR.
- Per-Series GK helpers (`pre_shot_gk_x` etc.) silently emit all-NaN when
  `defending_gk_player_id` is absent from `actions` — required by VAEP's
  `feature_column_names` introspection path. The aggregator
  `add_pre_shot_gk_position` raises `ValueError` (user-direct boundary).
  Documented in helper docstrings + `pre_shot_gk_default_xfns`.

## [2.8.0] — 2026-05-01

### Added — Tracking-aware action_context features (PR-S20)

- **`silly_kicks.tracking.features`** --- public per-feature surface for
  standard SPADL: `nearest_defender_distance`, `actor_speed`,
  `receiver_zone_density`, `defenders_in_triangle_to_goal`. Plus aggregator
  `add_action_context(actions, frames, *, receiver_zone_radius=5.0) -> pd.DataFrame`
  that enriches input actions with the 4 features + 4 linkage-provenance
  columns (`frame_id`, `time_offset_seconds`, `link_quality_score`,
  `n_candidate_frames`). Decorated with `@nan_safe_enrichment` per ADR-003.
  Plus `tracking_default_xfns` (4 `lift_to_states` wrappers) for
  HybridVAEP integration.
- **`silly_kicks.atomic.tracking.features`** --- atomic-SPADL parity with
  the same public surface (`atomic_tracking_default_xfns`). Mirrors the
  standard surface with atomic-shaped column reads (`x, y, dx, dy`).
- **`silly_kicks.tracking.feature_framework`** --- `ActionFrameContext`
  frozen dataclass + `lift_to_states` (lifts an `(actions, frames) -> pd.Series`
  helper to a `(states, frames) -> Features` transformer). Re-exports
  `frame_aware`, `is_frame_aware`, `Frames`, `FrameAwareTransformer`.
- **`silly_kicks.tracking._kernels`** (private) --- schema-agnostic compute
  kernels shared between standard and atomic public surfaces. Per
  ADR-005 §3 (kernel-extraction pattern).
- **`silly_kicks.tracking.utils._resolve_action_frame_context`** (private)
  --- builds the linked-context structure (linkage pointers + per-action
  actor row + opposite-team frame rows) once per `add_action_context()` call.
- **`silly_kicks.vaep.feature_framework`** --- extended with `frame_aware`
  decorator, `is_frame_aware` predicate, and `Frames` / `FrameAwareTransformer`
  type aliases. Marker-decorator pattern parallels the existing
  `@nan_safe_enrichment` contract (ADR-003).
- **`silly_kicks.vaep.base.VAEP.compute_features` / `rate`** --- additive
  `frames=None` keyword-only parameter. Frame-aware xfn dispatch via
  `is_frame_aware`. `HybridVAEP` and `AtomicVAEP` inherit the extension
  automatically (no code changes in their files). Symmetric LTR-normalization
  via lazy import of `tracking.utils.play_left_to_right` only when
  `frames is not None` (no module-import-time vaep <-> tracking cycle).
- **`silly_kicks._nan_safety`** --- new `is_nan_safe_enrichment(fn)` peer
  predicate to the existing `nan_safe_enrichment` decorator. Mirrors the
  new `is_frame_aware` introspection API.
- **ADR-005** ([docs/superpowers/adrs/ADR-005-tracking-aware-features.md](docs/superpowers/adrs/ADR-005-tracking-aware-features.md))
  --- tracking-aware feature integration contract. Captures the seven
  cross-cutting decisions PR-S20 introduces so PR-S21+ tracking-aware
  features inherit them without re-litigation.
- **`NOTICE`** --- canonical academic-attribution record at repo root,
  mirroring the lakehouse pattern. Cross-linked from `README.md` and
  `CLAUDE.md`. Cites Lucey et al. (2014), Anzer & Bauer (2021),
  Spearman (2018), Power et al. (2017), Pollard & Reep (1997) for the 4
  PR-S20 features, plus the foundational SPADL / VAEP / Atomic-SPADL / xT
  literature.
- **`TODO.md` restructured** to the lakehouse-style "On Deck" table.
  Eleven follow-up tracking-aware features (TF-1..TF-10) tracked with
  Size / Source / Notes columns and academic citations; TF-11 tracks the
  baselines-JSON backfill.
- **Loop 0 lakehouse probe** --- `scripts/probe_action_context_baselines.py`
  pulls slim-slice action+frame parquets per provider into
  `tests/datasets/tracking/action_context_slim/` (sportec / metrica /
  skillcorner; ~10 actions + linked frames each). Probe + outputs
  committed; real datasets are not. Backbone for the cross-provider
  parity test.
- **Tier-3 cross-provider parity test** ---
  `tests/tracking/test_action_context_cross_provider.py` runs
  `add_action_context` against the lakehouse-derived slim parquets per
  provider; asserts bounds + linkage rate >= 95% + actor_speed populated
  >= 80%.
- **e2e real-data sweep** ---
  `tests/tracking/test_action_context_real_data_sweep.py` (4
  e2e-marked tests, env-gated). Mirrors PR-S19's sweep shape: PFF via
  `PFF_TRACKING_DIR`; IDSSE / Metrica / SkillCorner via Databricks SQL.
  Skips with explicit reason on missing env.

### Backward compatibility

- All existing call sites (`v.compute_features(game, actions)`,
  `v.rate(game, actions)`) work verbatim --- `frames=None` is the
  default and walks the same code path. Regression-tested in
  `test_compute_features_frames_none_is_regression_equivalent`.
- No changes to `xfns_default`, `hybrid_xfns_default`, or atomic
  `xfns_default`. Tracking-aware features must be opted in by appending
  `tracking_default_xfns` (or `atomic_tracking_default_xfns`) to the
  caller's xfns list.

## [2.7.0] — 2026-04-30

### Added

- **`silly_kicks.tracking` namespace** --- first-class tracking-data
  support, parallel to `silly_kicks.spadl`. Hexagonal pure-function
  contract: `convert_to_frames(...) -> tuple[pd.DataFrame,
  TrackingConversionReport]`, zero I/O, zero global-state mutation.
  Nineteen-column long-form canonical schema
  (`TRACKING_FRAMES_COLUMNS`), per-provider dtype variants
  (`KLOPPY_TRACKING_FRAMES_COLUMNS`, `SPORTEC_TRACKING_FRAMES_COLUMNS`,
  `PFF_TRACKING_FRAMES_COLUMNS`), 105 x 68 m SPADL coordinates,
  long-form ball-row encoding (`is_ball=True`), `team_attacking_direction` /
  `ball_state` / `speed_source` provenance columns.
- **Four-provider adapter coverage** --- Sportec/IDSSE
  (`silly_kicks.tracking.sportec`, native), PFF
  (`silly_kicks.tracking.pff`, native), Metrica + SkillCorner
  (`silly_kicks.tracking.kloppy`, gateway via `kloppy.TrackingDataset`).
  PFF native is preferred over kloppy's PFF tracking parser for
  symmetry with `silly_kicks.spadl.pff` (PR-S18) and shared use of the
  `_direction.home_attacks_right_per_period` helper.
- **Linkage primitive**
  (`silly_kicks.tracking.utils.link_actions_to_frames` +
  `slice_around_event`) --- the load-bearing cross-pipeline operation
  that PR-S20+ tracking-aware features will build on. Returns pointer
  DataFrame plus `LinkReport` audit. Default tolerance 0.2 s, pinned
  by an explicit default-stability test.
- **Hybrid speed policy** --- adapters trust native speed where
  provided (PFF, Sportec); derive via `_derive_speed` (per-player
  groupby + diff) where missing (Metrica, SkillCorner). The
  `speed_source` column records provenance.
- **Empirical-probe-driven synthetic fixtures** ---
  `scripts/probe_tracking_baselines.py` measures real-data statistics
  (frame rates, NaN-rate-per-column, off-pitch tail rates,
  ball-visibility rates, distance-to-ball percentiles) from the
  lakehouse mart + local PFF; the committed JSON baseline at
  `tests/datasets/tracking/empirical_probe_baselines.json` parameterizes
  the per-provider synthetic generators. `realistic.parquet` fixtures
  inject baseline-calibrated edge cases (off-pitch tail, ball-out
  interval, ball-x throw-in tail) for CI; deterministic
  `tiny.parquet` / `medium_halftime.parquet` remain available for
  exact-answer unit tests.
- **`tests/test_tracking_real_data_sweep.py`** --- e2e-marked sweep
  exercising all four adapters against real data (local PFF JSONL.bz2 +
  lakehouse-derived Sportec / Metrica / SkillCorner samples). Skipped
  in CI; run locally before each tracking PR's single commit.
- **ADR-004**
  (`docs/superpowers/adrs/ADR-004-tracking-namespace-charter.md`) ---
  silly_kicks.tracking namespace charter; nine invariants locking the
  schema + adapter taxonomy + linkage contract for PR-S20+ to inherit.
- **`pyproject.toml`** --- `kloppy` optional minimum bumped to >= 3.18.0
  (kloppy 3.18 ships Metrica + SkillCorner tracking parsers used by the
  gateway). Pytest `pythonpath` config now includes `["", "tests"]` so
  per-provider synthetic-fixture generators are importable in test code
  via `datasets.tracking.<provider>.generate_synthetic`.

### Changed

- **`silly_kicks/spadl/pff.py`** --- the per-period direction lookup
  (`home_attacks_right_per_period`) is extracted into
  `silly_kicks/tracking/_direction.py` so events PFF, tracking PFF,
  and tracking Sportec adapters share one implementation. Pure
  refactor; the events test suite (127 tests) passes unchanged.

### Deferred

Tracking-aware features deferred to follow-up scoping cycles, in
priority order (per ADR-004 invariant 9): `action_context()` (PR-S20,
target 2.8.0), `pressure_on_carrier()`, `infer_ball_carrier()`,
`sync_score()`, pitch-control models (Spearman / Voronoi), smoothing
primitives (Savitzky-Golay, EMA), multi-frame interpolation /
gap filling, ReSpo.Vision adapter (licensing-gated).

## [2.6.0] — 2026-04-30

### Added

- **`silly_kicks.spadl.pff`** — first-class PFF FC / Gradient Sports
  events-data converter. Hexagonal pure-function contract (events
  DataFrame in, SPADL DataFrame + ConversionReport out, zero I/O).
  Mirrors the sportec / metrica converter shape. Dispatch table covers
  PFF's hierarchical event vocabulary (`gameEvents` × `possessionEvents`
  + `set_piece_type`): pass / cross / shot / clearance / dribble (BC) /
  tackle (CH) / keeper_save+keeper_pick_up (RE) / bad_touch (TC) +
  set-piece compositions (kickoff / open play / corner / free kick /
  throw-in / goal kick / penalty) + foul row synthesis with card
  result mapping. Excludes `OUT` / `SUB` / period-boundary / `OTB+IT`
  rows with full ConversionReport audit trail.
- **`silly_kicks.spadl.PFF_SPADL_COLUMNS`** — extended output schema:
  `SPADL_COLUMNS` + four nullable `Int64` tackle-passthrough columns
  (`tackle_winner_player_id`, `tackle_winner_team_id`,
  `tackle_loser_player_id`, `tackle_loser_team_id`) per ADR-001.
  `Int64` (pandas nullable) is a deliberate dtype departure from
  `SPORTEC_SPADL_COLUMNS`'s `object` dtype: PFF identifiers are integers
  whereas kloppy hands sportec strings.
- **Per-period direction-of-play normalization** — first silly-kicks
  converter requiring perspective-real coordinate handling. Two new
  parameters (`home_team_start_left`, `home_team_start_left_extratime`)
  carry the metadata-derived flip information per period.
- **`tests/datasets/pff/`** — synthetic match fixture
  (`synthetic_match.json`) plus deterministic generator
  (`_generate_synthetic_match.py`). Synthetic-only test policy until
  PFF licensing for redistributable real-data slices is confirmed.
- **`docs/examples/pff_wc2022_walkthrough.py`** — end-to-end pipeline
  demonstration (documentation, not test). Reads from a user-supplied
  PFF directory and walks events → SPADL → Atomic-SPADL → coverage /
  boundary metrics → VAEP labels.
- **`TODO.md` Tracking namespace entry** — captures the deferred
  `silly_kicks.tracking.*` design with verified luxury-lakehouse prior
  art (3 providers / 20 matches / ~38M player-frames in
  `soccer_analytics.dev_gold.fct_tracking_frames` as of 2026-04-30) and
  library-native architectural rules.

### Changed

- **`silly_kicks.spadl._finalize_output`** recognizes pandas extension
  dtypes (`Int64`, `Float64`, `boolean`, `string`, etc.) on schema
  entries — small surface-area generalization, fully backwards-
  compatible with existing object/int64 dtype handling. Required for
  `PFF_SPADL_COLUMNS` `Int64` tackle columns.
- **`tests/spadl/test_cross_provider_parity.py`** — PFF added as a
  parametrize entry; participates in the keeper-action emission gate,
  schema-shape gate, and ADR-001 team_id-mirror gate alongside the five
  pre-existing converters.
- **Pre-release empirical validation** — converter validated against the
  full WC 2022 dataset (64 matches, 144,541 events → 91,931 SPADL actions,
  zero conversion failures, zero unrecognized vocabulary). The sweep
  surfaced 6 vocabulary patterns the hand-authored synthetic-fixture suite
  missed (OFF / ON / G / THIRDKICKOFF / FOURTHKICKOFF game_event_types and
  OTB+empty initialNonEvent markers); all are now in the converter's
  excluded vocabulary, exercised by the synthetic fixture, and asserted by
  test_pff.py. Also surfaced a real-data schema detail: PFF stores
  ``fouls`` as a single dict per event (not a JSON array, contrary to
  initial fixture authoring); fixture + loaders updated. Standalone
  ``FOUL`` gameEventType events with ``possessionEventType="FO"`` now
  convert in-place to the canonical foul SPADL row (no phantom non_action
  parent).

## [2.5.0] — 2026-04-30

### Added

- **`silly_kicks._nan_safety.nan_safe_enrichment`** — marker decorator
  declaring an enrichment helper satisfies the NaN-safety contract
  (ADR-003). Sets `fn._nan_safe = True`; CI gates auto-discover decorated
  helpers via this attribute.
- **`goalkeeper_ids: set | None = None`** keyword-only parameter on
  `silly_kicks.spadl.utils.add_gk_role` and
  `silly_kicks.atomic.spadl.utils.add_gk_role`. When provided,
  distribution-detection extends with two additional matching rules:
  (a) `current player_id ∈ goalkeeper_ids` AND prev keeper-type same-team;
  (b) NaN-team fallback — both player_ids NaN AND same team_id AND prev
  keeper-type. Closes the lakehouse coverage gap on IDSSE/Metrica data
  with sparse player attribution. When `None` (default), behavior is
  byte-for-byte unchanged.
- **`tests/test_enrichment_nan_safety.py`** — auto-discovered NaN-fuzz
  test (15 cases). Parametrizes over every `@nan_safe_enrichment` helper
  × synthetic NaN-laced SPADL fixture; asserts no crash + sensible
  defaults. Includes registry-floor sanity assertions that catch silent
  discovery breakage.
- **`tests/test_enrichment_provider_e2e.py`** — auto-discovered
  cross-provider e2e regression (21 cases). Parametrizes over every
  `@nan_safe_enrichment` standard helper × vendored fixtures from
  StatsBomb / IDSSE / Metrica; atomic helpers run on the
  StatsBomb-derived atomic-SPADL fixture.
- **`tests/test_gk_role_goalkeeper_ids.py`** — feature tests for the new
  `goalkeeper_ids` parameter (8 cases): backward-compat, rule (a)
  known-GK match, rule (b) NaN-team fallback, edge cases (atomic, empty
  set, team-boundary respect).
- **`docs/superpowers/adrs/ADR-003-nan-safety-enrichment-helpers.md`** —
  formalizes the NaN-safety contract for public enrichment helpers,
  alternatives considered, and the registry-floor sanity assertion as
  the bulletproof for the auto-discovery mechanism.
- **CLAUDE.md "Key conventions" amendment** pointing to ADR-003.

### Fixed

- **`silly_kicks.spadl.utils.add_pre_shot_gk_context`** —
  `ValueError: cannot convert float NaN to integer` at line 543 when
  the most-recent defending-keeper-action's `player_id` is NaN
  (e.g. IDSSE bronze data with sparse player attribution). Surfaced
  2026-04-30 by the luxury-lakehouse `compute_spadl_vaep` task. Fix:
  detect NaN before the `int(...)` cast; `continue` to next shot
  (defending_gk_player_id stays NaN per the function's documented
  contract). Symmetric fix at
  `silly_kicks.atomic.spadl.utils.add_pre_shot_gk_context` line 826.
- **`silly_kicks.spadl.utils.add_gk_distribution_metrics`** — latent
  `ValueError: cannot convert float NaN to integer` at lines 374-377
  on `.astype(int)` zone-binning when a distribution-eligible row has
  NaN coordinates. Fix: filter `eligible` mask by `np.isfinite(...)`
  on all four coords. Symmetric fix at
  `silly_kicks.atomic.spadl.utils.add_gk_distribution_metrics`
  lines 665-668.
- **`silly_kicks.spadl.utils.coverage_metrics`** (defensive) — same
  `int(NaN)` crash class on `int(tid)` at line 1074 if input has NaN
  `type_id`. Fix: NaN guard before the cast; NaN type_ids tally as
  "unknown". Symmetric fix at
  `silly_kicks.atomic.spadl.utils.coverage_metrics` line 1036. Not
  under ADR-003 (TypedDict-returning, not enrichment helper) — fixed
  while we're here.

### Changed

- 10 public enrichment helpers (5 standard + 5 atomic) decorated with
  `@nan_safe_enrichment`: `add_possessions`, `add_names`, `add_gk_role`,
  `add_gk_distribution_metrics`, `add_pre_shot_gk_context` × 2 packages.

### Notes

- **Hyrum's Law surface:** `add_gk_role.__signature__` gains the new
  `goalkeeper_ids` keyword-only parameter. Consumers using
  `inspect.signature(add_gk_role)` would see the addition. Documented
  in ADR-003 as accepted exposure.
- **Test count:** 884 → 928 passing, 4 deselected (+44 net delta:
  15 fuzz + 21 e2e + 8 goalkeeper_ids feature tests). Pyright clean
  (0 errors / 0 warnings / 0 informations).
- Future direction: nullable-Int64 dtype migration for `player_id` /
  `team_id` columns is the long-term answer to type-level NaN-safety;
  out of scope for this PR (ADR-003 § Notes / Future direction).

## [2.4.0] — 2026-04-30

### Added

- **`silly_kicks.vaep.feature_framework`** — new public module holding the 7
  framework primitives both standard and atomic VAEP feature stacks build on:
  4 type aliases (`Actions`, `Features`, `FeatureTransfomer`, `GameStates`),
  `gamestates`, `simple`, and the promoted helper
  `actiontype_categorical(actions, spadl_cfg)`. Cross-package framework
  boundary now has a name; atomic-VAEP no longer reaches into
  `vaep.features.core` for framework primitives.
- **`actiontype_categorical(actions, spadl_cfg)`** — promoted from the
  previously-private `_actiontype` helper in `vaep.features.core` to a public,
  SPADL-config-parameterized framework helper. Both standard-VAEP and
  atomic-VAEP wrap it with `@simple` to produce their respective `actiontype`
  feature transformers. Drops the implicit-None config fallback (the function
  is meaningless without a config); positional `spadl_cfg` parameter.
  Examples-section docstring per the public-API discipline.
- **`tests/vaep/test_feature_framework_layout.py`** — 7-case framework-layout
  lock (T-D). Asserts each framework primitive's canonical home is
  `silly_kicks.vaep.feature_framework`.
- **`docs/superpowers/adrs/ADR-002-shared-vaep-feature-framework-boundary.md`** —
  captures the framework-extraction decision, the 4 alternatives considered,
  and the `_actiontype → actiontype_categorical` rename rationale.

### Changed

- **`silly_kicks.vaep.features.core` slimmed** to its standard-SPADL-specific
  helpers (`play_left_to_right`, `feature_column_names`); re-exports the
  framework primitives from `silly_kicks.vaep.feature_framework` so existing
  `from silly_kicks.vaep.features.core import gamestates` paths continue to
  resolve (Hyrum's-Law preservation).
- **`silly_kicks.atomic.vaep.features` imports framework directly from
  `vaep.feature_framework`** (no longer reaches into `vaep.features.core`);
  per-concern feature reuse from `bodypart` / `context` / `temporal` is
  preserved (intentional verbatim code-share, not framework leak).
- **`silly_kicks.vaep.features.actiontype` body updated** to call
  `actiontype_categorical(actions, spadlcfg)` instead of the private
  `_actiontype(actions)` (the latter relied on an implicit-None spadlcfg
  fallback; the new call passes spadlcfg explicitly — same resolved
  behaviour).
- **T-A backcompat (`tests/vaep/test_features_backcompat.py`)** gains one row
  for `actiontype_categorical`. 33 → 34 cases.
- **T-B layout (`tests/vaep/test_features_submodule_layout.py`)** drops the 6
  framework rows now living outside the features package. 33 → 27 cases.
- **T-C atomic-coupling (`tests/atomic/test_features_per_concern_import.py`)
  rewritten** to forbid `vaep.features.core` import for framework primitives
  and require import from `vaep.feature_framework`. Retains the existing
  package-root-import forbid + 3 per-concern-import requirements.
- **Examples-gate file list** (`tests/test_public_api_examples.py`) adds
  `silly_kicks/vaep/feature_framework.py`. 26 → 27 cases.

### Removed

- **`silly_kicks.vaep.features.core._actiontype`** — promoted to public
  `actiontype_categorical(actions, spadl_cfg)` in the new framework module.
  Was a leading-underscore-private symbol; never in `__all__`; never
  documented as public surface.

### Closed

- **TODO A9** — `atomic/vaep/features.py` per-concern coupling — closed via
  framework extraction (the trigger-condition resolution from PR-S15's
  deferral). The `## Architecture` section of `TODO.md` is now empty.
  See ADR-002.

### Notes

- **Hyrum's Law surface:** `gamestates.__module__` (and `simple.__module__`)
  flips from `silly_kicks.vaep.features.core` to
  `silly_kicks.vaep.feature_framework`. Consumers introspecting via
  `inspect.getmodule(gamestates)` would see the new value. Documented in
  ADR-002 as accepted exposure.
- **Test count:** 881 → 884 passing, 4 deselected (+3 net delta: +1 T-A row,
  -6 T-B rows, +7 T-D cases, +1 Examples-gate parametrize). Pyright clean
  (0 errors / 0 warnings / 0 informations).

## [2.3.0] — 2026-04-30

### Changed

- **`silly_kicks.vaep.features` decomposed from a 1170-line monolith into a
  package** of 8 concern-focused submodules (`core`, `actiontype`, `result`,
  `bodypart`, `spatial`, `temporal`, `context`, `specialty`). Hybrid visibility:
  every previously-public symbol remains importable via the package path
  (`from silly_kicks.vaep.features import startlocation` keeps working
  unchanged); submodule paths are also importable for advanced/atomic-internal
  use. Closes the long-standing TODO architecture entry. **Pure structural
  refactor — zero behavior change; every existing test passes through every
  step.**
- **`silly_kicks.atomic.vaep.features` updated to import per-concern.** 12
  symbols imported across 4 grouped statements against
  `vaep.features.{core,bodypart,context,temporal}` (was: single 12-symbol
  monolith import). TODO A9 partially addressed (severity Medium → Low) —
  full decoupling deferred until atomic features need to diverge independently.
  Local type alias duplicates (`Actions = pd.DataFrame` etc.) replaced by a
  single import from `vaep.features.core` (DRY cleanup).

### Added

- **8 new public-API submodule paths** (`silly_kicks.vaep.features.core`,
  `.actiontype`, `.result`, `.bodypart`, `.spatial`, `.temporal`, `.context`,
  `.specialty`). Documented as implementation detail of where each symbol
  lives — the canonical entry point remains the package itself.
- **3 new test files locking the structure:** T-A backcompat (33 parametrized
  cases asserting every public symbol stays importable from the package path),
  T-B submodule layout (33 parametrized cases asserting each symbol's
  `__module__` matches the design contract), T-C atomic-per-concern (1 test
  asserting atomic imports from per-concern submodules, not the package root).
- **CI gate (`tests/test_public_api_examples.py::_PUBLIC_MODULE_FILES`)
  widened from 19 → 26 entries** to cover all 8 new submodule paths. Net +7
  parametrize cases.

### Closed

- **TODO A19** (default hyperparameters scattered across 3 learner functions):
  reviewed and closed without code change. Already centralized as
  `_XGBOOST_DEFAULTS` / `_CATBOOST_DEFAULTS` / `_LIGHTGBM_DEFAULTS` module-level
  constants since 1.9.0; the audit description ("scattered across 3 functions")
  predates that extraction.
- **TODO O-M1** (full `events.copy()` at top of StatsBomb `convert_to_actions`):
  reviewed and closed without code change. The defensive copy is correct by
  design — `_flatten_extra` mutates the DataFrame by adding ~22 underscore
  columns; without the copy, caller's events would be mutated in place.
- **TODO O-M6** (temporary n×3 DataFrame for StatsBomb fidelity version check):
  reviewed and closed without code change. ~50 KB peak per match; could be
  numpy-fied for marginal gain (~25 KB savings); no measurable impact.

No API breakage. 881 tests passing (807 baseline + 33 T-A + 33 T-B + 1 T-C +
7 net gate delta), 4 deselected.

## [2.2.0] — 2026-04-30

### Added

- **`silly_kicks.atomic.spadl.coverage_metrics`** — Atomic-SPADL counterpart to
  the standard `silly_kicks.spadl.coverage_metrics` utility (added in 1.10.0).
  Resolves `type_id` against the atomic 33-type vocabulary
  (`silly_kicks.atomic.spadl.config.actiontypes`) including atomic-only types
  (`receival`, `interception`, `out`, etc.) and post-collapse names (`corner`,
  `freekick`). Reuses the standard `CoverageMetrics` TypedDict from
  `silly_kicks.spadl.utils` as the single source of truth — both standard and
  atomic surfaces import the same type. Closes TODO C-1 (deferred from 1.10.0).
- **Examples sections on 25 previously-uncovered public-API surfaces** across
  `silly_kicks/vaep/labels.py` (5), `silly_kicks/vaep/formula.py` (3),
  `silly_kicks/atomic/vaep/features.py` (9), `silly_kicks/atomic/vaep/labels.py` (5),
  and `silly_kicks/atomic/vaep/formula.py` (3). Closes the PR-S13 documentation
  coverage gap.

### Changed

- **CI guardrail (`tests/test_public_api_examples.py`) widened from 14 → 19
  module files.** The gate now mechanically enforces Examples coverage across
  the entire public API surface; future PRs that add a public function
  without an Example fail CI.

No API breakage. New public symbols (`coverage_metrics`, `CoverageMetrics`
re-export) are additive only.

## [2.1.1] — 2026-04-30

### Added

- **Examples sections on all public API surfaces.** Closes the long-standing D-8
  documentation gap. Every public function / class / method in
  `silly_kicks.spadl`, `silly_kicks.atomic.spadl`, `silly_kicks.vaep`,
  `silly_kicks.atomic.vaep`, and `silly_kicks.xthreat` now has a 3-7 line
  illustrative example showing typical usage. ~50 surfaces newly documented.
- **CI guardrail at `tests/test_public_api_examples.py`.** AST-based parametrized
  test asserts every public symbol has an `Examples` section in its docstring.
  Future PRs that add a public function without an Example fail CI; the failure
  message points to canonical-style references (`add_possessions`,
  `boundary_metrics`).

### Changed

- **D-9 entry removed from `TODO.md`.** Tech-debt entry was stale — all 9
  module-level helpers in `silly_kicks/xthreat.py` are already underscore-
  prefixed; the entry tracked work that was completed prior to silly-kicks 2.0.0.

No API or behavior changes.

## [2.1.0] — 2026-04-29

### ⚠️ Breaking

- **`add_possessions` default for `max_gap_seconds` changed from 5.0 to 7.0**
  in both `silly_kicks.spadl.add_possessions` and
  `silly_kicks.atomic.spadl.add_possessions`. Empirically Pareto-optimal at
  the per-match recall floor on 64 StatsBomb WorldCup-2018 matches (full
  campaign data:
  `docs/superpowers/specs/2026-04-29-add-possessions-precision-improvement-design.md`).
  Same input DataFrame produces different `possession_id` values for any
  pair of actions where the time gap is in `[5, 7)` seconds AND the team
  did not change.

  **Opt-out:** explicit `add_possessions(actions, max_gap_seconds=5.0)`.

  This default change is shipped as a minor bump under pragmatic semver
  (luxury-lakehouse is the only known consumer; one-line opt-out preserves
  prior behavior). Strict semver would call this 3.0.0.

### Added

- **`silly_kicks.spadl.add_possessions` (and atomic counterpart)** new
  opt-in keyword-only parameters for precision-improvement rules:

  - `merge_brief_opposing_actions: int = 0` + `brief_window_seconds: float = 0.0`
    (paired) — brief-opposing-action merge rule. Suppresses team-change
    boundaries when team B has 1..N consecutive actions sandwiched between
    team A actions within the time window. Both must be > 0 to enable;
    both 0 to disable; exactly one > 0 raises `ValueError`.
  - `defensive_transition_types: tuple[str, ...] = ()` — defensive-transition
    rule. Listed action types do not trigger team-change boundaries on
    their own. Recommended: `("interception", "clearance")`.

  All defaults disable the rules, preserving 2.0.x algorithmic behavior
  except for the `max_gap_seconds` default change above.

- **`tests/datasets/statsbomb/spadl-WorldCup-2018.h5`** regenerated with
  `preserve_native=["possession"]` — the 64-match HDF5 fixture is now a
  reusable regression corpus for `add_possessions`. New file size ~6 MB
  (one extra `possession` column on ~128K rows under zlib compression).

- **`tests/spadl/test_add_possessions.py::TestBoundaryAgainstStatsBomb64Match`**
  64-match parametrized regression gate complementing the existing 3-fixture
  cross-competition gate. Each match independently gated at
  `recall >= 0.83 AND precision >= 0.30`.

### Changed

- **`silly_kicks/spadl/utils.py`** boundary-detection logic refactored
  into a private `_compute_possession_boundaries` helper, mirroring the
  atomic-side `_compute_possessions` factoring. Public API unchanged;
  internal seam for the new opt-in rules.

- **`tests/spadl/test_add_possessions.py::TestBoundaryAgainstStatsBombNative`**
  per-match recall threshold lowered from 0.85 to 0.83. Absorbs the
  slightly reduced recall margin at the new `max_gap_seconds=7.0` default
  (worst observed across 64 matches: R_min=0.854) plus pandas/numpy
  version-drift safety margin.

### Behavior baselines

`add_possessions` empirical performance at the new default (no opt-in
rules, 64 WC-2018 matches):

| Metric | Mean | sd | Min |
|---|---|---|---|
| Precision | 0.439 | 0.035 | 0.350 |
| Recall | 0.939 | 0.023 | 0.854 |
| F1 | 0.597 | — | — |

(Compare to 2.0.x at `max_gap_seconds=5.0`: P=0.412, R=0.950, F1=0.574.)

Recommended opt-in settings: see `add_possessions` docstring and
`docs/superpowers/specs/2026-04-29-add-possessions-precision-improvement-design.md`.

## [2.0.0] — 2026-04-29

### ⚠️ Breaking

- **`silly_kicks.spadl.sportec.convert_to_actions` no longer overrides
  `team_id` / `player_id` from DFL `tackle_winner` / `tackle_winner_team`
  qualifiers.** Per ADR-001
  (`docs/superpowers/adrs/ADR-001-converter-identifier-conventions.md`),
  the SPADL converter contract is "caller's identifier conventions are
  sacred — never overridden from qualifiers." Caller-supplied `team` /
  `player_id` values mirror verbatim into the output. Pre-2.0.0 behavior
  silently rewrote ~56% of tackle rows on consumers using a
  caller-normalized `team` convention (see luxury-lakehouse PR-LL2
  close-out report).
- **Sportec output schema changes from `KLOPPY_SPADL_COLUMNS` to
  `SPORTEC_SPADL_COLUMNS`** — 14 + 4 = 18 columns. The 4 new columns
  surface DFL qualifier values: `tackle_winner_player_id`,
  `tackle_winner_team_id`, `tackle_loser_player_id`,
  `tackle_loser_team_id`. NaN on non-tackle rows; NaN when the qualifier
  is absent. Sportec consumers asserting against `KLOPPY_SPADL_COLUMNS`
  must switch to `SPORTEC_SPADL_COLUMNS`.

### Migration

If your pre-2.0.0 sportec consumer relied on the tackle-winner override
AND your upstream `team` / `player_id` columns are in the same
identifier convention as DFL's `tackle_winner_team` / `tackle_winner`
qualifiers (raw `DFL-CLU-...` / `DFL-OBJ-...`), call the new helper
post-conversion:

```python
from silly_kicks.spadl import sportec, use_tackle_winner_as_actor
actions, _ = sportec.convert_to_actions(events, home_team_id="DFL-CLU-XXX")
actions = use_tackle_winner_as_actor(actions)
```

If your `team` / `player_id` columns use any other convention, the
post-1.10.0 behavior already preserved your conventions correctly — no
migration needed; the bug fix is automatic on upgrade.

### Added

- **First silly-kicks ADR.** `docs/superpowers/adrs/ADR-001-converter-identifier-conventions.md`
  + `docs/superpowers/adrs/ADR-TEMPLATE.md` (vendored verbatim from
  luxury-lakehouse) establish the silly-kicks ADR pattern. Future
  decisions that add an exception to project-wide conventions, change
  schema ownership, or hardcode a workaround for a platform constraint
  get an ADR.
- **`silly_kicks.spadl.SPORTEC_SPADL_COLUMNS`** schema constant (18-key
  dict) — extends `KLOPPY_SPADL_COLUMNS` with the 4 tackle qualifier
  passthrough columns. Re-exported from `silly_kicks.spadl`.
- **`silly_kicks.spadl.use_tackle_winner_as_actor(actions) -> pd.DataFrame`**
  — pure post-conversion enrichment that restores pre-2.0.0 sportec
  SPADL "actor = winner" semantic for callers whose upstream identifier
  convention matches DFL's qualifier format. Raises `ValueError` early
  on missing required columns. Mirrors the `add_*` helper family pattern.
- **Cross-provider parity regression gate**
  (`tests/spadl/test_cross_provider_parity.py::test_team_id_mirrors_input_team`).
  Parametrized over all 5 DataFrame converters; asserts each output's
  `team_id` values are a subset of the input `team` values. Locks the
  ADR-001 contract per-provider going forward; would have caught the
  1.7.0 sportec bug.
- **e2e on the IDSSE production fixture**
  (`TestSportecAdrContractOnProductionFixture`, 5 tests). Verifies the
  contract works on production-shape data: caller's labels survive
  through the converter; the 4 new columns are populated for qualifier
  rows; the migration helper round-trips correctly; 1.10.0 keeper
  coverage is preserved.

### Changed

- **CLAUDE.md "Key conventions" section** gains one rule citing ADR-001:
  "Converter identifier conventions are sacred. SPADL DataFrame
  converters never override the caller's `team_id` / `player_id`
  columns from provider-specific qualifiers..."
- **Sportec module docstring** documents the 4 tackle qualifier
  passthrough columns + the `SPORTEC_SPADL_COLUMNS` schema + the
  migration helper. References ADR-001.

### Removed

- **`silly_kicks.spadl.sportec` tackle override block** at the previous
  `sportec.py:559-565`. The 6-line override that silently rewrote
  `team_id` / `player_id` from raw DFL qualifier values is gone.
- **`tests/spadl/test_sportec.py::TestSportecActionMappingShotsTacklesFoulsGK::test_tackle_uses_winner_as_actor`**
  — was asserting the now-removed override. Covered by the new
  `TestSportecTackleNoOverride` + `TestSportecTackleWinnerColumns`
  classes.

### Audit findings

Manual cross-converter review (this cycle) confirmed sportec.tackle
was the unique violator of the ADR-001 contract:

| Converter | Override `player_id` / `team_id`? | Notes |
|---|---|---|
| `silly_kicks.spadl.sportec` | YES (removed) | The bug. |
| `silly_kicks.spadl.metrica` | NO | 1.10.0 GK routing only changes `type_id` / `bodypart_id`. |
| `silly_kicks.spadl.wyscout` | NO | 1.0.0 aerial-duel reclassification only changes `type_id` / `subtype_id`. |
| `silly_kicks.spadl.statsbomb` | NO | No qualifier-driven overrides. |
| `silly_kicks.spadl.opta` | NO | No qualifier-driven overrides. |
| `silly_kicks.spadl.kloppy` | NO | Gateway path. |

The 2.0.0 change is surgical (one converter), but the parity gate locks
the contract for all future converter additions.

### Notes

- silly-kicks 2.0.0 is the project's first semver-major release. The
  library is ~3 weeks old (0.1.0 shipped 2026-04-06); major versions
  aren't precious — bumping locks the contract before more downstream
  consumers pin against pre-2.0.0 behavior.
- luxury-lakehouse can bump `silly-kicks>=2.0.0,<3.0` and (optionally)
  drop their `_team_label_to_dfl_id` shim from PR-LL2 close-out, OR
  keep it as a documented winner-attribution post-conversion pattern.

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
