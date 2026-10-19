# TF-46 Corner identification + offence routine inference (Arm 1) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans — **inline execution** (owner's standing default). No subagent fan-out: the 15 tasks are a dependency chain that accretes shared files (`corners/__init__.py` across Tasks 2/4/6/8/9; `_routines.py` across 7/8), so parallel agents would risk interface drift. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the shared corner-identification scaffolding plus the label-free offence routine-inference pipeline (Shaw & Gopaladesikan playbook) as raw primitives in a new `silly_kicks/corners/` package.

**Architecture:** A hexagonal `compute_*`/`add_*` family cloned from `silly_kicks/shot_stopping/`, but tracking-consuming (reads tracking frames + TF-79 contacts) rather than event-only. Two ADR-011 params-only fitted models (`RunZoneModel` GMM, `CornerRoutineModel` NMF) follow the `GhostGkModel` npz-of-raw-arrays template (sklearn at fit only, pure-numpy serve). Two `scripts/_driver.py` corpus drivers fit the models and run the construct-validity battery.

**Tech Stack:** Python, pandas, numpy, scipy, scikit-learn (GMM/NMF at fit only), silly_kicks internal seams (`id_compat`, `reflection`, `tracking._ball_contact` TF-79, `tracking._chirality`/`_feature_contract`, `spadl.config`, `resolve_defended_goals`/`GoalMap`), `scripts/_driver.py`.

**Spec:** `docs/superpowers/specs/2026-10-16-tf46-corner-offence-design.md` (read it; this plan argues from it). Spec `1f483f0f`, both reviewers APPROVE r2.

> **Amendment 2026-10-18 (SPECD-01 reconcile).** Grain is **`game_id`** (SPADL-native), superseding the
> draft's `match_id` per the owner-approved spec §6.1 amendment (2026-10-16) the implementation follows
> (`CORNER_KEYS = ["game_id", "corner_id"]`). All `match_id` in this plan now reads `game_id`; cross-table
> joins use `(game_id, period_id[, t_kick])`.

## Global Constraints

Every task's requirements implicitly include this section.

- **Commit gate (spec §14 C3, verbatim).** NO `git commit`/`git push` without the owner's explicit approval for that specific commit. Tasks accumulate on the branch `tf-46-corner-offence`; each task's terminal state is a **tested deliverable, not a commit**. No micro-commits, no per-step/per-task commit cadence, whatever an upstream skill template suggests. The whole cycle lands as a small number of owner-approved commits at cycle end (Task 15).
- **No version/ADR number pinned** until commit-prep (Task 15), after CI green + part-deux (#284) landed. Release is held for the part-deux hold; the owner publishes.
- **No doc commits until cycle end.** Spec, this plan, and the ADR stay uncommitted on the feature branch through the cycle.
- **Branch:** `tf-46-corner-offence` off `main` (already cut). No worktrees.
- **CI-faithful verification before any "green" claim:** `python -m pytest tests/ -m "not e2e"`, `python -m ruff check silly_kicks/ tests/ scripts/` + `--format --check`, `pyright` (bare, whole package), and the SB360 cascade. A targeted-local run misses full-suite-only gates.
- **Hexagonal purity:** pure functions (pandas in, pandas out, zero mutation, zero I/O in core). Models: sklearn imported LOCALLY inside `fit()` only; serve path imports no sklearn.
- **id discipline (ADR-019):** every id routed through `silly_kicks.id_compat` (`canonical_id_series` for grouping, raw ids emitted via `.first()`); never raw `==`/`astype(str)` on ids.
- **ASCII-only in `scripts/` driver source** (the driver ASCII gate; `§` passes ruff but fails it).
- **Execution: inline** (`executing-plans`), owner's standing default (`feedback_inline_execution_default`, no worktrees). Tasks accrete shared files, so no subagent fan-out.

### Spec→plan mechanism refinements (flagged to owner 2026-10-16; coverage intent unchanged)

- Corner-column orientation invariance is enforced by a **dedicated corners test** (`tests/corners/test_corner_orientation_invariance.py`), NOT the `tracking.__all__`-pinned `test_mirror_registry` Gate A/B (spec §8/§10.6 named that gate; it does not cover `corners/`).
- SB360 verdict for `add_corner_windows` is obtained by registering it as a **`BOUNDARY_ENTRY_POINT`** in `tests/sb360/` (spec §10.7), since the SB360 gate covers `tracking.add_*` + boundary points only.
- The id-scalar gate is extended by **adding `silly_kicks.corners` to `PACKAGES`** in `tests/invariants/conftest_id_scalar.py` (spec §10.8), so corners public fns are discovered + gated.

---

## File structure

**New package `silly_kicks/corners/`:**
- `__init__.py` — public surface; alphabetically-sorted `__all__` (constants, classes, fns). Docstring: hexagonal allowed-imports incl. `silly_kicks.tracking` (tracking-consuming — unlike shot_stopping), "NOTHING imports corners", "See NOTICE".
- `_config.py` — `CornerWindowParams`, `RunEncodingParams`, `RoutineParams`, `ChanceQualityParams` (frozen dataclasses, `.default()`/`.for_provider()`/`.is_default()` boilerplate, empty `_PROVIDER_*` dicts).
- `_columns.py` — `CORNER_*_KEYS` (lists), `CORNER_*_METRIC_COLUMNS` (lists), `CORNER_*_COLUMNS` (dict[str,str] dtype literals), closed source-vocab `frozenset`s.
- `_identify.py` — `identify_corners`.
- `_windows.py` — `add_corner_windows`.
- `_reflect.py` — `CORNER_REFLECTION_KINDS` + `reflect_corners_to_canonical_side`.
- `_zones.py` — `RunZoneModel` + `encode_runs`.
- `_routines.py` — `CornerRoutineModel` + `cluster_corner_routines`.
- `_encode.py` — `compute_corner_run_encoding`.
- `_effectiveness.py` — `compute_corner_chance_quality` + `aggregate_chance_quality_by_feature`.
- `_report.py` — the `*Report` census dataclasses.

**Fitted-model weights (bundled):** `silly_kicks/corners/_run_zone_weights/default/{model.npz,metadata.json,SHA256SUMS}` and `silly_kicks/corners/_routine_weights/default/{...}`.

**Tests `tests/corners/`:** `test_columns.py`, `test_config.py`, `test_import_allowlist.py`, `test_purity.py`, `test_identify.py`, `test_windows.py`, `test_reflect.py`, `test_corner_orientation_invariance.py`, `test_corner_column_liveness.py`, `test_zones.py`, `test_routines.py`, `test_encode.py`, `test_effectiveness.py`, `test_run_zone_model.py`, `test_routine_model.py`, conftest fixtures.

**Registry edits (existing files):** `silly_kicks/metric_contracts.py`, `silly_kicks/feature_glossary.py`, `NOTICE`, `silly_kicks/reflection.py` (new `_CORNER_REFLECTION_KINDS` + proxy, if any corner geometry columns are added to an enumerated schema — otherwise the corners-local map), `tests/test_metric_contracts.py` (`_PKG`), `tests/test_add_star_purity.py` (`PURITY_ENTRIES`, `_PKG_MOD`), `tests/invariants/glossary_emitted_columns.py` (`_corners_columns()` leg), `tests/invariants/conftest_id_scalar.py` (`PACKAGES` + entries), `tests/sb360/_regenerate.py` (`FAMILY` + `BOUNDARY_ENTRY_POINTS`), `tests/test_public_api_examples.py` (`_PUBLIC_MODULE_FILES`), `tests/tracking/test_geometry_constant_enumeration.py` (`DECLARED_CONSTANT_SOURCES`/`_EXEMPT` for any geometry consts).

**Drivers:** `scripts/build_corner_routines.py`, `scripts/validate_corners.py`; `tests/scripts/test_build_corner_routines.py`, `tests/scripts/test_validate_corners.py`; `tests/scripts/test_provenance_wiring.py` (`ARTIFACT_DRIVERS` += both stems).

**Docs:** C4 regen `docs/c4/architecture.{dsl,html}`; `AGENTS.md` TF-table row + count; memo `docs/research/corner_routines/{README.md,metrics.json}` (Task 14). ADR drafted at commit-prep (Task 15).

---

## Task 1: Package skeleton + output contract

**Files:**
- Create: `silly_kicks/corners/__init__.py`, `_config.py`, `_columns.py`, `_report.py`
- Modify: `silly_kicks/metric_contracts.py` (add `"corners"` to `METRIC_CONTRACTS`)
- Modify: `tests/test_metric_contracts.py` (`_PKG["corners"]`)
- Test: `tests/corners/test_columns.py`, `tests/corners/test_config.py`, `tests/corners/test_import_allowlist.py`

**Interfaces:**
- Produces: `CORNER_IDENTIFY_KEYS`, `CORNER_IDENTIFY_COLUMNS`; `CORNER_ROUTINE_METRIC_COLUMNS`, `CORNER_ROUTINE_COLUMNS`; `CORNER_CHANCE_METRIC_COLUMNS`, `CORNER_CHANCE_COLUMNS`; `CORNER_KEYS = ["game_id","corner_id"]` grain; `CornerWindowParams`, `RunEncodingParams`, `RoutineParams`, `ChanceQualityParams`; `CornersReport`.

- [ ] **Step 1: Write `tests/corners/test_columns.py` (failing).** Assert the constant shapes:
```python
from silly_kicks.corners import (
    CORNER_KEYS, CORNER_CHANCE_METRIC_COLUMNS, CORNER_CHANCE_COLUMNS,
)
def test_keys_are_plain_lists():
    assert CORNER_KEYS == ["game_id", "corner_id"]
def test_chance_columns_superset_of_metric_plus_keys():
    assert set(CORNER_CHANCE_METRIC_COLUMNS) <= set(CORNER_CHANCE_COLUMNS)
    assert set(CORNER_KEYS) <= set(CORNER_CHANCE_COLUMNS)
    assert all(isinstance(v, str) for v in CORNER_CHANCE_COLUMNS.values())  # dtype strings
```
- [ ] **Step 2: Run → fail** (`ModuleNotFoundError`). `python -m pytest tests/corners/test_columns.py -q`
- [ ] **Step 3: Write `_columns.py`.** Mirror `shot_stopping/_columns.py`: `CORNER_KEYS = ["game_id", "corner_id"]`; metric-column name constants; `*_METRIC_COLUMNS` lists (value columns only); `*_COLUMNS: dict[str, str]` literal (keys + metrics + provenance, dtype strings — e.g. `n_shots: "Int64"`, `sum_xg: "float64"`, `corner_source: "object"`); closed vocab `frozenset`s `CORNER_SOURCE_VALUES={"skillcorner_gi","spadl_corner","sb360_event"}`, `CORNER_KIND_VALUES={"direct","short","unclear"}`, `TARGET_SNAPSHOT_SOURCE_VALUES={"first_contact","kick_plus_2s","sb360_event"}`.
- [ ] **Step 4: Write `_config.py`.** Four frozen dataclasses with the `shot_stopping` boilerplate. `CornerWindowParams(init_lead_s=2.0, target_contact_lead_s=1.0, target_cap_s=2.0, outcome_window_s=6.0)`; `RunEncodingParams`; `RoutineParams(linkage="ward", ...)`; `ChanceQualityParams`. Each: `.default(force_universal=False)`, `.for_provider(provider)`, `.is_default()`, `_is_universal_default` sentinel, empty `_PROVIDER_*_PARAMS: dict = {}`. Every method carries a doctest (public-api-examples gate).
- [ ] **Step 5: Write `_report.py`.** `@dataclass(frozen=True) class CornersReport` with census fields (e.g. `n_corners`, `n_short_excluded`, `n_unclear`, `n_unresolved_winner`) + a conservation doctest.
- [ ] **Step 6: Write `__init__.py`.** Alphabetically-sorted `__all__`; docstring states allowed imports INCLUDING `silly_kicks.tracking` + `silly_kicks.spadl` + `silly_kicks.id_compat`, "NOTHING imports corners", "See NOTICE".
- [ ] **Step 7: Write `tests/corners/test_config.py`** asserting dataclass defaults, frozen-ness, `.default().is_default() is True`, `.for_provider("skillcorner")` equals default (empty provider dict).
- [ ] **Step 8: Write `tests/corners/test_import_allowlist.py`.** Clone `tests/shot_stopping/test_import_allowlist.py` but ALLOW `silly_kicks.tracking` (corners is tracking-consuming); still assert nothing in `silly_kicks` imports `silly_kicks.corners` (leaf discipline), with a planted-violation meta-test.
- [ ] **Step 9: Register in `metric_contracts.py`.** Add `"corners": {"keys": (...), "metric_columns": (...), "columns": (...), "column_types": {...}}` mirroring the `_columns.py` constants literally (tuples; `column_types` the dtype dict). Register one `MetricContract` per emitted mart grain the family exposes (identify / routine-features / chance-quality) if they differ in grain — otherwise one combined `corners` entry at the `(game_id, corner_id)` grain.
- [ ] **Step 10: Register `_PKG["corners"]`** in `tests/test_metric_contracts.py`: `("silly_kicks.corners", "CORNER_KEYS", "CORNER_CHANCE_METRIC_COLUMNS", "CORNER_CHANCE_COLUMNS")` (and one row per registered mart).
- [ ] **Step 11: Run contract gate.** `python -m pytest tests/corners/ tests/test_metric_contracts.py -q` → PASS. Accumulate on branch (NO commit).

## Task 2: `identify_corners` (provider-aware anchor)

**Files:**
- Create: `silly_kicks/corners/_identify.py`
- Modify: `silly_kicks/corners/__init__.py` (export `identify_corners`)
- Test: `tests/corners/test_identify.py`

**Interfaces:**
- Consumes: `CORNER_IDENTIFY_COLUMNS`, vocab frozensets (Task 1); `silly_kicks.id_compat`.
- Produces: `def identify_corners(actions: pd.DataFrame, *, provider: str, params: CornerWindowParams = _DEFAULT_WINDOW_PARAMS) -> pd.DataFrame` — per-corner table at grain `(game_id, corner_id)` with columns `period_id, t_kick, attacking_team_id, corner_side{left,right}, kick_x, kick_y, corner_kind{direct,short,unclear}, corner_source`.

- [ ] **Step 1: Write failing tests** for three provider paths on tiny SPADL fixtures:
```python
def test_skillcorner_gi_anchor_corner_reception():
    # a corner_for restart followed by a corner_reception possession -> one direct corner
    out = identify_corners(_sc_actions_corner_reception(), provider="skillcorner")
    assert len(out) == 1 and out.iloc[0]["corner_kind"] == "direct"
    assert out.iloc[0]["corner_source"] == "skillcorner_gi"
def test_skillcorner_gi_other_start_type_is_unclear():
    out = identify_corners(_sc_actions_corner_for_then_pass(), provider="skillcorner")
    assert out.iloc[0]["corner_kind"] == "unclear"
def test_spadl_corner_anchor_optical():
    out = identify_corners(_optical_actions_corner_crossed(), provider="sportec")
    assert out.iloc[0]["corner_source"] == "spadl_corner"
def test_corner_side_from_kick_y():
    out = identify_corners(_two_corners_left_and_right(), provider="sportec")
    assert set(out["corner_side"]) == {"left", "right"}
```
- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement `identify_corners`.** Vectorised, `np.select`-style. SkillCorner: first `player_possession` after a `game_interruption_before=="corner_for"` whose `start_type ∈ {corner_reception, corner_interception}` → `direct`; `corner_for` + other start type → `unclear`; never the ball-at-flag. Optical/SB360: SPADL `corner_crossed`/`corner_short` action (`short`→`corner_kind="short"`). `corner_side` from `kick_y` vs pitch midline in frame-LTR. Grain key `corner_id` = per-match chronological positional int64 (within-table only; join on `(game_id, period_id, t_kick)` for cross-table — spec §13). Return `.reindex(columns=list(CORNER_IDENTIFY_COLUMNS)).astype(CORNER_IDENTIFY_COLUMNS)`; empty-schema frame on no-corners. Raise canonical `KeyError` on a missing required column.
- [ ] **Step 4: Run → pass.** Accumulate (no commit).

## Task 3: Corner reflection + dedicated orientation-invariance test

**Files:**
- Create: `silly_kicks/corners/_reflect.py`
- Test: `tests/corners/test_reflect.py`, `tests/corners/test_corner_orientation_invariance.py`

**Interfaces:**
- Consumes: `silly_kicks.reflection` (`reflect_columns`, `ReflectionKind`).
- Produces: `CORNER_REFLECTION_KINDS: Mapping[str, ReflectionKind]` (corners-local map; `kick_x→point_x`, `kick_y→point_y`, snapshot `x/y`, run coords; zone labels/probabilities `invariant`); `def reflect_corners_to_canonical_side(df, corners, *, kinds=CORNER_REFLECTION_KINDS) -> pd.DataFrame`.
- **Declaration-completeness (P2, A-02):** because the mechanism swap off `test_mirror_registry` dropped Gate-A/B's declaration-completeness property, a corners-local completeness meta-assertion carries it — every emitted corner geometry column must be declared in `CORNER_REFLECTION_KINDS`, so a future column cannot silently escape reflection.

- [ ] **Step 1: Write failing `test_reflect.py`.** A corner on the `right` side reflects to canonical `left`; a corner already `left` is unchanged; coordinate y flips, invariant columns untouched; pure (returns new frame).
- [ ] **Step 2: Write failing `test_corner_orientation_invariance.py`** (the dedicated gate replacing the mis-stated shared Gate A/B): (a) construct a left-corner scene and its mirrored right-corner twin of identical geometry; after `reflect_corners_to_canonical_side` the downstream run-encoding is mirror-equal; **non-vacuity probe**: an off-centre point that MUST move under reflection does move. (b) **Completeness meta-assertion (P2):** enumerate every emitted corner geometry column (the `CORNER_*_COLUMNS` names matching `silly_kicks.reflection.GEOMETRIC_NAME`) and assert each is a key of `CORNER_REFLECTION_KINDS` — a geometry-named column left undeclared reds this test (so it cannot silently escape reflection).
- [ ] **Step 3: Run → fail.**
- [ ] **Step 4: Implement `_reflect.py`.** Build a per-corner boolean mask (side == non-canonical) and call `reflection.reflect_columns(df, mask, point_y=[...], vector_y=[...], ...)` with `CORNER_REFLECTION_KINDS`. Reflection runs BEFORE zone assignment (spec §8).
- [ ] **Step 5: Run → pass.** Accumulate.

## Task 4: `add_corner_windows` (TF-79 timing + winner)

**Files:**
- Create: `silly_kicks/corners/_windows.py`
- Modify: `silly_kicks/corners/__init__.py`
- Modify: `tests/test_add_star_purity.py` (`PURITY_ENTRIES["corners:add_corner_windows"]`, `_PKG_MOD["corners"]`)
- Modify: `tests/sb360/_regenerate.py` (`FAMILY` + `BOUNDARY_ENTRY_POINTS`), then run `_regenerate.py`/`_adjudicate.py` → `tests/sb360/_entries/corners.py`
- Create: `tests/corners/test_windows.py`, `tests/corners/test_purity.py`, `tests/corners/test_corner_column_liveness.py`

**Interfaces:**
- Consumes: `silly_kicks.tracking._ball_contact.first_contact_in_window(contacts, period_id, t0, t1)` (TF-79, shipped); `CornerWindowParams`; `@nan_safe_enrichment` from `silly_kicks._nan_safety`.
- Produces: `def add_corner_windows(corners: pd.DataFrame, frames: pd.DataFrame, contacts: pd.DataFrame, *, params: CornerWindowParams = _DEFAULT_WINDOW_PARAMS) -> pd.DataFrame` adding `t_init, t_target, t_outcome, first_contact_time, first_possession_time, winner_team_id, winner_player_id, target_snapshot_source`.

- [ ] **Step 1: Write failing `test_windows.py`.** Cases:
```python
def test_target_uses_first_contact_plus_lead_when_resolved():
    out = add_corner_windows(corners, frames, contacts_with_contact_at_1s)
    assert out.iloc[0]["t_target"] == pytest.approx(min(1.0 + 1.0, t_kick + 2.0))
    assert out.iloc[0]["target_snapshot_source"] == "first_contact"
def test_target_falls_back_to_kick_plus_2s_when_no_contact():
    out = add_corner_windows(corners, frames, _empty_contacts())
    assert out.iloc[0]["t_target"] == pytest.approx(t_kick + 2.0)
    assert out.iloc[0]["target_snapshot_source"] == "kick_plus_2s"
    assert pd.isna(out.iloc[0]["winner_team_id"])          # no GI fallback
def test_sb360_no_contacts_winner_is_na_source_sb360_event():
    out = add_corner_windows(sb360_corners, sb360_frames, _no_contacts())
    assert out.iloc[0]["target_snapshot_source"] == "sb360_event"
def test_first_possession_diagnostic_emitted():
    assert "first_possession_time" in add_corner_windows(corners, frames, contacts).columns
```
- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement `add_corner_windows`.** `@nan_safe_enrichment`, pure, `frames` positional-or-keyword, kwargs keyword-only (ADR-078). Per corner: `t_init = t_kick - params.init_lead_s`; call `first_contact_in_window(contacts, period_id, t_kick, t_kick + params.outcome_window_s)`; `t_fc = contact.time_seconds` or NaN; `t_target = min(t_fc + 1, t_kick + 2)` when `t_fc` present else `t_kick + 2`; `target_snapshot_source` per branch (+ `sb360_event` when provider is SB360 / no tracking); `winner_*` from the contact (NaN when unresolved, no GI fallback); `first_possession_time` from GI diagnostic; `t_outcome = t_kick + 6`. Return a NEW frame.
- [ ] **Step 4: Run → pass.**
- [ ] **Step 5: Register purity.** `PURITY_ENTRIES["corners:add_corner_windows"]` with ≥2 variants (contact-resolved branch + no-contact branch — it conditionally sets `target_snapshot_source`). Add `"corners"` to `_PKG_MOD`. Build inputs from fresh uncached builders.
- [ ] **Step 6: Write `tests/corners/test_purity.py`** (local): `add_corner_windows` does not mutate `corners`/`frames`/`contacts`; `out is not corners`.
- [ ] **Step 7: Write `tests/corners/test_corner_column_liveness.py`** (C2, spec §10.3): meta-assert the gate population equals the corner `add_*`/`compute_*` exports in `corners.__all__`; assert no emitted metric column is 100%-null on an in-domain fixture and every float metric column with ≥2 obs carries >1 distinct value; include a fixture-precondition test (in-domain, no constant feature).
- [ ] **Step 8: Register SB360 boundary verdict.** Add `"corners.add_corner_windows"` to `BOUNDARY_ENTRY_POINTS` and the `FAMILY` dict in `tests/sb360/_regenerate.py`; run `python tests/sb360/_regenerate.py` then `python tests/sb360/_adjudicate.py` then `python -m ruff format tests/sb360/`. Continuous-only columns dark on single freeze frames → `not_exercised`; bump `NOT_EXERCISED_BUDGET` + record the reason (do not fabricate coverage).
- [ ] **Step 9: Run** `pytest tests/corners/ tests/test_add_star_purity.py tests/sb360/ -q` → PASS. Accumulate.

## Task 5: `RunZoneModel` (GMM zones, ADR-011 params-only)

**Files:**
- Create: `silly_kicks/corners/_zones.py`
- Create: `tests/corners/test_run_zone_model.py`, `tests/corners/test_zones.py`
- Modify: `tests/tracking/test_geometry_constant_enumeration.py` (`DECLARED_CONSTANT_SOURCES`/`_EXEMPT` for any geometry const in `_zones.py`)

**Interfaces:**
- Consumes: `silly_kicks.tracking._chirality` (`chirality_fingerprint`, `verify_chirality`), `silly_kicks.tracking._feature_contract` (`feature_contract`, `verify_feature_contract`), `silly_kicks.tracking._geometry` (`in_penalty_area_absolute`, verified at `_geometry.py:186` — **NOT** `spadl.config`), `silly_kicks.spadl.config` (penalty-area dimension constants only).
- Produces: `class RunZoneModel` with `fit(init_xy, target_xy) -> RunZoneModel`, `encode_runs(init_xy, target_xy, active_mask) -> np.ndarray` (probabilistic `(n_i*n_t)`-vector), `save(path)`, `load(path, *, legacy_override=False)`; module-local `class IntegrityError(Exception)`.

- [ ] **Step 1: Write failing `test_run_zone_model.py`** (mirror `tests/tracking/test_chirality_enforcement.py` + `test_feature_contract.py` patterns):
```python
def test_round_trip_predicts_identically(tmp_path):
    m = _toy_run_zone_model(); v0 = m.encode_runs(*_toy_probe())
    m.save(tmp_path); v1 = RunZoneModel.load(tmp_path).encode_runs(*_toy_probe())
    np.testing.assert_allclose(v0, v1)
def test_load_raises_on_tampered_chirality(tmp_path): ...   # IntegrityError
def test_legacy_override_allows_with_warning(tmp_path): ...
def test_load_imports_no_sklearn(monkeypatch, tmp_path): ...  # block sklearn import, load() still works
```
- [ ] **Step 2: Write failing `test_zones.py`** (mechanics): synthetic corners with planted zones → GMM recovers them (BIC picks ≈ planted k); non-vacuity (encoded vector populates >1 run); active-zone = components inside penalty area via `in_penalty_area_absolute`.
- [ ] **Step 3: Run → fail.**
- [ ] **Step 4: Implement `_zones.py`.** `fit()` does `from sklearn.mixture import GaussianMixture` LOCALLY; fit initial GMM + target GMM; select component counts by BIC/log-likelihood elbow; keep target components inside the penalty area (`from silly_kicks.tracking._geometry import in_penalty_area_absolute`) as active a–g; store `means_`/`covariances_`/`weights_` as plain `np.ndarray` on `self`. `encode_runs()` pure-numpy: per player P(init zone) × P(target zone) (manual Gaussian log-density + softmax responsibilities), summed over active players → `(n_i*n_t)`-vector. `save()`: `np.savez_compressed(path/"model.npz", ...)`, `metadata.json` (LF) with `chirality=_chirality_block(self)` + `feature_contract=_feature_contract_block(...)` + selected-k + BIC curve, `SHA256SUMS`. `load()`: verify SHA256SUMS → `IntegrityError`; `np.load(allow_pickle=False)`; `verify_chirality(..., error_cls=IntegrityError)` + `verify_feature_contract(..., error_cls=IntegrityError)`. Any geometry-named module const registered in `DECLARED_CONSTANT_SOURCES` (or `_EXEMPT` with reason).
- [ ] **Step 5: Run → pass.** Accumulate.

## Task 6: `compute_corner_run_encoding`

**Files:**
- Create: `silly_kicks/corners/_encode.py`; Modify `__init__.py`; Test: `tests/corners/test_encode.py`

**Interfaces:**
- Consumes: `RunZoneModel` (Task 5), reflected snapshots (Task 3/4), `RunEncodingParams`.
- Produces: `def compute_corner_run_encoding(corners, frames, *, model: RunZoneModel, params: RunEncodingParams = ...) -> pd.DataFrame` — per-corner `(n_i*n_t)` float columns `run_{i}{t}` at grain `(game_id, corner_id)`.

- [ ] **Step 1: Write failing test.** Known two-player corner → expected probabilistic run vector; excludes `corner_kind ∈ {short,unclear}`; active-player filter (target near an active zone); `.reindex().astype()` schema.
- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement.** Pull init snapshot at `t_init`, target snapshot at `t_target` from `frames` (reflected), apply `model.encode_runs`, assemble the term-matrix-column frame. Return `tuple[pd.DataFrame, CornersReport]` per family convention.
- [ ] **Step 4: Run → pass.** Accumulate.

## Task 7: `CornerRoutineModel` (NMF) + routine features

**Files:**
- Create: `silly_kicks/corners/_routines.py` (model part); Modify `__init__.py`; Test: `tests/corners/test_routine_model.py`

**Interfaces:**
- Produces: `class CornerRoutineModel` with `fit(term_matrix, *, rank=None) -> CornerRoutineModel` (rank elbow-selected if None), `transform(term_matrix) -> np.ndarray` (per-corner r-dim H coeffs), `save`/`load` (own `IntegrityError`); `def compute_corner_routine_features(encoding, *, model) -> pd.DataFrame`.

- [ ] **Step 1: Write failing tests.** Planted NMF basis recovered (synthetic); reconstruction-error `‖X−WH‖_F` decreases and the chosen `rank` sits at the elbow; round-trip save/load numerically identical; `load()` imports no sklearn; seeded init → deterministic fit for the artifact digest.
- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement.** `fit()`: `from sklearn.decomposition import NMF` LOCALLY, α=1 sparse, multiplicative update, seeded init; rank by reconstruction-elbow; store `W` (`(n_i*n_t) × r`) as `np.ndarray`. `transform()`: pure-numpy non-negative least-squares / multiplicative-update solve for H given frozen W. `save`/`load` as Task 5. `compute_corner_routine_features` → per-corner `routine_feat_{k}` columns.
- [ ] **Step 4: Run → pass.** Accumulate.

## Task 8: `cluster_corner_routines` (agglomerative, raw structure)

**Files:** Modify `silly_kicks/corners/_routines.py`; `__init__.py`; Test: `tests/corners/test_routines.py`

**Interfaces:**
- Produces: `def cluster_corner_routines(features, *, params: RoutineParams = ...) -> pd.DataFrame` — per-corner `cluster_id` (`int64`) + the linkage returned (not a column). RAW structure only; no names/ranking (ADR-110).

- [ ] **Step 1: Write failing test.** Two clearly-separated synthetic routine groups → two clusters; deterministic given seed/linkage; output is raw assignments (no archetype strings).
- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement.** `scipy.cluster.hierarchy` (ward) over the H coefficient vectors; return assignments + linkage.
- [ ] **Step 4: Run → pass.** Accumulate.

## Task 9: Per-routine chance quality (descriptive effectiveness)

**Files:** Create `silly_kicks/corners/_effectiveness.py`; `__init__.py`; Test: `tests/corners/test_effectiveness.py`

**Interfaces:**
- Produces: `def compute_corner_chance_quality(corners, actions, *, xg_column: str, params: ChanceQualityParams = ...) -> pd.DataFrame` (per-corner `n_shots,n_goals,sum_xg,chance_quality_source` inside `[t_kick, t_outcome]`); `def aggregate_chance_quality_by_feature(features, chance_quality) -> pd.DataFrame` (H-coefficient-weighted mean; descriptive, not ranking).

- [ ] **Step 1: Write failing tests.** Known shots/goals/xG inside the 6 s window aggregate exactly; a corner with no outcome keeps its row with 0/NaN + reason (ADR-042); `xg_column` injected as a port (KeyError if absent); the by-feature aggregate is a soft-weighted mean (not a sort/rank).
- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement.** Window-join shots to corners on `(game_id, period_id)` within `[t_kick, t_outcome]`; sum via `id_compat`-safe grouping; the by-feature aggregate weights per-corner chance quality by the H coefficient.
- [ ] **Step 4: Run → pass.** Accumulate.

## Task 10: Glossary + NOTICE + id-scalar + public-api-examples

**Files:**
- Modify: `silly_kicks/feature_glossary.py` (FeatureColumn per emitted column + `_M_CORNERS` home-module const + `_A_SHAW` citation token)
- Modify: `tests/invariants/glossary_emitted_columns.py` (`_corners_columns()` leg, OR'd into the aggregate set)
- Modify: `NOTICE` (Shaw & Gopaladesikan; GMM Dempster et al.; NMF Lee & Seung / Kuang et al. / Hoyer / Févotte & Idier)
- Modify: `tests/invariants/conftest_id_scalar.py` (`PACKAGES += "silly_kicks.corners"`; `IdScalarEntry` per public fn with an entity `*_id` param — e.g. `attacking_team_id`)
- Modify: `tests/test_public_api_examples.py` (`_PUBLIC_MODULE_FILES += corners modules`)

- [ ] **Step 1: Run the coverage gates → fail** (they red the moment the family emits columns / exports id-fns / exposes public modules). `python -m pytest tests/test_feature_glossary_coverage.py tests/invariants/test_public_id_scalar_registry.py tests/test_public_api_examples.py -q`
- [ ] **Step 2: Add `FeatureColumn` entries** for every emitted column (`emitting_module` = the corners compute module, never `features.py`); `attribution=_A_SHAW` where methodology-derived, else `None`; add `_corners_columns()` leg running a tiny fixture through the compute fns.
- [ ] **Step 3: Add the matching `NOTICE` paragraph(s)** for any non-None attribution token (verbatim-linkage gate).
- [ ] **Step 4: Extend `PACKAGES`** with `silly_kicks.corners` and register an `IdScalarEntry` (dtype-invariant: `5` vs `"5"` give identical output) per public id-param fn, or justify into `NOT_INVARIANT`/`NOT_EXERCISABLE` (≥40-char reason).
- [ ] **Step 5: Add corners modules to `_PUBLIC_MODULE_FILES`** and ensure every public symbol has a real `Examples` section (runnable doctest or house illustrative block) — documented on creation, not deferred.
- [ ] **Step 6: Run the gates → pass.** Accumulate.

## Task 11: C4 + AGENTS.md count

**Files:** Modify `docs/c4/architecture.{dsl,html}`, `AGENTS.md` (TF-46 row + aggregator count)

- [ ] **Step 1:** Add the `corners/` subsystem + TF-46 row to the `AGENTS.md` architecture map/TF-table; bump the documented aggregator count.
- [ ] **Step 2:** Regenerate C4 via the `mad-scientist-skills:c4` skill with Graphviz `dot` (`-graphvizdot "C:/Users/Karsten/.claude/tools/graphviz/dot.exe"`); verify the `dot` render (non-Smetana viewBox), per CLAUDE.md.
- [ ] **Step 3:** Run the C4-pinning + AGENTS-md-budget tests (`tests/test_agents_md_budget.py`) → pass. Accumulate.

## Task 12: `scripts/build_corner_routines.py` (fit driver)

**Files:** Create `scripts/build_corner_routines.py`; `tests/scripts/test_build_corner_routines.py`; Modify `tests/scripts/test_provenance_wiring.py` (`ARTIFACT_DRIVERS += "build_corner_routines"`)

**Interfaces:**
- Consumes: `scripts/_driver.py` (`for_each`), `scripts/_loader_pining.py` (`pining_source`), `scripts/_provenance.py` (`require_clean_tree`, `git_provenance`), `scripts/_input_contract.py` (`declare_inputs`), `RunZoneModel`/`CornerRoutineModel`.
- Produces: fitted `model.npz` artifacts + a `metrics.json` fit summary.

- [ ] **Step 1: Write failing CI-safe kernel test** (`tests/scripts/test_build_corner_routines.py`): the pure combine/fit-summary helpers on a tiny in-memory frame (no corpus); assert `_SHARD_SCHEMA_VERSION` is referenced (not re-typed) in `token_inputs`.
- [ ] **Step 2: Add both stems to `ARTIFACT_DRIVERS`** (this task + Task 13) with justifying comments — else `test_the_artifact_driver_population_is_EXACT` reds the moment the scripts exist with `--out` + a write call.
- [ ] **Step 3: Run → fail.**
- [ ] **Step 4: Implement** per the research skeleton: `main()` calls `require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)` FIRST; `--out/--token/--cache-dir/--match-ids-json/--max-matches/--allow-dirty/--list-matches`; `pining_source(providers=["skillcorner", ...])`; `for_each(refs, key=lambda r: r.key, load=load, work=_work, shard_root=dest/"shards", token_inputs={"metric":"corner_routines","schema":_SHARD_SCHEMA_VERSION}, label="match")`; combine from `res.shard_keys`; fit the two models; write npz + `metrics.json` (`run_commit`, `run_tree_dirty`, `declare_inputs(driver="build_corner_routines", ...)`). `_EMITTED_SHARD_COLUMNS` + runtime drift assertion. ASCII-only.
- [ ] **Step 5: Run the CI-safe slice + provenance-wiring parametrized guards → pass.** Accumulate. (Real corpus fit is opt-in/`e2e`, Task 14.)

## Task 13: `scripts/validate_corners.py` (construct-validity harness)

**Files:** Create `scripts/validate_corners.py`; `tests/scripts/test_validate_corners.py`; Modify `tests/scripts/test_provenance_wiring.py` (`ARTIFACT_DRIVERS += "validate_corners"`)

**Interfaces:**
- Consumes: the Task 5–9 compute surface; `silly_kicks.causal.power` (`InjectionSpec`, `att_power_curve`, `n_degenerate_by_size`); `scripts/_driver.py` seam.
- Produces: `docs/research/corner_routines/metrics.json` verdicts (responsiveness / discrimination-ICC-vs-null / net-of-team / transfer) + §12 reported checks (non-vacuity, orientation invariance, synthetic recovery, NMF curve, cross-provider stability, convergent timing, **§12.7 SB360 init-divergence**).

- [ ] **Step 1: Write failing CI-safe kernel tests:** the pure stat kernels (`icc_vs_permutation`, the SB360 init-zone-reassignment-rate computation, the InjectionSpec power curve) on tiny fixtures; assert `att_power_curve` takes NO observed-outcome vector (the firewall — `TypeError` if a `Y=` kwarg is passed).
- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement** per the research skeleton: clean-tree guard first; `for_each` over corner-bearing matches; `reduce_samples` computes pooled verdicts (responsiveness means + t-stats; `discrimination_one_way` per metric `{icc, null_p95, p, n}` vs a structure-preserving permutation null + the negative control; `net_of_team`; `transfer`); §12.7 derives the SB360 init-divergence on a both-tracking+freeze provider (per-player `−2s→kick` positional-delta + initial-zone reassignment rate); the chance-quality construct-validity triad takes an `InjectionSpec` recipe (observed outcome unrepresentable) and counts `n_degenerate_by_size`. Write `metrics.json` with provenance + `input_contract`.
- [ ] **Step 4: Run CI-safe slice + provenance guards → pass.** Accumulate.

## Task 14: Run the corpus fit + validation (owner-tier, heavy)

**Files:** Produces `silly_kicks/corners/_*_weights/default/*` (fitted) + `docs/research/corner_routines/{README.md, metrics.json}`.

> Heavy corpus run — delegate to MEDIA-PC / DGX per the compute-delegation memory (AI-PC crashes on hot RAM). Requires a CLEAN tree (`require_clean_tree`), so run against a committed checkpoint OR with `--allow-dirty` recorded. Reversibility policy: aggregate-only artifacts, no reversible ids in committed outputs.

- [ ] **Step 1:** Run `build_corner_routines.py` on the owner-tier SkillCorner corpus (+ optical for cross-provider) → fitted npz artifacts; verify corpus size (owner token present).
- [ ] **Step 2:** Run `validate_corners.py` → `metrics.json`; read the GO/NO-GO reading of the three gates + the §12 reported checks.
- [ ] **Step 3:** Land the memo `docs/research/corner_routines/README.md` (the GO/NO-GO prose + "what it does NOT measure" caveat) + `metrics.json` with `run_commit`/`run_tree_dirty:false`/`input_contract`. Run the committed-research-artifact provenance gate + the no-ids gate LOCALLY against the memo before it lands.

> Decision surfaced to owner (do not self-decide): whether to extend the two hard-coded 3-model registries (`test_feature_contract.py::test_every_bundled_artifact_carries_a_verified_contract`, `test_declared_constant_values.py`) with the two new models if they ship bundled weights + declared constants (the registries are inconsistently applied today — `ReceiverModel`/`GhostOutfieldModel` absent).

## Task 15: Full CI-faithful gate + commit-prep (owner-gated)

- [ ] **Step 1: Run the full CI-faithful suite:** `python -m pytest tests/ -m "not e2e"`, `python -m ruff check silly_kicks/ tests/ scripts/` + `--format --check`, `pyright` bare, the SB360 cascade, `--doctest-modules`. Fix any red. (Full-suite-only gates: metric-contracts completeness, glossary coverage, purity meta, id-scalar discovery, public-api-examples, reflection completeness, SB360 budget, C4 count, AGENTS-md budget.)
- [ ] **Step 2: Draft the ADR(s)** — `docs/superpowers/adrs/ADR-<next-free>-tf46-corner-offence.md` **and** `ADR-<next-free+1>-sportec-corner-crossed-discriminator.md` (numbers assigned now, part-deux-reserved-aware). Uncommitted.
- [ ] **Step 2.5: Rebase onto part-deux (#284, MERGED to `main` 2026-10-18).** With the branch's code commits landed and the tree **CLEAN** (never rebase/merge into a dirty tree, and never while a DGX run is in flight), `git fetch origin main` and rebase (or merge) `origin/main` — now carrying **part-deux's HF-jobs logic + its version bump** — into `tf-46-corner-offence`. Expected conflict surfaces: `silly_kicks/_version.py` (part-deux bumped it; TF-46's bump in Step 3 goes OFF part-deux's new base, not the pre-part-deux 4.137.0), `CHANGELOG.md`, and `scripts/` (part-deux HF-publish jobs vs TF-46's `scripts/_loader_pining.py` `with_velocity`/native-xg edits). Re-run the full CI-faithful suite after the rebase.
- [ ] **Step 3: Pin the version** in `silly_kicks/_version.py` (additive minor, OFF part-deux's post-#284 base) and run `uv lock`. Only now.
- [ ] **Step 4: COMMIT GATE (binding).** Present the full diff + file list to the owner. **STOP. Wait for the owner's explicit per-commit approval.** Nothing — not green tests, not this plan — authorises the commit. On approval, commit as a small number of coherent, fully-tested commits (one feature + its tests green), NOT per-task micro-commits. Do not push without explicit approval. Release is held for part-deux (#284); the owner publishes any model artifacts (ADR-088 `publish_model_with_card`).

---

## Self-review

**Spec coverage:** §2 ships 1–9 → Tasks 2 (id), 4 (windows), 3 (reflection), 6 (encoder), 5 (GMM model), 7 (NMF), 8 (clustering), 9 (chance quality), 12–14 (drivers + memo). §3 forks → encoded in Tasks 4/5/7/the provider handling. §7 providers → Task 2 + Task 4. §8 orientation → Task 3. §9 models → Tasks 5/7. §10 gates → Tasks 1/4/10/11 + the gate edits throughout. §11 drivers → Tasks 12/13. §12 validation incl §12.7 → Task 13/14. §13 risks (corner_id, fit provenance, kick-onset, NMF determinism) → Tasks 2/5/7/12. §14 release/commit gate → Global Constraints + Task 15.

**Placeholder scan:** no "TBD"/"handle edge cases"; each step names the file, the test, and the mechanism. The heavy corpus numbers (Task 14) are produced at run time by design (reported-not-gated), not fabricated here.

**Type consistency:** `RunZoneModel.encode_runs` → `np.ndarray`; `compute_corner_run_encoding` consumes `model=RunZoneModel` and emits `run_{i}{t}`; `CornerRoutineModel.transform` → H coeffs consumed by `compute_corner_routine_features` → `routine_feat_{k}` consumed by `cluster_corner_routines` + `aggregate_chance_quality_by_feature`. Grain `(game_id, corner_id)` consistent across identify/windows/encoding/features/chance-quality. `CornerWindowParams` fields (`init_lead_s`/`target_contact_lead_s`/`target_cap_s`/`outcome_window_s`) used consistently in Task 4.
