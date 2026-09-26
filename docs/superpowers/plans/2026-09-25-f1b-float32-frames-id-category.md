# F1b — float32 tracking-frame storage + id→category + atomic-dedup residue — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan
> task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Halve at-rest tracking-frame memory (float32 coordinates + `category` identifiers), retrain every
bundled frame-geometry model on float32 frames with clean provenance, and clear the ADR-096 atomic-dedup
in-repo residue.

**Architecture:** float32 **storage**, float64 **compute** (upcast at kernel boundaries → drift is
deterministic storage-rounding only; numba keeps its float64 ADR-076 bit-identity). Frames only (SPADL
actions stay float64). id→`category` is value-neutral (routed through `id_compat`, not a model feature).
Two commits: commit-1 = schema migration (no weights); commit-2 = DGX-trained weights (clean
`training_commit`). Each commit has an explicit human-approval gate.

**Tech Stack:** pandas (float32/category dtypes, CoW under pandas 3), numpy, numba (`@njit` kernels),
xgboost/sklearn (bundled models, function-local), the `id_compat`/`reflection`/`schema` seams.

**Spec:** `docs/superpowers/specs/2026-09-25-f1b-float32-frames-id-category-design.md` (APPROVED r2). Reviews:
`D:\Development\_reviews\2026-09-25-sk-f1b-float32-frames-id-category-spec{,-r2}.md`.

## Global Constraints

- **Gold-standard; scope/breaking not a constraint** (owner). Fix at the correct seam, no workarounds.
- **float32 STORAGE, float64 COMPUTE.** Every frame-coord read into a numeric kernel upcasts to float64 at
  the boundary (`np.asarray(x, dtype=np.float64)`). No native-float32 compute. numba signatures stay float64.
- **Frames only.** SPADL action coords stay float64. Event-only models (`xsuccess`/`expected_passing`/
  `match_outcome`/`win_probability`) are NOT retrained.
- **Part B (off-frame provenance) is OUT.** Do not add a `FrameProvenance` object.
- **id→category is value-neutral** → forces no retrain; the retrain is float32-only.
- **Retrain-all-frame-geometry** (no measure-and-skip); the measurement CLASSIFIES `gk_completion`, it does
  not gate whether-to-retrain.
- **2 commits, each with an explicit human-approval gate immediately before it.** No auto-commit. No
  micro-commits — each commit is a fully-tested coherent state, suite green. Present the diff, wait for an
  explicit yes.
- **No version number until commit-prep.** ADR = ADR-106.
- **Bundled retrains: `assert_public_corpus` (public arms) + `require_clean_tree` + `training_commit` =
  commit-1 SHA.** Artifact drivers stamp provenance.
- Branch: one feature branch off `main` (`feat/f1b-float32-frames-id-category`), no worktree.
- Every new `warnings.warn(..., stacklevel=2)`; lint at CI scope; mirror ci.yml's pytest invocation.

---

## PHASE A — COMMIT 1 (schema migration, no weights)

### Task 1: float32 coordinate schema pin (RED first)

**Files:**
- Modify: `silly_kicks/tracking/schema.py` (`TRACKING_FRAMES_COLUMNS`: `x`/`y`/`z` → `"float32"`)
- Test: `tests/test_tracking_schema.py`

**Interfaces:**
- Produces: `TRACKING_FRAMES_COLUMNS["x"|"y"|"z"] == "float32"`, consumed by every producer (Task 2) and the
  enumeration pin.

- [ ] **Step 1: Write the failing test** — assert the base coordinate dtypes are float32 and NA-safe.

```python
def test_frame_coordinate_dtypes_are_float32():
    from silly_kicks.tracking.schema import TRACKING_FRAMES_COLUMNS
    for c in ("x", "y", "z"):
        assert TRACKING_FRAMES_COLUMNS[c] == "float32", c
```

- [ ] **Step 2: Run — expect FAIL** (`float64 != float32`).
- [ ] **Step 3: Implement** — set `x`/`y`/`z` to `"float32"` in `TRACKING_FRAMES_COLUMNS`.
- [ ] **Step 4: Extend the enumeration pin** — the existing complete-by-enumeration test over every
  `*_TRACKING_FRAMES_COLUMNS` asserts each variant's coord dtypes are float32 (variants inherit the base;
  update the expected map). Keep the ADR-058 NA-preservation assertions.
- [ ] **Step 5: Run the schema test file** — expect PASS (producers still to migrate → other suites RED until
  Task 2; that is expected mid-phase).

### Task 2: Producer float32 casts

**Files:**
- Modify: `silly_kicks/tracking/preprocess/_velocity.py`, `.../preprocess/_smoothing.py` (emit `vx`/`vy`/
  `speed`/`x_smoothed`/`y_smoothed` as float32 at the end of `derive_velocities`/`smooth_frames`)
- Modify: native adapters `silly_kicks/tracking/sportec.py`, `.../gradientsports.py`, the SkillCorner/Metrica
  native bronze→frame builders, `silly_kicks/tracking/kloppy.py` (cast coords to float32 on output)
- Modify: `silly_kicks/tracking/_snapshot.py` `_cast_to_declared_schema` (the `snapshot_to_tracking_frames`
  seam; coord columns → float32, per-column, the ADR-058 idiom)
- Test: `tests/tracking/test_preprocess.py`, provider frame tests, `tests/providers/statsbomb/...`

**Interfaces:**
- Consumes: `TRACKING_FRAMES_COLUMNS` (Task 1).
- Produces: every `convert_to_frames`/`snapshot_to_tracking_frames`/`derive_velocities`/`smooth_frames`
  output carries float32 coord + kinematic columns.

- [ ] **Step 1: Write failing tests** — one per producer: output frame's `x/y/z` dtype is float32, and
  post-`derive_velocities` `vx/vy/speed`, post-`smooth_frames` `x_smoothed/y_smoothed` are float32.

```python
def test_derive_velocities_emits_float32(sample_frames):
    out = derive_velocities(sample_frames)
    for c in ("vx", "vy", "speed"):
        assert out[c].dtype == "float32", c
```

- [ ] **Step 2: Run — expect FAIL.**
- [ ] **Step 3: Implement** — cast at each producer's output seam (`.astype("float32")` on the coord/kinematic
  columns; NA-safe — float32 holds NaN). For `_cast_to_declared_schema`, coord columns follow the declared
  float32 (ids handled in Task 4).
- [ ] **Step 4: Run all producer tests — expect PASS.**
- [ ] **Step 5: Guard reassignment** — confirm no producer reassigns a float64 result back onto a coord column
  (the schema-pin test catches a reverted dtype; add an assertion in the highest-traffic producer test).

### Task 3: Compute-boundary float64 upcasts (reads) + storage-dtype write-backs + AST gate

**Files:**
- Modify: every frame-coord-consuming kernel/extractor (spec §4.1 list): `pitch_control/_spearman.py`
  (`_extract_frame_players`), `_spearman_batch.py`, `_kernels.py`, `_gk_influence.py`, `_player_influence.py`,
  `_cover_shadows.py`, `_das.py`, `_obso.py`, `_pausa.py`, `_space_creation.py`, `_ghost_gk.py`,
  `_ghost_outfield.py`, `_receiver.py`, `_shot_goalmouth.py`, `_packing.py`, `_defensive_line.py`,
  `_structural_pass.py`, `_shape_graph.py`, `_line_breaking.py`, `_off_ball_runs.py`, `_ball_carrier.py`,
  `_elastic_sync.py`
- Modify (the **write-back-cast class** — every site that WRITES a float64 result back onto a float32 coord
  column via a masked/scalar setitem, which pandas 3 rejects with `LossySetitemError` where pandas 2 silently
  upcast, ADR-057): `silly_kicks/reflection.py` (`reflect_columns` point_x/point_y/vector_x/vector_y),
  `silly_kicks/tracking/direction.py` (flip block x/y/vx/vy), `silly_kicks/tracking/utils.py`
  (`speed` fill), `silly_kicks/positioning/_optimizer.py` (working frame cast to float64 at entry),
  `silly_kicks/gkdv/_engine.py`, `silly_kicks/gkdv/_probe.py`, `silly_kicks/restdefense/_counterfactual.py`,
  `silly_kicks/restdefense/_probe.py`, `silly_kicks/tracking/_model_eval.py`
- Create: `tests/tracking/test_frame_coord_upcast_gate.py`

**Interfaces:**
- Consumes: float32 frames (Tasks 1–2).
- Produces: every kernel READS frame coords as float64 → drift = storage-rounding only; numba kernels keep
  float64 signatures (ADR-076 bit-identity re-anchored on float64 inputs). Every WRITE-BACK of a float64
  result onto a coord column casts to the column's storage dtype (`.astype(str(col.dtype))` for numpy-array
  RHS, `.astype(col.dtype)` for a Series RHS), so a float32 column stays float32 and pandas 3 does not raise.

- [ ] **Step 1: Write the AST gate (RED first)** — enumerate modules that read a frame coordinate column
  (`frame[...][["x","y",...]]` / `.to_numpy()` on coord cols) and assert each upcasts to float64 at the read,
  or is listed in `_UPCAST_EXEMPT` with a reason. Land it RED (before the upcasts) and observe it fail.

```python
# tests/tracking/test_frame_coord_upcast_gate.py — enumerate coord-reading kernels; each must
# call np.asarray(..., dtype=np.float64) / .astype(np.float64) on the coord slice, or be _UPCAST_EXEMPT.
```

- [ ] **Step 2: Run — expect FAIL** (kernels don't yet upcast).
- [ ] **Step 3: Implement upcasts** — at each kernel's frame-coord read: `pos = np.asarray(cols, dtype=np.float64)`.
  numba kernel callers upcast before the `@njit` call so the kernel receives float64 (no float32
  specialization).
- [ ] **Step 4: Re-anchor the ghost-GK numba parity gate** — run the numpy/numba leaf-walk parity
  (`SILLY_KICKS_GHOST_FORCE_NUMPY` both legs) on float64-upcast inputs; assert `np.array_equal` still holds
  (ADR-076).
- [ ] **Step 5: Implement the write-back-cast class** — at every masked/scalar coord setitem whose RHS is a
  float64 array/Series, cast the RHS back to the column's storage dtype before assignment
  (`out.loc[m, col] = rhs.astype(str(out[col].dtype))`; Series RHS uses `.astype(out[col].dtype)`; a working
  compute frame — `positioning/_optimizer.py` — is cast to float64 once at entry so its in-place `.at[...]`
  scalar writes never re-touch a float32 column). This is the reverse boundary of the read upcast: reads widen
  float32→float64, write-backs narrow float64→storage.
- [ ] **Step 6: Red-green a storage-dtype write-back guard** — add a test that a float32-storage coord column
  survives a `reflect_columns` round-trip as float32 (red on pandas 2 pre-fix via an explicit dtype assert;
  green after). The **forward-guard for the write-back class is the pandas-3 CI leg** (ADR-057 pandas-major
  span: CI runs pandas 3 on 3.11/3.12/windows-3.12), which raises `LossySetitemError` on any NEW un-cast
  write-back — a bespoke write-back AST gate is a brittle narrow duplicate of that leg and is NOT added
  (F1B-PLAN-06 decision; recorded in ADR-106). Confirm the pandas-3 leg exercises every write-back site's
  consuming test.
- [ ] **Step 7: Run the gate + kernel suites (local pandas 2 AND a pandas-3 venv) — expect PASS.**

### Task 4: id→category (team_id ONLY) + id_compat extension

**Scope (option A, owner-approved 2026-09-25):** `team_id` → `category`; **`player_id` stays Int64/object**
(dynamic post-build — `_das.py` `.loc[ball_mask,"player_id"]="ball"` masked new-category setitem crashes a
categorical; ADR-103 rule). The original "player_id + team_id" was narrowed; the spec as first drafted
erred. See spec §4.2.

**Files:**
- Modify: `silly_kicks/tracking/schema.py` (`team_id` → `"category"`; `player_id` unchanged Int64/object),
  the variant constants (drop the kloppy `team_id` override so it inherits base `category`), producer casts
  (Task 2 sites) and `_cast_to_declared_schema` (numeric `team_id` → Int64-first-then-category over the
  caller's underlying dtype)
- Modify: `silly_kicks/id_compat.py` (`canonical_id`/`canonical_id_series` decategorize first; `ids_equal`/
  `ids_differ`/`ids_match`/`same_id`/`align_join_keys`/`restore_id_dtype` + `_raw_comparable` handle category)
- Modify: `tests/tracking/test_id_dtype_invariance.py` (+ category axis), `tests/invariants/conftest_id_scalar.py`
  / `test_public_id_scalar_registry.py` (+ category axis), `tests/test_tracking_schema.py` (id pins)
- Test: `tests/test_id_compat.py` (or wherever id_compat is tested)

**Interfaces:**
- Consumes: category-dtyped frame ids.
- Produces: id_compat treats category as its underlying dtype (value-neutral); frames carry category ids.

- [ ] **Step 1: Write failing tests** — (a) `canonical_id_series` on a category-of-Int64 with a ball-row NA
  returns the same canonical values as the Int64 version; (b) `ids_match`/`ids_equal` on category vs int/str/
  float scalars match the underlying-dtype behavior; (c) schema pin: id columns are `category`, NA-preserving.

```python
def test_canonical_id_series_category_matches_int64():
    s_int = pd.Series([1, 2, pd.NA], dtype="Int64")
    s_cat = s_int.astype("category")
    pd.testing.assert_series_equal(
        canonical_id_series(s_cat).reset_index(drop=True),
        canonical_id_series(s_int).reset_index(drop=True),
        check_dtype=False,
    )
```

- [ ] **Step 2: Run — expect FAIL.**
- [ ] **Step 3: Implement** — decategorize at the top of `canonical_id`/`canonical_id_series`
  (`s = s.astype(s.cat.categories.dtype)` when `isinstance(s.dtype, CategoricalDtype)`); route the other
  comparators + `_raw_comparable` through the same decategorize; schema/producer id casts to category.
- [ ] **Step 4: Extend the ADR-019 gates** — add a category axis to the `add_*` dtype-invariance gate and the
  `PUBLIC_ID_SCALAR_ENTRIES` registry (category-of-Int64 frames × {int, str, float} scalars). Land the new
  axis RED against un-migrated id_compat, then green.
- [ ] **Step 5: Run the id suites + full tracking suite — expect PASS** (value-neutral: outputs unchanged
  except id dtype).

### Task 5: Re-materialize goldens (float32-input) + SB360 audit

**Files:**
- Modify: committed tracking-feature goldens (pitch-control/threat/obso/pausa/space-creation/etc.) + bundled
  model-output goldens (float32-input/float64-weights, §4.5 first pass)
- Modify: `tests/sb360/` registry (re-derive observation tokens; review adjudication)

- [ ] **Step 1: Run the golden suites — observe RED** (float32 storage-rounding shifts pinned values ~1e-5).
- [ ] **Step 2: Regenerate tracking-feature goldens** via their committed regeneration scripts; record
  before/after max|Δ| in the commit message.
- [ ] **Step 3: Regenerate model-output goldens** (float32-input, weights still float64) — these regenerate
  AGAIN in Task 12 after retrain; note the double-regen is intended.
- [ ] **Step 4: Re-derive the SB360 audit registry** (`tests/sb360/_regenerate.py` + `_adjudicate.py`);
  review each `identical`→`differs` flip (float32 numeric change is legitimate, adjudicate `differs_by_design`
  where the value merely shifted within storage-eps). Round-trip verify byte-identical.
- [ ] **Step 5: Run the golden + SB360 suites — expect PASS.**

### Task 6: Commit-1 model-load CI leg (F1B-SPEC-01)

**Files:**
- Create: `tests/trained_models/test_bundled_models_load_on_float32_frames.py`

**Interfaces:**
- Consumes: unchanged float64 probe builders (NOT touched in commit-1) + float32 frames.

- [ ] **Step 1: Write the test** — for every bundled model, `load()` (chirality + feature-contract pass on
  its unchanged float64 probe) and `predict`/`serve` finite output on a float32 frame.

```python
@pytest.mark.parametrize("loader", ALL_BUNDLED_MODEL_LOADERS)
def test_bundled_model_loads_and_finite_predicts_on_float32(loader, float32_frame):
    model = loader()                      # float64 probe fingerprints unchanged → load passes
    out = model_predict(model, float32_frame)
    assert np.all(np.isfinite(np.asarray(out, dtype=float)))
```

- [ ] **Step 2: Run — expect PASS** (proves un-retrained models are clean before the retrain lands). If a
  model's `load()` RAISES here, a probe builder was wrongly routed through the float32 schema → fix that
  builder to stay float64 (the F1B-SPEC-01 red-gate trap).

### Task 7: Bundled-weights-dir classification gate (F1B-SPEC-02)

**Files:**
- Create: `tests/trained_models/test_bundled_weights_classification.py`

- [ ] **Step 1: Write the gate** — enumerate every bundled weights dir (mechanical `silly_kicks/**/weights` +
  `_*_weights` glob), require each classified in a `WEIGHTS_CLASSIFICATION` map as `"frame_geometry"`,
  `"event_only"`, or `"out"` (with reason); `_UNDERIVABLE` asserted empty; a dir absent from the map fails.

```python
def test_every_bundled_weights_dir_is_classified():
    dirs = discover_bundled_weights_dirs()          # mechanical glob
    assert set(dirs) == set(WEIGHTS_CLASSIFICATION), (
        set(dirs) ^ set(WEIGHTS_CLASSIFICATION))
    assert not _UNDERIVABLE
```

- [ ] **Step 2: Populate the map** — frame_geometry: xshot×2, xcross×2, ghost_gk×4, ghost_outfield×2,
  receiver; event_only: expected_passing, match_outcome, win_probability, xsuccess; out: xtgk/_retention (#4);
  gk_completion = classified in Task 10's measurement (default `frame_geometry` unless measurement proves
  action-only → then `event_only` with the measured reason).
- [ ] **Step 3: Run — expect PASS.**

### Task 8: Atomic-dedup residue (#2)

**Files:**
- Modify: committed atomic fixtures/goldens encoding the action-type list/count/index-dependent values
- Test: the atomic test surface (assert `len(atomicconfig.actiontypes) == 32`)

- [ ] **Step 1: Enumerate** — grep the atomic test surface + `tests/datasets` + committed atomic parquet
  goldens for the pre-dedup count/ordering or an action-type-index-dependent value.
- [ ] **Step 2: Write/repair the assertion** — a test pins `len(atomicconfig.actiontypes) == 32` and that no
  test still asserts the pre-dedup count.
- [ ] **Step 3: Re-materialize** each affected atomic fixture/golden against the 32-type config (via its
  regeneration path; verify byte-reproducible).
- [ ] **Step 4: Run the atomic suite — expect PASS.**

### Task 9: Commit-1 docs, numbering, and the HUMAN-APPROVAL GATE

**Files:**
- Create: `docs/superpowers/adrs/ADR-106-f1b-float32-frames-and-id-category.md`
- Modify: `CHANGELOG.md`, `AGENTS.md` (Key-conventions bullet ≤600 chars + `docs/context/` pointer),
  `docs/context/tracking-features.md` (WHY), `TODO.md`, `silly_kicks/_version.py` (at commit-prep)

- [ ] **Step 1: Write ADR-106** — the float32-storage/float64-compute decision, Part-B-dropped rationale,
  retrain-all policy, id→category value-neutral, the F1B-SPEC-01/02 gates, Hyrum block.
- [ ] **Step 2: CHANGELOG + AGENTS.md bullet (≤600, run `test_agents_md_budget`) + docs/context WHY + TODO
  NOW-block.** Version/PR at commit-prep.
- [ ] **Step 3: Run the FULL non-e2e suite** (`python -m pytest tests/ -m "not e2e" -p no:randomly -q`) +
  ruff check/format --check at CI scope + bare pyright. All green.
- [ ] **Step 4: /final-review** (catches full-suite-only gate misses per the new-sibling-gates class).
- [ ] **Step 5: HUMAN-APPROVAL GATE — present the full commit-1 diff/file list, STOP, await explicit owner
  approval. Only then commit** commit-1 (schema migration, no weights).

---

## PHASE B — COMMIT 2 (DGX-trained weights, clean provenance)

### Task 10: Feature-delta measurement + gk_completion classification (DGX)

**Files:**
- Create: `scripts/measure_f1b_feature_delta.py` (argparse; `require_clean_tree`; stamps provenance)
- Create: `docs/research/f1b_float32/` (metrics.json + findings.md)

- [ ] **Step 1: Write the driver** — for each frame-geometry model, on a fixed corpus slice, compute the
  feature vector on float64 frames vs float32-stored frames; report max/mean |Δ| per feature. For
  gk_completion, additionally record WHICH coordinate source it reads (frame vs action).
- [ ] **Step 2: Run on the DGX** (clean tree). Confirm: drift bounded ~storage-eps; drift > `atol=1e-6`
  (justifies retrain); gk_completion classification (frame-geometry → retrain; action-only → exclude + set
  Task 7 map to `event_only` with the measured reason).
- [ ] **Step 3: Commit the artifact** (in commit-2) with `run_commit`/`run_tree_dirty` provenance.

### Task 11: Cast probe builders to float32 + retrain (DGX)

**Files:**
- Modify: each frame-geometry model module's probe builder (`canonical_probe_frame` / feature-contract probe
  frame → float32 explicitly, NOT via the schema)
- Regenerate: `_xshot_weights`×2, `_xcross_weights`×2, `_ghost_gk_weights`×4, `_ghost_outfield_weights`×2,
  `_receiver_weights`, `_gk_completion_weights`×2 (if Task 10 classifies frame-geometry) — npz/booster +
  metadata + chirality/feature-contract fingerprints + `SHA256SUMS`

- [ ] **Step 1: Cast probe builders to float32** in each model module; regenerate the stored fingerprints
  against the float32 probes.
- [ ] **Step 2: Retrain each model on float32 frames** (the same public/owner corpus it currently uses;
  `assert_public_corpus` for bundled public arms; `require_clean_tree`; `training_commit` = commit-1 SHA), via
  each model's existing `train_*.py` (`publish_model_with_card` unchanged where publish applies).
- [ ] **Step 3: Verify `load()` passes** on the regenerated float32 probes (chirality + feature-contract),
  and the Task 6 leg now runs against retrained models (float32 probe + float32 frame).

### Task 12: Second-pass golden regen + wheel verification

**Files:**
- Modify: bundled model-output goldens (float32-weights, §4.5 second pass)
- Modify: `.gitattributes` (binary pin for any byte-exact-load weights dir that lacks one)

- [ ] **Step 1: Regenerate model-output goldens** to the float32-input/float32-weights values.
- [ ] **Step 2: Run the golden + load + Task-6 + Task-7 gates — expect PASS.**
- [ ] **Step 3: Wheel verification** — build the wheel; verify each regenerated weights dir is bundled (npz/
  booster + SHA256SUMS, byte-exact); `.gitattributes ... binary` pin present for byte-exact-load dirs.

### Task 13: Commit-2 docs + the HUMAN-APPROVAL GATE

- [ ] **Step 1: CHANGELOG note** for the weights commit (measurement artifact reference, retrained set,
  clean `training_commit`).
- [ ] **Step 2: Run the FULL non-e2e suite + lint + pyright — all green.** /final-review.
- [ ] **Step 3: HUMAN-APPROVAL GATE — present the full commit-2 diff/file list, STOP, await explicit owner
  approval. Only then commit** commit-2 (DGX-trained weights).

---

## Self-review

- **Spec coverage:** §4.1 float32 storage/compute (read-boundary upcasts AND the reverse-boundary
  write-back-cast class, guarded by the pandas-3 CI leg, F1B-PLAN-05/06) → Tasks 1–3; §4.2 id→category → Task 4; §4.3 retrain +
  measurement + gk_completion classification → Tasks 10–11; §4.4 atomic residue → Task 8; §4.5 goldens →
  Tasks 5 + 12; §5 Hyrum → Task 9 ADR/CHANGELOG; §6 gates → Tasks 3/6/7 + full suite; §7 2-commit + human
  gates → Tasks 9 + 13. F1B-SPEC-01/02/03 → Tasks 6/7/10. All covered.
- **Placeholder scan:** none (version/PR at commit-prep is the standing rule, not a TBD; ADR-106 named).
- **Type consistency:** `canonical_id_series`, `WEIGHTS_CLASSIFICATION`, `_UPCAST_EXEMPT`, `_UNDERIVABLE`
  referenced consistently; the retrain set matches the spec's 11(+gk_completion) exactly.
- **Commit discipline:** the human-approval gate is the LAST step of Task 9 and Task 13; no task commits mid
  phase; no micro-commits.

## Execution Handoff

Plan complete. Recommend **inline execution** (executing-plans) with checkpoints at each phase's human gate,
matching the owner's established flow (the DGX retrain in Phase B is owner-run). Ready for review.
