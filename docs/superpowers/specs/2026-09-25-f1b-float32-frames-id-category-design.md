# F1b — float32 tracking-frame storage + id→category + atomic-dedup residue (design)

**Status:** Proposed (awaiting review)
**Decision ADR:** ADR-106 (to be written)
**Version / PR:** assigned at commit-prep (do not hardcode; next-free after 4.127.0 / PR-S199 / ADR-105)
**Supersedes / consumes:** the ADR-103 deferred "F1b" block (float32 + id→category + off-frame provenance) and the ADR-096 atomic-`interception`-dedup "retrain owed" note.

---

## 1. Context

ADR-103 (4.125.0) shipped the *no-retrain* subset of the tracking frame-memory work (`category` on the four
static low-cardinality columns, bounded-LRU `PitchControlCache`, the batched pitch-control seam). ADR-105
(4.127.0) shipped the *no-retrain* CPU/streaming work (vectorized spearman kernel + scorer batching + DAS
cost guardrail). Both explicitly deferred the **retrain-carrying** memory work as "F1b".

F1b is the last silly-kicks-internal optimization that forces a bundled-model retrain. This cycle clears it,
bundled with the one remaining in-repo residue of the ADR-096 atomic-`interception` dedup.

Owner has set: gold-standard / best-practice; scope and breaking changes are not a constraint. The owner
selected item **#1 (F1b)** and item **#2 (atomic-dedup residue)**; items #3 (GS-dribble re-fit) and #4
(xt-gk-v2 ρ retrain) are explicitly **out** of this cycle.

## 2. Goals

- Halve the at-rest memory of a tracking frame table's dominant columns (coordinates + identifiers), which at
  ~0.95–3.2 M rows/match dominate frame RAM.
- Do it to gold-standard: deterministic, exact-as-designed, train/serve-consistent, fully re-provenanced.
- Clear the ADR-096 atomic residue so the repo's committed atomic fixtures/goldens reflect the shipped
  `len(atomicconfig.actiontypes) == 32`.

## 3. Scope

### In scope
- **Part C-1 — float32 coordinate/kinematic storage** on tracking frames: `x`, `y`, `z`, `vx`, `vy`,
  `speed`, `x_smoothed`, `y_smoothed` → `float32`.
- **Part C-2 — id→`category`** on tracking frames: **`team_id` → `category`** (over its existing
  `Int64`/`object` underlying dtype), with an `id_compat` extension. **`player_id` stays Int64/object**
  (dynamic post-build; ADR-103 rule) — team_id-only, option A, owner-approved 2026-09-25 (see §4.2).
- **Bundled-model retrain** (commit-2) of every model that trains on tracking-frame geometry, on float32
  frames, with clean `training_commit`, regenerated chirality + feature-contract probes, and a documented
  feature-delta measurement.
- **#2 — atomic-dedup residue**: re-materialize / correct every committed atomic fixture and golden that
  encodes the pre-dedup action-type count or ordering; confirm the atomic test surface asserts 32.
- Re-materialize every committed tracking-feature golden whose pinned numeric values move under
  storage-rounding.

### Out of scope (explicit, owner-set)
- **Part B — off-frame `FrameProvenance`.** Dropped. Its target (constant per-row provenance strings
  `source_provider` / `is_goalkeeper_source` / `_preprocessed_with`) was **already captured by ADR-103's
  `category` treatment** of those exact columns; the off-frame move buys ~zero incremental memory while
  adding a companion-object API ripple (`pandas.attrs` does not propagate — every `frames` consumer would
  need a paired `provenance` arg) plus the xt-gk per-row variant-read behavior risk. YAGNI + Chesterton's
  Fence. Not done.
- **float32 on SPADL action coordinates.** Frames only. The memory is in frames (millions of rows/match) not
  events (thousands); float32 on the canonical event-coordinate contract would degrade every event metric's
  precision for negligible memory. Action coords stay `float64` → **event-only models
  (`xsuccess` / `expected_passing` / `match_outcome` / `win_probability`) are untouched, no retrain.**
- **#3 GS-dribble re-fit** and **#4 xt-gk-v2 ρ retrain** (owner-deferred; ρ's model is xt-gk-v2, quarantined).
- **Consumer / lakehouse retrains** (AtomicVAEP, VAEP, xT are consumer-fit — silly-kicks bundles no VAEP/xT
  weights). Out by the owner's "downstream irrelevant" rule; noted for the CHANGELOG Hyrum block only.

## 4. Design

### 4.1 float32 storage, float64 compute (the load-bearing decision)

**Storage is float32; computation upcasts to float64 at the compute boundary.** The two are separated on
purpose:

- Storage rounding float64→float32 introduces a deterministic, bounded error of ~1e-5 m on pitch
  coordinates (float32 has ~7 significant figures; on a ≤105 m coordinate the absolute step is ~6e-6 m,
  i.e. ~1e-5 m worst case) — far below tracking-sensor precision (~cm), so **zero information loss** at the
  domain level, but **above the trained-model feature contract's `atol=1e-6`**, so it is a real
  feature-value change and a real retrain trigger.
- Every downstream compute path (pitch-control TTI, geometry extractors, numba leaf/PPCF kernels) **upcasts
  the frame coordinate slice to `float64` at its boundary** (`np.asarray(pos, dtype=np.float64)`), so:
  1. The numba `@njit` kernels keep their existing `float64` numeric behavior — no float32 signature
     specialization, no float32-accumulation variance, the ADR-076 bit-identity contract is re-anchored
     against float64 inputs.
  2. The only numeric drift anywhere is the **storage rounding**, which is deterministic and reproducible —
     not a moving target that depends on which ops ran in float32.

This means the retrain is driven by a single, well-characterized cause (storage rounding), not by
float32-compute noise. It also means we do **not** chase a CPU win from float32 SIMD in this cycle (that
would be a separate, measured decision; native-float32 compute is a non-goal here).

**Schema + producer changes:**
- `tracking/schema.py`: `TRACKING_FRAMES_COLUMNS` sets `x`/`y`/`z` to `float32` (base pin).
- Every provider variant constant (`*_TRACKING_FRAMES_COLUMNS` — kloppy/sportec/skillcorner/metrica/
  gradientsports) inherits the float32 coordinate dtypes; the `test_tracking_schema.py`
  complete-by-enumeration pin is updated to float32 and asserts NA-preservation is unaffected (ids handled
  in 4.2).
- `preprocess` emits `vx`/`vy`/`speed`/`x_smoothed`/`y_smoothed` as `float32` (these are added by preprocess,
  not in `TRACKING_FRAMES_COLUMNS`; the Savitzky-Golay / EMA / SG-derivative outputs cast to float32 at the
  end of `derive_velocities` / `smooth_frames`).
- `snapshot_to_tracking_frames._cast_to_declared_schema` casts coordinate columns to float32 (per-column, the
  ADR-058 idiom).
- Native `convert_to_frames` adapters + the kloppy gateway cast coords to float32 on output.

**Compute-boundary upcast sites** (each reads frame coords into a numeric kernel — upcast to float64 there):
`pitch_control/_spearman.py` (`_extract_frame_players`), `pitch_control/_spearman_batch.py`, `_kernels.py`
context/defensive-line/team-shape extractors, `_gk_influence.py`, `_player_influence.py`, `_cover_shadows.py`,
`_das.py`, `_obso.py`, `_pausa.py`, `_space_creation.py`, `_ghost_gk.py` / `_ghost_outfield.py` extractors,
`_receiver.py`, `_shot_goalmouth.py`, `_packing.py`, `_defensive_line.py`, `_structural_pass.py`,
`_shape_graph.py`, `_line_breaking.py`, `_off_ball_runs.py`, `_ball_carrier.py`, `_elastic_sync.py`. The exact
set is derived by AST (any module that reads a frame coordinate column into numpy) and pinned by an
enumeration gate so a new coord-consuming kernel must upcast or be listed exempt-with-reason (ADR-056 idiom).

### 4.2 id→`category` — `team_id` ONLY (option A; owner-approved 2026-09-25)

- **`team_id` on tracking frames → `category`**, over its existing underlying dtype (`Int64` for the
  numeric-id providers, `object` for the kloppy family / SkillCorner string ids). `category` supports NA
  (the ball row's NA-id is preserved as a category-NA; ADR-058 base pin updated). `team_id` is
  static/set-once (~2 distinct/match) — its only post-build touch is a masked `=None`→NaN, allowed on a
  categorical.
- **`player_id` STAYS Int64/object — NOT category.** It is MUTATED post-build, and `category` is not
  transparent to a new category at setitem (the ADR-103 dynamic-column rule): `_das.py` does
  `out.loc[ball_mask,"player_id"]="ball"` (a masked new-category setitem → raises on a categorical),
  `_run_values` reassigns `player_id` to Int64, and the keeper/actor identity bridges whole-column-replace
  it. **This narrows the original "player_id + team_id" scope (the spec as first drafted erred): a
  categorical player_id crashes `_das`.** `player_id → category` is reconsiderable once DAS is
  reimplemented natively (that removes the `_das` `"ball"` sentinel blocker — the next cycle). Owner
  approved team_id-only on 2026-09-25 (recorded here + in ADR-106 + reviewer finding F1B-IMPL-05).
- **Value-neutral: ids are not model features, so this part forces NO retrain.** It is bundled here only
  because it is a tracking-frame schema change that co-migrates with float32.
- **`id_compat` extension (ADR-019):** `canonical_id` / `canonical_id_series` decategorize first
  (`.astype(underlying)` via `Series.cat` / `.cat.categories`), so the canonical (integral-float-collapse)
  logic runs on the underlying values. `ids_equal` / `ids_differ` / `ids_match` / `same_id` /
  `align_join_keys` / `restore_id_dtype` and the `_raw_comparable` content-probe all decategorize a category
  operand before their existing paths. `group_rows` already carries `observed=True` (4.127.0); its canonical
  collision detection runs on decategorized keys.
- **Gate extension:** the ADR-019 `add_*` dtype-invariance gate and the `PUBLIC_ID_SCALAR_ENTRIES` registry
  gain a `category` axis (category-of-Int64 frames × int/str/float scalars), so an id comparison that breaks
  on category input fails CI. `test_tracking_schema.py` enumeration asserts the id columns are `category` and
  NA-preserving.

### 4.3 Retrain plan (commit-2)

**Retrain-all-frame-geometry, do not measure-and-skip.** Serving now runs on float32-stored frames, so every
model that trains on tracking-frame geometry is retrained on float32-stored frames — train/serve dtype- and
value-consistency, no train(float64)/serve(float32) skew, one clean float32 `training_commit` per model.

**Confirmed frame-geometry set (11):** `_xshot_weights` {default, position_only}, `_xcross_weights`
{default, position_only}, `_ghost_gk_weights` {default, position_only, sweeper, sweeper_position_only},
`_ghost_outfield_weights` {default, position_only}, `_receiver_weights` {default}.

**Classify-then-include (NOT measure-and-skip):** `_gk_completion_weights` {default, skillcorner}. The open
question is a **classification, not a skip**: does it read tracking-frame geometry, or only action
coordinates (float64) via xt-gk-v1 + `resolve_gk_geometry`? The measurement step (below) answers *that* — it
does not apply an atol threshold to decide whether to retrain. If it reads **any** tracking-frame coordinate
it is frame-geometry and retrains **unconditionally** (the retrain-all-frame-geometry policy). Only if it
reads solely action coords is it excluded, genuinely unaffected like the event-only models.

**Explicitly excluded** (event-only, float64 actions — no frame geometry): `expected_passing` (PassCompletion),
`match_outcome` (DependenceModel ρ), `win_probability`, `xsuccess`. And **`xtgk/_retention_weights` (ρ)** —
item #4, out of cycle.

**Per-model retrain procedure** (each, on the DGX, from the SAME public/owner corpus it currently uses,
`assert_public_corpus`-gated for the bundled public arms, clean tree via `require_clean_tree`):
1. Rebuild the training frames with float32 storage.
2. Re-fit; regenerate the artifact (npz / booster-JSON) + `metadata.json` + `SHA256SUMS`.
3. In **commit-2 only**, cast the probe BUILDERS (`canonical_probe_frame` / the feature-contract probe
   frame) to float32 **explicitly in the model module** — NOT by routing them through the float32
   `TRACKING_FRAMES_COLUMNS` schema — then regenerate the stored chirality + feature-contract fingerprints
   against the float32 probes. **The probe builders are code-built and schema-INDEPENDENT** (verified), so
   commit-1's schema change does not touch them: an un-retrained (float64) model's float64 probe still
   matches its stored float64 fingerprint, and `load()` therefore passes in commit-1 (see §7). An
   implementer who routes a probe through the float32 schema in commit-1 turns that model's load gate red —
   the builders must stay float64 until commit-2 casts them explicitly.
4. Record `training_commit` = the commit-1 SHA (clean tree).

**Feature-delta measurement artifact** (`docs/research/f1b_float32/`, provenance-stamped): for each model,
the max/mean |Δ| per feature between the float64-frame and float32-frame feature vectors on a fixed corpus
slice — to (a) confirm the drift is bounded at ~storage-eps (sanity), (b) confirm it exceeds `atol=1e-6`
(justifies the retrain), (c) settle the `gk_completion` include/exclude question with a number, not a guess
(measure the gating quantity). The measurement documents the change; it does **not** gate whether to retrain
(retrain-all-frame-geometry is the chosen policy for consistency).

### 4.4 Atomic-dedup residue (#2)

The ADR-096 config change already shipped (`len(atomicconfig.actiontypes) == 32`, verified). The residue is
in-repo only (the AtomicVAEP model retrain is consumer-side, out):
- Enumerate every committed atomic fixture / golden that encodes the action-type list, count, or an
  action-type-index-dependent value (grep the atomic test surface + `tests/datasets` + any committed atomic
  parquet golden).
- Re-materialize each against the 32-type config; confirm the atomic test surface asserts 32 (and that no
  test still asserts the pre-dedup count).
- No behavior change beyond the already-shipped config; this is fixture/golden hygiene the dedup deferred.

### 4.5 Re-materialize (commit-1)

- Tracking-feature goldens pinning pitch-control / threat / obso / pausa / space-creation / etc. numeric
  values regenerate under float32 storage-rounding (bounded ~1e-5 drift). Each regeneration is committed with
  the before/after max|Δ| noted.
- **Bundled-MODEL-output goldens regenerate TWICE, once per commit, and each commit stays green.** In
  commit-1 a model's *input* frames become float32 while its *weights* are still float64, so its output
  shifts ~1e-5 → the golden regenerates to the float32-input/float64-weights value. In commit-2 the model is
  retrained, so the same golden regenerates again to the float32-input/float32-weights value. This
  double-regen is the intended, honest consequence of splitting inputs (commit-1) from weights (commit-2);
  the intermediate state never ships (one PR, one tag — the TF-53/57/61 two-phase precedent).
- The SB360 audit registry (ADR-053) is re-derived — float32 changes numeric outputs, so the machine
  observation tokens may shift (`identical`→`differs`); the human adjudication is reviewed, not auto-locked.
- Chirality / feature-contract probes for the 11(+) retrained models regenerate in commit-2 (§4.3), not
  commit-1 (§4.3 step 3: the probe builders stay float64 in commit-1 so un-retrained models load clean).

## 5. Breaking changes / Hyrum

Scope/breaking is not a constraint (owner). Documented for the CHANGELOG Hyrum block:
- `TRACKING_FRAMES_COLUMNS` coordinate dtypes float64→float32 and id dtypes Int64/object→category: any
  consumer asserting the old frame dtypes breaks. This is the intended change.
- Bundled frame-geometry model weights change (new numeric outputs within storage-eps): downstream consumers
  that persist those outputs re-materialize; downstream VAEP/xT fit on tracking features is a **consumer**
  retrain trigger (out of this cycle, noted only).
- `id_compat` gains category handling (additive; no break).

## 6. Testing / CI gates

- `test_tracking_schema.py`: float32 coord pins + category id pins + NA-preservation (base + enumeration
  over all `*_TRACKING_FRAMES_COLUMNS`).
- ADR-019: id-dtype-invariance `add_*` gate + `PUBLIC_ID_SCALAR_ENTRIES` gain a category axis.
- ADR-033 purity, ADR-053 SB360 audit re-derivation, aggregator column-liveness — all re-run; goldens
  regenerated where storage-rounding moves a pinned value.
- Trained-model `load()` gates (chirality + feature-contract) pass against the regenerated float32 probes;
  a bundled-weights load+finite-predict CI leg per model.
- New: the compute-boundary upcast enumeration gate (§4.1) — every frame-coord-consuming kernel upcasts or
  is exempt-with-reason.
- New (F1B-SPEC-02 forward guard): a **bundled-weights-dir classification gate** — the upcast gate covers
  compute SITES but not the MODEL SET, so this gate enumerates every bundled weights dir (the mechanical
  `silly_kicks/**/weights` + `_*_weights` listing) and requires each classified `frame_geometry` (retrained
  on float32) or `event_only`/`out` (exempt, with reason). A FUTURE frame-geometry model that ships
  un-retrained on float32 frames — a silent train/serve skew — fails CI (ADR-056 complete-by-enumeration,
  `_UNDERIVABLE` asserted empty).
- Commit-1 model-load leg (F1B-SPEC-01): every bundled model `load()`s on its unchanged float64 probe and
  finite-predicts on a float32 frame, proving the un-retrained models pass before the retrain lands.
- Full `-m "not e2e"` suite green on the merged tree before the commit gate; owner-run e2e where a retrain
  needs real data.
- The wheel-bundles-weights verification for each regenerated weights dir (published wheel carries the new
  npz/booster + SHA256SUMS); `.gitattributes binary` pin present for any byte-exact-load weights dir.

## 7. Commit structure

Two commits, **each with an explicit human-approval gate before it is made** (no auto-commit; no
micro-commits; each commit is a fully-tested coherent state with the suite green):

- **Commit 1 — schema migration (no weights):** float32 storage + compute-boundary upcasts + id→category +
  `id_compat` extension + all schema/producer changes + re-materialized tracking-feature goldens +
  double-regenerated model-output goldens (float32-input/float64-weights, §4.5) + SB360 audit re-derivation +
  #2 atomic-dedup residue + version/CHANGELOG/AGENTS.md/`docs/context`/ADR-106/TODO.
  **The probe builders (`canonical_probe_frame` / feature-contract probe) stay float64 — commit-1 does NOT
  touch them (they are schema-independent) — so the un-retrained bundled models load and finite-predict
  clean.** A commit-1 CI leg asserts exactly that: every bundled model `load()`s (chirality + feature-contract
  pass on its unchanged float64 probe) and finite-predicts on a float32 frame. Suite green. **Stop, present
  the diff, await explicit owner approval to commit.**
- **Commit 2 — DGX-trained bundled weights (clean provenance):** cast each retrained model's probe builders
  to float32 explicitly (in the model module, not via the schema; §4.3 step 3) + the 11(+) retrained
  frame-geometry model artifacts + regenerated chirality/feature-contract fingerprints + `SHA256SUMS`,
  `training_commit` = commit-1 SHA + second-pass model-output golden regen (float32-weights, §4.5) + the
  `docs/research/f1b_float32/` measurement artifact. Suite green. **Stop, present, await explicit owner
  approval.**

The TF-53/57/61 two-phase precedent. The plan must place the human-approval gate immediately before each
commit; a plan that lands code without that gate is a blocker.

## 8. Risks

- **numba float32 leakage.** If any kernel forgets the float64 upcast, numba specializes a float32 signature
  and the ADR-076 bit-identity contract silently shifts. Mitigation: the §4.1 upcast enumeration gate + the
  ghost-GK numpy/numba parity gate re-run on float64-upcast inputs.
- **pandas float32 auto-upcast.** Arithmetic with a float64 scalar upcasts to float64 (harmless — storage
  stays float32 until reassigned). Risk is an accidental `frames["x"] = <float64 result>` reassigning float64
  back onto the column; the schema-pin test catches a reverted dtype.
- **category id operations.** A raw `.astype(str)` / dict-key / groupby on a category id that skips the
  `id_compat` decategorize is the ADR-019 trap in a new dtype; the category-axis gate extension is the
  backstop.
- **Retrain corpus provenance.** Each bundled public arm must retrain on a public-only corpus
  (`assert_public_corpus`), clean tree; a dirty-tree or restricted-corpus retrain is refused.
- **`gk_completion` mis-classification.** Resolved by measurement (§4.3), not assumption.

## 9. Non-goals

- Native float32 compute / SIMD CPU win (separate measured decision).
- Part B off-frame provenance (dropped, §3).
- float32 on SPADL actions / event-only model retrains.
- GS-dribble re-fit (#3), xt-gk-v2 ρ retrain (#4), and all consumer-side (AtomicVAEP/VAEP/xT) retrains.

## 10. Numbering

Version / PR-S / tag assigned at commit-prep (not hardcoded). Target ADR: **ADR-106**
(`docs/superpowers/adrs/ADR-106-f1b-float32-frames-and-id-category.md`), written in commit-1.
