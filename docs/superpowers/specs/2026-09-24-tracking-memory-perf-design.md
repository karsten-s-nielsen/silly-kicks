# Tracking-scorer memory & performance (F1a/F5/F6/F2/F3) — Design Spec

| Field | Value |
|---|---|
| **Date** | 2026-09-24 |
| **Author** | silly-kicks session (Karsten) |
| **Status** | Draft (awaiting review) |
| **Repo HEAD at authoring** | `628f397` (v4.124.0) |
| **Target version** | 4.125.0 |
| **Extends** | ADR-008 (pitch-control cache), ADR-058 (nullable-id frame schema), ADR-063 (velocity tiering / speed_source), ADR-069 (detection-aware visibility), ADR-076 (numba parity-gate pattern), ADR-081 (rest_defense spearman hard constraint) |

## Context

The luxury-lakehouse `compute_tracking_marts` drain OOM-killed (exit 137) every worker on a Gradient
Sports match. Root-cause profiling (handoff `D:\Development\_reviews\2026-09-24-sk-tracking-scorer-memory-perf-handoff.md`,
Databricks serverless 16 GB, GS match `10502` p1 — 1.91 M rows / 84 K frames / 16 players, sk 4.123.0)
showed the per-unit scoring path peaks **~3.4 GB** and `rest_defense` runs **361 s**. None of it is
lakehouse-specific — it is inherent to the silly-kicks tracking frame schema and the pitch-control
scorers, so fixing it in silly-kicks benefits **every** tracking consumer/provider (off_ball_runs,
defensive_credit, rest_defense, gkdv, territorial_defense). All findings were verified against the live
tree at `628f397`.

Two levers dominate: **the frame schema** (1.27 GB held for every scorer; ~84 % is low-cardinality/
constant object strings) and **pitch control** (per-frame CPU + retained per-player surfaces).

**This cycle** ships the five **no-retrain** findings. Deferred (owner-approved, separate cycles):
- **F4** (bounded-memory streaming) → *next* cycle; ownership decided = **"Both"** (sk-internal
  streaming default + an exposed per-batch primitive), re-scoped after F1-3 shrink the baseline. Not
  designed here.
- **F1b** (float32 positions + id→category) → a *later* cycle: it changes numeric output (float32) and
  crosses the ADR-019 `id_compat` surface (id→category), so it is a **VAEP/tracking retrain +
  re-materialize + golden/chirality/feature-contract regeneration** event. Out of scope here.

## Decision

One cycle, one PR, one ADR (**ADR-103**), branch `feat/tracking-memory-perf`, version 4.125.0.
**Implemented: F1a (static-subset category + drop `confidence`), F5 (cache LRU), F6 (docs + contract
test), F2 (batched pitch-control API).** **F3 DROPPED** (memory regression, no CPU win — Change 5); F4
and F1b are separate later cycles (Non-goals). Fully **additive / value-neutral** — off-frame provenance,
float32, and id→category all deferred to F1b (they force retrains). **No retrain, no re-materialize**
(every change is value-neutral or byte-identical, parity-gated).

---

### Change 1 (F1a) — Frame schema: category the STATIC low-cardinality columns; drop the dead column

The oriented frame is 1268 MB for 1.91 M rows; object columns are ~84 %. **Revised after implementation
(r3): `category` applies ONLY to STATIC, set-once low-cardinality columns.** The full-category attempt
(all 6 low-card object columns) hit a measured 22-failure ripple: **`category` is NOT transparent to
`setitem` OR `.fillna(new_value)`** — assigning a value not already in the categories raises
`TypeError: Cannot setitem on a Categorical with a new category`. Three columns are MUTATED post-build:
`team_attacking_direction` (orientation `.loc="ltr"/"rtl"`, reflect swap), `speed_source` (velocity
`.loc="derived"`), `visibility` (`_truthy_bool` `fillna("")`, `_gk_geometry.py:316`). Making those
`category` would require full-domain `CategoricalDtype` + a category-preserving orient/velocity/reflect/
`_truthy_bool` chain — broad, and permanently fragile for the last ~23 % of memory. **Declined.**
`category` suits static columns; forcing it on dynamic ones is over-engineering + fragility (see
`feedback_category_dtype_only_for_static_columns`).

**Column dispositions** (verified against `tracking/schema.py:10` + the builders + all consumers):

| column | now | → | rationale |
|---|---|---|---|
| `ball_state` | object | **category** | static, set-once; only READ post-build (`== "dead"`/`"alive"`) |
| `source_provider` | object | **category** | static; post-build only whole-column `= None` (object-safe) + value_counts (guarded) |
| `is_goalkeeper_source` | object | **category** | static; set during build, only READ post-build (keeper map, gkdv) |
| `_preprocessed_with` | object | **category** | static, match-constant; the **biggest single column** (~210 MB); only re-stamped whole-column (idempotent) + reflect-invariant. Cast at `_smoothing.py:138` |
| `team_attacking_direction` | object | **object (stays)** | DYNAMIC: orientation `.loc="ltr"/"rtl"` + reflect swap → category setitem crashes |
| `speed_source` | object | **object (stays)** | DYNAMIC: velocity `.loc="derived"` (`utils.py:133`) → category setitem crashes |
| `visibility` | object | **object (stays)** | DYNAMIC: `_truthy_bool` `fillna("")` (`_gk_geometry.py:316`) → category fillna crashes; no string domain |
| `confidence` | object | **DROP** | all-null across **all** providers; no value consumer |
| `player_id` / `team_id` | Int64/object | **unchanged** | id→category is F1b (ADR-019 `id_compat` cross-cut) |
| `x,y,z,speed,vx,vy,x_smoothed,y_smoothed` | float64 | **unchanged** | float32 is F1b (retrain trigger) |

No return-shape change → **additive, non-breaking, no consumer migration** (category is transparent to
the READS these static columns receive). **Expected reduction is ~2×** frame memory (the 4 static
columns ≈ 586 MB of object → a few MB, `confidence` −46 MB), NOT the ~6× of the full-category/id/float32
set — the dynamic-3 (~204 MB) + ids (~230 MB, F1b) + floats (F1b) stay. The bigger reduction is F1b's
job (it carries the retrain).

**Dropping `confidence`** touches its two schema-contract references (both are shape refs, NOT feature
value reads → no retrain): remove it from the empty-frame template `vaep/features/core.py:65` and the
reflection registry `reflection.py:140` (and its per-provider `out["confidence"] = None` stamps). Verify
the reflection registry-completeness meta-assertion (ADR-045) stays satisfied after removal.

**Schema + gate updates.** `TRACKING_FRAMES_COLUMNS` 20 → 19 (−`confidence`; the 3 static in-schema
columns become `category`, the dynamic 3 stay `object`); `_preprocessed_with` (preprocess-added) becomes
`category` at `_smoothing.py:138`; `tests/test_tracking_schema.py` (`628f397:17-63`, gated column-set +
per-variant dtype asserts, incl. `test_tracking_frames_columns_is_20_columns` → 19) updates. The
category columns are never id-compared, so `id_compat` is untouched — but the schema / id-dtype-invariance
gates traverse all columns, so **confirm category dtypes don't trip them** (add coverage if needed).
ADR-058's nullable-Int64 id contract is untouched.

**Category caveat — `value_counts` / `groupby` are NOT transparent (review r2, D2-SPEC-09).** `category`
is value-transparent for `==` / `.dropna()` / `pd.unique` / column-presence, but `Series.value_counts()`
on a category returns **all** categories incl. zero-count ones, and a `groupby(observed=False)` **keyed**
on a category enumerates all categories. So a value_counts on a now-category column changes its output
dict. **Audit every `value_counts` / category-keyed `groupby` on the 4 static category columns and guard each** (the 3 dynamic columns stay `object`, so they cannot hit the trap) (cast the
column to `object` at the call site — `.astype("object").value_counts()` — restoring only-observed keys,
provably byte-identical). Two confirmed sites: `tracking/utils.py:479` (`source_provider.value_counts()`
→ `LinkReport.per_provider_link_rate` would gain `{sportec:0.0,…}`) and `tracking/utils.py:833`
(`speed_source.value_counts(dropna=False)` → the velocity counts dict). The `_action_orientation.py:115/258`
groupbys are keyed on `team_id` (not a category) → unaffected. Guarding these keeps the LinkReport /
velocity-count outputs byte-identical, so value-neutrality is a **tested guarantee**, not a claim.

**Expected effect:** the 4 static columns ≈ 586 MB object → a few MB + `confidence` −46 MB ≈ **~2×**
frame reduction (~1268 MB → ~640 MB) per unit, all
consumers, **value-neutral** (no VALUE changes → no retrain; every scorer output AND the LinkReport /
velocity-count dicts byte-identical, once the value_counts sites are guarded).

**Resolved when:** the **4 static** columns are `category` (each materially smaller per-column, ≥5×) and
total frame memory drops (≈2×); `confidence` is gone (schema + template + reflection registry updated); a
schema test asserts the new dtypes (static→category, dynamic→object); every scorer output + the
`LinkReport` are byte-identical (category value-neutrality), proven on a fixture.

**Off-frame provenance is deferred to F1b** (with the float32 + id→category work): it is only worth doing
if the `source_provider`→variant migration is done together with its xt_gk retrain, which this cycle's
no-retrain boundary excludes.

### Change 2 (F5) — `PitchControlCache` bounded eviction

`PitchControlCache._store` (`tracking/pitch_control/_cache.py:45`) is a plain unbounded `dict`. Add an
optional `maxsize: int | None = None` LRU (`OrderedDict.move_to_end`/`popitem(last=False)`); `None`
preserves today's unbounded behaviour. "Aggregate-only" already exists — `surface(..., decompose=False)`
returns no per-player grids — so F5's only new work is the bound; document `decompose=False` as the
aggregate path.

**Resolved when:** `PitchControlCache(maxsize=N)` retains ≤ N surfaces (test asserts eviction + a
non-decomposed surface's smaller footprint); default `maxsize=None` is byte-identical to today.

### Change 3 (F6) — Document + demonstrate one shared `pitch_control_cache=` across the PC scorers

**Corrected after plan review (r1).** The handoff's premise (scorers recompute pitch control 3–4×) is
wrong: only **two** public scorers use pitch control — `value_off_ball_runs` (`_run_values.py:441`) and
`compute_rest_defense` (`restdefense/_compute.py:195`) — and **both already accept `pitch_control_cache=`.
`defensive_credit` uses NO pitch control** (verified: zero `pitch_control`/`compute_pitch_control`/
`.surface` in `tracking/defensive_credit/` or `add_defensive_credit`); `compute_gk_decision_value` is
lean (F7). So there is **nothing to thread** — F6's remaining work is (a) confirm both PC scorers accept
the kwarg (they do), and (b) **document + test the one-cache-per-unit pattern** (one bounded
`PitchControlCache` passed to off_ball + rest_defense so a unit's pitch control is computed once), with
F5's `maxsize` bound.

**Resolved when:** a documented test shows one shared bounded cache serving off_ball + rest_defense
computes each shared frame's surface once — asserted as **real reuse** (the shared cache's surface count
is strictly less than the sum of two independent per-scorer caches on the same unit), not merely that the
cache grew.

### Change 4 (F2) — Batched pitch-control API

No batched compute exists — `compute_pitch_control` (`pitch_control/_dispatch.py:31`) and
`compute_pitch_control_at_points` (`:117`) are single-frame; every scorer loops (`value_off_ball_runs`
`actions.iterrows()` → `cache.surface()` per action, `_run_values.py:538/565`). Add public
`compute_pitch_control_batch(frames, requests, *, method, decompose)` → N surfaces in one call (one
vectorised pass over the request set), plus a `PitchControlCache` batch-warm path. Additive; the
single-frame API is unchanged.

**Parity gate (load-bearing, ADR-076 precedent):** the batched path is **byte-identical**
(`np.array_equal`, max |Δ| 0.0) to the per-frame loop over the same requests, for both `spearman` and
`fernandez_bornn`, `decompose` True/False. No value change → no retrain.

**Resolved when:** the batched entry point exists, is parity-gated byte-identical, and a benchmark shows
a material speedup on a multi-hundred-frame batch vs the per-frame loop.

### Change 5 (F3) — `rest_defense` layer-2 batching — **DROPPED (owner-approved, r3)**

**Removed from this cycle.** Implementation established that `_score_samples` already computes exactly
**one** `spearman, decompose=True` surface per sample and reuses it within the sample (`_danger.py:113`),
and samples map to **distinct** frames — so warm-routing removes **no** redundant work. Worse, warming
all ~254 decompose surfaces into a shared cache **holds them all at once** (~330 MB) versus one-at-a-time
today → a **memory regression** with **no** CPU win. The real 361 s→60 s requires a **vectorized
cross-sample spearman kernel** — the piece explicitly deferred (F2's batched API is its seam). So F3's
warm-routing is not done; `_score_samples` is unchanged (byte-identical). The vectorized kernel is a
follow-up (next cycle, with F4). No change to `compute_rest_defense`.

### Testing (TDD, non-vacuous)

- **F1a:** a realistic-cardinality synthetic (or committed) tracking half asserts (i) each static
  low-card column drops **≥5× per-column** as `category` vs `object`, and `memory_usage(deep=True).sum()`
  is materially smaller (total ≈2× on real frames — the dynamic 3 + ids + float coords stay `object`, so
  the per-column win does not multiply out to a per-frame ≥3×); (ii) `confidence` absent from
  `frames.columns`; (iii) the **4 static** columns (`ball_state`, `source_provider`,
  `is_goalkeeper_source`, `_preprocessed_with`) are `category`, and the **3 dynamic** columns
  (`team_attacking_direction`, `speed_source`, `visibility`) stay `object` (category is not
  setitem/`fillna`-transparent — the governing rule); (iv)
  every scorer output (off_ball / defensive_credit / rest_defense / gk_decision) is byte-identical to the
  pre-change (object-dtype) frame — category value-neutrality; (v) the variant-selecting consumers
  (`_xt_gk` `pd.unique(source_provider)`, keeper-identity `is_goalkeeper_source` map) resolve identically
  on category vs object; (vi) **`LinkReport.per_provider_link_rate`** (utils.py:479) **and the
  `speed_source` velocity-counts dict** (utils.py:833) are byte-identical pre/post — the guard against the
  category `value_counts` zero-count trap (D2-SPEC-09). Reflection registry-completeness still passes
  after the `confidence` removal.
- **F5:** `maxsize=2` over 5 distinct surfaces retains 2 (LRU order asserted); `decompose=False` surface
  footprint < `decompose=True`; `maxsize=None` unchanged.
- **F6:** the two PC scorers (`value_off_ball_runs` + `compute_rest_defense`) both accept
  `pitch_control_cache=`, so one shared `PitchControlCache` serves both — cross-scorer reuse proven by
  **composition** (`inspect.signature` presence here + the cache's reuse/evict-by-canonical-key gates in
  `test_cache.py` / `test_batch.py::test_cache_warm_makes_surface_calls_hit`), not a heavy end-to-end
  `len(shared) < len(c_off)+len(c_rd)` fixture; plus `inspect.signature(add_defensive_credit)` has **no**
  `pitch_control_cache` param (defensive_credit is PC-free — the handoff's "3-4× recompute" premise was
  wrong, nothing to thread).
- **F2:** parity — batched == per-frame loop, `np.array_equal`, both methods × decompose T/F; a benchmark
  (structural op-count or wall, per the repo's perf convention) shows the speedup.
- **F3:** DROPPED (see Change 5) — no code change to `compute_rest_defense`, so no new test.

### ADR + release + rollout

- **ADR-103** (extends ADR-008/058/063/069/076/081): the frame-schema memory model (category-on-frame for
  every low-cardinality object column + drop-the-dead-column; why off-frame provenance is NOT done here —
  per-row contracts + the xt_gk retrain), the cache bound, the batched pitch-control API + parity
  contract, and the rest_defense batching. Records the F4/F1b deferrals + the F1b retrain trigger.
- **Version** 4.125.0 via `silly_kicks/_version.py` (ADR-079 single source).
- **TODO.md** updated in-commit.
- **Commit gate:** reach "ready to commit," present the diff/file list, **STOP for the owner's explicit
  approval** (CLAUDE.md — no `git commit`/`push`/PR without it). One coherent commit after the yes.
- **Consumer note (out of sk's tree):** the lakehouse realizes F1a only if its oriented-frame build uses
  sk's schema dtypes — its handoff profiled GS `player_id`/`team_id` as **object** while sk's GS schema is
  **Int64** (`test_gradientsports_variant_uses_nullable_int64_identifiers`). Flag back: confirm their
  frame build applies `GRADIENTSPORTS_TRACKING_FRAMES_COLUMNS`, else the win won't transfer.

## Alternatives considered

| Option | Why rejected |
|---|---|
| Category the id columns (`player_id`/`team_id`) too | Crosses the ADR-019 `id_compat` surface (50 modules); ADR-058 chose Int64 deliberately. Deferred to F1b |
| float32 positions this cycle | Changes pitch-control/xT/distance outputs → retrain + re-materialize + golden/chirality/feature-contract regeneration. Deferred to F1b |
| Drop `visibility` (per the handoff) | Real for SkillCorner + consumed by GK geometry/ghost-GK (ADR-069); dropping silently breaks SkillCorner |
| Move ANY provenance off-frame this cycle (`source_provider`/`is_goalkeeper_source`/`_preprocessed_with`) | Review (r1) verified per-row contracts: `source_provider` drives xt_gk variant selection (off-frame → **output change = retrain**), `is_goalkeeper_source` is per-keeper (scalar is lossy), `_preprocessed_with` is required on-frame by `derive_velocities`. Buys ~6 MB over a constant `category`. Deferred to F1b (with its xt_gk retrain) |
| `FrameProvenance` returned-object (`(frames, provenance)`) | Was the r0 draft; a BREAKING return-shape migration across every builder/consumer for ~6 MB. `category`-on-frame is additive, non-breaking, value-neutral. Rejected |
| `.attrs` for provenance | pandas drops `.attrs` across merge/groupby/concat — fragile (moot now nothing moves off-frame) |
| Swap `rest_defense` off `spearman` for speed | ADR-081 hard constraint (GK-blindness); batching is the win, not a method swap |

## Consequences

**Positive** — ~2× smaller frames (F1a) + a batched pitch-control API (F2, the future-kernel CPU seam)
+ one shared cross-scorer `PitchControlCache` (F6, off_ball + rest_defense overlap computed once) for
every tracking consumer/provider; the precondition for F4's bounded-memory path. (The larger
"rest_defense minutes → tens of seconds" wall-clock win belonged to F3's layer-2 batching, which is
**DROPPED** — the remaining rest_defense win is the F6 shared-cache reuse only, and the batched-kernel
speedup lands when the vectorized spearman kernel ships next cycle.)

**Negative / cost** — small: a dtype migration (object→category on the **4 static** low-card columns) +
drop `confidence` with
its 2 schema-contract refs (template + reflection registry). **No return-shape change, no consumer
migration** (category is transparent to `==`/`unique`/presence). Schema + `TRACKING_CATEGORICAL_DOMAINS`
+ `test_tracking_schema` updated.

**Neutral** — **No retrain, no re-materialize** (category is value-neutral once the two `value_counts`
sites are guarded — D2-SPEC-09; batching byte-identical, parity-gated). C4: `compute_pitch_control_batch`
is a primitive, not an action-coupled aggregator — action-coupled count unchanged (confirm).

## Non-goals (owner-approved deferrals)

- **F4** — bounded-memory streaming; next cycle; ownership = "Both"; re-scoped after F1-3.
- **F1b** — float32 positions + id→category + (if still wanted) off-frame provenance for
  `source_provider`/`is_goalkeeper_source`/`_preprocessed_with`; later cycle; retrain + re-materialize event.
- **F7** — gk_decision already lean; no action.

## Anchors (verify against the tree)

- `tracking/schema.py:10` `TRACKING_FRAMES_COLUMNS`, `:34-40` object cols, `:43/64` kloppy/GS variants;
  `TRACKING_CATEGORICAL_DOMAINS`.
- builders: `gradientsports.py:142/143`, `metrica.py:142/168/181`, `skillcorner.py:242/268/283`,
  `sportec.py:151/152` (confidence/visibility); `convert_to_frames` at `gradientsports.py:54`/
  `sportec.py:59`/`skillcorner.py:152`/`metrica.py:81`/`kloppy.py:39`.
- `tracking/preprocess/_smoothing.py:100/138` (`_preprocessed_with` write + idempotency);
  `tracking/preprocess/_velocity.py:41` (`derive_velocities` REQUIRES `_preprocessed_with` on-frame).
- `tracking/_velocity_availability.py:30-59` (speed_source per-row contract); `tracking/_gk_geometry.py:262`
  (visibility consumer).
- **category consumers that must stay green:** `tracking/_xt_gk.py:344` (`pd.unique(frames["source_provider"])`
  → variant), `tracking/_gk_completion.py:276` + `xtgk/_retention.py:38` (variant keys),
  `keeper_identity.py:946-953` (source_provider link) + `:984-995` (per-`(game,team,player)`
  `is_goalkeeper_source` map), `gkdv/_engine.py:74/371` (`is_goalkeeper_source`), `tracking/utils.py:410/640`
  (source_provider).
- **category `value_counts` guard sites (D2-SPEC-09):** `tracking/utils.py:479`
  (`source_provider.value_counts()` → `LinkReport.per_provider_link_rate`), `tracking/utils.py:833`
  (`speed_source.value_counts(dropna=False)` → velocity counts). `_action_orientation.py:115/258` groupbys
  are `team_id`-keyed → unaffected.
- **confidence-drop refs:** `vaep/features/core.py:65` (empty-frame template) + `reflection.py:140`
  (`"confidence":"invariant"`).
- `tracking/pitch_control/_cache.py:45` (`_store` dict), `:63` `surface`; `pitch_control/_dispatch.py:31`
  `compute_pitch_control`, `:117` `compute_pitch_control_at_points`.
- `tracking/_run_values.py:441` `value_off_ball_runs(pitch_control_cache=)`, `:538/565` iterrows loop.
- `restdefense/_compute.py:131` `_score_samples`, `:195` `compute_rest_defense(pitch_control_cache=)`;
  `restdefense/_danger.py:38 _SPEARMAN`, `:79 layer2_metrics`, `:108-114` per-sample decompose surface.
- `tests/test_tracking_schema.py:17-63` (schema gate).
