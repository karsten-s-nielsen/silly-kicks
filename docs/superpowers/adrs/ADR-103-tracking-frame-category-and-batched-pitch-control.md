# ADR-103: Tracking-frame static-column `category` + batched pitch-control (memory & perf)

| Field | Value |
|---|---|
| **Date** | 2026-09-24 |
| **Status** | Accepted |
| **Deciders** | Karsten S. Nielsen (owner), silly-kicks session |
| **Extends** | ADR-008 (pitch-control cache), ADR-058 (nullable-id frame schema), ADR-063 (velocity / speed_source), ADR-069 (detection-aware visibility), ADR-076 (numba parity-gate), ADR-081 (rest_defense spearman) |

## Context

A luxury-lakehouse `compute_tracking_marts` drain OOM-killed (exit 137) on a Gradient Sports match.
Profiling one GS half (1.91 M rows / 84 K frames) showed the per-unit scoring path peaks ~3.4 GB and
`rest_defense` runs 361 s. Root causes are in silly-kicks, not the lakehouse: the tracking frame schema
holds ~84 % of its 1268 MB/unit as low-cardinality/constant **object** strings, and the pitch-control
scorers have no batched compute. Six findings (F1a frame schema, F2 batched pitch control, F3
rest_defense, F4 bounded-memory streaming, F5 cache bound, F6 shared cache) were handed off. This cycle
ships the **no-retrain** subset; F4 and the retrain-forcing pieces are deferred.

## Decision

`category`-dtype the **static, set-once** low-cardinality frame columns (`ball_state`,
`source_provider`, `is_goalkeeper_source`, `_preprocessed_with`), drop the all-null `confidence`, add a
bounded LRU to `PitchControlCache`, and add a parity-gated batched `compute_pitch_control_batch`. All
changes are value-neutral or byte-identical — **no VAEP/tracking retrain, no re-materialize.**

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. `category` ALL 6 low-card columns (full-domain `CategoricalDtype`) | ~2.4× frame reduction | `category` is not transparent to `setitem`/`.fillna(new)`; the dynamic 3 (`team_attacking_direction`/`speed_source`/`visibility`) are mutated post-build (orient/velocity/`_truthy_bool`) → 22-failure ripple; needs a category-preserving orient/velocity/reflect chain = broad + permanently fragile | Rejected — over-engineering + fragility for the last ~23 % of memory |
| B. Off-frame `FrameProvenance` for the constant provenance | Purest schema | `source_provider` is read per-row for xt_gk variant selection (output change = retrain); `is_goalkeeper_source` is per-keeper (scalar lossy); `_preprocessed_with` required on-frame by `derive_velocities` | Deferred to F1b (with its retrain) |
| C. float32 positions + id→category | The bulk of the memory | Changes pitch-control/xT numeric output + crosses ADR-019 `id_compat` → retrain + re-materialize + golden/chirality regeneration | Deferred to F1b |
| D. F3 warm-route rest_defense through a shared cache | Plan's stated CPU fix | `_score_samples` already computes ONE surface/sample + reuses it; samples are distinct frames → no redundant work removed; warming holds all ~254 decompose surfaces at once (~330 MB) → **memory regression**, no CPU win | Rejected — real win needs a vectorized cross-sample kernel (deferred) |
| E. (chosen) `category` the STATIC 4 + drop `confidence` + F5 LRU + F2 batch API | ~2× frame reduction, zero mutation ripple, zero fragility, byte-identical | Smaller win than A/C | — |

**The durable rule:** `category` suits **static** low-cardinality columns; on **dynamic** (post-build
mutated via `.loc`/`.fillna`) columns it crashes on a new-category value and must stay `object`. See
`feedback_category_dtype_only_for_static_columns`.

## Consequences

### Positive
- ~2× per-unit frame memory (the 4 static columns ≈ 586 MB object → a few MB; `confidence` −46 MB), for
  every tracking consumer/provider, value-neutrally.
- A public byte-identical batched `compute_pitch_control_batch` (+ `PitchControlCache.warm`) — the seam
  a future vectorized spearman kernel plugs into; a bounded `PitchControlCache(maxsize=…)` lets a
  whole-unit scorer cap peak memory.
- `value_counts` on the `category` `source_provider` is guarded (`utils.py:479` casts to object) so
  `LinkReport.per_provider_link_rate` stays byte-identical (no zero-count categories).

### Negative / cost
- Two schema-contract refs updated on the `confidence` drop (empty-frame template, reflection registry).
- `category` non-transparency to `setitem`/`.fillna` is a standing constraint on the 4 static columns
  (a future mutation of one must cast to object first) — accepted as fail-loud + documented.

### Neutral / deferred
- **F3 dropped** (memory regression, no CPU win); the rest_defense 361 s→60 s needs a **vectorized
  cross-sample spearman kernel** — next cycle, with **F4** (bounded-memory streaming).
- **F1b** (float32 positions + id→category + off-frame provenance) is the retrain-carrying cycle for the
  bigger memory win.
- No C4 change (`compute_pitch_control_batch` is a primitive, not an action-coupled aggregator).
